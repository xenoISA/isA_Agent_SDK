#!/usr/bin/env python3
"""
Connection Pool Circuit Breaker
Lightweight concurrent connection limiting
"""
import asyncio
import time
from typing import Dict, Optional
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from isa_agent_sdk.utils.logger import api_logger

class ConnectionLimiter:
    """Lightweight connection pool limiter"""
    
    def __init__(self):
        # 活跃连接跟踪
        self.active_connections = 0
        self.connection_lock = asyncio.Lock()
        
        # 每IP连接数跟踪
        self.ip_connections: Dict[str, int] = defaultdict(int)
        self.ip_lock = asyncio.Lock()
        
        # 配置
        self.limits = {
            "max_global_connections": 100,    # 全局最大连接数
            "max_connections_per_ip": 10,     # 每IP最大连接数
            "max_concurrent_requests": 50     # 最大并发请求数
        }
        
        # 统计
        self.stats = {
            "total_requests": 0,
            "rejected_requests": 0,
            "peak_connections": 0,
            "start_time": time.time()
        }
        
    async def acquire_connection(self, client_ip: str) -> Optional[Dict]:
        """获取连接许可"""
        async with self.connection_lock:
            # 检查全局连接数
            if self.active_connections >= self.limits["max_global_connections"]:
                self.stats["rejected_requests"] += 1
                return {
                    "rejected": True,
                    "reason": "max_global_connections",
                    "limit": self.limits["max_global_connections"],
                    "current": self.active_connections
                }
            
            # 检查每IP连接数
            if self.ip_connections[client_ip] >= self.limits["max_connections_per_ip"]:
                self.stats["rejected_requests"] += 1
                return {
                    "rejected": True,
                    "reason": "max_connections_per_ip",
                    "limit": self.limits["max_connections_per_ip"],
                    "current": self.ip_connections[client_ip]
                }
            
            # 获得许可
            self.active_connections += 1
            self.ip_connections[client_ip] += 1
            self.stats["total_requests"] += 1
            
            # 更新峰值
            if self.active_connections > self.stats["peak_connections"]:
                self.stats["peak_connections"] = self.active_connections
            
            return {"rejected": False}
    
    async def release_connection(self, client_ip: str):
        """释放连接许可"""
        async with self.connection_lock:
            if self.active_connections > 0:
                self.active_connections -= 1
            if self.ip_connections[client_ip] > 0:
                self.ip_connections[client_ip] -= 1
                
                # 清理空的IP记录
                if self.ip_connections[client_ip] == 0:
                    del self.ip_connections[client_ip]
    
    def get_stats(self) -> Dict:
        """获取连接统计"""
        uptime = time.time() - self.stats["start_time"]
        return {
            **self.stats,
            "active_connections": self.active_connections,
            "active_ips": len(self.ip_connections),
            "uptime_seconds": uptime,
            "requests_per_second": self.stats["total_requests"] / max(uptime, 1),
            "rejection_rate": (self.stats["rejected_requests"] / max(self.stats["total_requests"], 1)) * 100
        }

class ConnectionContextManager:
    """连接上下文管理器"""
    
    def __init__(self, limiter: ConnectionLimiter, client_ip: str):
        self.limiter = limiter
        self.client_ip = client_ip
        self.acquired = False
    
    async def __aenter__(self):
        result = await self.limiter.acquire_connection(self.client_ip)
        if result["rejected"]:
            raise ConnectionLimitExceeded(result)
        self.acquired = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            await self.limiter.release_connection(self.client_ip)

class ConnectionLimitExceeded(Exception):
    """Connection limit exceeded exception"""
    def __init__(self, info: Dict):
        self.info = info
        super().__init__(f"Connection limit exceeded: {info}")

class ConnectionLimitMiddleware:
    """Connection limiting middleware"""
    
    def __init__(self):
        self.limiter = ConnectionLimiter()
        
        # 白名单路径（不限制连接）
        self.excluded_paths = {
            "/health", "/api/info", "/stats", "/docs", "/openapi.json"
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP"""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        elif request.client:
            return request.client.host
        else:
            # Fallback: use a generic IP if client is None (e.g., in tests or connection errors)
            return "unknown"
    
    async def __call__(self, request: Request, call_next):
        """Middleware execution"""
        path = request.url.path
        
        # 跳过静态文件和排除路径
        if (path.startswith("/static") or 
            path in self.excluded_paths or
            request.method == "OPTIONS"):
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        
        try:
            # 使用连接上下文管理器
            async with ConnectionContextManager(self.limiter, client_ip):
                response = await call_next(request)
                
                # 添加连接统计头
                stats = self.limiter.get_stats()
                response.headers["X-Active-Connections"] = str(stats["active_connections"])
                response.headers["X-Connection-Limit"] = str(self.limiter.limits["max_global_connections"])
                
                return response
                
        except ConnectionLimitExceeded as e:
            # 连接数超限
            info = e.info
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Connection limit exceeded",
                    "message": "Too many concurrent connections. Please try again later.",
                    "reason": info["reason"],
                    "limit": info["limit"],
                    "current": info["current"],
                    "retry_after": 5
                },
                headers={
                    "Retry-After": "5",
                    "X-Connection-Limit": str(info["limit"]),
                    "X-Active-Connections": str(info["current"])
                }
            )

# Global instance
connection_limit_middleware = ConnectionLimitMiddleware()