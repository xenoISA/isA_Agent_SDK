#!/usr/bin/env python3
"""
System Resource Protection Circuit Breaker
Lightweight system resource monitoring and protection
"""
import psutil
import asyncio
import time
from typing import Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from isa_agent_sdk.utils.logger import api_logger

class SystemResourceMonitor:
    """Lightweight system resource monitor"""
    
    def __init__(self):
        self.last_check = 0
        self.check_interval = 5.0  # 5秒检查一次
        self.cached_status = {
            "memory_percent": 0,
            "cpu_percent": 0,
            "healthy": True,
            "timestamp": time.time()
        }
        
        # 阈值配置 - Relaxed to prevent shutdowns
        self.thresholds = {
            "memory_critical": 98,   # 98%内存使用率 (was 90%)
            "memory_warning": 95,    # 95%内存使用率 (was 80%)
            "cpu_critical": 99,      # 99%CPU使用率 (was 95%)
            "cpu_warning": 98        # 98%CPU使用率 (was 85%)
        }
        
        # 初始化CPU监控（需要第一次调用）
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        
    def get_system_status(self) -> Dict:
        """获取系统状态（带缓存）"""
        now = time.time()
        
        # 使用缓存避免频繁系统调用
        if now - self.last_check < self.check_interval:
            return self.cached_status
        
        try:
            # 获取系统资源使用情况
            memory = psutil.virtual_memory()
            
            # 优化：使用非阻塞方式获取CPU使用率，避免100ms阻塞
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # 如果获取失败，使用上次的值
            if cpu_percent is None:
                cpu_percent = self.cached_status.get("cpu_percent", 0)
                api_logger.debug("CPU percent unavailable, using cached value")
            
            # 获取进程级别的CPU使用率（用于调试）
            try:
                process = psutil.Process()
                process_cpu = process.cpu_percent()
            except Exception:
                process_cpu = 0
            
            # 判断健康状态
            memory_critical = memory.percent > self.thresholds["memory_critical"]
            cpu_critical = cpu_percent > self.thresholds["cpu_critical"]
            
            healthy = not (memory_critical or cpu_critical)
            
            # 只在状态变化时记录日志，避免重复告警
            previous_memory_critical = self.cached_status.get("memory_critical", False)
            previous_cpu_critical = self.cached_status.get("cpu_critical", False)
            
            self.cached_status = {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_percent": cpu_percent,
                "process_cpu_percent": process_cpu,  # 新增：进程CPU使用率
                "healthy": healthy,
                "memory_critical": memory_critical,
                "cpu_critical": cpu_critical,
                "timestamp": now
            }
            
            self.last_check = now
            
            # 只在状态变化时记录日志
            if memory_critical and not previous_memory_critical:
                api_logger.warning(f"🚨 Memory usage critical: {memory.percent:.1f}%")
            if cpu_critical and not previous_cpu_critical:
                api_logger.warning(f"🚨 CPU usage critical: {cpu_percent:.1f}% (process: {process_cpu:.1f}%)")
                
        except Exception as e:
            api_logger.error(f"System resource check failed: {e}")
            # 发生错误时认为系统不健康，但保持上次的CPU值
            self.cached_status["healthy"] = False
            # 保持上次的CPU值，避免丢失监控数据
            if "cpu_percent" not in self.cached_status:
                self.cached_status["cpu_percent"] = 0
            
        return self.cached_status

class SystemProtectionMiddleware:
    """System resource protection middleware"""
    
    def __init__(self):
        self.monitor = SystemResourceMonitor()
        self.protection_enabled = False  # Disabled to prevent service interruptions
        
        # 保护策略
        self.protection_rules = {
            # 严重时拒绝所有非关键请求
            "critical": {
                "reject_paths": ["/api/chat", "/api/chat/test"],
                "allow_paths": ["/health", "/api/info", "/stats"],
                "response_code": 503,
                "message": "Service temporarily unavailable due to high system load"
            },
            # 警告时只拒绝高负载请求
            "warning": {
                "reject_paths": ["/api/chat/test"],
                "throttle_delay": 0.1  # 100ms延迟
            }
        }
    
    def _should_reject_request(self, path: str, status: Dict) -> Optional[Dict]:
        """判断是否应该拒绝请求"""
        if not self.protection_enabled or status["healthy"]:
            return None
            
        # 关键路径总是允许
        critical_paths = ["/health", "/api/info", "/stats"]
        if any(path.startswith(p) for p in critical_paths):
            return None
            
        # 系统严重负载
        if status.get("memory_critical") or status.get("cpu_critical"):
            rules = self.protection_rules["critical"]
            if any(path.startswith(p) for p in rules["reject_paths"]):
                return {
                    "code": rules["response_code"],
                    "message": rules["message"],
                    "details": {
                        "memory_percent": status["memory_percent"],
                        "cpu_percent": status["cpu_percent"],
                        "available_memory_gb": status.get("memory_available_gb", 0)
                    }
                }
        
        return None
    
    async def __call__(self, request: Request, call_next):
        """Middleware execution"""
        path = request.url.path
        
        # 跳过静态文件
        if path.startswith("/static") or path.startswith("/docs"):
            return await call_next(request)
        
        # 检查系统状态
        status = self.monitor.get_system_status()
        
        # 判断是否需要拒绝请求
        rejection = self._should_reject_request(path, status)
        if rejection:
            return JSONResponse(
                status_code=rejection["code"],
                content={
                    "error": "System Protection",
                    "message": rejection["message"],
                    "system_status": rejection["details"],
                    "retry_after": 30  # 建议30秒后重试
                },
                headers={
                    "Retry-After": "30",
                    "X-System-Memory": f"{status['memory_percent']:.1f}%",
                    "X-System-CPU": f"{status['cpu_percent']:.1f}%"
                }
            )
        
        # 警告级别添加延迟（非阻塞）
        warning_rules = self.protection_rules["warning"]
        if (not status["healthy"] and 
            "throttle_delay" in warning_rules and
            any(path.startswith(p) for p in warning_rules.get("reject_paths", []))):
            await asyncio.sleep(warning_rules["throttle_delay"])
        
        # 执行请求
        response = await call_next(request)
        
        # 添加系统状态头
        response.headers["X-System-Status"] = "healthy" if status["healthy"] else "degraded"
        response.headers["X-System-Memory"] = f"{status['memory_percent']:.1f}%"
        
        return response

# Global instance  
system_protection_middleware = SystemProtectionMiddleware()