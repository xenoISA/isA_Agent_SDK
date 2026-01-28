#!/usr/bin/env python3
"""
Lightweight Rate Limiting Middleware
Non-intrusive request rate limiting with memory-efficient sliding window
"""
import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import ipaddress

class SlidingWindowRateLimiter:
    """Memory-efficient sliding window rate limiter"""
    
    def __init__(self):
        # 使用deque实现滑动窗口，内存效率高
        self._windows: Dict[str, deque] = defaultdict(lambda: deque())
        self._last_cleanup = time.time()
        
    def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, Dict]:
        """Check if request is allowed under rate limit"""
        now = time.time()
        window_start = now - window_seconds
        
        # Get or create window for this key
        window = self._windows[key]
        
        # Remove expired entries (sliding window)
        while window and window[0] < window_start:
            window.popleft()
        
        # Check if limit exceeded
        current_count = len(window)
        if current_count >= limit:
            # Calculate reset time
            reset_time = int(window[0] + window_seconds) if window else int(now + 1)
            return False, {
                "current": current_count,
                "limit": limit,
                "reset_time": reset_time,
                "retry_after": reset_time - int(now)
            }
        
        # Add current request
        window.append(now)
        
        # Periodic cleanup (every 5 minutes)
        if now - self._last_cleanup > 300:
            self._cleanup_expired_windows(now)
            self._last_cleanup = now
        
        return True, {
            "current": current_count + 1,
            "limit": limit,
            "reset_time": int(now + window_seconds)
        }
    
    def _cleanup_expired_windows(self, now: float):
        """Clean up completely expired windows to save memory"""
        expired_keys = []
        for key, window in self._windows.items():
            if not window or (window and window[-1] < now - 3600):  # 1 hour old
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._windows[key]

class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self):
        self.limiter = SlidingWindowRateLimiter()
        
        # 可配置的限流规则 (requests per minute)
        self.rules = {
            # API endpoints
            "/api/chat": {"limit": 30, "window": 60, "per": "ip"},
            "/api/chat/test": {"limit": 10, "window": 60, "per": "ip"},
            
            # Health/info endpoints (more lenient)
            "/health": {"limit": 60, "window": 60, "per": "ip"},
            "/api/info": {"limit": 60, "window": 60, "per": "ip"},
            
            # Default fallback
            "_default": {"limit": 120, "window": 60, "per": "ip"}
        }
        
        # 白名单IP (本地开发)
        self.whitelist_ips = {
            "127.0.0.1", "::1", "localhost"
        }
    
    def _get_client_key(self, request: Request, rule: Dict) -> str:
        """Generate rate limiting key"""
        if rule["per"] == "ip":
            # Get real IP (考虑代理)
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            elif request.client:
                client_ip = request.client.host
            else:
                # Fallback: use a generic key if client is None (e.g., in tests or connection errors)
                client_ip = "unknown"
            
            # Check whitelist
            if client_ip in self.whitelist_ips:
                return None  # Skip rate limiting
            
            return f"ip:{client_ip}"
        
        elif rule["per"] == "session":
            # Use session_id from request body/query
            session_id = request.query_params.get("session_id")
            if not session_id and hasattr(request, "_json"):
                session_id = request._json.get("session_id")
            if session_id:
                return f"session:{session_id}"
            else:
                # Fallback to IP
                if request.client:
                    return f"ip:{request.client.host}"
                else:
                    return f"ip:unknown"
        
        # Default fallback
        if request.client:
            return f"ip:{request.client.host}"
        else:
            return f"ip:unknown"
    
    def _get_rule(self, path: str) -> Dict:
        """Get rate limit rule for path"""
        # Exact match first
        if path in self.rules:
            return self.rules[path]
        
        # Prefix match
        for rule_path, rule in self.rules.items():
            if rule_path != "_default" and path.startswith(rule_path):
                return rule
        
        return self.rules["_default"]
    
    async def __call__(self, request: Request, call_next):
        """Middleware execution"""
        path = request.url.path
        
        # Skip rate limiting for static files and certain paths
        if (path.startswith("/static") or 
            path.startswith("/docs") or 
            path.startswith("/openapi.json") or
            request.method == "OPTIONS"):
            return await call_next(request)
        
        # Get rate limit rule
        rule = self._get_rule(path)
        client_key = self._get_client_key(request, rule)
        
        # Skip if whitelisted
        if client_key is None:
            return await call_next(request)
        
        # Check rate limit
        allowed, info = self.limiter.is_allowed(
            client_key, 
            rule["limit"], 
            rule["window"]
        )
        
        if not allowed:
            # Rate limit exceeded
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rule['limit']} requests per {rule['window']} seconds",
                    "limit": info["limit"],
                    "current": info["current"],
                    "reset_time": info["reset_time"],
                    "retry_after": info["retry_after"]
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(max(0, info["limit"] - info["current"])),
                    "X-RateLimit-Reset": str(info["reset_time"]),
                    "Retry-After": str(info["retry_after"])
                }
            )
        
        # Add rate limit headers to successful response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(max(0, info["limit"] - info["current"]))
        response.headers["X-RateLimit-Reset"] = str(info["reset_time"])
        
        return response

# Global instance
rate_limit_middleware = RateLimitMiddleware()