"""
Custom middleware for FastAPI application
"""

import time
import logging
import uuid
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import json

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""
    
    def __init__(self, app, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time,
            }
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "endpoints": {},
            "start_time": time.time()
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Increment total requests
        self.metrics["total_requests"] += 1
        
        # Track endpoint usage
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in self.metrics["endpoints"]:
            self.metrics["endpoints"][endpoint] = {"count": 0, "avg_time": 0}
        self.metrics["endpoints"][endpoint]["count"] += 1
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        self.metrics["response_times"].append(response_time)
        
        # Update endpoint average time
        endpoint_metrics = self.metrics["endpoints"][endpoint]
        endpoint_metrics["avg_time"] = (
            (endpoint_metrics["avg_time"] * (endpoint_metrics["count"] - 1) + response_time) /
            endpoint_metrics["count"]
        )
        
        # Track success/failure
        if 200 <= response.status_code < 400:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Keep only last 1000 response times for memory efficiency
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = time.time() - self.metrics["start_time"]
        avg_response_time = (
            sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            if self.metrics["response_times"] else 0
        )
        requests_per_minute = self.metrics["total_requests"] / (uptime / 60) if uptime > 0 else 0
        
        return {
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "average_response_time_ms": avg_response_time * 1000,
            "requests_per_minute": requests_per_minute,
            "uptime_seconds": uptime,
            "endpoints": self.metrics["endpoints"]
        }


class SecurityMiddleware(BaseHTTPMiddleware):
    """Basic security middleware"""
    
    def __init__(self, app, api_key: str = None):
        super().__init__(app)
        self.api_key = api_key
        self.protected_paths = {"/model/reload", "/metrics"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check API key for protected endpoints
        if self.api_key and request.url.path in self.protected_paths:
            provided_key = request.headers.get("X-API-Key")
            if provided_key != self.api_key:
                return Response(
                    content=json.dumps({"error": "Invalid or missing API key"}),
                    status_code=401,
                    media_type="application/json"
                )
        
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response


def setup_middleware(app, config: Dict[str, Any] = None):
    """
    Set up all middleware for the FastAPI application
    
    Args:
        app: FastAPI application instance
        config: Configuration dictionary
    """
    if config is None:
        config = {}
    
    # CORS middleware (should be first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=config.get("cors_credentials", True),
        allow_methods=config.get("cors_methods", ["*"]),
        allow_headers=config.get("cors_headers", ["*"]),
    )
    
    # Security middleware
    app.add_middleware(
        SecurityMiddleware,
        api_key=config.get("api_key")
    )
    
    # Metrics middleware
    metrics_middleware = MetricsMiddleware(app)
    app.add_middleware(MetricsMiddleware)
    
    # Store metrics middleware instance for access in routes
    app.state.metrics_middleware = metrics_middleware
    
    # Request logging middleware (should be last)
    app.add_middleware(
        RequestLoggingMiddleware,
        log_body=config.get("log_request_body", False)
    )
    
    logger.info("All middleware configured successfully")
    return app
