"""Trace ID 中间件。

从请求头 ``X-Trace-ID`` 提取 trace_id，注入到 ``ContextVar``，
确保同一请求的所有异步协程都能获取到相同的 trace_id 用于日志追踪。

如果请求头中没有 ``X-Trace-ID``，则自动生成 UUID。
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from deerflow.core.context import trace_id_ctx_var


class TraceMiddleware(BaseHTTPMiddleware):
    """Middleware that extracts/injects trace_id and sets ContextVar."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 从请求头提取 trace_id，没有则自动生成
        trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())

        # 设置 ContextVar（贯穿所有异步协程）
        token = trace_id_ctx_var.set(trace_id)

        # 在 request.state 中保留，方便后续读取
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
            # 在响应头中回传 trace_id，方便前端定位
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            # 请求结束后重置 ContextVar，避免影响下一个请求
            trace_id_ctx_var.reset(token)
