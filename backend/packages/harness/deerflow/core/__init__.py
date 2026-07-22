"""请求级上下文管理 — trace_id 贯穿前端请求到 agent 执行。

提供 ``trace_id_ctx_var``（ContextVar）和 ``logger`` 用于全链路日志追踪。
"""

from .context import trace_id_ctx_var
from .log import logger

__all__ = [
    "trace_id_ctx_var",
    "logger",
]
