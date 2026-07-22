"""请求级上下文管理器。

使用 ``contextvars.ContextVar`` 存储请求级 trace_id，
确保在同一次请求的所有异步协程中都能获取到相同的值。

用法::

    from deerflow.core.context import trace_id_ctx_var

    # 在请求入口设置
    trace_id_ctx_var.set("abc123")

    # 在任意深度嵌套的协程中读取
    trace_id = trace_id_ctx_var.get()
"""

from contextvars import ContextVar

# 请求级 trace_id — 从前端请求到 agent 执行贯穿全链路
trace_id_ctx_var: ContextVar[str] = ContextVar("trace_id", default="")
