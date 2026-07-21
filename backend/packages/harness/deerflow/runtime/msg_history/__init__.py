"""聊天记录存储模块。

提供 PostgreSQL 连接池管理和消息写入操作。
独立于 LangGraph checkpointer，仅用于存储用户与 AI 的对话记录。

消息只记录：
- 用户发送的消息（role=user）
- AI 最终回复（role=assistant）
- 中间过程由 LangGraph checkpointer 在 checkpoint 表中管理。
"""

from __future__ import annotations

from .pool import make_msg_history_pool
from .repository import record_message

__all__ = [
    "make_msg_history_pool",
    "record_message",
]
