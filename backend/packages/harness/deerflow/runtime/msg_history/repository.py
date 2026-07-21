"""消息历史写入操作。

提供 ``record_message()`` 用于向 ``message_history`` 表写入记录。
只包含写入操作，读取通过后续 API 层实现。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# fmt: off
_INSERT_SQL = """
INSERT INTO message_history
    (user_id, thread_id, run_id, role, content, model_name, metadata)
VALUES
    ($1,    $2,        $3,    $4,   $5,      $6,        $7)
"""
# fmt: on


async def record_message(
    pool,
    *,
    user_id: str,
    thread_id: str,
    run_id: str | None = None,
    role: int,
    content: str,
    model_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """向 ``message_history`` 表写入一条消息记录。

    Args:
        pool: 从 ``make_msg_history_pool`` 获取的连接池实例。
        user_id: 用户标识（``get_effective_user_id()``）。
        thread_id: 会话标识（对应 LangGraph thread_id）。
        run_id: 可选的运行标识（对应 RunRecord.run_id）。
        role: 角色（1=user, 2=assistant）。
        content: 消息纯文本内容。
        model_name: 可选的模型名称（仅 assistant 消息需要）。
        metadata: 可选的扩展元数据（如 token_usage）。
    """
    try:
        async with pool.connection() as conn:
            await conn.execute(
                _INSERT_SQL,
                (user_id, thread_id, run_id, role, content, model_name, metadata),
            )
    except Exception:
        logger.exception(
            "Failed to record message: user_id=%s thread_id=%s role=%s",
            user_id,
            thread_id,
            role,
        )
        raise
