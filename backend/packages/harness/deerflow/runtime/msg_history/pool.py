"""PostgreSQL 连接池管理。

复用 ``deerflow.runtime.checkpointer.async_provider._build_postgres_pool``
构建连接池，包装为 async context manager 供 FastAPI lifespan 使用。
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator

from deerflow.config.msg_history_config import MsgHistoryDatabaseConfig

logger = logging.getLogger(__name__)


def _build_postgres_pool(conn_string: str):
    """Build an AsyncConnectionPool with TCP keepalive and connection checking."""
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    return AsyncConnectionPool(
        conn_string,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 60,
            "keepalives_interval": 10,
            "keepalives_count": 6,
        },
        check=AsyncConnectionPool.check_connection,
    )


@contextlib.asynccontextmanager
async def make_msg_history_pool(config: MsgHistoryDatabaseConfig) -> AsyncIterator:
    """Async context manager 创建并管理 msg_history 数据库连接池。

    用法::

        async with make_msg_history_pool(config) as pool:
            app.state.msg_history_pool = pool
            yield
            # 退出时自动关闭 pool
    """
    pool = _build_postgres_pool(config.connection_string)
    logger.info(
        "msg_history pool created: %s",
        config.connection_string.split("@")[-1] if "@" in config.connection_string else config.connection_string,
    )
    async with pool:
        yield pool
