"""聊天记录存储数据库配置。

独立于 checkpointer 和应用数据库，仅用于存储 message_history 表。
目前仅支持 PostgreSQL。"""

from pydantic import BaseModel, Field


class MsgHistoryDatabaseConfig(BaseModel):
    """聊天记录存储数据库配置。

    Attributes:
        type: 固定为 "postgres"。暂不支持其他后端。
        connection_string: PostgreSQL DSN，例如 ``postgresql://user:pass@host:5432/db``。
    """

    type: str = Field(
        default="postgres",
        description="数据库类型，固定为 postgres。",
    )
    connection_string: str = Field(
        ...,
        description="PostgreSQL 连接字符串 DSN。支持 $VAR 环境变量引用。",
    )
