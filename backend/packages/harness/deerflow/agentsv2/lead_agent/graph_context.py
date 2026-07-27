from langchain.chat_models import BaseChatModel
from langfuse import Langfuse
from pydantic import BaseModel, ConfigDict, Field

from deerflow.agentsv2.plan_storage import PlanStorage
from deerflow.config.app_config import AppConfig


class GraphContext(BaseModel):
    """Graph runtime context (LangGraph Runtime 依赖注入)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    app_config: AppConfig
    """全局配置。"""

    plan_llm: BaseChatModel = Field(
        default=None,
        exclude=True,
        description="用于计划和执行的 LLM。",
    )

    langfuse_client: Langfuse = Field(
        default=None,
        exclude=True,
        description="Langfuse 追踪客户端。",
    )

    plan_storage: PlanStorage = Field(
        default=None,
        exclude=True,
        description="Plan 存储（内存或 Redis）。",
    )
