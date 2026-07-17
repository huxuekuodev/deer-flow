from langchain.chat_models import BaseChatModel
from langfuse import Langfuse
from pydantic import BaseModel, ConfigDict

from deerflow.config.app_config import AppConfig


class GraphContext(BaseModel):
    """Graph context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    """App config. Required. 获取config.yaml的配置"""
    app_config: AppConfig

    """Plan 模型. Required. 用于生成计划"""
    plan_llm: BaseChatModel

    """Langfuse 客户端. Required. 用于调用Langfuse API"""
    langfuse_client: Langfuse
