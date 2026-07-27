import logging

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse

from deerflow.config.agents_config import load_agent_config, validate_agent_name
from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.models import create_chat_model

logger = logging.getLogger(__name__)
langfuse_client = Langfuse()


def _get_runtime_config(config: RunnableConfig) -> dict:
    """Merge legacy configurable options with LangGraph runtime context."""
    cfg = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, dict):
        cfg.update(context)
    return cfg


def _resolve_model_name(requested_model_name: str | None = None, *, app_config: AppConfig | None = None) -> str:
    """Resolve a runtime model name safely, falling back to default if invalid."""
    app_config = app_config or get_app_config()
    default_model_name = app_config.models[0].name if app_config.models else None
    if default_model_name is None:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")

    if requested_model_name and app_config.get_model_config(requested_model_name):
        return requested_model_name

    if requested_model_name and requested_model_name != default_model_name:
        logger.warning(f"Model '{requested_model_name}' not found in config; fallback to default model '{default_model_name}'.")
    return default_model_name


def create_llm(config: RunnableConfig, *, app_config: AppConfig | None = None):
    """
    创建计划 + 执行 LLM。
    关闭 thinking 以支持 structured output 和 tool binding。
    """
    cfg = _get_runtime_config(config)
    resolved_app_config = app_config or get_app_config()

    is_bootstrap = cfg.get("is_bootstrap", False)
    requested_model_name: str | None = cfg.get("model_name") or cfg.get("model")
    agent_name = validate_agent_name(cfg.get("agent_name"))

    agent_config = load_agent_config(agent_name) if not is_bootstrap else None
    agent_model_name = agent_config.model if agent_config and agent_config.model else None
    model_name = _resolve_model_name(requested_model_name or agent_model_name, app_config=resolved_app_config)

    llm = create_chat_model(
        name=model_name,
        thinking_enabled=False,  # Thinking mode does not support tool_choice / structured output
        app_config=resolved_app_config,
        attach_tracing=False,
    )
    return llm


def create_execution_llm(config: RunnableConfig, *, app_config: AppConfig | None = None):
    """
    创建步骤执行用的 LLM（支持 thinking）。
    与 plan_llm 不同，执行 LLM 需要 thinking 能力来推理工具调用。
    """
    cfg = _get_runtime_config(config)
    resolved_app_config = app_config or get_app_config()

    requested_model_name: str | None = cfg.get("model_name") or cfg.get("model")
    agent_name = validate_agent_name(cfg.get("agent_name"))

    agent_config = load_agent_config(agent_name) if not cfg.get("is_bootstrap", False) else None
    agent_model_name = agent_config.model if agent_config and agent_config.model else None
    model_name = _resolve_model_name(requested_model_name or agent_model_name, app_config=resolved_app_config)

    llm = create_chat_model(
        name=model_name,
        thinking_enabled=True,  # 执行步骤需要 thinking
        app_config=resolved_app_config,
        attach_tracing=True,
    )
    return llm
