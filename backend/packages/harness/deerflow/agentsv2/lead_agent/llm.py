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
    """Resolve a runtime model name safely, falling back to default if invalid. Returns None if no models are configured."""
    app_config = app_config or get_app_config()
    default_model_name = app_config.models[0].name if app_config.models else None
    if default_model_name is None:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")

    if requested_model_name and app_config.get_model_config(requested_model_name):
        return requested_model_name

    if requested_model_name and requested_model_name != default_model_name:
        logger.warning(f"Model '{requested_model_name}' not found in config; fallback to default model '{default_model_name}'.")
    return default_model_name


def create_plan_llm(config: RunnableConfig, *, app_config: AppConfig | None = None):
    """
    创建计划LLM
    """
    cfg = _get_runtime_config(config)
    resolved_app_config = app_config or get_app_config()

    is_bootstrap = cfg.get("is_bootstrap", False)
    # thinking_enabled = cfg.get("thinking_enabled", False)

    requested_model_name: str | None = cfg.get("model_name") or cfg.get("model")
    agent_name = validate_agent_name(cfg.get("agent_name"))

    agent_config = load_agent_config(agent_name) if not is_bootstrap else None
    agent_model_name = agent_config.model if agent_config and agent_config.model else None
    model_name = _resolve_model_name(requested_model_name or agent_model_name, app_config=resolved_app_config)

    # Plan LLM uses with_structured_output() which internally sets tool_choice.
    # Thinking mode (e.g. DeepSeek) rejects tool_choice with
    # "Thinking mode does not support this tool_choice".
    # Force thinking off so structured output works.
    return create_chat_model(name=model_name, thinking_enabled=False, app_config=resolved_app_config, attach_tracing=False)


# 创建执行agent
def create_react_agent(config: RunnableConfig, *, app_config: AppConfig | None = None):
    """
    创建反应agent
    """
