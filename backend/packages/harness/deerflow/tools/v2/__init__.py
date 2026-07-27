"""
V2 工具注册表：按 stage (plan / execute / both) 分区获取工具。

与 v1 (tools/tools.py) 的区别:
  - 支持 ToolConfig.stage 字段：plan / execute / both
  - plan 工具：只在 plan_model_node 中使用（如 ask_clarification）
  - execute 工具：只在 step subgraph 中使用（如 weather, web_search, bash）
  - both 工具：两边都可用（默认，v1 兼容）
  - v1 的 get_available_tools() 不受影响
"""

import logging

from langchain.tools import BaseTool

from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.reflection import resolve_variable
from deerflow.tools.builtins import ask_clarification_tool
from deerflow.tools.sync import make_sync_tool_wrapper

logger = logging.getLogger(__name__)

# 内置工具及其所属 stage
BUILTIN_TOOLS: list[tuple[BaseTool, str]] = [
    (ask_clarification_tool, "plan"),  # ask_clarification 只在 plan 阶段
]

EXECUTION_BUILTIN_TOOLS: list[tuple[BaseTool, str]] = [
    # 未来执行阶段的固定内置工具放在这里
]


def _ensure_sync_invocable_tool(tool: BaseTool) -> BaseTool:
    """Attach a sync wrapper to async-only tools used by sync agent callers."""
    if getattr(tool, "func", None) is None and getattr(tool, "coroutine", None) is not None:
        tool.func = make_sync_tool_wrapper(tool.coroutine, tool.name)
    return tool


def _deduplicate_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """去重。"""
    seen_names: set[str] = set()
    result: list[BaseTool] = []
    for t in tools:
        if t.name not in seen_names:
            result.append(t)
            seen_names.add(t.name)
    return result


def load_config_tools(
    stage: str | None = None,
    exact: bool = False,
    *,
    app_config: AppConfig | None = None,
) -> list[BaseTool]:
    """从 config.yaml 加载工具，按 stage 过滤。

    Args:
        stage: "plan" | "execute" | None（不过滤）
        exact: True → 只匹配 stage 完全相同的工具
               False（默认）→ "both" 匹配 plan 和 execute
        app_config: AppConfig 实例

    Returns:
        符合条件的工具列表
    """
    config = app_config or get_app_config()

    if stage:
        if exact:
            tool_configs = [t for t in config.tools if getattr(t, "stage", "both") == stage]
        else:
            tool_configs = [t for t in config.tools if getattr(t, "stage", "both") in (stage, "both")]
    else:
        tool_configs = list(config.tools)

    loaded: list[BaseTool] = []
    for cfg in tool_configs:
        try:
            loaded_tool = resolve_variable(cfg.use, BaseTool)
            loaded.append(_ensure_sync_invocable_tool(loaded_tool))
        except Exception as e:
            logger.warning(f"Failed to load tool {cfg.name} ({cfg.use}): {e}")

    return loaded


def get_plan_tools(*, app_config: AppConfig | None = None) -> list[BaseTool]:
    """获取 plan 阶段可用的工具（显式绑定到 Plan LLM）。

    只包括配置中 stage="plan" 的工具。
    stage="both" 的工具属于执行阶段，不绑定到 Plan LLM（仅通过 describe 文字描述）。
    """
    config_tools = load_config_tools(stage="plan", exact=True, app_config=app_config)

    builtin_tools = [t for t, s in BUILTIN_TOOLS if s == "plan"]

    all_tools = _deduplicate_tools(config_tools + builtin_tools)
    logger.info(f"[v2] Plan tools: {len(all_tools)} total ({len(config_tools)} from config, {len(builtin_tools)} built-in)")
    return all_tools


def get_execute_tools(*, app_config: AppConfig | None = None) -> list[BaseTool]:
    """获取 execute 阶段可用的工具。

    包括：
      - 来自 config.yaml 中 stage="execute" 或 stage="both" 的工具
    """
    config_tools = load_config_tools(stage="execute", app_config=app_config)

    builtin_tools = [t for t, s in EXECUTION_BUILTIN_TOOLS if s == "execute"]

    all_tools = _deduplicate_tools(config_tools + builtin_tools)
    logger.info(f"[v2] Execute tools: {len(all_tools)} total ({len(config_tools)} from config, {len(builtin_tools)} built-in)")
    return all_tools


def describe_execute_tools(*, app_config: AppConfig | None = None) -> str:
    """生成执行能力描述，供 Plan agent 参考。

    ⚠️ 这里列出工具名和用途，但 Plan agent 不可以调用它们。
    Plan agent 唯一可调用的是 create_plan / update_plan / get_plan_status / ask_clarification。
    """
    tools = get_execute_tools(app_config=app_config)
    if not tools:
        return ""

    lines = [
        "## 参考：执行阶段可用的工具（仅用于规划参考，你不可调用）",
        "",
        "以下工具将在步骤执行阶段可用。你应根据它们设计步骤，",
        "但不要直接调用它们——你只能调用 create_plan 等规划工具。",
        "",
    ]

    for t in tools:
        name = t.name
        desc = t.description if hasattr(t, "description") else ""
        summary = desc.split("\n")[0].strip() if desc else name
        lines.append(f"- `{name}` — {summary}")

    lines.append("")
    lines.append("设计步骤时参考上面的工具能力。如果没有任何工具能满足用户需求，直接告知用户当前不支持，不要创建计划。")
    return "\n".join(lines)
