"""Tool definitions for step execution in StepSubgraph ReAct loop."""

from langchain_core.tools import tool


def make_execution_tools():
    """创建步骤执行工具列表。"""

    @tool
    def weather(city: str, date: str = "") -> str:
        """查询某个城市的天气信息。"""
        return f"天气查询: {city}, 日期: {date if date else '今日'}"

    return []


def describe_execution_tools() -> str:
    """以可读文本描述所有执行工具的能力清单，供 Plan agent 参考。

    Plan agent 根据此清单了解执行阶段有什么能力可用，从而生成合理的步骤描述。
    此描述会注入 create_plan 的 system prompt，确保 Planner 知道执行器能做什么。
    """
    tools = make_execution_tools()

    if not tools:
        return "当前没有可用的执行工具。"

    lines = ["## 可用的执行能力（参考：执行时可以使用以下工具完成任务）", ""]
    for t in tools:
        name = t.name
        desc = t.description if hasattr(t, "description") else ""
        summary = desc.split("\n")[0].strip() if desc else name
        lines.append(f"- **{name}**: {summary}")

    lines.append("")
    lines.append("如果上方的工具列表无法满足用户的需求，请直接告知用户当前不支持该功能，不要创建计划。")

    return "\n".join(lines)
