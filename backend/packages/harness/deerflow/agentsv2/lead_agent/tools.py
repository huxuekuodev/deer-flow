"""Tool definitions for standard LangGraph ToolNode."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool


def make_tools(llm):
    """创建 ToolNode 使用的工具列表。"""

    @tool
    def weather(city: str, date: str = "") -> str:
        """查询某个城市的天气信息。"""
        prompt = f"请查询城市「{city}」的天气情况。"
        if date:
            prompt += f" 日期: {date}"
        result = llm.invoke(
            [
                SystemMessage(content="你是一个天气查询助手。根据城市名返回天气信息。"),
                HumanMessage(content=prompt),
            ]
        )
        return result.content

    @tool
    def general(query: str) -> str:
        """通用信息查询。可查询任何事实性信息。"""
        result = llm.invoke(
            [
                SystemMessage(content="你是一个通用信息查询助手。根据用户问题返回准确简洁的信息。"),
                HumanMessage(content=query),
            ]
        )
        return result.content

    return [weather, general]
