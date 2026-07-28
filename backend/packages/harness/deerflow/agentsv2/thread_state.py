"""
V2 状态定义，支持澄清 + 策划 + 执行 + 审查循环。

状态字段：
  - messages: 消息列表
  - plan_tasks: List[SubTask] — 计划中的子任务列表（持久化在 ThreadState 中）
  - completed: bool — 是否全部完成
  - user_message: str — 原始用户问题
  - final_answer: str — 最终的 AIMessage.content
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from deerflow.agentsv2.subtask import SubTask


class ThreadState(TypedDict, total=False):
    # LangGraph 消息列表
    messages: Annotated[list[BaseMessage], add_messages]

    # 子任务列表（Plan agent 输出）
    plan_tasks: list[SubTask]

    # 完成标记
    completed: bool

    # 用户原始消息
    user_message: str

    # 最终答案
    final_answer: str
