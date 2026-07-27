"""
DAG 模式的状态定义（Co-Sight 风格）。

关键变化：
  - 移除旧的顺序 phases (TodoItem / current_phase)
  - 用 plan_id 引用 PlanDocument（内存或 Redis 中的 DAG 计划）
  - active_steps: 记录当前正在执行的步骤（子图）
"""

from typing import Annotated

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ThreadState(AgentState):
    # LangGraph 消息列表（含 ToolMessages）
    messages: Annotated[list[BaseMessage], add_messages]

    # === DAG Plan 相关字段 ===

    # 当前计划的 ID（引用 PlanDocument）
    plan_id: str

    # 已完成且仍需传递的上下文摘要（Co-Sight 风格 step_notes 汇总）
    plan_context: str

    # 当前正在活跃执行的步骤索引列表（支持并行子图）
    active_steps: list[int]

    # 所有步骤是否已完成
    plan_completed: bool

    # 用户原始消息（用于 review_node 评审时判断结果是否满足需求）
    user_message: str
