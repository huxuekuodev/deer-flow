"""标准 LangGraph ToolNode 模式的状态定义。

PLAN 产出 todo_list + AIMessage.tool_calls（当前阶段）
  → ToolNode 自动读取并执行工具
  → OBSERVE 读 ToolMessages，更新 todo，产出下一阶段 tool_calls"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(default="")
    tool_name: str = Field(default="")
    tool_args: dict = Field(default_factory=dict)
    task_desc: str = Field(default="")
    done: bool = False
    result: str = Field(default="")


class TodoItem(BaseModel):
    phase_desc: str = Field(default="")
    todo: list[Task] = Field(default_factory=list)
    done: bool = False
    context: str = Field(default="")


class ThreadState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # 含 ToolMessages
    todo_list: list[TodoItem]
    current_phase: int  # 0-based，-1 全部完成
