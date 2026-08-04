"""
V2 状态定义，支持澄清 + 策划 + 执行 + 审查循环。

plan_tasks 使用 merge_plan_tasks reducer：
  - 每个节点返回的 plan_tasks 会与现有列表 merge
  - 相同 plan_id 的 subtask 状态字段（step_statuses/result/blocked_message）会被更新
  - 不可变字段（name/desc/deps/execution_agent/sort）不会被覆盖
  - 新的 subtask 会被追加
  - 新计划（create 场景）由 plan_model_node 用 Overwrite 整体替换，绕过此 reducer
"""

from copy import deepcopy
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict

from deerflow.agentsv2.subtask import SubTask

# 状态变更时允许覆盖的字段（其他字段不可变）
_MUTABLE_FIELDS = {"step_statuses", "result", "blocked_message"}


def merge_plan_tasks(current: list[SubTask], update: list[SubTask]) -> list[SubTask]:
    """Merge plan_tasks：相同 plan_id 只更新状态字段，新 plan_id 追加。

    对于已存在的任务（相同 plan_id），仅更新 step_statuses / result / blocked_message，
    name / desc / deps / execution_agent / sort 保持不变。
    新 plan_id 的任务直接追加。
    """
    if not current:
        return [deepcopy(t) for t in (update or [])]
    if not update:
        return current

    existing = {t.plan_id: t for t in current}
    changed = False

    for t in update:
        if t.plan_id in existing:
            # 已有任务：仅更新可变字段
            orig = existing[t.plan_id]
            for field in _MUTABLE_FIELDS:
                new_val = getattr(t, field, None)
                if new_val is not None and new_val != getattr(orig, field, None):
                    setattr(orig, field, deepcopy(new_val) if isinstance(new_val, (dict, list)) else new_val)
                    changed = True
        else:
            # 新任务：整体追加
            existing[t.plan_id] = deepcopy(t)
            changed = True

    if changed:
        return list(existing.values())
    return current


class ThreadState(TypedDict, total=False):
    # LangGraph 消息列表
    messages: Annotated[list[BaseMessage], add_messages]

    # 子任务列表（Plan agent 输出），使用自定义 merge reducer
    plan_tasks: Annotated[list[SubTask], merge_plan_tasks]

    # 完成标记
    completed: bool

    # 用户原始消息
    user_message: str

    # 最终答案
    final_answer: str

    # 由 step_fan_out_router 通过 Send 派发到执行 agent 时注入的字段
    plan_id: str
    task_name: str
    task_desc: str
