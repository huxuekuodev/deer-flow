"""
PLAN 节点：意图识别 + 任务拆解。

工作方式：
  1. 以结构化输出生成 todo_list
  2. 手动构造 AIMessage.tool_calls（第一阶段的任务）
  3. ToolNode 自动读取 tool_calls 并执行
"""

import datetime
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from lxml import etree
from pydantic import BaseModel, Field

from deerflow.agentsv2 import ThreadState
from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.nodes.state_bar import format_todo_state_bar
from deerflow.agentsv2.thread_state import Task, TodoItem
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger

CURRENT_TIME_TAG = ["<current_time>", "</current_time>"]


class PlanOutput(BaseModel):
    """计划节点结构化输出。"""

    need_plan: bool = Field(description="是否需要拆解为多阶段执行计划")
    todo_phases: list[dict] = Field(
        default_factory=list,
        description="""按顺序的执行阶段列表。每个元素：
{
  "phase_desc": "阶段描述",
  "tasks": [
    {"tool_name": "weather/general", "tool_args": {...}, "task_desc": "描述"}
  ]
}""",
    )
    direct_answer: str = Field(default="", description="直接回答（need_plan=False）")


PLAN_SYSTEM_PROMPT = """你是一个任务规划助手。分析用户需求，拆解为按顺序执行的多个阶段。

可用工具：
  - weather(city, date): 查询天气。city=城市名, date=日期(可选)
  - general(query): 通用信息查询。query=问题描述

输出 todo_phases 列表，每个 phase 包含：
  - phase_desc: 阶段描述
  - tasks: 本阶段要调用的工具列表（同阶段可并行）

{state_bar}"""


def _make_tool_call(tool_name: str, tool_args: dict) -> dict:
    """构造 ToolNode 识别的 tool_call 格式。"""
    return {
        "name": tool_name,
        "args": tool_args,
        "id": str(uuid.uuid4()),
        "type": "tool_call",
    }


async def plan_model_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    context = runtime.context
    llm = context.plan_llm
    writer = get_stream_writer()
    writer({"type": THINK_MES, "messages": "助手开始规划任务", "trace_id": trace_id_ctx_var.get()})

    structured = llm.with_structured_output(PlanOutput)

    # 构建消息
    cp = state.get("current_phase", 0)
    todo = state.get("todo_list", [])
    bar = format_todo_state_bar(todo, cp)

    system_text = PLAN_SYSTEM_PROMPT.format(state_bar=bar)
    messages: list[BaseMessage] = [HumanMessage(content=system_text)]
    messages.extend(state.get("messages", []))

    # 时间戳
    has_time = False
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage) and str(msg.content).strip().startswith(CURRENT_TIME_TAG[0]):
            try:
                el = etree.fromstring(str(msg.content))
                d = datetime.datetime.strptime(str(el.text), "%Y-%m-%d").date()
                if d == datetime.date.today():
                    has_time = True
            except Exception:
                pass
    if not has_time:
        messages.append(HumanMessage(content=f"{CURRENT_TIME_TAG[0]}{datetime.date.today().strftime('%Y-%m-%d')}{CURRENT_TIME_TAG[1]}"))

    for attempt in range(1, 4):
        try:
            output: PlanOutput = structured.invoke(messages)
            if not output.need_plan:
                return {"todo_list": [], "current_phase": -1, "messages": [AIMessage(content=output.direct_answer)]}

            todo_list = []
            for i, phase_data in enumerate(output.todo_phases):
                tasks = []
                for t in phase_data.get("tasks", []):
                    tn = t.get("tool_name", "")
                    if tn not in ("weather", "general"):
                        raise ValueError(f"未知工具: {tn}")
                    tasks.append(
                        Task(
                            tool_name=tn,
                            tool_args=t.get("tool_args", {}),
                            task_desc=t.get("task_desc", ""),
                        )
                    )
                todo_list.append(
                    TodoItem(
                        phase_desc=phase_data.get("phase_desc", f"阶段{i + 1}"),
                        todo=tasks,
                    )
                )

            if not todo_list:
                raise ValueError("空的 todo_list")

            # SSE 推送计划
            for i, phase in enumerate(todo_list):
                writer(
                    {
                        "type": THINK_MES,
                        "step": i + 1,
                        "total_steps": len(todo_list),
                        "phase_desc": phase.phase_desc,
                        "tasks": [{"tool": t.tool_name, "args": t.tool_args, "desc": t.task_desc} for t in phase.todo],
                        "step_status": "pending",
                        "trace_id": trace_id_ctx_var.get(),
                    }
                )

            # 构造第一阶段 tool_calls 的 AIMessage
            phase0 = todo_list[0]
            tool_calls = [_make_tool_call(t.tool_name, t.tool_args) for t in phase0.todo]
            ai_msg = AIMessage(content=f"开始执行: {phase0.phase_desc}", tool_calls=tool_calls)

            return {
                "todo_list": todo_list,
                "current_phase": 0,
                "messages": [ai_msg],
            }

        except Exception as e:
            logger.error(f"Plan 第 {attempt} 次失败: {e}", extra={"trace_id": trace_id_ctx_var.get()})
            if attempt == 3:
                return {"todo_list": [], "current_phase": -1, "messages": [AIMessage(content="抱歉，计划生成失败")]}
