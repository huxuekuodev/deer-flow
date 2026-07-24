"""
OBSERVE 节点：观察执行结果，判断完成，改写和推进计划。

工作方式：
  1. 读取 ToolMessages（工具执行结果），更新 todo_list
  2. 判断当前阶段是否全部完成
  3. 如果完成 → 汇总 context 注入下一阶段
  4. 产出 AIMessage.tool_calls（下一阶段任务）或普通 AIMessage（结束）
"""

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.nodes.state_bar import format_todo_state_bar
from deerflow.agentsv2.thread_state import ThreadState, TodoItem
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger


class ObserveOutput(BaseModel):
    """Observe 的结构化输出。"""

    current_phase_done: bool = Field(description="当前阶段是否全部完成")
    next_phase_index: int = Field(default=-1, description="下一阶段索引，-1 全部完成")
    reasoning: str = Field(default="", description="判断依据")


OBSERVE_PROMPT = """你是一个任务观察者。根据当前状态判断当前阶段是否完成。

当前阶段: 第 {current_phase} 阶段
状态栏:
{state_bar}

最近工具执行结果:
{tool_results}

职责：
1. 判断当前阶段所有任务是否已完成。
2. 如果完成但还有后续阶段, next_phase_index = 下一阶段序号。
3. 如果全部完成, next_phase_index = -1。"""


async def _summarize_tool_results(messages: list) -> str:
    """提取最近的 ToolMessages 作为观察依据。"""
    lines = []
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "tool":
            lines.append(f"[{m.name}] {str(m.content)[:300]}")
        if len(lines) >= 5:
            break
    return "\n".join(reversed(lines))


def _make_tool_call(tool_name: str, tool_args: dict, task_id: str) -> dict:
    import uuid

    return {
        "name": tool_name,
        "args": tool_args,
        "id": task_id or str(uuid.uuid4()),
        "type": "tool_call",
    }


async def observe_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    context = runtime.context
    llm = context.plan_llm
    writer = get_stream_writer()

    cp = state.get("current_phase", 0)
    todo = state.get("todo_list", [])

    if not todo or cp < 0 or cp >= len(todo):
        return {"current_phase": -1}

    phase = todo[cp]
    all_done = all(t.done for t in phase.todo)

    if not all_done:
        # 更新 todo_list：从 ToolMessages 同步结果
        updated_todo = _sync_tool_results(todo, state.get("messages", []))
        writer({"type": THINK_MES, "messages": "[观察] 等待工具结果", "trace_id": trace_id_ctx_var.get()})
        return {"todo_list": updated_todo}

    # 阶段完成，LLM 观察判断
    bar = format_todo_state_bar(todo, cp)
    tool_results = await _summarize_tool_results(state.get("messages", []))
    prompt = OBSERVE_PROMPT.format(current_phase=cp + 1, state_bar=bar, tool_results=tool_results)

    structured = llm.with_structured_output(ObserveOutput)
    try:
        out: ObserveOutput = structured.invoke([SystemMessage(content=prompt)])
    except Exception as e:
        logger.error(f"Observe 失败: {e}", extra={"trace_id": trace_id_ctx_var.get()})
        out = ObserveOutput(
            current_phase_done=True,
            next_phase_index=cp + 1 if cp + 1 < len(todo) else -1,
            reasoning=f"降级处理 ({e})",
        )

    # 标记完成，汇总 context
    todo[cp].done = True
    next_idx = out.next_phase_index
    ctx = _synthesize_context(todo[cp])
    if next_idx >= 0 and next_idx < len(todo):
        todo[next_idx].context = ctx

    writer(
        {
            "type": THINK_MES,
            "messages": f"[观察] {out.reasoning}",
            "next_phase": next_idx,
            "trace_id": trace_id_ctx_var.get(),
        }
    )

    if next_idx < 0 or next_idx >= len(todo):
        # 全部完成
        return {
            "todo_list": todo,
            "current_phase": -1,
            "messages": [AIMessage(content="所有任务已完成！")],
        }

    # 还有下一阶段 → 构造 tool_calls 让 ToolNode 继续
    next_phase = todo[next_idx]
    tool_calls = [_make_tool_call(t.tool_name, t.tool_args, t.task_id) for t in next_phase.todo]
    ai_msg = AIMessage(content=f"开始执行: {next_phase.phase_desc}", tool_calls=tool_calls)

    return {
        "todo_list": todo,
        "current_phase": next_idx,
        "messages": [ai_msg],
    }


def _sync_tool_results(todo_list: list[TodoItem], messages: list) -> list[TodoItem]:
    """从 ToolMessages 更新 todo_list 中的 task 状态。"""
    import copy

    result = copy.deepcopy(todo_list)

    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "")
            name = getattr(msg, "name", "")
            content = str(msg.content) if msg.content else ""

            for phase in result:
                for task in phase.todo:
                    if task.task_id == tool_call_id or (not task.done and task.tool_name == name):
                        task.done = True
                        task.result = content
                        break
    return result


def _synthesize_context(phase: TodoItem) -> str:
    """汇总阶段执行结果。"""
    if not phase.todo:
        return ""
    parts = [f"## {phase.phase_desc} 执行结果"]
    for t in phase.todo:
        if t.result:
            preview = t.result[:500] + "..." if len(t.result) > 500 else t.result
            parts.append(f"[{t.tool_name}] {t.task_desc}: {preview}")
    return "\n".join(parts)
