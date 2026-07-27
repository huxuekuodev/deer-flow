"""
步骤调度节点：从 Plan Storage 中读取 DAG 计划，并行解析执行 ready steps。

DAG 执行语义（仿 Co-Sight）:
  1. get_ready_steps() → 获取所有就绪的步骤
  2. asyncio.gather() → 并行执行所有就绪步骤（每个步骤一个子图）
  3. 全部完成后 → 重新 get_ready_steps() → 继续并行
  4. 无就绪步骤 → 完成
"""

import asyncio
from asyncio import Semaphore
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.plan_document import StepStatus
from deerflow.agentsv2.plan_storage import get_plan_storage
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger

MAX_CONCURRENT_STEPS = 3  # 最大同时执行的步骤数


# ---------------------------------------------------------------
# 步骤子图状态
# ---------------------------------------------------------------


class StepSubgraphState(TypedDict, total=False):
    """步骤子图状态。通过 config 传递 LLM 等依赖。"""

    messages: Annotated[list[BaseMessage], add_messages]
    step_index: int
    plan_id: str
    step_description: str
    plan_context: str  # 全局计划上下文
    iteration_count: int
    completed: bool


# ---------------------------------------------------------------
# mark_step 工具工厂
# ---------------------------------------------------------------


def make_mark_step_tool(plan_id: str, step_index: int):
    """创建绑定到特定 plan/step 的 mark_step 工具。"""

    @tool
    async def mark_step(step_status: str = "completed", notes: str = "") -> str:
        """
        标记当前步骤执行完成。

        当你完成当前步骤的所有任务后，调用此工具来标记完成。

        Args:
            step_status: "completed"（已完成）或 "blocked"（阻塞）
            notes: 执行结果摘要，包含关键发现、输出文件路径、重要数据。

        Returns:
            确认信息及当前计划进度。
        """
        storage = get_plan_storage()
        await storage.update_status(plan_id, step_index, step_status, notes)

        plan = await storage.load(plan_id)
        progress = plan.get_progress() if plan else {}

        return f"✅ 步骤 {step_index} 已标记为 {step_status}！  进度: {progress.get('completed', 0)}/{progress.get('total', 0)}"

    return mark_step


# ---------------------------------------------------------------
# 构建步骤子图（供 step_dispatch_node 内部调用）
# ---------------------------------------------------------------


def build_step_subgraph(step_index: int, plan_id: str, plan_llm, execution_tools: list):
    """
    构建步骤执行子图。

    每次 step_dispatch_node 调用时动态构建（因为 mark_step 绑定了 plan_id/step_index）。

    子图结构:
      llm_call → route: tool_calls → tools → llm_call
                         no tools → mark_complete → END
    """
    mark_step_tool = make_mark_step_tool(plan_id, step_index)
    all_tools = list(execution_tools) + [mark_step_tool]

    async def step_llm_call(state: StepSubgraphState, config: RunnableConfig) -> dict:
        """LLM 调用节点（不依赖 Runtime context，直接使用传来的 plan_llm）。"""
        writer = get_stream_writer()
        trace_id = trace_id_ctx_var.get()
        iteration = state.get("iteration_count", 0) + 1
        messages = list(state.get("messages", []))

        if iteration == 1:
            step_desc = state.get("step_description", "")
            plan_ctx = state.get("plan_context", "")
            context_block = f"\n## 上下文（之前步骤结果）\n{plan_ctx}\n" if plan_ctx else ""

            sys_prompt = f"""你是一个任务执行专家。执行以下步骤。

## ⚡ 规则
1. 完成后必须调用 **mark_step** 工具标记完成。
2. 如果**所有可用工具都无法完成**该步骤的目标（例如需要的能力都不在工具列表中），则调用 mark_step(status="blocked", notes="当前系统不支持该功能") 并说明原因。

## 当前步骤（Step {step_index}）
{step_desc}{context_block}

## 可用工具
按需调用工具来完成任务。所有操作完成后，调用 mark_step 结束。"""

            messages = [
                SystemMessage(content=sys_prompt),
            ]

        bound = plan_llm.bind_tools(all_tools)
        result = await bound.ainvoke(messages)

        tool_count = len(result.tool_calls) if hasattr(result, "tool_calls") and result.tool_calls else 0
        writer(
            {
                "type": THINK_MES,
                "messages": f"[Step {step_index}] iter {iteration}: {tool_count} tool(s)",
                "step_index": step_index,
                "iteration": iteration,
                "trace_id": trace_id,
            }
        )

        return {"messages": [result], "iteration_count": iteration}

    async def mark_complete(state: StepSubgraphState) -> dict:
        """步骤完成。"""
        return {"completed": True}

    def route_step(state: StepSubgraphState) -> str:
        msgs = state.get("messages", [])
        if not msgs:
            return "mark_complete"
        last = msgs[-1]
        if isinstance(last, AIMessage) and hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "mark_complete"

    builder = StateGraph(StepSubgraphState)
    builder.add_node("llm_call", step_llm_call)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_node("mark_complete", mark_complete)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        route_step,
        {
            "tools": "tools",
            "mark_complete": "mark_complete",
        },
    )
    builder.add_edge("tools", "llm_call")
    builder.add_edge("mark_complete", END)

    return builder.compile()


# ---------------------------------------------------------------
# 单步骤执行协程（供 asyncio.gather 并行调用）
# ---------------------------------------------------------------


async def _execute_single_step(
    step_index: int,
    plan_id: str,
    plan_llm,
    exec_tools: list,
    completed_notes: list,
    config: RunnableConfig,
) -> None:
    """
    执行单个 ready step（由 asyncio.gather 并发调用多个此函数）。

    对应 Co-Sight `CoSight._execute_single_step()`:
      每个步骤拥有独立的 subgraph + LLM + tools + mark_step。
    """
    writer = get_stream_writer()
    trace_id = trace_id_ctx_var.get()

    storage = get_plan_storage()
    plan = await storage.load(plan_id)
    step_desc = plan.steps[step_index] if plan and step_index < len(plan.steps) else f"步骤 {step_index}"

    # 标记 in_progress
    await storage.update_status(plan_id, step_index, StepStatus.IN_PROGRESS)
    step_context = "\n".join(completed_notes[-3:]) if completed_notes else ""

    writer(
        {
            "type": THINK_MES,
            "messages": f"▶️ Step {step_index}: {step_desc[:80]}",
            "step_index": step_index,
            "trace_id": trace_id,
        }
    )

    # 构建子图并执行
    subgraph = build_step_subgraph(step_index, plan_id, plan_llm, exec_tools)

    try:
        async for _ in subgraph.astream(
            input={
                "step_index": step_index,
                "plan_id": plan_id,
                "step_description": step_desc,
                "plan_context": step_context,
                "messages": [],
                "iteration_count": 0,
                "completed": False,
            },
            config=config,
        ):
            pass  # streaming events handled by subgraph's get_stream_writer()

        # 子图完成后，从 storage 读最终结果
        plan = await storage.load(plan_id)
        note = plan.step_notes.get(str(step_index), "") if plan else ""
        current_status = plan.step_statuses.get(str(step_index), StepStatus.IN_PROGRESS) if plan else StepStatus.IN_PROGRESS

        # 如果子图结束时步骤仍处于 in_progress（LLM 直接返回文本没调 mark_step），
        # 自动标记为 completed 并把 LLM 的最终回复写入 notes
        if current_status == StepStatus.IN_PROGRESS:
            # 收集子图中 LLM 的最后一次回复作为步骤结果
            if not note:
                note = step_desc
            await storage.update_status(plan_id, step_index, StepStatus.COMPLETED, note)
            plan = await storage.load(plan_id)
            note = plan.step_notes.get(str(step_index), note) if plan else note

        completed_notes.append(f"Step {step_index}: {note or step_desc}")

        writer(
            {
                "type": THINK_MES,
                "messages": f"✅ Step {step_index} 完成",
                "step_index": step_index,
                "trace_id": trace_id,
            }
        )

    except Exception as e:
        logger.error(f"Step {step_index} failed: {e}", extra={"trace_id": trace_id})
        await storage.update_status(plan_id, step_index, StepStatus.BLOCKED, str(e))
        writer(
            {
                "type": THINK_MES,
                "messages": f"❌ Step {step_index}: {str(e)[:200]}",
                "step_index": step_index,
                "trace_id": trace_id,
            }
        )


# ---------------------------------------------------------------
# 主调度节点（DAG 并行执行）
# ---------------------------------------------------------------


async def step_dispatch_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    """
    步骤调度节点（主图节点）。

    DAG 并行执行语义（完全对齐 Co-Sight）:
      1. 从 storage 读取 plan
      2. get_ready_steps() → 一层 DAG 中所有就绪步骤
      3. asyncio.gather() → **并行**执行所有就绪步骤
      4. 等待该层所有步骤完成后，重新 get_ready_steps()
      5. 重复直到没有就绪步骤（全部完成 或 全部阻塞）

    对比 Co-Sight CoSight.execute():
      - Co-Sight: ThreadPoolExecutor per step + while True 轮询
      - 本设计:   asyncio.gather per DAG layer + while 分层
    """
    # 通过 Runtime 注入获取 plan_llm（LangGraph 1.2 特性）
    plan_llm = runtime.context.plan_llm

    writer = get_stream_writer()
    trace_id = trace_id_ctx_var.get()
    plan_id = state.get("plan_id", "")

    if not plan_id:
        return {"plan_completed": True}

    storage = get_plan_storage()
    plan = await storage.load(plan_id)
    if not plan:
        logger.error(f"Plan not found: {plan_id}", extra={"trace_id": trace_id})
        return {"plan_completed": True}

    from deerflow.agentsv2.lead_agent.tools import make_execution_tools
    from deerflow.tools.v2 import get_execute_tools

    # 合并 v2 config 工具的 execute 部分 + 本地 fallback 工具
    v2_exec_tools = get_execute_tools()
    exec_tools = make_execution_tools() + v2_exec_tools

    # 并发信号量：最多 MAX_CONCURRENT_STEPS 个步骤同时运行
    step_semaphore = Semaphore(MAX_CONCURRENT_STEPS)

    # 共享的已完成步骤记录
    completed_notes: list[str] = []
    max_layers = 25  # safety limit (DAG 分层数不等于步骤数)

    writer(
        {
            "type": THINK_MES,
            "messages": f"🚀 开始执行 DAG 计划: {plan.title} ({len(plan.steps)} 步)",
            "step_count": len(plan.steps),
            "trace_id": trace_id,
        }
    )

    # === DAG 分层并行执行 ===
    while max_layers > 0:
        max_layers -= 1

        # DAG 依赖解析：获取当前层所有就绪步骤
        ready_steps = plan.get_ready_steps()
        if not ready_steps:
            break

        logger.info(
            f"DAG layer: {len(ready_steps)} ready steps: {ready_steps}",
            extra={"trace_id": trace_id},
        )

        # asyncio.gather → 并行执行所有就绪步骤，受 semaphore 限制并发数
        # 如果 ready_steps 有 5 个，但 MAX_CONCURRENT_STEPS=3，
        # 则同时跑 3 个，有 2 个等待 slot
        sem = step_semaphore

        async def _run_step(si):
            async with sem:
                return await _execute_single_step(
                    step_index=si,
                    plan_id=plan_id,
                    plan_llm=plan_llm,
                    exec_tools=exec_tools,
                    completed_notes=completed_notes,
                    config=config,
                )

        await asyncio.gather(*[_run_step(si) for si in ready_steps])

        # 该层执行完毕后重新读取 plan（已由每个步骤的 mark_step 更新）
        plan = await storage.load(plan_id)

        # 检查是否有步骤被阻塞（blocked），已阻塞的步骤不阻碍其他步骤
        progress = plan.get_progress() if plan else {}
        blocked_count = progress.get("blocked", 0)
        if blocked_count > 0:
            logger.warning(
                f"{blocked_count} step(s) blocked, continuing with remaining DAG",
                extra={"trace_id": trace_id},
            )

    # === 汇总结果 ===
    plan = await storage.load(plan_id)
    progress = plan.get_progress() if plan else {}
    total = progress.get("total", 0)
    completed = progress.get("completed", 0)
    blocked = progress.get("blocked", 0)
    all_done = completed == total > 0

    summary_lines = []
    step_outputs = []
    if plan:
        for i, s in enumerate(plan.steps):
            st = plan.step_statuses.get(str(i), "?")
            nt = plan.step_notes.get(str(i), "")
            summary_lines.append(f"Step {i} [{st}]: {s}")
            if nt:
                summary_lines.append(f"  → {nt[:300]}")
                step_outputs.append(f"**Step {i}**：{nt[:300]}")
            else:
                step_outputs.append(f"**Step {i}**：*无详细输出*")

    summary = "\n".join(summary_lines)
    step_content = "\n\n".join(step_outputs)

    status_msg = ""
    if all_done:
        status_msg = f"🎉 计划全部完成！({completed}/{total})"
    elif blocked == total:
        status_msg = f"❌ 所有步骤均阻塞 ({total} blocked)"
    elif blocked > 0:
        status_msg = f"⚠️ 部分完成 ({completed}/{total})，{blocked} 步阻塞"
    else:
        status_msg = f"⏸️ 计划执行暂停 ({completed}/{total})"

    final_msg = AIMessage(content=(f"## {status_msg}\n\n{step_content}\n\n---\n*计划进度: {completed}/{total} 步骤完成*"))

    writer(
        {
            "type": THINK_MES,
            "messages": status_msg,
            "progress": progress,
            "trace_id": trace_id,
        }
    )

    return {
        "plan_completed": all_done or blocked == total,
        "plan_context": summary,
        "messages": [final_msg],
    }
