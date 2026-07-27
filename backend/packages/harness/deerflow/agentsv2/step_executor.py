"""
步骤执行器：LangGraph 子图（Subgraph），封装单个步骤的 ReAct 循环。

Co-Sight 模式:
  在 Co-Sight 中，每个步骤由独立的 TaskActorAgent 线程执行，
  包含完整的 LLM + tools ReAct 循环，直到 LLM 调用 mark_step 结束。

此处改造成 LangGraph 子图，同样提供 ToolNode 执行工具，
但利用 LangGraph 自身的图结构管理循环，无需手动线程池。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.plan_document import PlanDocument
from deerflow.agentsv2.plan_storage import get_plan_storage
from deerflow.core.context import trace_id_ctx_var

# ---------------------------------------------------------------
# Step State
# ---------------------------------------------------------------


class StepState(TypedDict, total=False):
    """步骤子图的状态。"""

    messages: Annotated[list[BaseMessage], add_messages]
    step_index: int
    plan_id: str
    step_description: str
    iteration_count: int


# ---------------------------------------------------------------
# mark_step 工具（由步骤执行的 LLM 调用）
# ---------------------------------------------------------------


def _make_mark_step_tool(plan_id: str, step_index: int):
    """工厂函数：创建绑定到特定 plan/step 的 mark_step 工具。"""

    @tool
    async def mark_step(step_status: str = "completed", notes: str = "") -> str:
        """
        标记当前步骤执行完成或更新状态。

        当你完成当前步骤的所有工作时，调用此工具来标记完成，
        以便计划推进到下一个步骤。

        Args:
            step_status: 步骤状态，可选值: "completed", "blocked"
            notes: 执行结果摘要或备注，包含关键发现和输出。

        Returns:
            确认信息。
        """
        storage = get_plan_storage()
        await storage.update_status(plan_id, step_index, step_status, notes)

        plan = await storage.load(plan_id)
        progress = plan.get_progress() if plan else {}

        return f"✅ 步骤 {step_index} 已标记为 {step_status}！\n  当前计划进度: {progress.get('completed', 0)}/{progress.get('total', 0)}\n  备注已保存。"

    return mark_step


# ---------------------------------------------------------------
# 步骤提示词工厂
# ---------------------------------------------------------------


def _build_step_system_prompt(plan: PlanDocument, step_index: int) -> str:
    """构建步骤执行的系统提示。"""
    step_desc = plan.steps[step_index] if step_index < len(plan.steps) else ""
    plan_context = plan.format()

    return f"""你是一个步骤执行专家。你需要完成以下步骤：

## 当前步骤（Step {step_index}）
{step_desc}

## 全局计划状态
{plan_context}

## 你的职责
1. 理解步骤目标，调用必要的工具来执行任务。
2. 你可以多次调用工具，步骤内部可以多轮交互。
3. 当步骤目标达成时，调用 mark_step 工具标记完成。
4. 如果遇到无法解决的障碍，调用 mark_step(status="blocked", notes="原因")。

## 可用工具
你可以使用以下工具来完成任务。按需调用。"""


# ---------------------------------------------------------------
# 子图节点
# ---------------------------------------------------------------


async def step_llm_node(state: StepState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    """步骤 LLM 调用节点。"""
    context = runtime.context
    llm = context.plan_llm  # re-use plan_llm for step execution
    writer = get_stream_writer()

    plan_id = state.get("plan_id", "")
    step_index = state.get("step_index", 0)
    step_desc = state.get("step_description", "")

    # 构建消息
    messages = state.get("messages", [])
    if not any(isinstance(m, HumanMessage) and m.content == step_desc for m in messages):
        # 将步骤描述作为首条 HumanMessage
        storage = get_plan_storage()
        plan = await storage.load(plan_id)
        sys_prompt = _build_step_system_prompt(plan, step_index) if plan else ""
        msg = [
            HumanMessage(content=sys_prompt),
            HumanMessage(content=f"请执行步骤 {step_index}: {step_desc}"),
        ]
        messages = msg + messages

    # 注入 step context
    bound_llm = llm.bind_tools([_make_mark_step_tool(plan_id, step_index)])
    result = await bound_llm.ainvoke(messages)

    writer(
        {
            "type": THINK_MES,
            "messages": f"[Step {step_index}] LLM 响应",
            "content": str(result.content)[:200] if result.content else "",
            "tool_calls": len(result.tool_calls) if hasattr(result, "tool_calls") and result.tool_calls else 0,
            "trace_id": trace_id_ctx_var.get(),
        }
    )

    return {"messages": [result]}


async def mark_step_node(state: StepState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    """标记步骤完成的汇聚节点。"""
    writer = get_stream_writer()
    step_index = state.get("step_index", 0)

    writer(
        {
            "type": THINK_MES,
            "messages": f"✅ 步骤 {step_index} 执行完毕",
            "trace_id": trace_id_ctx_var.get(),
        }
    )

    return {"step_index": -1}  # signal done


def route_step(state: StepState) -> str:
    """路由：LLM 输出 → 有 tool_calls 则去 ToolNode，否则标记完成。"""
    messages = state.get("messages", [])
    if not messages:
        return "mark_step_node"

    last = messages[-1]
    has_tool_calls = isinstance(last, AIMessage) and hasattr(last, "tool_calls") and last.tool_calls

    if has_tool_calls:
        return "tools"

    # 没有 tool_calls → 步骤完成（或 LLM 直接返回文本）
    return "mark_step_node"


# ---------------------------------------------------------------
# 构建子图
# ---------------------------------------------------------------


def build_step_subgraph(context: GraphContext) -> StateGraph:
    """
    构建步骤执行子图。

    结构:
      START → step_llm_node
        step_llm_node → route_step
          - tool_calls → tools → step_llm_node  (ReAct 循环)
          - no tool_calls → mark_step_node → END
    """
    mark_step_tool = _make_mark_step_tool("{{plan_id}}", 0)  # placeholder, will be bound at runtime

    # 执行工具：从现有的 lead_agent.tools 获取
    from deerflow.agentsv2.lead_agent.tools import make_tools

    execution_tools = make_tools(context.plan_llm) + [mark_step_tool]

    builder = StateGraph(StepState)

    builder.add_node("step_llm_node", step_llm_node)
    builder.add_node("tools", ToolNode(execution_tools))
    builder.add_node("mark_step_node", mark_step_node)

    builder.add_edge(START, "step_llm_node")

    builder.add_conditional_edges(
        "step_llm_node",
        route_step,
        {
            "tools": "tools",
            "mark_step_node": "mark_step_node",
        },
    )
    builder.add_edge("tools", "step_llm_node")
    builder.add_edge("mark_step_node", END)

    return builder.compile()


# ---------------------------------------------------------------
# 入口：在父图中调用的节点
# ---------------------------------------------------------------


async def step_dispatch_node(state, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    """
    从存储中读取 plan，获取下一个 ready step，构造 state 传给子图。

    此节点在父图中作为一个 "invoke subgraph" 节点存在。
    """
    pass  # 实际调用在父图中通过 Command / subgraph 显式完成
