"""
PLAN 节点（Co-Sight 模式）。

变化（vs 旧版 Pydantic structured output）：
  1. LLM 通过 function calling 调用 create_plan/update_plan 工具
  2. 工具执行写入 storage（内存/Redis）
  3. 返回 plan_id → step_dispatch_node 读取执行
"""

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.plan_toolkit import create_plan, get_plan_status, update_plan
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger
from deerflow.tools.v2 import describe_execute_tools

PLAN_SYSTEM_PROMPT_HEADER = """你是一个任务规划助手。分析用户需求，拆解为多步骤 DAG 计划。

## 规划工具
你只能使用以下工具来规划任务：

1. **create_plan(title, steps, dependencies?)** — 创建 DAG 计划
   - title: 计划标题
   - steps: 步骤描述列表（数组）
   - dependencies (可选): 依赖关系。如 {{"1": [0]}} 表示步骤1依赖步骤0
   - 不传 dependencies：默认顺序依赖

2. **update_plan(plan_id, title?, steps?, dependencies?)** — 修改计划（保留已完成步骤）
3. **get_plan_status(plan_id)** — 查询计划进度

## 规划原则
- 每个步骤应是一个具体可执行的任务描述
- 步骤间可指定依赖关系（DAG），无依赖的步骤可自动并行
- 步骤数量控制在 3-8 个之间
- 使用 create_plan 创建计划


{execution_tool_descriptions}"""


def build_plan_system_prompt(existing_plan_id: str = "", existing_context: str = "") -> str:
    """动态构建 Plan system prompt，注入执行工具能力描述。"""

    tool_desc = describe_execute_tools()
    actual_desc = tool_desc or ""

    text = PLAN_SYSTEM_PROMPT_HEADER.format(execution_tool_descriptions=actual_desc)

    if existing_plan_id:
        text += f"\n\n已有 plan_id={existing_plan_id}。如需修改，请用 get_plan_status 查看进度后用 update_plan 更新。"
    if existing_context:
        text += f"\n\n之前已完成的上下文:\n{existing_context}"

    text += """

## 步骤设计指南
- 依赖执行能力的步骤，要描述清楚**要做什么**而不是直接写工具名
- 例如：「调用天气查询接口获取北京今日天气数据」而不是「weather(北京)」
- 步骤描述应让执行 LLM 理解目标并自主选择工具

## 能力边界
- 如果上方的「可用的执行能力」中没有任何工具能满足用户的需求，则**不要创建计划**
- 此时应明确告知用户当前系统不支持该需求，并说明你有哪些能力范围"""

    return text


def _extract_plan_id(result_text: str) -> str:
    """从工具结果中提取 plan_id。"""
    m = re.search(r"Plan\s*ID:\s*([a-f0-9]{32})", result_text, re.IGNORECASE)
    return m.group(1) if m else ""


async def _execute_plan_tool(tool_call: dict, writer) -> str:
    """执行单个 plan tool call。"""
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    if name == "create_plan":
        result = await create_plan.ainvoke(args)
        return str(result)
    elif name == "update_plan":
        result = await update_plan.ainvoke(args)
        return str(result)
    elif name == "get_plan_status":
        result = await get_plan_status.ainvoke(args)
        return str(result)
    else:
        return f"未知工具: {name}"


async def plan_model_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    """
    PLAN 节点：LLM 通过 function calling 创建 DAG 计划。

    流程：
      1. LLM + create_plan/update_plan/get_plan_status 工具绑定
      2. LLM 调用 create_plan 工具
      3. 本节点内部执行工具（实时写入 storage）
      4. 返回 plan_id 给 thread_state
    """
    context = runtime.context
    llm = context.plan_llm
    writer = get_stream_writer()
    trace_id = trace_id_ctx_var.get()

    writer({"type": THINK_MES, "messages": "🤔 分析需求，创建执行计划...", "trace_id": trace_id})

    existing_plan_id = state.get("plan_id", "")
    existing_context = state.get("plan_context", "")

    # 构建 messages
    system_text = build_plan_system_prompt(
        existing_plan_id=existing_plan_id,
        existing_context=existing_context,
    )

    messages: list[BaseMessage] = [HumanMessage(content=system_text)]
    user_msgs = state.get("messages", [])
    messages.extend(user_msgs)

    # 绑定 plan 工具：规划工具 + ask_clarification
    from deerflow.tools.v2 import get_plan_tools

    plan_tools = [create_plan, update_plan, get_plan_status] + get_plan_tools()
    bound_llm = llm.bind_tools(plan_tools)

    for attempt in range(1, 4):
        try:
            result = await bound_llm.ainvoke(messages)

            if not hasattr(result, "tool_calls") or not result.tool_calls:
                # 不需要规划，直接回答
                writer({"type": THINK_MES, "messages": "无需拆解，直接回答", "trace_id": trace_id})
                return {"plan_id": "", "plan_completed": True, "messages": [result]}

            # 检查是否调用了 ask_clarification
            # 如果有，不执行工具，直接把 AIMessage 原样返回给下游处理
            if any(tc.get("name") == "ask_clarification" for tc in result.tool_calls):
                writer(
                    {
                        "type": THINK_MES,
                        "messages": "需要用户澄清",
                        "tool_calls": [tc.get("name") for tc in result.tool_calls],
                        "trace_id": trace_id,
                    }
                )
                return {"plan_id": "", "plan_completed": True, "messages": [result]}

            # 执行 LLM 调用的 plan 工具
            # 注意：LLM 可能一次返回多个 create_plan（重复调用），只执行第一个
            tool_messages: list[ToolMessage] = []
            plan_id_found = ""
            has_created = False

            for tc in result.tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")

                # 跳过重复的 create_plan
                if tc_name == "create_plan":
                    if has_created:
                        continue
                    has_created = True

                tool_result_text = await _execute_plan_tool(tc, writer)

                # 从 create_plan 结果提取 plan_id
                if tc_name == "create_plan":
                    pid = _extract_plan_id(tool_result_text)
                    if pid:
                        plan_id_found = pid

                tool_messages.append(ToolMessage(content=tool_result_text, tool_call_id=tc_id, name=tc_name))

            writer(
                {
                    "type": THINK_MES,
                    "messages": f"📋 规划完成，plan_id={plan_id_found or '未知'}",
                    "plan_id": plan_id_found,
                    "tool_calls": [tc.get("name") for tc in result.tool_calls],
                    "trace_id": trace_id,
                }
            )

            # 如果创建了 plan，直接进入执行阶段
            if plan_id_found:
                return {
                    "plan_id": plan_id_found,
                    "plan_completed": False,
                    "active_steps": [],
                    "messages": tool_messages,
                }

            # 没有 create_plan（只是查询或更新），但仍有 tool_calls
            # 保留 messages 以便继续
            return {
                "messages": [result] + tool_messages,
            }

        except Exception as e:
            logger.error(f"Plan 第 {attempt} 次失败: {e}", extra={"trace_id": trace_id})
            if attempt == 3:
                fallback = AIMessage(content="计划生成失败，请重新描述需求。")
                return {"plan_completed": True, "messages": [fallback]}

    return {"plan_completed": True}
