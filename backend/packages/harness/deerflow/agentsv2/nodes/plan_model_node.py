"""
规划节点（合并澄清 + 规划 + 审查）。

职责：
  1. 澄清：分析用户输入，模糊或缺失信息时调用 ask_clarification
  2. 规划：需求明确后拆解为 SubTask DAG
  3. 审查：执行后审查结果，决定完成或 replan

简化说明：
  - 使用 create_agent 处理完整的 ReAct 循环，plan_model_node 不再手动管理 tool_calls
  - create_agent 返回的 messages 中取最后一条（最终 AI 回复）写入 state
  - plan_model_node 通过 _bridge_var (ContextVar) 从工具获取 plan_tasks 输出，用于路由到执行节点
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.plan_toolkit import _bridge_var, create_plan, get_plan_status, update_plan
from deerflow.agentsv2.subtask import SubTask
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger


def _build_system_prompt(agent_descriptions: str = "", capability_descriptions: str = "") -> str:
    langfuse = Langfuse()
    return langfuse.get_prompt("deerflow_v2/plan_system_prompt_v2", type="text").compile(
        agent_descriptions=agent_descriptions or "- general_agent: 通用执行 agent，可调用所有工具",
        capability_descriptions=capability_descriptions or "",
    )


async def plan_model_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    context = runtime.context
    llm = context.plan_llm
    writer = get_stream_writer()
    trace_id = trace_id_ctx_var.get()

    writer({"type": THINK_MES, "messages": "📋 分析需求，制定执行计划...", "trace_id": trace_id})

    # 构建 system prompt
    from deerflow.tools.v2 import describe_execute_tools, get_plan_tools

    capability_desc = describe_execute_tools()

    # 获取已有任务列表（review/replan 场景）
    existing_tasks = state.get("plan_tasks", [])
    plan_context = ""
    if existing_tasks:
        plan_context = "\n".join(f"- [{t.step_statuses}] {t.name}: {t.result[:200] if t.result else '待执行'}" for t in existing_tasks)

    # 构建消息
    messages: list[BaseMessage] = []
    context_lines = []
    user_msg = state.get("user_message", "")
    if user_msg:
        context_lines.append(f"<UserRequest>{user_msg}</UserRequest>")
    if plan_context:
        context_lines.append(f"<PlanStatus>{plan_context}</PlanStatus>")
    if context_lines:
        messages.append(HumanMessage(content="\n".join(context_lines)))
    user_msgs = state.get("messages", [])
    messages.extend(user_msgs)

    # 将 ThreadState 中的 plan_tasks 注入桥接层，供 get_plan_status 工具读取
    bridge = _bridge_var.get()
    bridge["plan_tasks"] = list(existing_tasks)
    bridge["created_task_dicts"] = None  # 重置
    _bridge_var.set(bridge)

    # 绑定工具并创建 agent（create_agent 内部自动处理 ReAct 循环）
    plan_tools = [create_plan, update_plan, get_plan_status] + get_plan_tools()
    bound_llm = llm.bind_tools(plan_tools)
    agent = create_agent(
        bound_llm,
        plan_tools,
        system_prompt=_build_system_prompt(capability_descriptions=capability_desc),
    )

    for attempt in range(1, 4):
        try:
            # create_agent.ainvoke({"messages": [...]}) 返回 {"messages": [完整 ReAct 消息列表]}
            agent_output = await agent.ainvoke({"messages": messages}, config=config)

            # 取最终回复（最后一条 message）
            agent_msgs = agent_output.get("messages", [])
            final_msg = agent_msgs[-1] if agent_msgs else AIMessage(content="计划生成失败")

            # 检查桥接层：create_plan/update_plan 是否创建了新任务
            bridge = _bridge_var.get()
            created_dicts = bridge.get("created_task_dicts")

            if created_dicts:
                subtasks = [SubTask(**t) for t in created_dicts]
                bridge["created_task_dicts"] = None
                _bridge_var.set(bridge)
                writer(
                    {
                        "type": THINK_MES,
                        "messages": f"📋 规划完成，共 {len(subtasks)} 个子任务",
                        "task_count": len(subtasks),
                        "trace_id": trace_id,
                    }
                )
                return {"messages": [final_msg], "plan_tasks": subtasks}

            # 没有创建计划 → agent 直接回复（澄清、审查结论等）
            writer({"type": THINK_MES, "messages": "📋 规划完成", "trace_id": trace_id})
            return {"messages": [final_msg], "completed": True}

        except Exception as e:
            logger.error("Plan 第 {} 次失败: {}", attempt, e, extra={"trace_id": trace_id})
            if attempt == 3:
                return {"messages": [AIMessage(content="计划生成失败，请重新描述需求。")], "completed": True}

    return {"completed": True}
