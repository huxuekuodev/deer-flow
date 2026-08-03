"""
规划节点（合并澄清 + 规划 + 审查）。

职责：
  1. 澄清：分析用户输入，模糊或缺失信息时调用 ask_clarification
  2. 规划：需求明确后拆解为 SubTask DAG（模型直接输出计划 JSON）
  3. 审查：执行后审查结果，决定完成或 replan

设计说明：
  - 不再使用 create_plan / update_plan 工具（绕了三层间接：工具→bridge→哨兵/reducer）
  - 模型通过结构化输出直接产出计划（PlanOutput），plan_model_node 解析为 SubTask
  - 新计划（用户新需求）→ 用 Overwrite 整体替换旧计划（绕过 merge reducer）
  - 状态更新（执行节点回写）→ 继续用 merge reducer 合并
  - 仅保留 ask_clarification 工具（经 get_plan_tools 注入）
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, get_client
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from pydantic import BaseModel, Field

from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.subtask import SubTask
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger


class PlanTask(BaseModel):
    """计划中的单个子任务（模型结构化输出）。"""

    plan_id: str = Field(description="子任务唯一标识，如 task1 / task2")
    name: str = Field(description="子任务名称（简短）")
    desc: str = Field(description="子任务详细描述。可用 {其他任务plan_id} 引用依赖任务的结果")
    execution_agent: str = Field(default="general_agent", description="执行此任务的 agent")
    sort: int = Field(default=0, description="执行顺序序号")
    deps: list[str] = Field(default_factory=list, description="依赖的子任务 plan_id 列表")


class PlanOutput(BaseModel):
    """规划节点的结构化输出。"""

    action: str = Field(description="create: 创建全新计划（替换旧计划）；update: 更新现有计划状态")
    title: str = Field(default="", description="计划标题")
    tasks: list[PlanTask] = Field(default_factory=list, description="子任务列表")


def _build_system_prompt(agent_descriptions: str = "", capability_descriptions: str = "") -> str:
    langfuse = Langfuse()
    return langfuse.get_prompt("deerflow_v2/plan_system_prompt_v2", type="text").compile(
        agent_descriptions=agent_descriptions or "- general_agent: 通用执行 agent，可调用所有工具",
        capability_descriptions=capability_descriptions or "",
    )


def _to_subtask(t: PlanTask) -> SubTask:
    """将 PlanTask 转换为 SubTask。"""
    return SubTask(
        plan_id=t.plan_id,
        name=t.name,
        desc=t.desc,
        execution_agent=t.execution_agent,
        sort=t.sort,
        deps=t.deps,
    )


async def plan_model_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    context = runtime.context
    llm = context.plan_llm
    writer = get_stream_writer()
    trace_id = trace_id_ctx_var.get()

    # 获取已有任务列表（review/replan 场景）
    existing_tasks = state.get("plan_tasks", [])
    writer({"type": THINK_MES, "messages": "📋 分析需求，制定执行计划...", "trace_id": trace_id})

    # 构建 system prompt
    from deerflow.tools.v2 import describe_execute_tools, get_plan_tools

    capability_desc = describe_execute_tools()

    plan_context = ""
    if existing_tasks:
        plan_context = "\n".join(f"- [{t.step_statuses}] {t.name}: {t.result if t.result else '待执行'}" for t in existing_tasks)

    # 构建消息
    messages: list[BaseMessage] = []
    user_msgs = state.get("messages", [])
    messages.extend(user_msgs)
    context_lines = []
    if plan_context:
        context_lines.append(f"""<PlanStatus>\n当前计划
        {plan_context}\n\n
        </PlanStatus>""")
    if context_lines:
        messages.append(HumanMessage(content="\n".join(context_lines)))
    # 注入当前时间（供 agent 处理日期相关任务，如"今日天气"）
    # TODO 验证state中是否有current_time, 如果有判断是否是今日，如果不是注入新的日期，如果是不重复注入当日日期
    messages.append(HumanMessage(content=f"<current_time>{context.current_time}</current_time>"))

    # 绑定工具：仅 ask_clarification（get_plan_tools 已包含）
    plan_tools = get_plan_tools()
    bound_llm = llm.bind_tools(plan_tools)
    agent = create_agent(
        bound_llm,
        plan_tools,
        middleware=[ClarificationMiddleware()],
        name="plan_agent",
        response_format=PlanOutput,
        system_prompt=_build_system_prompt(capability_descriptions=capability_desc),
    )

    for attempt in range(1, 4):
        try:
            agent_output = await agent.ainvoke({"messages": messages}, config=config)
            agent_msgs: list[Any] = agent_output.get("messages", []) if isinstance(agent_output, dict) else []
            langfuse = get_client()
            with langfuse.start_as_current_observation(as_type="span", name="call-research-sub-agent", trace_context={"trace_id": trace_id}) as span:
                span.update(input=messages[-1].content, output=agent_msgs[-1].content)
            # 检查是否有澄清（agent 调用了 ask_clarification → 会 interrupt）
            # ainvoke 返回的最终 state 里若出现 ToolMessage 且无 plan 输出，说明是澄清
            has_clarification = any(isinstance(m, AIMessage) and getattr(m, "tool_calls", None) and any(tc.get("name") == "ask_clarification" for tc in m.tool_calls) for m in agent_msgs)
            if has_clarification:
                writer({"type": THINK_MES, "messages": "📋 需要澄清需求", "trace_id": trace_id})
                return {"messages": agent_msgs, "completed": True}

            # 尝试从最终回复解析结构化输出
            plan_output = _extract_plan_output(agent_output)
            if plan_output and plan_output.tasks:
                subtasks = [_to_subtask(t) for t in plan_output.tasks]
                writer(
                    {
                        "type": THINK_MES,
                        "messages": f"📋 规划完成，共 {len(subtasks)} 个子任务",
                        "task_count": len(subtasks),
                        "trace_id": trace_id,
                    }
                )
                if plan_output.action == "create":
                    # 新计划：整体替换旧计划（Overwrite 绕过 merge reducer）
                    return {"messages": agent_msgs, "plan_tasks": Overwrite(value=subtasks)}
                # update：合并到现有计划
                return {"messages": agent_msgs, "plan_tasks": subtasks}

            # 没有计划输出 → agent 直接回复（澄清、审查结论等）
            writer({"type": THINK_MES, "messages": "📋 规划完成", "trace_id": trace_id})
            return {"messages": agent_msgs, "completed": True}

        except Exception as e:
            logger.error("Plan 第 {} 次失败: {}", attempt, e, extra={"trace_id": trace_id})
            if attempt == 3:
                return {"messages": [AIMessage(content="计划生成失败，请重新描述需求。")], "completed": True}

    return {"completed": True}


def _extract_plan_output(agent_output: dict) -> PlanOutput | None:
    """从 agent 输出中提取结构化计划。

    兼容两种形态：
      1. 模型直接输出 PlanOutput 对象（structured output）
      2. 最终 AIMessage 里带结构化内容（部分模型）
    """
    if not isinstance(agent_output, dict):
        return None

    messages = agent_output.get("messages", [])
    if not messages:
        return None

    # 查找带结构化输出的消息
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            # 结构化输出通常放在 additional_kwargs 或 content 的 JSON 里
            plan = _try_parse_plan(msg)
            if plan:
                return plan
    return None


def _try_parse_plan(msg: AIMessage) -> PlanOutput | None:
    """尝试从 AIMessage 解析 PlanOutput。"""

    # 1. 结构化输出注入到 content（JSON 字符串）
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        try:
            return PlanOutput.model_validate_json(content)
        except Exception:
            pass

    # 2. additional_kwargs 里的 parsed
    try:
        kwargs = getattr(msg, "additional_kwargs", {}) or {}
        for key in ("parsed", "tool_call", "structured_output"):
            if key in kwargs:
                val = kwargs[key]
                if isinstance(val, dict):
                    return PlanOutput.model_validate(val)
    except Exception:
        pass

    return None
