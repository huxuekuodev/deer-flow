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
from langfuse import Langfuse
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from pydantic import BaseModel, Field

from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agentsv2.evaluation.plan_evaluator import (
    EvaluationInput,
    PlanEvaluationConfig,
    PlanEvaluator,
)
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

    # 捕获评估输入轨迹（规划 agent 实际看到的全部上下文，用于公平评估）
    eval_input = _capture_eval_input(
        messages=messages,
        plan_context=plan_context,
        current_time=context.current_time,
    )

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

            # 判定 agent 的实际输出类型（不管进哪个分支，都先评估）
            has_clarification = any(isinstance(m, AIMessage) and getattr(m, "tool_calls", None) and any(tc.get("name") == "ask_clarification" for tc in m.tool_calls) for m in agent_msgs)
            plan_output = _extract_plan_output(agent_output)

            # 填充评估输入，统一触发评估（覆盖澄清 / 规划 / 直接回复 全部分支）
            eval_input.clarification_requested = has_clarification
            if plan_output:
                eval_input.plan_action = plan_output.action
                eval_input.tasks = [t.model_dump() for t in plan_output.tasks]
            await _maybe_evaluate(
                trace_id=trace_id,
                eval_input=eval_input,
                messages=messages,
                config=config,
                runtime=runtime,
            )

            # 澄清：需求含糊时 agent 调用了 ask_clarification → 中断等待用户
            if has_clarification:
                writer({"type": THINK_MES, "messages": "📋 需要澄清需求", "trace_id": trace_id})
                return {"messages": agent_msgs, "completed": True}

            # 规划：模型输出了有效计划
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


# ----------------------------------------------------------------------
# 规划评估（实验性）
# ----------------------------------------------------------------------


def _capture_eval_input(
    *,
    messages: list[BaseMessage],
    plan_context: str,
    current_time: str,
) -> EvaluationInput:
    """捕获规划节点实际看到的输入轨迹，供评估器公平判断。"""
    user_messages: list[str] = []
    history: list[dict] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, str) and content.strip():
                user_messages.append(content)
        history.append(
            {
                "type": type(m).__name__,
                "content": str(getattr(m, "content", ""))[:2000],
            }
        )
    return EvaluationInput(
        user_messages=user_messages,
        history=history,
        plan_status=plan_context,
        current_time=current_time,
    )


async def _maybe_evaluate(
    *,
    trace_id: str,
    eval_input: EvaluationInput,
    messages: list[BaseMessage],
    config: RunnableConfig,
    runtime: Runtime[GraphContext],
) -> None:
    """按配置触发规划评估；未启用或被采样时静默跳过。"""
    try:
        app_config = runtime.context.app_config
        plan_eval_cfg = getattr(app_config, "plan_evaluation", None)
        if plan_eval_cfg is None or not getattr(plan_eval_cfg, "enabled", False):
            # 未配置或未启用 → 跳过（实验性功能默认关闭）
            return

        # 合并维度：部分覆盖时保留默认开关（Pydantic default_factory 不自动合并）
        _DEFAULT_DIMENSIONS = {
            "clarification_quality": True,
            "task_atomicity": True,
            "agent_selection_validity": True,
        }
        dimensions = dict(getattr(plan_eval_cfg, "dimensions", {}) or {})
        for key, default_val in _DEFAULT_DIMENSIONS.items():
            dimensions.setdefault(key, default_val)

        cfg = PlanEvaluationConfig(
            enabled=getattr(plan_eval_cfg, "enabled", False),
            sample_rate=getattr(plan_eval_cfg, "sample_rate", 1.0),
            judge_model=getattr(plan_eval_cfg, "judge_model", None),
            dimensions=dimensions,
        )
        if not cfg.enabled:
            return

        # 允许的 execution_agent 集合：内置 general_agent + 配置的 custom_agents
        enabled_agents: set[str] = set()
        subagents_cfg = getattr(app_config, "subagents", None)
        if subagents_cfg is not None:
            custom = getattr(subagents_cfg, "custom_agents", {}) or {}
            enabled_agents = set(custom.keys())

        # Judge LLM：优先用配置的 judge_model，否则用 plan_llm
        judge_llm = None
        if cfg.judge_model:
            try:
                from deerflow.agentsv2.lead_agent import create_llm_with_name

                judge_llm = create_llm_with_name(config, model_name="evaluate_plan")
            except Exception:
                judge_llm = None
        else:
            judge_llm = runtime.context.plan_llm

        evaluator = PlanEvaluator(
            langfuse=runtime.context.langfuse_client,
            enabled_agents=enabled_agents,
            config=cfg,
            judge_llm=judge_llm,
        )
        await evaluator.evaluate(
            trace_id=trace_id,
            eval_input=eval_input,
            messages=messages,
            config=config,
        )
    except Exception as exc:
        logger.warning("Plan evaluation skipped: %s", exc)
