"""
规划节点（合并澄清 + 规划 + 审查）。

职责：
  1. 澄清：分析用户输入，模糊或缺失信息时调用 ask_clarification
  2. 规划：需求明确后拆解为 SubTask DAG
  3. 审查：执行后审查结果，决定完成或 replan

可用工具：
  - ask_clarification — 用户输入不清晰时追问
  - create_plan — 创建计划（输出 SubTask 列表）
  - update_plan — 更新计划状态
  - get_plan_status — 查询计划状态
"""

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.plan_toolkit import create_plan, get_plan_status, update_plan
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger

CLARIFICATION_SECTION = """
<Clarification>
## 澄清职责
在**开始规划之前**，必须先分析用户输入是否清晰。

### 逐项检查
1. **主体（谁/什么）**：用户明确说要操作什么对象或获取什么信息？
   ❌ "帮我查一下" → 查什么？
   ✅ "查北京天气" → 主体=天气

2. **范围（时间/地点/条件）**：范围是否完整？
   ❌ "今天天气" → 哪个城市？
   ❌ "北京天气" → 哪一天？
   ✅ "北京今天天气" → 完整

3. **动作/目标**：要求做什么？产出是什么？
   ❌ "看看这个文件" → 看什么？
   ✅ "搜索人工智能进展" → 明确

4. **指代消解**：用了代词（它、这个、那里）？有明确指代吗？
   ❌ "帮我优化一下"（无前文）
   ❌ "我在那" → 完全模糊

### 决策
- 四项中有任何不明确 → 调用 ask_clarification，不要开始规划
- 全部明确后再进行规划
- 每次只追问最关键的缺失信息
</Clarification>
"""

PLAN_SYSTEM_PROMPT_TEMPLATE = """<Role>
你是一个高级任务规划与审查助手。你负责：
1. **澄清**：分析用户输入是否清晰，不清晰则追问
2. **规划**：拆解为可并行执行的子任务
3. **审查**：审查执行结果，决定完成或 replan
</Role>

{clarification_section}

<AvailableAgents>
以下 agent 可用于执行子任务，在 SubTask.execution_agent 字段中使用：
{agent_descriptions}
</AvailableAgents>

<AgentCapabilities>
以下系统能力在执行阶段可用，规划时请据此设计子任务步骤：
{capability_descriptions}
</AgentCapabilities>

<PlanningTools>
你可使用以下工具：
1. **ask_clarification(question, clarification_type, context?, options?)** — 用户需求不清晰时追问
2. **create_plan(title, steps, dependencies?)** — 创建计划
3. **update_plan(plan_id, title?, steps?, dependencies?)** — 更新计划
4. **get_plan_status(plan_id)** — 查询计划状态
</PlanningTools>

<OutputFormat>
子任务 JSON Schema：
{{
  "tasks": [
    {{
      "plan_id": "唯一标识",
      "name": "简短名称",
      "desc": "详细描述",
      "execution_agent": "general_agent",
      "sort": 0,
      "deps": [],
      "step_statuses": "not_started",
      "blocked_message": "",
      "result": ""
    }}
  ],
  "success": false,
  "result": ""
}}
</OutputFormat>

<TaskDesignRules>
1. 每个子任务聚焦一个具体可执行的目标
2. 通过 sort 和 deps 表达依赖关系
3. 依赖任务的结果用 ${{plan_id_result}} 引用
4. 无依赖的子任务可以同时执行
5. 执行过程中可根据中间结果新增子任务
</TaskDesignRules>

<ReviewWorkflow>
### Step 1: 检查是否全部完成
- 所有子任务 completed → success=true, 输出最终答案
- 有 blocked 且系统无法解决 → 输出已获得结果 + 阻塞说明

### Step 2: 如果未完成
- 使用 get_plan_status 查询最新状态
- 分析已完成子任务结果
- 判断是否需要追加新子任务
- 调用 update_plan 更新计划
</ReviewWorkflow>

<CapabilityBoundary>
- 如果系统能力无法满足用户需求，直接告知用户不支持
</CapabilityBoundary>"""


def _build_system_prompt(agent_descriptions: str = "", capability_descriptions: str = "") -> str:
    return PLAN_SYSTEM_PROMPT_TEMPLATE.format(
        clarification_section=CLARIFICATION_SECTION,
        agent_descriptions=agent_descriptions or "- general_agent: 通用执行 agent，可调用所有工具",
        capability_descriptions=capability_descriptions or "",
    )


def _format_clarification_message(args: dict) -> str:
    question = args.get("question", "")
    clarification_type = args.get("clarification_type", "missing_info")
    context = args.get("context")
    options = args.get("options", [])

    if isinstance(options, str):
        try:
            options = json.loads(options)
        except (json.JSONDecodeError, TypeError):
            options = [options]
    if options is None:
        options = []
    elif not isinstance(options, list):
        options = [options]

    type_icons = {
        "missing_info": "❓",
        "ambiguous_requirement": "🤔",
        "approach_choice": "🔀",
        "risk_confirmation": "⚠️",
        "suggestion": "💡",
    }
    icon = type_icons.get(clarification_type, "❓")

    parts = []
    if context:
        parts.append(f"{icon} {context}")
        parts.append(f"\n{question}")
    else:
        parts.append(f"{icon} {question}")
    if options:
        parts.append("")
        for i, option in enumerate(options, 1):
            parts.append(f"  {i}. {option}")

    return "\n".join(parts)


async def plan_model_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    context = runtime.context
    llm = context.plan_llm
    writer = get_stream_writer()
    trace_id = trace_id_ctx_var.get()

    writer({"type": THINK_MES, "messages": "📋 分析需求，制定执行计划...", "trace_id": trace_id})

    # 构建 system prompt
    from deerflow.tools.v2 import describe_execute_tools, get_plan_tools

    capability_desc = describe_execute_tools()
    system_prompt = _build_system_prompt(capability_descriptions=capability_desc)

    # 获取已有任务列表（review/replan 场景）
    existing_tasks = state.get("plan_tasks", [])
    plan_context = ""
    if existing_tasks:
        plan_context = "\n".join(f"- [{t.step_statuses}] {t.name}: {t.result[:200] if t.result else '待执行'}" for t in existing_tasks)

    # 构建消息
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
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

    # 绑定工具：ask_clarification + 规划工具
    plan_tools = [create_plan, update_plan, get_plan_status] + get_plan_tools()
    bound_llm = llm.bind_tools(plan_tools)

    for attempt in range(1, 4):
        try:
            result = await bound_llm.ainvoke(messages)

            if not hasattr(result, "tool_calls") or not result.tool_calls:
                writer({"type": THINK_MES, "messages": "无需拆解，直接回答", "trace_id": trace_id})
                return {"messages": [result], "completed": True}

            # 检查是否调用了 ask_clarification
            if any(tc.get("name") == "ask_clarification" for tc in result.tool_calls):
                tool_messages: list[ToolMessage] = []
                for tc in result.tool_calls:
                    if tc.get("name") == "ask_clarification":
                        tc_id = tc.get("id", "")
                        tc_args = tc.get("args", {})
                        tool_result = _format_clarification_message(tc_args)
                        tool_messages.append(ToolMessage(content=tool_result, tool_call_id=tc_id, name="ask_clarification"))
                writer({"type": THINK_MES, "messages": "需要用户补充信息", "trace_id": trace_id})
                return {"messages": [result] + tool_messages, "completed": True}

            # 执行规划工具
            tool_messages = []
            has_create = False

            for tc in result.tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})

                if tc_name == "create_plan":
                    if has_create:
                        continue
                    has_create = True

                result_text = ""
                if tc_name == "create_plan":
                    result_text = str(await create_plan.ainvoke(tc_args))
                elif tc_name == "update_plan":
                    result_text = str(await update_plan.ainvoke(tc_args))
                elif tc_name == "get_plan_status":
                    result_text = str(await get_plan_status.ainvoke(tc_args))
                else:
                    result_text = f"未知工具: {tc_name}"

                tool_messages.append(ToolMessage(content=result_text, tool_call_id=tc_id, name=tc_name))

            writer(
                {
                    "type": THINK_MES,
                    "messages": f"📋 规划完成，tool_calls: {[tc.get('name') for tc in result.tool_calls]}",
                    "trace_id": trace_id,
                }
            )
            return {"messages": [result] + tool_messages}

        except Exception as e:
            logger.error("Plan 第 {} 次失败: {}", attempt, e, extra={"trace_id": trace_id})
            if attempt == 3:
                return {"messages": [AIMessage(content="计划生成失败，请重新描述需求。")], "completed": True}

    return {"completed": True}
