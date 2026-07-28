"""
PLAN 节点（Co-Sight 模式）。

变化（vs 旧版 Pydantic structured output）：
  1. LLM 通过 function calling 调用 create_plan/update_plan 工具
  2. 工具执行写入 storage（内存/Redis）
  3. 返回 plan_id → step_dispatch_node 读取执行
"""

import re
from datetime import datetime

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
from deerflow.tools.v2 import describe_execute_tools

PLAN_SYSTEM_PROMPT_HEADER = """<Role>
你是一个任务规划助手。分析用户需求，拆解为多步骤 DAG 计划。
</Role>
<planTools>
你只能使用以下工具来规划任务：
1. **create_plan(title, steps, dependencies?)** — 创建 DAG 计划
   - title: 计划标题
   - steps: 步骤描述列表（数组）
   - dependencies (可选): 依赖关系。如 {{"1": [0]}} 表示步骤1依赖步骤0
   - 不传 dependencies：默认顺序依赖

2. **update_plan(plan_id, title?, steps?, dependencies?)** — 修改计划（保留已完成步骤）
3. **get_plan_status(plan_id)** — 查询计划进度
<planTools>
<thinking_style>
## 规划原则
- 每个步骤应是一个具体可执行的任务描述
- 步骤间可指定依赖关系（DAG），无依赖的步骤可自动并行
- 步骤数量控制在 3-8 个之间
- 使用 create_plan 创建计划
 ## 步骤设计指南
 - 依赖执行能力的步骤，要描述清楚**要做什么**而不是直接写工具名
 - 例如：「调用天气查询接口获取北京今日天气数据」而不是「weather(北京)」
 - 步骤描述应让执行 LLM 理解目标并自主选择工具

 ## 能力边界
 - 如果上方的「可用的执行能力」中没有任何工具能满足用户的需求，则**不要创建计划**
 - 此时应明确告知用户当前系统不支持该需求，并说明你有哪些能力范围"
</thinking_style>

<clarification_system>
**工作流优先级：澄清 → 规划 → 行动**
1. **第一步**：在思考中分析请求——识别不清晰、缺失或模糊之处
2. **第二步**：如果需要澄清，立即调用 `ask_clarification` 工具——不要开始工作
3. **第三步**：所有澄清问题解决后，才进行规划

**关键规则：澄清永远先于行动。绝不要在执行过程中才开始澄清。**

**必须澄清的场景——在开始工作前必须调用 ask_clarification：**

1. **信息缺失**（`missing_info`）：未提供必要的细节
   - 例如：用户说"创建一个网页爬虫"但未指定目标网站
   - 例如："部署应用"但未指定环境
   - **必要操作**：调用 ask_clarification 获取缺失信息

2. **需求模糊**（`ambiguous_requirement`）：存在多种合理解释
   - 例如："优化代码"可能指性能、可读性或内存使用
   - 例如："让它更好"不清楚要改进哪个方面
   - **必要操作**：调用 ask_clarification 澄清确切需求

3. **方案选择**（`approach_choice`）：存在多种可行方案
   - 例如："添加认证"可以用JWT、OAuth、会话或API密钥
   - 例如："存储数据"可以用数据库、文件、缓存等
   - **必要操作**：调用 ask_clarification 让用户选择方案

4. **风险操作**（`risk_confirmation`）：破坏性操作需要确认
   - 例如：删除文件、修改生产配置、数据库操作
   - 例如：覆盖已有代码或数据
   - **必要操作**：调用 ask_clarification 获取明确确认

5. **建议**（`suggestion`）：你有推荐但希望获得批准
   - 例如："我建议重构这段代码。是否继续？"
   - **必要操作**：调用 ask_clarification 获取批准

**严格执行：**
- ❌ 不要先开始工作再在执行中提出澄清——先澄清
- ❌ 不要为了"效率"而跳过澄清——准确性比速度更重要
- ❌ 不要在信息缺失时做假设——总是要提问
- ❌ 不要靠猜测推进——停下来先调用 ask_clarification
- ✅ 在思考中分析 → 识别不清晰之处 → 在行动前提出
- ✅ 如果在思考中识别到需要澄清，必须立即调用该工具
- ✅ 调用 ask_clarification 后，执行将自动中断
- ✅ 等待用户回复——不要带着假设继续

**使用方法：**
```python
ask_clarification(
    question="你的具体问题？",
    clarification_type="missing_info",  # 或其他类型
    context="为什么需要这个信息",  # 可选但推荐
    options=["选项1", "选项2"]  # 可选，用于选择场景
)
```

**示例：**
用户："部署应用"
你（思考）：缺少环境信息——必须提出澄清
你（行动）：ask_clarification(
    question="应该部署到哪个环境？",
    clarification_type="approach_choice",
    context="我需要知道目标环境以进行正确配置",
    options=["开发环境", "预发布环境", "生产环境"]
)
[执行停止——等待用户回复]

用户："预发布环境"
你："正在部署到预发布环境..." [继续执行]
</clarification_system>

<execution_tool_descriptions>
## 以下工具只作为规划参考工具，规划助手不能调用任何工具
{execution_tool_descriptions}
</execution_tool_descriptions>

"""


def _extract_plan_id(result_text: str) -> str:
    """从工具结果中提取 plan_id。"""
    m = re.search(r"Plan\s*ID:\s*([a-f0-9]{32})", result_text, re.IGNORECASE)
    return m.group(1) if m else ""


async def _full_messages(state: ThreadState) -> list[BaseMessage]:
    """返回包含 ToolMessage 的完整消息列表。"""
    # 1.系统提示词
    existing_plan_id = state.get("plan_id", "")
    existing_context = state.get("plan_context", "")

    tool_desc = describe_execute_tools()
    actual_desc = tool_desc or ""
    system_text = PLAN_SYSTEM_PROMPT_HEADER.format(execution_tool_descriptions=actual_desc)
    base_messages: list[BaseMessage] = [SystemMessage(content=system_text)]
    # 2. 规划记录
    text = "<plan_record>"
    if existing_plan_id:
        text += f"\n\n已有 plan_id={existing_plan_id}。如需修改，请用 get_plan_status 查看进度后用 update_plan 更新。"
    if existing_context:
        text += f"\n\n之前已完成的上下文:\n{existing_context}"
    text += "</plan_record>"
    base_messages.append(HumanMessage(content=text))
    # 3. 时间追踪
    base_messages.append(HumanMessage(content=f"<current_time>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</current_time>"))
    # 4. 历史轨迹 + 用户输入
    user_msgs = state.get("messages", [])
    base_messages.extend(user_msgs)

    return base_messages


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
    base_messages = await _full_messages(state)
    # 绑定 plan 工具：规划工具 + ask_clarification
    from deerflow.tools.v2 import get_plan_tools

    plan_tools = [create_plan, update_plan, get_plan_status] + get_plan_tools()
    bound_llm = llm.bind_tools(plan_tools)

    for attempt in range(1, 4):
        try:
            # 重试时只保留原始消息（不包含之前失败的 ToolMessage）
            attempt_messages = list(base_messages)

            result = await bound_llm.ainvoke(attempt_messages)

            if not hasattr(result, "tool_calls") or not result.tool_calls:
                # 不需要规划，直接回答
                writer({"type": THINK_MES, "messages": "无需拆解，直接回答", "trace_id": trace_id})
                return {"plan_id": "", "plan_completed": True, "messages": [result]}

            # 执行 LLM 调用的 plan 工具
            # 注意：LLM 可能一次返回多个 create_plan（重复调用），只执行第一个
            tool_messages: list[ToolMessage] = []
            plan_id_found = ""
            has_created = False
            has_clarification = False

            for tc in result.tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")

                # 检查是否有 ask_clarification
                if tc_name == "ask_clarification":
                    has_clarification = True
                    continue

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

            # 如果 LLM 调用了 ask_clarification，不产生 ToolMessage，
            # 清除 AIMessage 的 tool_calls，仅保留提问文本返回。
            # 这样 checkpoint 恢复时 OpenAI 不会看到孤立的 tool_calls。
            if has_clarification:
                writer(
                    {
                        "type": THINK_MES,
                        "messages": "需要用户澄清",
                        "trace_id": trace_id,
                    }
                )
                # 清除 tool_calls，问题已提出，不应持久化 pending 调用
                result.tool_calls = []
                result.additional_kwargs.pop("tool_calls", None)
                return {"plan_id": "", "plan_completed": True, "messages": [result]}

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
            logger.error("Plan 第 {} 次失败: {}", attempt, e, extra={"trace_id": trace_id})
            if attempt == 3:
                fallback = AIMessage(content="计划生成失败，请重新描述需求。")
                return {"plan_completed": True, "messages": [fallback]}

    return {"plan_completed": True}
