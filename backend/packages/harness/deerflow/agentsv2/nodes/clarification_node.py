"""
澄清节点：在规划之前对用户的问题进行澄清。

职责：
  1. 读取用户消息，让 LLM 判断是否信息不足、需求模糊
  2. 如果需要澄清 → 调用 ask_clarification 工具 → 执行并产生 ToolMessage → 返回等待用户
  3. 不需要澄清 → 直接透传给 plan_model_node
  4. 用户回复澄清后 → 将澄清内容拼接为完整描述传递给后续节点
"""

import json

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.core.context import trace_id_ctx_var

CLARIFICATION_PROMPT = """<Role>
你是一个需求澄清助手。你的职责是分析用户输入，判断用户的描述是否清晰、完整。
如果发现用户的描述缺少关键信息，必须调用 ask_clarification 工具来追问。
</Role>

<Principle>
**只要用户的输入存在任何不确定性，就必须追问，直到需求完全明确。**
</Principle>

<InputAnalysis>
逐项检查用户输入，每项不满足都必须追问：

1. **主体（谁/什么）**：用户明确说要操作什么对象或获取什么信息了吗？
   - ❌ "帮我查一下" → 查什么？缺少查询对象
   - ❌ "我在那" → 查找什么信息？缺少需求主体
   - ✅ "查北京天气" → 主体明确（天气）

2. **范围（时间/地点/条件）**：是否需要限定范围？范围是否已提供？
   - ❌ "今天天气" → 哪个城市？缺少地点
   - ❌ "北京天气" → 哪天的天气？缺少时间
   - ❌ "我在那" → 你在哪里？缺少上下文
   - ✅ "北京今天天气" → 范围明确

3. **动作/目标**：用户要求做什么？产出是什么？
   - ❌ "帮我看看这个文件" → 看什么？分析什么？缺少目标
   - ❌ "运行一下" → 运行什么？缺少动作对象
   - ✅ "搜索一下人工智能的最新进展" → 动作和产出明确

4. **指代消解**：用户是否使用了代词（它、这个、那里等）？上文是否有明确指代？
   - ❌ "帮我优化一下"（没有前文）→ 优化什么？缺少指代
   - ❌ "处理一下这个"（没有前文）→ 处理什么？指代不明

**决策规则**：
- 以上四项中有任何一项不满足/不明确 → 立即调用 ask_clarification
- 只有当四项全部明确时 → 才认为描述清晰，无需澄清
</InputAnalysis>

<Examples>
用户输入："北京今天天气"
  分析：主体=天气 ✓，范围=北京+今天 ✓，动作=查询 ✓，指代=无 ✓
  → 清晰，无需澄清

用户输入："今天天气"
  分析：主体=天气 ✓，范围=今天+缺地点 ✗，动作=查询 ✓
  → 缺失地点 → 追问"请问您想查哪个城市的天气？"

用户输入："帮我查一下"
  分析：主体=缺 ✗，范围=缺 ✗，动作=查 ✓但查什么不明确，指代=无
  → 缺失太多 → 追问"请问您想查什么信息？"

用户输入："我在那"
  分析：主体=缺 ✗，范围=缺 ✗，动作=缺 ✗，指代="那"无指代
  → 完全模糊 → 追问"您好，我没有理解您的意思，请问您想做什么？"

用户输入："处理一下这个"
  分析：主体="这个"无指代 ✗，范围=缺 ✗，动作=缺 ✗，指代不明 ✗
  → 指代不明 + 缺信息 → 追问

用户输入："优化代码"
  分析：主体=代码 ✓，范围=缺（优化性能/可读性？），动作=优化但方向不明确 ✗
  → 需求模糊 → 追问"请问您希望优化代码的哪个方面？"
</Examples>

<Style>
- 每次只追问最关键的缺失信息，不要一次问太多
- 提问要具体、直接："请问您想查哪个城市的天气？"
- 不要替用户做假设，不知道就问
- 不要问"是否需要我帮您查天气？"这种废话，直接问缺少的信息
</Style>"""


def _filter_consumed_tool_pairs(messages: list[BaseMessage]) -> list[BaseMessage]:
    """过滤掉已经结束的 tool_calls + ToolMessage 配对，只保留纯文本消息。

    OpenAI 要求每个 ToolMessage 前面必须有对应的 AIMessage.tool_calls。
    当 checkpointer 恢复历史消息时，之前已经结束的 tool 调用会产生孤立的配对。
    这里只保留纯文本（HumanMessage / AIMessage 无 tool_calls），
    确保 LLM 不会看到历史 tool 调用痕迹。
    """
    result: list[BaseMessage] = []
    for m in messages:
        # 跳过 ToolMessage（前面配对已经在历史中结束）
        if isinstance(m, ToolMessage):
            continue
        # 跳过带 tool_calls 的 AIMessage（它们需要对应的 ToolMessage 跟随）
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
            continue
        # 只保留纯文本消息
        result.append(m)
    return result


def _format_clarification_message(args: dict) -> str:
    """Format the clarification arguments into a user-friendly message.

    Args:
        args: The tool call arguments containing clarification details

    Returns:
        Formatted message string
    """
    question = args.get("question", "")
    clarification_type = args.get("clarification_type", "missing_info")
    context = args.get("context")
    options = args.get("options", [])

    # Some models (e.g. Qwen3-Max) serialize array parameters as JSON strings
    # instead of native arrays. Deserialize and normalize so `options`
    # is always a list for the rendering logic below.
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except (json.JSONDecodeError, TypeError):
            options = [options]

    if options is None:
        options = []
    elif not isinstance(options, list):
        options = [options]

    # Type-specific icons
    type_icons = {
        "missing_info": "❓",
        "ambiguous_requirement": "🤔",
        "approach_choice": "🔀",
        "risk_confirmation": "⚠️",
        "suggestion": "💡",
    }

    icon = type_icons.get(clarification_type, "❓")

    # Build the message naturally
    message_parts = []

    # Add icon and question together for a more natural flow
    if context:
        # If there's context, present it first as background
        message_parts.append(f"{icon} {context}")
        message_parts.append(f"\n{question}")
    else:
        # Just the question with icon
        message_parts.append(f"{icon} {question}")

    # Add options in a cleaner format
    if options and len(options) > 0:
        message_parts.append("")  # blank line for spacing
        for i, option in enumerate(options, 1):
            message_parts.append(f"  {i}. {option}")

    return "\n".join(message_parts)


async def clarification_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    """澄清节点。

    职责：
      - 读取用户最新消息
      - 让 LLM 识别用户输入中是否有模糊、缺失信息
      - 如果需要澄清 → 调用 ask_clarification → 返回等待用户回复
      - 如果不需要 → 透传给 plan_model_node

    澄清节点不关心系统能力，只管把模糊问题变清晰。
    """
    context = runtime.context
    llm = context.plan_llm
    writer = get_stream_writer()
    trace_id = trace_id_ctx_var.get()

    user_msgs = state.get("messages", [])
    if not user_msgs:
        return {}

    writer({"type": THINK_MES, "messages": "🤔 分析需求是否明确...", "trace_id": trace_id})

    # 构建 prompt — 保留完整消息，让 LLM 看到上下文全貌
    messages: list[BaseMessage] = [SystemMessage(content=CLARIFICATION_PROMPT)]
    messages.extend(user_msgs)

    # 绑定 ask_clarification 工具
    from deerflow.tools.v2 import get_plan_tools

    plan_tools = get_plan_tools()  # 只含 ask_clarification
    bound_llm = llm.bind_tools(plan_tools)

    result = await bound_llm.ainvoke(messages)

    if not hasattr(result, "tool_calls") or not result.tool_calls:
        # LLM 认为不需要澄清
        writer({"type": THINK_MES, "messages": "需求明确，无需澄清", "trace_id": trace_id})
        return {}

    # 执行 ask_clarification

    tool_messages: list[ToolMessage] = []
    for tc in result.tool_calls:
        if tc.get("name") == "ask_clarification":
            tc_id = tc.get("id", "")
            tc_args = tc.get("args", {})
            tool_result = _format_clarification_message(tc_args)
            tool_messages.append(ToolMessage(content=tool_result, tool_call_id=tc_id, name="ask_clarification"))

    writer(
        {
            "type": THINK_MES,
            "messages": "需要用户补充信息",
            "trace_id": trace_id,
        }
    )

    return {
        "messages": [result] + tool_messages,
        "completed": True,
    }
