"""
REVIEW 节点：执行完毕后评审步骤结果，决定是汇总输出还是重新规划。

职责：
  1. 接收 step_dispatch_node 的执行结果（plan_context + step_notes）
  2. 让 Plan LLM 评审当前结果是否能回答用户原始问题
  3. 如果能回答 → 生成友好的最终答案返回给用户
  4. 如果不能（缺信息/步骤不完整）→ 调用 update_plan 补充步骤（replan）
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.nodes.constants import THINK_MES
from deerflow.agentsv2.plan_storage import get_plan_storage
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger

REVIEW_SYSTEM_PROMPT = """你是一个任务评审助手。你的职责是评审计划执行结果。

## 上下文
- 以下是已执行的步骤及其结果
- 你的任务是根据这些结果判断：**当前信息是否足够回答用户的问题？**

## 判断标准

### 足够回答的情况
- 用户的问题已经被完整回答，所有必要信息都已获取
- 步骤执行结果中包含了用户问题的完整答案
- 此时你应输出一个友好的最终答案给用户

### 需要补充的情况
- 当前结果不足以回答用户问题
- 有明确缺失的信息可以通过添加步骤来补充
- 之前有步骤执行失败（blocked）但可以通过其他方式重试

### 无法回答的情况
- 系统没有相应的能力
- 多次尝试后仍然无法获取所需信息
- 此时如实告诉用户当前系统不支持

## 输出
1. 如果足够回答：直接给出完整的最终答案（格式友好，直接呈现给用户）
2. 如果需要补充：说明还需要什么，输出标记 [[REPLAN]] 并描述需要什么新步骤
3. 如果无法回答：如实告知用户"""


async def review_node(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    """评审节点：检查执行结果，决定汇总输出还是 replan。"""
    context = runtime.context
    llm = context.plan_llm
    writer = get_stream_writer()
    trace_id = trace_id_ctx_var.get()

    plan_id = state.get("plan_id", "")
    plan_context = state.get("plan_context", "")
    user_message = state.get("user_message", "")

    if not plan_id or not plan_context:
        # 没有计划或有执行结果，直接结束
        return {"plan_completed": True}

    storage = get_plan_storage()
    plan = await storage.load(plan_id)
    progress = plan.get_progress() if plan else {}
    total = progress.get("total", 0)
    completed = progress.get("completed", 0)
    blocked = progress.get("blocked", 0)
    all_done = completed == total > 0

    # 构建评审消息
    messages: list[BaseMessage] = [
        SystemMessage(content=REVIEW_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## 用户原始问题\n{user_message}\n\n"
                f"## 计划执行结果\n{plan_context}\n\n"
                f"## 计划状态\n"
                f"- 总步骤: {total}\n"
                f"- 已完成: {completed}\n"
                f"- 阻塞: {blocked}\n"
                f"- 是否全部完成: {'是' if all_done else '否'}\n\n"
                f"请评审这些结果是否能回答用户问题。"
                f"如果能，直接给出最终答案。如果需要更多步骤才能回答，请说明需要什么。"
                f"如果无法回答，如实告知用户。"
            )
        ),
    ]

    writer(
        {
            "type": THINK_MES,
            "messages": "📋 评审执行结果...",
            "trace_id": trace_id,
        }
    )

    try:
        result = await llm.ainvoke(messages)
        content = str(result.content) if result.content else ""

        # 检查 LLM 是否要求 replan
        if "[[REPLAN]]" in content:
            # 提取 replan 描述
            replan_desc = content.split("[[REPLAN]]")[-1].strip()

            writer(
                {
                    "type": THINK_MES,
                    "messages": f"🔄 需要补充步骤: {replan_desc[:200]}",
                    "trace_id": trace_id,
                }
            )

            # 让 Plan LLM 调用 update_plan 来添加新步骤
            # 构建包含当前上下文的 replan prompt
            from deerflow.agentsv2.plan_toolkit import update_plan

            replan_prompt = f"## 当前已有步骤和结果\n{plan_context}\n\n## 需要补充\n{replan_desc}\n\n请规划还需要什么步骤。使用 update_plan 工具来补充步骤。"

            replan_msg = HumanMessage(content=replan_prompt)
            from deerflow.agentsv2.nodes.plan_model_node import build_plan_system_prompt

            system_text = build_plan_system_prompt(
                existing_plan_id=plan_id,
                existing_context=plan_context,
            )

            bound_llm = llm.bind_tools([update_plan])
            replan_result = await bound_llm.ainvoke(
                [
                    HumanMessage(content=system_text),
                    replan_msg,
                ]
            )

            # 如果有 tool_calls，执行 update_plan
            tool_msgs = []
            if hasattr(replan_result, "tool_calls") and replan_result.tool_calls:
                for tc in replan_result.tool_calls:
                    if tc.get("name") == "update_plan":
                        tc_args = tc.get("args", {})
                        tc_id = tc.get("id", "")
                        result_text = await update_plan.ainvoke(tc_args)
                        from langchain_core.messages import ToolMessage

                        tool_msgs.append(ToolMessage(content=result_text, tool_call_id=tc_id, name="update_plan"))

            writer(
                {
                    "type": THINK_MES,
                    "messages": "🔄 计划已更新，继续执行...",
                    "trace_id": trace_id,
                }
            )

            # 返回 replan 标记，路由回到 step_dispatch_node
            return {
                "plan_completed": False,
                "messages": [replan_result] + tool_msgs,
            }

        else:
            # LLM 直接输出了最终答案
            writer(
                {
                    "type": THINK_MES,
                    "messages": "✅ 评审完成，生成最终答案",
                    "trace_id": trace_id,
                }
            )

            final_msg = AIMessage(content=content)

            return {
                "plan_completed": True,
                "messages": [final_msg],
            }

    except Exception as e:
        logger.error(f"Review 失败: {e}", extra={"trace_id": trace_id})
        # 降级：用已有结果直接返回
        fallback = AIMessage(content=(f"## 执行结果\n\n{plan_context}\n\n*计划状态: {completed}/{total} 步骤完成*"))
        return {
            "plan_completed": True,
            "messages": [fallback],
        }
