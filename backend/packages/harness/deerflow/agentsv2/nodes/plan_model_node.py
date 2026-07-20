import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from deerflow.agentsv2 import ThreadState
from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.thread_state import TodoItem

logger = logging.getLogger(__name__)


class PlanOutput(BaseModel):
    need_plan: bool = Field(description="判断当前用户需求是否需要拆解为执行计划")
    todo_list: list[TodoItem] = Field(default_factory=list, description="可并行的任务列表")
    direct_answer: str = Field(default="", description="直接回答用户的问题")


async def route_after_plan(state: ThreadState) -> str:
    """
    声明式路由：根据 state 中的 todo_list 是否为空决定去向
    """
    if len(state.get("todo_list", [])) > 0:
        return "dispatch_node"  # 有计划，去任务下发节点
    else:
        return END  # 没计划（直接回答/澄清/失败），直接结束


async def plan_model_node(state: ThreadState, runtime: Runtime[GraphContext]) -> dict:
    """
    计划节点：只负责意图识别、任务拆解、校验、SSE推送，并返回状态更新。
    """
    context = runtime.context
    plan_llm = context.plan_llm
    writer = get_stream_writer()

    available_agents = ["weather", "general"]
    structured_llm = plan_llm.with_structured_output(PlanOutput)
    system_prompt = context.langfuse_client.get_prompt(context.app_config.langfuse_prompt_config.plan_system_prompt).compile(
        subagent_list="""
       -  weather: 用于查询天气信息的智能体。 需要提供时间、地点列如 20260701 北京
       -  general: 用于一般任务的智能体。例如 查询时间、计算等。
    """
    )
    # 获取用户长期记忆
    full_messages = [system_prompt] + state["messages"]

    max_retries = 3
    attempt = 0
    valid_todo_list = []
    need_plan_flag = False
    direct_answer_content = ""

    while attempt < max_retries:
        attempt += 1
        try:
            output: PlanOutput = structured_llm.invoke(full_messages)

            # 场景 A：不需要生成 TODO（直接回答 or 澄清问题）
            if not output.need_plan:
                need_plan_flag = False
                direct_answer_content = output.direct_answer
                break

            # 场景 B：需要生成 TODO，校验 agent_name 合法性
            need_plan_flag = True
            invalid_agents = []

            for todo_item in output.todo_list:
                if len(todo_item.todo) > 3:
                    todo_item.todo = todo_item.todo[:3]

                for task in todo_item.todo:
                    if task.agent_name not in available_agents:
                        invalid_agents.append(task.agent_name)

            if not invalid_agents:
                valid_todo_list = output.todo_list
                break
            else:
                error_msg = f"第 {attempt} 次尝试失败。以下 agent 不存在: {invalid_agents}。可用的 agent 是: {available_agents}。请重新规划。"
                full_messages.append(HumanMessage(content=error_msg))

        except Exception as e:
            logger.error(f"第 {attempt} 次解析失败: {str(e)}")
            error_msg = f"第 {attempt} 次解析失败: {str(e)}。请重新生成。"
            full_messages.append(HumanMessage(content=error_msg))

    # --- 重试结束后的分支处理 ---

    # 情况 1：重试 3 次仍然失败，降级处理
    if attempt == max_retries and (not need_plan_flag or not valid_todo_list) and not direct_answer_content:
        writer({"type": "plan_error", "message": "任务规划失败，请重试。"})
        # 返回空 todo_list，条件边会据此路由到 END
        return {"todo_list": [], "messages": [AIMessage(content="抱歉，我在处理时遇到了困难，请提供更详细的指令。")]}

    # 情况 2：不需要生成 TODO（直接回答 / 澄清问题）
    if not need_plan_flag:
        # 返回空 todo_list，条件边会据此路由到 END
        return {"todo_list": [], "messages": [AIMessage(content=direct_answer_content)]}

    # 情况 3：成功生成并校验通过 TODO
    for index, todo_item in enumerate(valid_todo_list):
        sse_event = {"type": "plan_step_created", "step": index + 1, "total_steps": len(valid_todo_list), "tasks": todo_item.model_dump(), "step_status": "pending"}
        writer(sse_event)

    summary_message = AIMessage(content=f"我已经为您生成了执行计划，共包含 {len(valid_todo_list)} 个阶段。即将开始执行...")

    # 返回非空 todo_list，条件边会据此路由到 dispatch_node
    return {"messages": [summary_message], "todo_list": valid_todo_list}
