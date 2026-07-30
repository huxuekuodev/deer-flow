"""
通用执行 agent：接收 step_dispatch_node 通过 Send 派发的任务并执行。

流程：
  1. 从 state 获取 plan_id、task_name、task_desc（由 Send 注入）
  2. 从 plan_tasks 验证依赖任务是否已完成
  3. 调用 LLM 执行任务
  4. 修改任务状态为 completed，写入 result
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext, create_llm
from deerflow.agentsv2.subtask import SubTask
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.tools.v2 import describe_execute_tools_v2, get_execute_tools


async def general_agent(state: ThreadState, config: RunnableConfig, runtime: Runtime[GraphContext]) -> dict:
    """通用执行 agent。

    任务信息（plan_id / task_name / task_desc）由 step_fan_out_router 通过 Send 注入。
    依赖验证和状态回写通过 plan_tasks（ThreadState 中）完成。

    Args:
        state: 合并了 Send 注入字段的 ThreadState
            - plan_id: 当前要执行的任务 ID
            - task_name: 任务名称
            - task_desc: 已注入依赖结果的任务描述

    Returns:
        更新后的 plan_tasks（标记 completed + result）。
    """
    plan_id = state.get("plan_id", "")
    task_name = state.get("task_name", "未知任务")
    task_desc = state.get("task_desc", "")
    plan_tasks = state.get("plan_tasks", [])

    if not plan_id:
        return {"completed": True}

    # === 1. 验证依赖任务是否已完成 ===
    for dep_id in _get_deps_of(plan_id, plan_tasks):
        dep_task = _find_task(dep_id, plan_tasks)
        if dep_task is None or dep_task.step_statuses != "completed":
            return {"plan_tasks": [SubTask(plan_id=plan_id, step_statuses="not_started", blocked_message=f"依赖任务 [{dep_id}] 尚未完成")]}

    # === 2. 调用 LLM 执行任务 ===
    langfuse_client = runtime.context.langfuse_client
    system_prompt = langfuse_client.get_prompt("deerflow_v2/general_agent_system_prompt").compile(tools_desc=describe_execute_tools_v2())
    task_info = f"""任务名称：{task_name}
        任务描述：{task_desc}
        计划 ID：{plan_id}"""

    llm = create_llm(config)
    llm.bind_tools(get_execute_tools())
    agent = create_agent(model=llm, system_prompt=system_prompt)
    agent_result = await agent.ainvoke(
        {"messages": [HumanMessage(content=task_info)]},
        config=config,
    )
    agent_msgs = agent_result.get("messages", [])
    final_msg = agent_msgs[-1] if agent_msgs else AIMessage(content="")
    task_result = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    # === 3. 修改任务状态为 completed ===
    return {"plan_tasks": [SubTask(plan_id=plan_id, step_statuses="completed", result=str(task_result)[:2000])]}


def _find_task(plan_id: str, plan_tasks: list[SubTask]) -> SubTask | None:
    """从 plan_tasks 中查找指定 plan_id 的任务。"""
    return next((t for t in plan_tasks if t.plan_id == plan_id), None)


def _get_deps_of(plan_id: str, plan_tasks: list[SubTask]) -> list[str]:
    """获取指定任务的依赖列表。"""
    task = _find_task(plan_id, plan_tasks)
    return task.deps if task else []
