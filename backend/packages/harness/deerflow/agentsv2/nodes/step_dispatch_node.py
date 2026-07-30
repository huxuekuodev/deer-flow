"""
Step Dispatch Node + Fan-out Router：筛选可执行任务，标记 in_progress，并行派发。

职责分离：
  - step_dispatch_node（节点）：
      从 plan_tasks 筛选可执行任务（not_started + 依赖全部完成），
      标记为 in_progress，返回 state 更新。
  - step_fan_out_router（纯路由函数）：
      读取更新后的 state，返回 [Send("general_agent", {...})] 并行派发，
      或返回 END（全部完成）。
"""

from langgraph.constants import Send
from langgraph.graph import END

from deerflow.agentsv2.subtask import SubTask
from deerflow.agentsv2.thread_state import ThreadState


def _inject_dep_results(task: SubTask, plan_tasks: list[SubTask]) -> str:
    """将 task.deps 中已完成依赖的 result 注入到 desc 中。

    替换 desc 中的 {plan_id} 占位符为对应依赖任务的 result 内容。
    """
    filled = task.desc
    for dep_id in task.deps or []:
        dep_task = next((t for t in plan_tasks if t.plan_id == dep_id), None)
        if dep_task and dep_task.result:
            placeholder = "{" + dep_id + "}"
            filled = filled.replace(placeholder, dep_task.result)
    return filled


async def step_dispatch_node(state: ThreadState, **kwargs) -> dict:
    """Step Dispatch Node：从 plan_tasks 筛选可执行任务并标记为 in_progress。

    可执行条件：
      - step_statuses == "not_started"
      - 所有 deps 已完成 (step_statuses == "completed")

    Returns:
        plan_tasks 状态更新（由 merge reducer 写回 ThreadState）。
    """
    plan_tasks = state.get("plan_tasks", [])
    if not plan_tasks:
        return {}

    status_map = {t.plan_id: t.step_statuses for t in plan_tasks}
    status_updates: list[SubTask] = []

    for task in plan_tasks:
        if task.step_statuses != "not_started":
            continue

        # 检查依赖
        deps_ready = True
        for dep_id in task.deps or []:
            dep_status = status_map.get(dep_id)
            if dep_status != "completed":
                task.blocked_message = f"等待依赖任务 [{dep_id}] 完成"
                deps_ready = False
                break

        if deps_ready:
            task.step_statuses = "in_progress"
            task.blocked_message = ""
            status_updates.append(SubTask(plan_id=task.plan_id, step_statuses="in_progress"))

    if not status_updates:
        # 全部完成
        all_done = all(t.step_statuses == "completed" for t in plan_tasks)
        return {"completed": True} if all_done else {}

    return {"plan_tasks": status_updates}


def step_fan_out_router(state: ThreadState) -> list[Send] | str:
    """Fan-out 路由：读取 state，为每个 in_progress 任务创建 Send 派发。

    将已完成依赖的 result 注入到子任务 desc 中，使 general_agent 收到的描述完整。

    Returns:
        - list[Send]: 有可执行任务时并行派发到 execution_agent
        - END: 没有可派发任务
    """
    plan_tasks = state.get("plan_tasks", [])
    if not plan_tasks:
        return END

    sends: list[Send] = []
    for task in plan_tasks:
        if task.step_statuses != "in_progress":
            continue

        # 将依赖结果注入 desc
        filled_desc = _inject_dep_results(task, plan_tasks)

        sends.append(
            Send(
                task.execution_agent,
                {
                    "plan_id": task.plan_id,
                    "task_name": task.name,
                    "task_desc": filled_desc,
                    "plan_tasks": plan_tasks,
                },
            )
        )

    return sends if sends else END
