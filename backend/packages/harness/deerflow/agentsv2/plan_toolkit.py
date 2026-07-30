"""
Plan Toolkit（v2）：操作 SubTask 对象，通过 plan_tasks 存储到 ThreadState。

create_plan / update_plan / get_plan_status 都操作 SubTask 列表，
不再依赖 PlanDocument / PlanStorage。
"""

import json
import uuid
from contextvars import ContextVar
from typing import Any

from langchain_core.tools import tool

from deerflow.agentsv2.subtask import SubTask

# 桥接层（ContextVar，线程/异步安全）：plan_model_node 在调用 create_agent 前注入 plan_tasks，
# 工具将创建/查询结果写回，plan_model_node 在 agent.ainvoke 后读取。
_bridge_var: ContextVar[dict[str, Any]] = ContextVar("plan_toolkit_bridge", default={"plan_tasks": [], "created_task_dicts": None})


def _get_bridge() -> dict[str, Any]:
    """获取当前上下文的桥接层 dict。"""
    return _bridge_var.get()


def _generate_plan_id() -> str:
    """生成短 plan_id。"""
    return uuid.uuid4().hex[:8]


def _validate_tasks(tasks: list[dict]) -> list[str]:
    """验证任务列表的依赖完整性。

    自动补全缺失的 plan_id，然后检查：
    1. plan_id 不可重复
    2. deps 中引用的 plan_id 必须存在于 tasks 中
    3. 不可自依赖
    4. 不可循环依赖

    Returns:
        错误信息列表，空列表表示验证通过。
    """
    errors: list[str] = []
    plan_ids: set[str] = set()

    # 去重检查
    for t in tasks:
        pid = t.get("plan_id", "") or ""
        name = t.get("name", "unnamed")
        if pid in plan_ids:
            errors.append(f"plan_id '{pid}'（任务 '{name}'）重复")
        plan_ids.add(pid)

    # 检查 deps 完整性
    for t in tasks:
        pid = t.get("plan_id", "") or ""
        deps = t.get("deps", [])
        name = t.get("name", "unnamed")
        if not isinstance(deps, list):
            errors.append(f"任务 '{pid}'（{name}）的 deps 必须是 list")
            continue
        for dep_id in deps:
            if dep_id == pid:
                errors.append(f"任务 '{pid}'（{name}）不能依赖自身")
            elif dep_id not in plan_ids:
                errors.append(f"任务 '{pid}'（{name}）依赖的 plan_id '{dep_id}' 不存在于任务列表中")

    # 检查循环依赖（DFS）
    if not errors:
        adj: dict[str, list[str]] = {t.get("plan_id", ""): t.get("deps", []) for t in tasks if t.get("plan_id")}
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            rec_stack.add(node)
            for dep in adj.get(node, []):
                if dep in adj and _dfs(dep):
                    return True
            rec_stack.remove(node)
            return False

        for pid in list(adj.keys()):
            if pid not in visited and _dfs(pid):
                errors.append("检测到循环依赖，请检查任务间的依赖关系")
                break

    return errors


@tool
async def create_plan(
    title: str,
    tasks: list[dict],
) -> str:
    """
    创建 DAG 计划。将用户需求拆解为多个子任务。

    Args:
        title: 计划标题，概括任务目标
        tasks: 子任务列表。每个元素是 JSON 对象，包含：
            - plan_id (str, 可选): 子任务唯一标识，不传则自动生成
            - name (str, 必填): 子任务名称
            - desc (str, 必填): 子任务详细描述
            - execution_agent (str, 可选): 执行 agent，默认 general_agent
            - sort (int, 可选): 执行顺序序号
            - deps (list[str], 可选): 依赖的子任务 plan_id 列表

    Returns:
        创建结果 JSON 字符串。
    """
    # 1. 自动补全缺失的 plan_id
    for t in tasks:
        if not t.get("plan_id"):
            t["plan_id"] = _generate_plan_id()

    # 2. 验证依赖完整性
    errors = _validate_tasks(tasks)
    if errors:
        return json.dumps(
            {
                "action": "plan_created",
                "error": "; ".join(errors),
                "title": title,
                "tasks": [],
                "task_count": 0,
                "dep_count": 0,
            },
            ensure_ascii=False,
        )

    # 3. 写入 SubTask 列表，并通过桥接层通知 plan_model_node
    subtasks = []
    for t in tasks:
        subtasks.append(
            SubTask(
                plan_id=t["plan_id"],
                name=t.get("name", ""),
                desc=t.get("desc", ""),
                execution_agent=t.get("execution_agent", "general_agent"),
                sort=t.get("sort", 0),
                deps=t.get("deps", []),
            )
        )

    task_count = len(subtasks)
    dep_count = sum(len(t.deps) for t in subtasks)

    # 写回桥接层，plan_model_node 在 agent.ainvoke 后读取
    _get_bridge()["created_task_dicts"] = [t.model_dump() for t in subtasks]

    # 返回 JSON 序列化结果
    result = {
        "action": "plan_created",
        "title": title,
        "tasks": [t.model_dump() for t in subtasks],
        "task_count": task_count,
        "dep_count": dep_count,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
async def update_plan(
    tasks: list[dict],
) -> str:
    """
    更新已有子任务的状态或追加新子任务。

    已创建的任务内容（name/desc/deps/execution_agent/sort）不可修改，
    只能更新状态字段（step_statuses / result / blocked_message）。
    对于新追加的任务（新的 plan_id），所有字段均可设置。

    Args:
        tasks: 要更新的子任务列表。相同 plan_id 更新，新 plan_id 追加。
              每个元素包含：
            - plan_id (str, 必填): 子任务标识
            - step_statuses (str, 可选): 状态 (not_started/in_progress/completed/blocked)
            - result (str, 可选): 执行结果
            - blocked_message (str, 可选): 阻塞原因
            - desc (str, 可选): 新任务的描述（仅新任务生效）
            - name (str, 可选): 新任务的名称（仅新任务生效）
            - deps (list[str], 可选): 新任务的依赖（仅新任务生效）

    Returns:
        更新结果 JSON 字符串。
    """
    # 1. 自动补全缺失的 plan_id（新任务）
    for t in tasks:
        if not t.get("plan_id"):
            t["plan_id"] = _generate_plan_id()

    # 2. 验证依赖完整性
    errors = _validate_tasks(tasks)
    if errors:
        return json.dumps(
            {
                "action": "plan_updated",
                "error": "; ".join(errors),
                "tasks": [],
                "task_count": 0,
            },
            ensure_ascii=False,
        )

    # 3. 创建 SubTask 列表（保留现有状态的字段不变）
    subtasks = []
    for t in tasks:
        subtasks.append(
            SubTask(
                plan_id=t["plan_id"],
                name=t.get("name", ""),
                desc=t.get("desc", ""),
                execution_agent=t.get("execution_agent", "general_agent"),
                sort=t.get("sort", 0),
                deps=t.get("deps", []),
                step_statuses=t.get("step_statuses", "not_started"),
                result=t.get("result", ""),
                blocked_message=t.get("blocked_message", ""),
            )
        )

    # 写回桥接层，plan_model_node 在 agent.ainvoke 后读取
    _get_bridge()["created_task_dicts"] = [t.model_dump() for t in subtasks]

    result = {
        "action": "plan_updated",
        "tasks": [t.model_dump() for t in subtasks],
        "task_count": len(subtasks),
    }
    return json.dumps(result, ensure_ascii=False)


def _format_plan_status_from_bridge() -> str:
    """从桥接层读取 plan_tasks 并格式化为可读状态。"""
    tasks = _get_bridge().get("plan_tasks", [])
    if not tasks:
        return "当前没有活跃的计划。"
    lines = []
    for t_idx, subtask in enumerate(tasks):
        dep_str = f" (deps: {subtask.deps})" if subtask.deps else ""
        lines.append(f"[{subtask.step_statuses}] {subtask.name} ({subtask.plan_id}){dep_str}")
        if subtask.result:
            lines.append(f"  → {subtask.result[:200]}")
        if subtask.blocked_message:
            lines.append(f"  ⛔ {subtask.blocked_message}")
    return "当前计划状态：\n" + "\n".join(lines)


@tool
async def get_plan_status() -> str:
    """
    查询计划当前状态。

    从桥接层读取 plan_tasks，无需额外参数。
    """
    return _format_plan_status_from_bridge()
