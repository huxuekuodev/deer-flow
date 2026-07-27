"""
Plan Toolkit: 提供给 LLM 的 create_plan / update_plan 工具。

Co-Sight 模式:
  - Planner 不再用 Pydantic structured output 生成 todo_list
  - 而是注册 create_plan / update_plan 为 LLM 可调用的工具
  - LLM 通过 function calling 构造计划（Co-Sight 风格）

在 LangGraph 中，这些工具注册到 ToolNode，供 plan_model_node 调用。
"""

from langchain_core.tools import tool

from deerflow.agentsv2.plan_document import PlanDocument, StepStatus, _normalize_dependencies
from deerflow.agentsv2.plan_storage import get_plan_storage


@tool
async def create_plan(
    title: str,
    steps: list[str],
    dependencies: dict = None,
) -> str:
    """
    创建 DAG 计划。将用户需求拆解为多个步骤，并可指定步骤间的依赖关系。

    Args:
        title: 计划标题，概括任务目标
        steps: 按执行顺序的步骤描述列表。每个步骤应是一个具体可执行的任务描述。
        dependencies: 步骤依赖关系，可选。格式: {"步骤索引": ["依赖的步骤索引"]}。
                     例如 {1: [0]} 表示步骤 1 依赖步骤 0。
                     索引从 0 开始。不传则默认按顺序依赖（步骤1依赖步骤0，依此类推）。

    Returns:
        创建结果字符串，包含 plan_id。
    """
    deps = _normalize_dependencies(dependencies or {})
    plan = PlanDocument(title=title, steps=steps, dependencies=deps)
    storage = get_plan_storage()
    await storage.save(plan)

    step_count = len(steps)
    dep_count = len(deps)
    return f"✅ 计划创建成功！\n  Plan ID: {plan.plan_id}\n  标题: {title}\n  步骤数: {step_count}\n  依赖关系数: {dep_count}\n  使用 update_plan 可以修改计划，使用 mark_step 可以标记步骤完成。"


@tool
async def update_plan(
    plan_id: str,
    title: str = None,
    steps: list[str] = None,
    dependencies: dict = None,
) -> str:
    """
    更新已有计划。保留已完成步骤的进度。

    Args:
        plan_id: 要更新的计划 ID
        title: 新标题（可选）
        steps: 新的步骤列表（可选）。保留已有进度，新步骤以 not_started 开始。
        dependencies: 新的依赖关系（可选）。

    Returns:
        更新结果。
    """
    storage = get_plan_storage()
    plan = await storage.load(plan_id)
    if not plan:
        return f"❌ 未找到 plan_id={plan_id}"

    if title:
        plan.title = title

    if steps:
        # 保留已完成步骤的状态 (Co-Sight 风格)
        old_steps = plan.steps
        old_statuses = plan.step_statuses
        old_notes = plan.step_notes
        old_tool_calls = plan.step_tool_calls

        new_steps = []
        new_statuses = {}
        new_notes = {}
        new_tool_calls = {}

        for s in steps:
            if s in old_steps and old_statuses.get(old_steps.index(s), "") != StepStatus.NOT_STARTED:
                idx = old_steps.index(s)
                new_steps.append(s)
                new_statuses[str(len(new_steps) - 1)] = old_statuses.get(str(idx), StepStatus.NOT_STARTED)
                new_notes[str(len(new_steps) - 1)] = old_notes.get(str(idx), "")
                if str(idx) in old_tool_calls:
                    new_tool_calls[str(len(new_steps) - 1)] = old_tool_calls[str(idx)]
            elif s in old_steps:
                idx = old_steps.index(s)
                new_steps.append(s)
                new_statuses[str(len(new_steps) - 1)] = StepStatus.NOT_STARTED
                new_notes[str(len(new_steps) - 1)] = old_notes.get(str(idx), "")
            else:
                new_steps.append(s)
                new_statuses[str(len(new_steps) - 1)] = StepStatus.NOT_STARTED
                new_notes[str(len(new_steps) - 1)] = ""
                new_tool_calls[str(len(new_steps) - 1)] = []

        plan.steps = new_steps
        plan.step_statuses = new_statuses
        plan.step_notes = new_notes
        plan.step_tool_calls = new_tool_calls

    if dependencies:
        plan.dependencies = _normalize_dependencies(dependencies)
    elif steps:
        plan.dependencies = {i: [i - 1] for i in range(1, len(steps))} if len(steps) > 1 else {}

    await storage.save(plan)
    return f"✅ 计划已更新。当前进度: {plan.get_progress()}"


@tool
async def get_plan_status(plan_id: str) -> str:
    """
    查询计划的当前状态。

    Args:
        plan_id: 计划 ID

    Returns:
        格式化的计划状态字符串。
    """
    storage = get_plan_storage()
    plan = await storage.load(plan_id)
    if not plan:
        return f"❌ 未找到 plan_id={plan_id}"
    return plan.format(with_detail=True)
