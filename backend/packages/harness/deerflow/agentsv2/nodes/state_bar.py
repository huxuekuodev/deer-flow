"""Agent 状态栏工具：将 todo_list 渲染为 LLM 可见的 XML 状态栏。

每次 LLM 调用前，将 todo_list 的完整状态格式化为一段可见文本，
让 LLM 始终"看见"全局进度（Manus "复述操控注意力"机制）。
"""

from deerflow.agentsv2.thread_state import Task, TodoItem

STATE_BAR_TEMPLATE = """<agent_state_bar>
当前阶段: 第 {current_phase} 阶段 / 共 {total_phases} 阶段

{phase_items}

可用工具：
- update_todo_status(task_id, done) — 标记某个子任务完成
- rewrite_todo_list(new_todo_list) — 重写后续阶段的 todo_list
</agent_state_bar>"""


def _format_task_line(task: Task) -> str:
    status = "✅" if task.done else "⬜"
    line = f"  {status} [{task.tool_name}] {task.task_desc}"
    if task.result:
        preview = task.result[:200] + "..." if len(task.result) > 200 else task.result
        line += f"\n    ↳ 结果: {preview}"
    return line


def _format_phase_section(index: int, item: TodoItem, current_phase: int) -> str:
    if item.done:
        status_mark = "[✅ 已完成]"
    elif index == current_phase:
        status_mark = "[🔄 进行中]"
    else:
        status_mark = "[⬜ 待执行]"

    lines = [f"阶段{index + 1} {status_mark}: {item.phase_desc}"]

    if item.context:
        context_preview = item.context[:300] + "..." if len(item.context) > 300 else item.context
        lines.append(f"  📎 依赖上下文: {context_preview}")

    for task in item.todo:
        lines.append(_format_task_line(task))

    return "\n".join(lines)


def format_todo_state_bar(todo_list: list[TodoItem], current_phase: int) -> str:
    """把 todo_list 渲染为人类可读的状态栏 XML 块。

    用法：拼到 system prompt 尾部，让 LLM 在每次调用前"看见"全局进度。
    """
    if not todo_list:
        return "<agent_state_bar>\n当前没有待办事项。\n</agent_state_bar>"

    phase_items = []
    for i, item in enumerate(todo_list):
        phase_items.append(_format_phase_section(i, item, current_phase))

    return STATE_BAR_TEMPLATE.format(
        current_phase=current_phase + 1,
        total_phases=len(todo_list),
        phase_items="\n\n".join(phase_items),
    )
