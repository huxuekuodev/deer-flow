"""
路由函数：三节点循环 PLAN → ToolNode → OBSERVE。

路由规则：
  - route_after_plan:  todo_list 非空 → tools(ToolNode)，否则 END
  - route_after_tools: 有 AIMessage.tool_calls 需要继续执行吗？→ 继续tools或observe
  - route_after_observe: 还有下一阶段且未完成 → tools，否则 END

关键：ToolNode 产生 ToolMessages 后，OBSERVE 读取并决定是否产出新 tool_calls
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from deerflow.agentsv2.thread_state import ThreadState


async def route_after_plan(state: ThreadState) -> str:
    """Plan → ToolNode（有计划）或 END。"""
    todo = state.get("todo_list", [])
    return "tools" if todo else END


async def route_after_tools(state: ThreadState) -> str:
    """ToolNode → 最后的消息是 AIMessage.tool_calls？→ 继续tools，否则observe"""
    msgs = state.get("messages", [])
    if not msgs:
        return "observe_node"

    last = msgs[-1]
    # 如果最后是带 tool_calls 的 AIMessage，ToolNode 继续执行
    if isinstance(last, AIMessage) and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "observe_node"


async def route_after_observe(state: ThreadState) -> str:
    """Observe → 最后是 tool_calls？→ 继续tools，否则END"""
    msgs = state.get("messages", [])
    if not msgs:
        return END

    last = msgs[-1]
    if isinstance(last, AIMessage) and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    cp = state.get("current_phase", -1)
    return END if cp < 0 else "tools"
