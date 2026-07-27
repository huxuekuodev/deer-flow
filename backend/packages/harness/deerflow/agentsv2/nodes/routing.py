"""
路由函数：三节点循环 PLAN → step_dispatch_node → review_node → (PLAN|END)。

流程:
  START → plan_model_node
    plan_completed=True → END
    有 plan_id → step_dispatch_node

  step_dispatch_node → review_node
    (始终，由 edge 保证)

  review_node
    plan_completed=False → plan_model_node (replan 循环)
    plan_completed=True → END
"""

from langgraph.graph import END

from deerflow.agentsv2.thread_state import ThreadState


async def route_after_plan(state: ThreadState) -> str:
    """Plan 节点后路由。"""
    completed = state.get("plan_completed", False)
    plan_id = state.get("plan_id", "")

    if plan_id and not completed:
        return "step_dispatch_node"

    return END


async def route_after_dispatch(state: ThreadState) -> str:
    """dispatch 执行后，始终进入 review_node。"""
    return "review_node"


async def route_after_review(state: ThreadState) -> str:
    """review 后路由：
    - plan_completed=False → 回到 plan_model_node 做 replan
    - plan_completed=True → END
    """
    completed = state.get("plan_completed", False)
    return END if completed else "plan_model_node"
