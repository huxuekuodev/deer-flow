"""
三节点循环主图：PLAN → step_dispatch_node → review_node → (PLAN|END)。

循环流程:
  START → plan_model_node
    plan_completed → END（无 plan / 直接回答）
    创建 plan → step_dispatch_node

  step_dispatch_node
    执行 DAG 所有步骤 → review_node（始终）

  review_node
    评审执行结果
      足够回答 → 输出最终答案 → END
      需要补充 → replan 回到 plan_model_node（调用 update_plan 追加步骤）
      无法回答 → 如实告知用户 → END
"""

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.graph import END, START, StateGraph

from deerflow.agentsv2.lead_agent import GraphContext, create_llm
from deerflow.agentsv2.nodes import (
    plan_model_node,
    review_node,
    route_after_dispatch,
    route_after_plan,
    route_after_review,
    step_dispatch_node,
)
from deerflow.agentsv2.plan_storage import get_plan_storage
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.config.app_config import get_app_config
from deerflow.core.context import trace_id_ctx_var
from deerflow.runtime import RunContext


class GraphAgent:
    """
    三节点循环 GraphAgent (v2):
      START → plan_model_node → step_dispatch_node → review_node
                ↑                                    |
                └──────── replan ─────────────────────┘
    """

    def __init__(self, config: RunnableConfig, runcontext: RunContext):
        self.config = config
        self._app_config = get_app_config()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(ThreadState, context_schema=GraphContext)

        builder.add_node("plan_model_node", plan_model_node)
        builder.add_node("step_dispatch_node", step_dispatch_node)
        builder.add_node("review_node", review_node)

        builder.add_edge(START, "plan_model_node")

        # PLAN → plan_id? → step_dispatch_node : END
        builder.add_conditional_edges(
            "plan_model_node",
            route_after_plan,
            {
                "step_dispatch_node": "step_dispatch_node",
                END: END,
            },
        )

        # step_dispatch_node → 始终进入 review_node
        builder.add_conditional_edges(
            "step_dispatch_node",
            route_after_dispatch,
            {
                "review_node": "review_node",
            },
        )

        # review_node → 回到 plan_model_node (replan) : END
        builder.add_conditional_edges(
            "review_node",
            route_after_review,
            {
                "plan_model_node": "plan_model_node",
                END: END,
            },
        )

        return builder.compile()

    async def astream(self, messages, trace_id=None):
        tid = trace_id or trace_id_ctx_var.get()
        if tid:
            self.config["trace_id"] = tid

        gi = {"messages": messages} if not isinstance(messages, dict) else messages
        if isinstance(gi, dict):
            gi.setdefault("plan_id", "")
            gi.setdefault("plan_context", "")
            gi.setdefault("active_steps", [])
            gi.setdefault("plan_completed", False)
            gi.setdefault("user_message", "")

        agent = self._build_graph()
        ctx = self.get_context()

        async for st in agent.astream(
            stream_mode=["values", "messages", "custom"],
            input=gi,
            config=self.config,
            context=ctx,
            version="v2",
        ):
            yield st

    def get_context(self) -> GraphContext:
        return GraphContext(
            app_config=self._app_config,
            plan_llm=create_llm(self.config),
            langfuse_client=Langfuse(),
            plan_storage=get_plan_storage(),
        )
