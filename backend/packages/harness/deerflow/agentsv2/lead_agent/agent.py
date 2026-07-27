"""
三节点循环主图：PLAN → step_dispatch_node → review_node → (PLAN|END)。

支持 LangGraph checkpointer，相同 thread_id 自动恢复历史消息。

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

    checkpointer 支持:
      - __init__ 时从 runcontext 获取 checkpointer / msg_history_pool
      - _build_graph 编译时注入 checkpointer
      - 相同 thread_id 的多次调用自动恢复历史消息
    """

    def __init__(self, config: RunnableConfig, runcontext: RunContext):
        self.config = config
        self._app_config = get_app_config()
        self._checkpointer = runcontext.checkpointer if runcontext else None
        self._msg_history_pool = getattr(runcontext, "msg_history_pool", None) if runcontext else None
        self._agent = None  # 缓存编译好的图

    def _build_graph(self) -> StateGraph:
        """构建并编译图。使用缓存，避免每次重建。"""
        if self._agent is not None:
            return self._agent

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

        # 编译时注入 checkpointer
        if self._checkpointer is not None:
            self._agent = builder.compile(checkpointer=self._checkpointer)
        else:
            self._agent = builder.compile()

        return self._agent

    async def astream(self, messages, trace_id=None):
        tid = trace_id or trace_id_ctx_var.get()
        if tid:
            self.config["trace_id"] = tid

        agent = self._build_graph()
        ctx = self.get_context()

        # 关键：使用 checkpointer 时只传新消息，历史从 checkpoint 恢复
        input_data: dict = {}
        if isinstance(messages, dict):
            input_data = messages
        elif isinstance(messages, list):
            input_data = {"messages": messages}
        else:
            input_data = {"messages": [messages]}

        # 只有首次调用（没有已有 plan_id 时）才填充默认值
        for key, default in [
            ("plan_id", ""),
            ("plan_context", ""),
            ("active_steps", []),
            ("plan_completed", False),
            ("user_message", ""),
        ]:
            input_data.setdefault(key, default)

        async for st in agent.astream(
            stream_mode=["values", "messages", "custom"],
            input=input_data,
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
