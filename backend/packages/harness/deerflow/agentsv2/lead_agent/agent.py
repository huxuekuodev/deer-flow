"""
主图 v2：plan_model_node（澄清 + 规划 + 审查）→ step_dispatch_node → END。
"""

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.graph import END, START, StateGraph

from deerflow.agentsv2.lead_agent import GraphContext, create_llm
from deerflow.agentsv2.nodes import (
    plan_model_node,
)
from deerflow.agentsv2.plan_storage import get_plan_storage
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.config.app_config import get_app_config
from deerflow.core.context import trace_id_ctx_var
from deerflow.runtime import RunContext


class GraphAgent:
    def __init__(self, config: RunnableConfig, runcontext: RunContext):
        self.config = config
        self._app_config = get_app_config()
        self._checkpointer = runcontext.checkpointer if runcontext else None
        self._agent = None

    def _build_graph(self) -> StateGraph:
        if self._agent is not None:
            return self._agent

        builder = StateGraph(ThreadState, context_schema=GraphContext)

        builder.add_node("plan_model_node", plan_model_node)

        builder.add_edge(START, "plan_model_node")
        builder.add_edge("plan_model_node", END)

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

        input_data: dict = {}
        if isinstance(messages, dict):
            input_data = messages
        elif isinstance(messages, list):
            input_data = {"messages": messages}
        else:
            input_data = {"messages": [messages]}

        for key, default in [
            ("plan_tasks", []),
            ("completed", False),
            ("user_message", ""),
            ("final_answer", ""),
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
