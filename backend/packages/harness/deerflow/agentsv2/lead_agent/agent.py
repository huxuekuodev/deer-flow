from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.graph import StateGraph

from deerflow.agentsv2 import ThreadState
from deerflow.agentsv2.lead_agent import GraphContext, create_plan_llm
from deerflow.agentsv2.nodes import dispatch_node, plan_model_node
from deerflow.config.app_config import get_app_config
from deerflow.core.context import trace_id_ctx_var
from deerflow.runtime import RunContext


class GraphAgent:
    def __init__(self, config: RunnableConfig, runcontext: RunContext):
        self.config = config
        self._app_config = get_app_config()
        self._agent = (
            StateGraph(ThreadState, context_schema=GraphContext).add_node("plan_model_node", plan_model_node).add_node("dispatch_node", dispatch_node).set_entry_point("plan_model_node").compile(checkpointer=runcontext.checkpointer)
        )

    async def astream(self, messages, trace_id=None):
        """Stream the graph events for the given messages."""
        # Priority: explicit arg > ContextVar (set by TraceMiddleware)
        effective_trace_id = trace_id or trace_id_ctx_var.get()
        if effective_trace_id:
            self.config["trace_id"] = effective_trace_id
        async for state in self._agent.astream(
            stream_mode=["values", "messages", "custom"],
            input=messages,
            config=self.config,
            context=self.get_context(),
            version="v2",
        ):
            yield state

    def get_context(self) -> GraphContext:
        """Construct runtime context for the graph.

        Args:
            config: LangGraph runtime config (used for model resolution).
            app_config: DeerFlow application config.

        Returns:
            A GraphContext instance ready to be passed to ``astream(context=...)``.
        """
        plan_llm = create_plan_llm(self.config)
        return GraphContext(app_config=self._app_config, plan_llm=plan_llm, langfuse_client=Langfuse())
