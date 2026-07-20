from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.graph import StateGraph

from deerflow.agentsv2 import ThreadState
from deerflow.agentsv2.lead_agent import GraphContext, create_plan_llm
from deerflow.agentsv2.nodes import dispatch_node, plan_model_node
from deerflow.config.app_config import AppConfig, get_app_config

# Build the graph once at module level
_AGENT: StateGraph = StateGraph(ThreadState, context_schema=GraphContext).add_node("plan_model_node", plan_model_node).add_node("dispatch_node", dispatch_node).set_entry_point("plan_model_node").compile()


def get_context(config: RunnableConfig, *, app_config: AppConfig) -> GraphContext:
    """Construct runtime context for the graph.

    Args:
        config: LangGraph runtime config (used for model resolution).
        app_config: DeerFlow application config.

    Returns:
        A GraphContext instance ready to be passed to ``astream(context=...)``.
    """
    plan_llm = create_plan_llm(config)
    return GraphContext(app_config=app_config, plan_llm=plan_llm, langfuse_client=Langfuse())


class GraphAgent:
    def __init__(self, config: RunnableConfig):
        self.config = config

    async def astream(self, messages):
        ctx = get_context(self.config, app_config=get_app_config())

        async for state in _AGENT.astream(
            input=messages,
            config=self.config,
            context=ctx,
        ):
            yield state
