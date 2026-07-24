"""
三节点循环：PLAN → tools(ToolNode) → OBSERVE。

标准 LangGraph ToolNode 模式：
  - PLAN/OBSERVE 产出 AIMessage.tool_calls
  - ToolNode 自动读取并执行工具
  - OBSERVE 读 ToolMessages，决策下一批 tool_calls 或结束
"""

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from deerflow.agentsv2.lead_agent import GraphContext, create_llm
from deerflow.agentsv2.lead_agent.tools import make_tools
from deerflow.agentsv2.nodes import (
    observe_node,
    plan_model_node,
    route_after_observe,
    route_after_plan,
    route_after_tools,
)
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.config.app_config import get_app_config
from deerflow.core.context import trace_id_ctx_var
from deerflow.runtime import RunContext


class GraphAgent:
    def __init__(self, config: RunnableConfig, runcontext: RunContext):
        self.config = config
        self._app_config = get_app_config()

    def _build_graph(self):
        """构建图。每次 astream 调用时构建（因为 tools 依赖 llm）。"""
        ctx = self.get_context()
        tools = make_tools(ctx.plan_llm)

        builder = StateGraph(ThreadState, context_schema=GraphContext)

        builder.add_node("plan_model_node", plan_model_node)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("observe_node", observe_node)

        builder.add_edge(START, "plan_model_node")

        builder.add_conditional_edges(
            "plan_model_node",
            route_after_plan,
            {
                "tools": "tools",
                END: END,
            },
        )
        builder.add_conditional_edges(
            "tools",
            route_after_tools,
            {
                "tools": "tools",
                "observe_node": "observe_node",
            },
        )
        builder.add_conditional_edges(
            "observe_node",
            route_after_observe,
            {
                "tools": "tools",
                END: END,
            },
        )

        return builder.compile()

    async def astream(self, messages, trace_id=None):
        tid = trace_id or trace_id_ctx_var.get()
        if tid:
            self.config["trace_id"] = tid

        gi = {"messages": messages} if not isinstance(messages, dict) else messages

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
        )
