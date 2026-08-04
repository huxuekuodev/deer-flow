"""
主图 v2：plan_model_node → step_dispatch_node (fan-out via Send) → general_agent → 循环 → END。

step_fan_out_router 是路由函数（不是节点），由 add_conditional_edges 调用。
当返回 [Send(...)] 时 LangGraph 自动并行派发到 general_agent；
当返回 END 时流程结束。
"""

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from deerflow.agentsv2.errors import should_retry
from deerflow.agentsv2.lead_agent import GraphContext, create_llm
from deerflow.agentsv2.lead_agent.subagent.general_agent import general_agent
from deerflow.agentsv2.nodes import (
    plan_model_node,
    step_dispatch_node,
    step_fan_out_router,
)
from deerflow.agentsv2.plan_storage import get_plan_storage
from deerflow.agentsv2.thread_state import ThreadState
from deerflow.config.app_config import get_app_config
from deerflow.core.context import trace_id_ctx_var
from deerflow.runtime import RunContext

# 规划节点重试策略：仅对可恢复的 LLM 错误重试（超时/连接/5xx/429/服务繁忙），
# 欠费/认证失败等不可恢复错误不重试（直接返回友好提示）。
_PLAN_RETRY_POLICY = RetryPolicy(
    max_attempts=3,  # 首次 + 2 次重试
    retry_on=should_retry,  # 复用 v2 errors 模块的错误分类
    initial_interval=0.5,  # 首次重试前等待 0.5s
    backoff_factor=2.0,  # 指数退避：0.5 → 1 → 2s
    jitter=True,  # 加随机抖动防惊群
)


class GraphAgent:
    def __init__(self, config: RunnableConfig, runcontext: RunContext):
        self.config = config
        self._app_config = get_app_config()
        self._checkpointer = runcontext.checkpointer if runcontext else None
        # 编译后的图类型依赖 compile() 的参数，泛型过于复杂且与静态检查摩擦大，
        # 统一用 Any（运行时安全，LangGraph 自身也推荐将编译图当作黑盒）。
        self._agent: Any = None

    def _build_graph(self) -> Any:
        if self._agent is not None:
            return self._agent
        builder = StateGraph(ThreadState, context_schema=GraphContext)

        # 节点函数返回 Partial<State>（dict），与 StateNode 严格签名有摩擦，
        # 但这是 LangGraph 的标准用法；用 cast 消除静态噪音。
        builder.add_node("plan_model_node", cast(Any, plan_model_node), retry_policy=_PLAN_RETRY_POLICY)
        builder.add_node("step_dispatch_node", cast(Any, step_dispatch_node))
        builder.add_node("general_agent", cast(Any, general_agent))

        # START → 规划·
        builder.add_edge(START, "plan_model_node")

        # 规划 → 已完成（最终答案）直接结束；
        # 有子任务走调度；无任务（澄清/直接回复）也结束
        builder.add_conditional_edges(
            "plan_model_node",
            lambda s: END if s.get("completed") else ("step_dispatch_node" if s.get("plan_tasks") else END),
        )

        # 调度 → fan-out 路由：返回 [Send(...)] 并行派发，
        # 全部完成返回 "plan_model_node" 审查并给最终答案
        # step_fan_out_router 是纯路由函数（非节点），由 framework 调用
        builder.add_conditional_edges(
            "step_dispatch_node",
            step_fan_out_router,
        )

        # general_agent 完成 → 回到调度继续下一轮
        builder.add_edge("general_agent", "step_dispatch_node")

        if self._checkpointer is not None:
            self._agent = builder.compile(checkpointer=self._checkpointer)
        else:
            self._agent = builder.compile()

        return self._agent

    async def astream(self, messages, trace_id=None):
        tid = trace_id or trace_id_ctx_var.get()
        if tid:
            # trace_id 放进 configurable（RunnableConfig 的标准扩展字段），
            # 与 v1 保持一致，避免在 TypedDict 上写未声明键。
            configurable = dict(self.config.get("configurable") or {})
            configurable["trace_id"] = tid
            self.config["configurable"] = configurable

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

        async for st in agent.astream(stream_mode=["values", "messages", "custom"], input=input_data, config=self.config, context=ctx, version="v2", subgraphs=True):
            yield st

    def get_context(self) -> GraphContext:
        return GraphContext(
            app_config=self._app_config,
            plan_llm=create_llm(self.config),
            langfuse_client=Langfuse(),
            plan_storage=get_plan_storage(),
        )
