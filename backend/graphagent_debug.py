#!/usr/bin/env python
"""
Debug script for lead_agent.
Run this file directly in VS Code with breakpoints.

Requirements:
    Run with `uv run` from the backend/ directory so that the uv workspace
    resolves deerflow-harness and app packages correctly:

        cd backend && PYTHONPATH=. uv run python debug.py

Usage:
    1. Set breakpoints in agent.py or other files
    2. Press F5 or use "Run and Debug" panel
    3. Input messages in the terminal to interact with the agent
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from deerflow.agentsv2.lead_agent.agent import GraphAgent
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger
from deerflow.runtime import RunContext

load_dotenv()


class StreamPrinter:
    """成熟的消息打印器：处理 values / messages / custom 三种 stream 模式。

    - values：完整 state 快照，按消息 id 去重，只打印新消息
    - messages：LLM 消息 token 块，按 (id, step) 聚合后打印完整内容
    - custom：thinkMessage 状态消息，美化输出
    """

    def __init__(self):
        self._seen_msg_ids: set[str] = set()
        self._message_buffers: dict[tuple[str, int], list[str]] = {}
        self._last_ai_content: str = ""

    @property
    def final_answer(self) -> str:
        """最后一次 AI 回复内容（用于持久化记录）。"""
        return self._last_ai_content

    @staticmethod
    def _msg_type(msg) -> str:
        """返回消息的中文类型标签。"""
        if isinstance(msg, HumanMessage):
            return "👤 用户"
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                return "🤖 Agent(调用工具)"
            return "🤖 Agent"
        if isinstance(msg, ToolMessage):
            return "🔧 工具结果"
        if isinstance(msg, SystemMessage):
            return "⚙️ 系统"
        return "💬 消息"

    def _format_message(self, msg) -> str | None:
        """格式化单条消息为可读日志，返回 None 表示无需打印。"""
        # 跳过无 id 或已见过的消息（values 快照去重）
        msg_id = getattr(msg, "id", None)
        if msg_id:
            if msg_id in self._seen_msg_ids:
                return None
            self._seen_msg_ids.add(msg_id)

        label = self._msg_type(msg)
        name = getattr(msg, "name", None)

        # AIMessage with tool_calls
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            calls = ", ".join(tc.get("name", "?") for tc in tool_calls)
            return f"{label} → 调用工具: {calls}"

        # ToolMessage
        if isinstance(msg, ToolMessage):
            content = (msg.content or "")[:500]
            return f"{label} [{name}]:\n{content}"

        # 普通文本消息
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            return None
        if name:
            return f"{label} [{name}]:\n{content}"
        return f"{label}: {content}"

    def handle_chunk(self, chunk: dict) -> None:
        """处理单个 stream chunk（兼容 v1/v2 包装）。"""
        if not isinstance(chunk, dict):
            return
        chunk_type = chunk.get("type", "values")
        data = chunk.get("data", chunk)

        if chunk_type == "custom":
            self._handle_custom(data)
        elif chunk_type == "values":
            self._handle_values(data)
        elif chunk_type == "messages":
            self._handle_messages(data)

    def _handle_custom(self, data) -> None:
        """custom: {"type": "thinkMessage", "messages": "...", "trace_id": ...}"""
        if not isinstance(data, dict):
            return
        status_type = data.get("type", "")
        message = str(data.get("messages", ""))
        if status_type == "thinkMessage":
            logger.info(f"💡 {message}")
        else:
            logger.info(f"📦 [{status_type}] {message}")

    def _handle_values(self, data) -> None:
        """values: 完整 state 快照，按消息 id 去重后打印新消息。"""
        if not isinstance(data, dict):
            return
        messages = data.get("messages") or []
        for msg in messages:
            # 记录最后一条 AI 内容（用于持久化）
            if isinstance(msg, AIMessage):
                content = (getattr(msg, "content", None) or "").strip()
                if content:
                    self._last_ai_content = content
            line = self._format_message(msg)
            if line:
                logger.info(line)

    def _handle_messages(self, data) -> None:
        """messages: (message_chunk, metadata)，按 (id, step) 聚合 token 块。"""
        if not (isinstance(data, (tuple, list)) and len(data) == 2):
            return
        msg_chunk, metadata = data
        if not isinstance(msg_chunk, AIMessage):
            return

        msg_id = getattr(msg_chunk, "id", None) or ""
        step = (metadata or {}).get("langgraph_step", 0)
        key = (msg_id, step)

        content = getattr(msg_chunk, "content", "") or ""
        self._message_buffers.setdefault(key, []).append(content)

        # 聚合完成后打印完整消息
        if getattr(msg_chunk, "response_metadata", {}).get("finish_reason") is not None:
            full = "".join(self._message_buffers.pop(key, [])).strip()
            if full:
                logger.info(f"🖋️ Agent 回复: {full}")


async def main():
    trace_id = Langfuse.create_trace_id()
    trace_id_ctx_var.set(trace_id)
    logger.info("debug test start")
    from deerflow.config import get_app_config
    from deerflow.config.app_config import apply_logging_level
    from deerflow.runtime.checkpointer.async_provider import make_checkpointer
    from deerflow.runtime.msg_history import make_msg_history_pool, record_message

    app_config = get_app_config()
    apply_logging_level(app_config.log_level)
    from langchain_core.messages import HumanMessage

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "debug-thread-050",
            "thinking_enabled": True,
            "is_plan_mode": True,
            "model_name": "deepseek-reasoner",
            "trace_id": trace_id,
        },
        "callbacks": [CallbackHandler(trace_context={"trace_id": trace_id})],
    }
    # langfuse = get_client()
    async with (
        make_checkpointer(app_config=app_config) as checkpointer,
        make_msg_history_pool(app_config.msg_history_database) as msg_history_pool,
    ):
        runcontext = RunContext(checkpointer=checkpointer, msg_history_pool=msg_history_pool)
        agent = GraphAgent(config, runcontext)
        userquery = "查询北京的天气"
        state = {"messages": [HumanMessage(content=userquery)]}
        await record_message(msg_history_pool, content=userquery, role=1, user_id="huxuekuo", thread_id="debug-thread-012", run_id="trace_id", model_name="deepseek-reasoner", metadata={})

        # 使用成熟的消息打印器
        printer = StreamPrinter()
        async for chunk in agent.astream(state):
            printer.handle_chunk(chunk)

        # 提取最终答案并持久化
        final_ai = printer.final_answer
        logger.info(f"✅ 最终答案（{len(final_ai)} 字符）")
        await record_message(
            msg_history_pool,
            content=final_ai,
            role=2,
            user_id="huxuekuo",
            thread_id="debug-thread-020",
            run_id="trace_id",
            model_name="deepseek-reasoner",
            metadata={},
        )


if __name__ == "__main__":
    asyncio.run(main())
