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
import uuid

from dotenv import load_dotenv

from deerflow.agentsv2.lead_agent.agent import GraphAgent
from deerflow.core.context import trace_id_ctx_var
from deerflow.core.log import logger
from deerflow.runtime import RunContext

load_dotenv()


async def main():
    trace_id = uuid.uuid4().hex
    trace_id_ctx_var.set(trace_id)
    logger.info("debug test start")
    from deerflow.config import get_app_config
    from deerflow.config.app_config import apply_logging_level
    from deerflow.runtime.checkpointer.async_provider import make_checkpointer
    from deerflow.runtime.msg_history import make_msg_history_pool
    from deerflow.tracing import build_tracing_callbacks

    app_config = get_app_config()
    apply_logging_level(app_config.log_level)
    from langchain_core.messages import HumanMessage

    config = {
        "configurable": {
            "thread_id": "debug-thread-003",
            "thinking_enabled": True,
            "is_plan_mode": True,
            "model_name": "deepseek-reasoner",
            "trace_id": trace_id,
        }
    }
    tracing_callbacks = build_tracing_callbacks(trace_id=trace_id)
    if tracing_callbacks:
        existing = config.get("callbacks") or []
        if not isinstance(existing, list):
            existing = list(existing)
        config["callbacks"] = [*existing, *tracing_callbacks]

    async with make_checkpointer(app_config=app_config) as checkpointer, make_msg_history_pool(app_config.msg_history_database) as msg_history_pool:
        runcontext = RunContext(checkpointer=checkpointer, msg_history_pool=msg_history_pool)
        agent = GraphAgent(config, runcontext)
        state = {"messages": [HumanMessage(content="北京，今天的天气")]}
        ai_content = ""
        async for chunk in agent.astream(state):
            if chunk["type"] == "messages":
                message_chunk, metadata = chunk["data"]
                if message_chunk.content:
                    logger.info(message_chunk.content, end="|", flush=True)
            elif chunk["type"] == "custom":
                logger.info(f"Status: {chunk['data']['type']}")
            elif chunk["type"] == "values":
                ai_content = chunk["data"]["messages"][-1].content
                logger.info(ai_content)


if __name__ == "__main__":
    asyncio.run(main())
