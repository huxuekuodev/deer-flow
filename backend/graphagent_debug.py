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
from langchain_core.runnables import RunnableConfig

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
    from deerflow.runtime.msg_history import make_msg_history_pool, record_message
    from deerflow.tracing import build_tracing_callbacks

    app_config = get_app_config()
    apply_logging_level(app_config.log_level)
    from langchain_core.messages import HumanMessage

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "debug-thread-025",
            "thinking_enabled": True,
            "is_plan_mode": True,
            "model_name": "deepseek-reasoner",
            "trace_id": trace_id,
        }
    }
    tracing_callbacks = build_tracing_callbacks(trace_id=trace_id)
    if tracing_callbacks:
        config["callbacks"] = [*tracing_callbacks]

    async with (
        make_checkpointer(app_config=app_config) as checkpointer,
        make_msg_history_pool(app_config.msg_history_database) as msg_history_pool,
    ):
        runcontext = RunContext(checkpointer=checkpointer, msg_history_pool=msg_history_pool)
        agent = GraphAgent(config, runcontext)
        userquery = "河北最凉爽的城市是那个？"
        state = {"messages": [HumanMessage(content=userquery)]}
        await record_message(msg_history_pool, content=userquery, role=1, user_id="huxuekuo", thread_id="debug-thread-012", run_id="trace_id", model_name="deepseek-reasoner", metadata={})
        ai_content = ""
        async for chunk in agent.astream(state):
            if chunk["type"] == "custom":
                logger.info(f"Status: {chunk['data']['type']}, {chunk['data']['messages']}")
            elif chunk["type"] == "values":
                ai_content = chunk["data"]["messages"][-1].content
                logger.info(ai_content)
        await record_message(msg_history_pool, content=ai_content, role=2, user_id="huxuekuo", thread_id="debug-thread-020", run_id="trace_id", model_name="deepseek-reasoner", metadata={})


if __name__ == "__main__":
    asyncio.run(main())
