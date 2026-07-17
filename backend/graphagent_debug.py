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
import logging
import uuid

from dotenv import load_dotenv

from deerflow.agentsv2.lead_agent.agent import GraphAgent

load_dotenv()

_LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _setup_logging(log_level: int = logging.INFO) -> None:
    """Route logs to ``debug.log`` using *log_level* for the initial root/file setup.

    This configures the root logger and the ``debug.log`` file handler so logs do
    not print on the interactive console. It is idempotent: any pre-existing
    handlers on the root logger (e.g. installed by ``logging.basicConfig`` in
    transitively imported modules) are removed so the debug session output only
    lands in ``debug.log``.

    Note: later config-driven logging adjustments may change named logger
    verbosity without raising the root logger or file-handler thresholds set
    here, so the eventual contents of ``debug.log`` may not be filtered solely by
    this function's ``log_level`` argument.
    """
    root = logging.root
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.setLevel(log_level)

    file_handler = logging.FileHandler("debug.log", mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
    root.addHandler(file_handler)


async def main():
    # Install file logging first so warnings emitted while loading config do not
    # leak onto the interactive terminal via Python's lastResort handler.
    _setup_logging()

    from deerflow.config import get_app_config
    from deerflow.config.app_config import apply_logging_level

    app_config = get_app_config()
    apply_logging_level(app_config.log_level)
    from langchain_core.messages import HumanMessage

    # Create agent with default config
    trace_id = uuid.uuid4().hex
    print(f"自定义trace_id:{trace_id}")
    config = {
        "configurable": {
            "thread_id": "debug-thread-001",
            "thinking_enabled": True,
            "is_plan_mode": True,
            # Uncomment to use a specific model
            "model_name": "kimi-k2.5",
            "trace_id": trace_id,
        }
    }
    agent = GraphAgent(config)
    state = {"messages": [HumanMessage(content="我明天要去北京，我想知道北京的天气。")]}
    async for state in agent.astream(state):
        print(state)


if __name__ == "__main__":
    asyncio.run(main())
