"""Middleware for creating custom Langfuse observations around LLM calls.

Captures the **system prompt + current user question** (no history messages) as
input, and the **LLM's response** (text content + tool_calls as JSON) as output.
This creates a named ``observation`` (typed as a ``span``) in Langfuse that a
**Langfuse Evaluator** can target to verify "did the model correctly decide to
invoke a tool?".

Trace ID resolution
-------------------
The middleware reads ``trace_id`` from ``request.config["configurable"]["trace_id"]``
— the same field set by ``client.py`` (via ``_get_runnable_config``) and
``make_lead_agent`` (from the Gateway request's ``configurable``).  This avoids
relying on OTel context propagation (which is only active during the model call
itself, inside Langfuse's ``CallbackHandler.on_chat_model_start/end``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import Langfuse

logger = logging.getLogger(__name__)


class LangfuseObservationMiddleware(AgentMiddleware[AgentState]):
    """Wrap each LLM call with a custom Langfuse span for evaluator analysis.

    Reads ``trace_id`` from ``request.config["configurable"]["trace_id"]``
    to create the observation under the correct parent trace.
    """

    def __init__(self) -> None:
        self._lf = Langfuse()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_trace_id(request: ModelRequest) -> str | None:
        """Read the optional custom trace_id from the request's configurable."""
        request_config = getattr(request, "config", None)
        if not isinstance(request_config, dict):
            return None
        configurable = request_config.get("configurable")
        if isinstance(configurable, dict):
            return configurable.get("trace_id")
        return None

    @staticmethod
    def _is_langfuse_enabled() -> bool:
        try:
            from deerflow.config.tracing_config import get_enabled_tracing_providers

            return "langfuse" in get_enabled_tracing_providers()
        except Exception:
            return False

    @staticmethod
    def _get_last_user_question(messages: list) -> str | None:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                raw = msg.content
                if isinstance(raw, str) and raw.strip():
                    return raw
                if isinstance(raw, list):
                    texts: list[str] = []
                    for block in raw:
                        if isinstance(block, str):
                            texts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            candidate = block.get("text", "")
                            if isinstance(candidate, str):
                                texts.append(candidate)
                    joined = " ".join(texts).strip()
                    if joined:
                        return joined
                break
        return None

    @staticmethod
    def _get_system_text(system_message: SystemMessage | None, system_prompt: str | None) -> str | None:
        """Return the effective system text from either source."""
        if system_message is not None:
            raw = system_message.content
            if isinstance(raw, str) and raw.strip():
                return raw
        if system_prompt and system_prompt.strip():
            return system_prompt
        return None

    @staticmethod
    def _format_output(ai_msg: AIMessage) -> dict:
        output: dict = {}
        content = ai_msg.content
        if isinstance(content, str) and content.strip():
            output["content"] = content
        elif isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    candidate = block.get("text", "")
                    if isinstance(candidate, str):
                        texts.append(candidate)
            joined = "".join(texts).strip()
            if joined:
                output["content"] = joined

        if ai_msg.tool_calls:
            serialised = [{"name": tc["name"], "args": tc["args"], "id": tc.get("id")} for tc in ai_msg.tool_calls]
            output["tool_calls"] = json.dumps(serialised, ensure_ascii=False)
        return output

    @staticmethod
    def _resolve_output(response: ModelResponse) -> dict | None:
        """Extract the output dict from the last AIMessage in the response."""
        for msg in reversed(response.result):
            if isinstance(msg, AIMessage):
                return LangfuseObservationMiddleware._format_output(msg)
        return None

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model invocation — sync path."""
        if not self._is_langfuse_enabled():
            return handler(request)

        trace_id = self._get_trace_id(request)
        if not trace_id:
            return handler(request)

        user_question = self._get_last_user_question(request.messages)
        if not user_question:
            return handler(request)

        system_text = self._get_system_text(request.system_message, request.system_prompt)
        input_data: dict = {}
        if system_text:
            input_data["system_prompt"] = system_text
        input_data["user_question"] = user_question

        response = handler(request)

        try:
            output = self._resolve_output(response)
            observation = self._lf.span(
                name="llm-decider-check",
                trace_id=trace_id,
                input=input_data,
                output=output or {},
            )
            observation.end()
        except Exception as exc:
            logger.warning("Langfuse sync observation failed: %s", exc)

        return response

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap model invocation — async path (used by Gateway)."""
        if not self._is_langfuse_enabled():
            return await handler(request)

        trace_id = self._get_trace_id(request)
        if not trace_id:
            return await handler(request)

        user_question = self._get_last_user_question(request.messages)
        if not user_question:
            return await handler(request)

        system_text = self._get_system_text(request.system_message, request.system_prompt)
        input_data: dict = {}
        if system_text:
            input_data["system_prompt"] = system_text
        input_data["user_question"] = user_question

        response = await handler(request)

        try:
            output = self._resolve_output(response)
            observation = self._lf.span(
                name="llm-decider-check",
                trace_id=trace_id,
                input=input_data,
                output=output or {},
            )
            observation.end()
        except Exception as exc:
            logger.warning("Langfuse async observation failed: %s", exc)

        return response
