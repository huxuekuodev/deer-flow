"""Middleware for creating custom Langfuse observations around LLM calls.

Captures the **system prompt + current user question** (no history messages) as
input, and the **LLM's response** (text content + tool_calls as JSON) as output.
This creates a named ``observation`` (typed as a ``span``) in Langfuse that a
**Langfuse Evaluator** can target to verify "did the model correctly decide to
invoke a tool?".

How it works
------------
Uses ``wrap_model_call`` / ``awrap_model_call`` hooks — these wrap the actual
model invocation.  At that point the OTel context is active (pushed by the
graph-root Langfuse ``CallbackHandler``), so ``start_as_current_observation``
naturally becomes a child of the current OTel span.

``start_as_current_observation`` is a context manager — we wrap
``handler(request)`` with ``with ... as obs:``, which is exactly the design
pattern the SDK prescribes.  No manual ``__enter__``/``__exit__``.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langfuse import Langfuse

if sys.version_info >= (3, 12):
    from typing import override
elif TYPE_CHECKING:
    from typing import override as _override

    override = _override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class LangfuseObservationMiddleware(AgentMiddleware[AgentState]):
    """Wrap each LLM call with a custom Langfuse span for evaluator analysis.

    The span is created via ``start_as_current_observation`` around the model
    invocation in ``wrap_model_call`` / ``awrap_model_call``.
    """

    def __init__(self) -> None:
        self._lf = Langfuse()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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
            serialised = [
                {"name": tc["name"], "args": tc["args"], "id": tc.get("id")}
                for tc in ai_msg.tool_calls
            ]
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
        """Wrap model invocation in a Langfuse observation context manager.

        Extracts system prompt + current user question from the request,
        creates the observation around ``handler(request)``, then reads
        the output from the response.
        """
        if not self._is_langfuse_enabled():
            return handler(request)

        # Build input from request data
        user_question = self._get_last_user_question(request.messages)
        if not user_question:
            return handler(request)

        system_text = self._get_system_text(request.system_message, request.system_prompt)

        input_data: dict = {}
        if system_text:
            input_data["system_prompt"] = system_text
        input_data["user_question"] = user_question

        try:
            trace_id = self._lf.get_current_trace_id()
            if trace_id is None:
                return handler(request)

            with self._lf.start_as_current_observation(
                name="llm-decider-check",
                as_type="span",
                input=input_data,
            ) as obs:
                response = handler(request)
                output = self._resolve_output(response)
                if output:
                    obs.update(output=output)
                return response

        except Exception as exc:
            logger.debug("Langfuse observation failed, falling through: %s", exc)
            return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant — same logic as ``wrap_model_call``."""
        if not self._is_langfuse_enabled():
            return await handler(request)

        user_question = self._get_last_user_question(request.messages)
        if not user_question:
            return await handler(request)

        system_text = self._get_system_text(request.system_message, request.system_prompt)

        input_data: dict = {}
        if system_text:
            input_data["system_prompt"] = system_text
        input_data["user_question"] = user_question

        try:
            trace_id = self._lf.get_current_trace_id()
            if trace_id is None:
                return await handler(request)

            with self._lf.start_as_current_observation(
                name="llm-decider-check",
                as_type="span",
                input=input_data,
            ) as obs:
                response = await handler(request)
                output = self._resolve_output(response)
                if output:
                    obs.update(output=output)
                return response

        except Exception as exc:
            logger.debug("Langfuse observation failed, falling through: %s", exc)
            return await handler(request)
