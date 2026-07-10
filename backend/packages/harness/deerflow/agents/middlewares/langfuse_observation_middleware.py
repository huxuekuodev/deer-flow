"""Middleware for creating custom Langfuse observations around LLM calls.

Captures the **system prompt + current user question** (no history messages) as
input, and the **LLM's response** (text content + tool_calls as JSON) as output.
This creates a named ``observation`` (typed as a ``span``) in Langfuse that a
**Langfuse Evaluator** can target to verify "did the model correctly decide to
invoke a tool?".

Timing
------
The Langfuse ``CallbackHandler`` (registered at the graph-invocation root in
``agent.py``) pushes its OTel context when LangGraph calls the model node —
specifically during ``on_chat_model_start``.  That means inside
``awrap_model_call`` / ``wrap_model_call``, calling
``get_current_trace_id()`` **before** ``handler(request)`` will return ``None``
every time because the OTel context has not been activated yet.

Because of that, the observation cannot wrap ``handler(request)`` with
``start_as_current_observation`` (a context manager).  Instead we fire the
model call first, let the OTel context become active, then use the free
``Langfuse.span(...)`` API to submit the observation.
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

    Because the OTel context **is only activated** by the Langfuse
    ``CallbackHandler`` during the actual model invocation
    (``on_chat_model_start``), we:

    1. Call ``handler(request)`` first — this triggers the model call and
       pushes the OTel context.
    2. Then call ``self._lf.span(name="llm-decider-check", ...)`` to submit
       the observation as a child of the current active trace.
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
        """Wrap model invocation — sync path.

        OTel context timing (see class docstring) prevents wrapping
        ``handler(request)`` with ``start_as_current_observation``, so we
        submit a free ``Langfuse.span()`` after the model call instead.
        """
        if not self._is_langfuse_enabled():
            logger.info("Langfuse observation middleware disabled (sync), falling through")
            return handler(request)

        user_question = self._get_last_user_question(request.messages)
        if not user_question:
            logger.info("No user question found in sync request, falling through")
            return handler(request)

        system_text = self._get_system_text(request.system_message, request.system_prompt)

        input_data: dict = {}
        if system_text:
            input_data["system_prompt"] = system_text
        input_data["user_question"] = user_question

        # Fire model call first so the CallbackHandler activates the OTel
        # context (on_chat_model_start).
        response = handler(request)

        try:
            trace_id = self._lf.get_current_trace_id()
            if trace_id is None:
                logger.info("No Langfuse trace after sync model call (likely no tracing active), skipping observation")
                return response

            logger.info(
                "Langfuse observation created (sync) trace=%s",
                trace_id,
            )

            output = self._resolve_output(response)
            observation = self._lf.start_observation(
                name="llm-decider-check",
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
        """Wrap model invocation — async path (used by Gateway).

        OTel context timing (see class docstring) prevents wrapping
        ``handler(request)`` with ``start_as_current_observation``, so we
        submit a free ``Langfuse.span()`` after the model call instead.
        """
        if not self._is_langfuse_enabled():
            logger.info("Langfuse observation middleware disabled (async), falling through")
            return await handler(request)

        user_question = self._get_last_user_question(request.messages)
        if not user_question:
            logger.info("No user question found in async request, falling through")
            return await handler(request)

        system_text = self._get_system_text(request.system_message, request.system_prompt)

        input_data: dict = {}
        if system_text:
            input_data["system_prompt"] = system_text
        input_data["user_question"] = user_question

        # Fire model call first so the CallbackHandler activates the OTel
        # context (on_chat_model_start).
        try:
            response = await handler(request)
            with self._lf.start_as_current_observation(
                name="llm-decider-check",
                input=input_data,
            ) as observation:
                output = self._resolve_output(response)
                if output:
                    observation.update(output=output or {})
        except Exception as exc:
            logger.error("Langfuse async observation failed: %s", exc)

        return response
