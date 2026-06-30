"""Middleware for creating custom Langfuse observations around LLM calls.

Captures the **system prompt + current user question** (no history messages) as
input, and the **LLM's response** (text content + tool_calls as JSON) as output.
This creates a named ``span`` in Langfuse that a **Langfuse Evaluator** can
target to verify "did the model correctly decide to invoke a tool?".

Integration
-----------
This middleware is automatically registered in the lead-agent middleware chain
when ``LANGFUSE_TRACING=true`` and the required Langfuse credentials are
configured. It sits right before ``ClarificationMiddleware`` so that:

- ``before_model`` sees the **fully prepared state** (after all other middlewares
  have injected their context)
- ``after_model`` sees the **raw model output first** (before other ``after_model``
  hooks process it further)

When Langfuse is **not** enabled, the middleware is a no-op — no imports from
``langfuse`` are triggered, and no spans are created.

Usage in Langfuse
-----------------
In the Langfuse UI, each LLM turn produces a custom span named
``llm-decider-check`` nested under the root trace. Attach an Evaluator to this
span name to validate tool-calling correctness:

- **Input**: ``{"system_prompt": "...", "user_question": "..."}``
- **Output**: ``{"content": "model text reply", "tool_calls": "[{\"name\": \"...\", \"args\": {...}}]"}``
  (``tool_calls`` is omitted when the model did not call any tool)

Thread safety
-------------
``before_model`` / ``after_model`` are called sequentially within a single
LangGraph turn, so storing the span reference on ``self`` is safe.
"""

import json
import logging
import sys
from typing import TYPE_CHECKING

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

    The span captures:

    * **Input** — the system prompt and the *current* user question only
      (historical messages are excluded) so the evaluator has a clean signal.
    * **Output** — the model's text reply (``content``) and any tool calls
      serialised as JSON (``tool_calls``).

    The span is named ``llm-decider-check`` and appears as a child of the
    current Langfuse trace (created by the graph-root ``CallbackHandler``).
    """

    def __init__(self) -> None:
        """Initialize the middleware.

        The ``_current_span`` field holds a reference to the Langfuse span
        created in ``before_model`` and consumed in ``after_model``.
        """
        self._current_span: object | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_langfuse_enabled() -> bool:
        """Check whether Langfuse is among the enabled tracing providers.

        Returns ``False`` eagerly when Langfuse is not configured so callers
        never attempt to import ``langfuse`` unnecessarily.
        """
        try:
            from deerflow.config.tracing_config import get_enabled_tracing_providers

            return "langfuse" in get_enabled_tracing_providers()
        except Exception:
            return False

    @staticmethod
    def _get_last_user_question(messages: list) -> str | None:
        """Return the text content of the **last** ``HumanMessage``.

        Only the last non-tool, non-AI human message is returned — this is
        the *current* user question, not historical messages.
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                raw = msg.content
                if isinstance(raw, str) and raw.strip():
                    return raw
                # Handle list-of-block content (multimodal messages).
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
                break  # HumanMessage found but empty — stop looking
        return None

    @staticmethod
    def _get_system_prompt(messages: list) -> str | None:
        """Return the content of the first ``SystemMessage``."""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                raw = msg.content
                if isinstance(raw, str) and raw.strip():
                    return raw
                return None
        return None

    @staticmethod
    def _format_output(last: AIMessage) -> dict:
        """Build the output dict from the model's response.

        Returns a dict with at most two keys:

        * ``content`` — the text reply (omitted if empty).
        * ``tool_calls`` — JSON-dumped list of tool calls (omitted if empty).
        """
        output: dict = {}

        # --- text content ---
        content = last.content
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

        # --- tool calls ---
        if last.tool_calls:
            serialised = [
                {"name": tc["name"], "args": tc["args"], "id": tc.get("id")}
                for tc in last.tool_calls
            ]
            output["tool_calls"] = json.dumps(serialised, ensure_ascii=False)

        return output

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    @override
    def before_model(self, state: AgentState, runtime) -> dict | None:  # type: ignore[override]
        """Start a Langfuse span before the LLM call.

        Input: system prompt + current user question (no history).
        Skips silently when Langfuse is disabled or there is no user
        question to observe.
        """
        if not self._is_langfuse_enabled():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        system_prompt = self._get_system_prompt(messages)
        user_question = self._get_last_user_question(messages)

        if not user_question:
            return None

        # Build the input payload.
        input_data: dict = {}
        if system_prompt:
            input_data["system_prompt"] = system_prompt
        input_data["user_question"] = user_question

        try:
            from langfuse import get_client

            client = get_client()
            span = client.span(
                name="llm-decider-check",
                input=input_data,
            )
            self._current_span = span
        except Exception as exc:
            logger.debug("Failed to create Langfuse observation span: %s", exc)
            self._current_span = None

        return None

    @override
    def after_model(self, state: AgentState, runtime) -> dict | None:  # type: ignore[override]
        """Close the Langfuse span started in ``before_model``.

        Output: the model's text reply + serialised tool calls.
        """
        span = self._current_span
        if span is None:
            return None

        # Clear eagerly so a re-entrant call (should not happen) does not
        # accidentally re-use a stale span.
        self._current_span = None

        try:
            messages = state.get("messages", [])
            if not messages:
                span.end()
                return None

            last = messages[-1]
            if not isinstance(last, AIMessage):
                span.end()
                return None

            output = self._format_output(last)
            span.update(output=output)
            span.end()

        except Exception as exc:
            logger.debug("Failed to finalise Langfuse observation span: %s", exc)
            try:
                span.end()  # ensure span is always closed
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # Async variants — delegate to sync implementations
    # ------------------------------------------------------------------

    @override
    async def abefore_model(self, state: AgentState, runtime) -> dict | None:  # type: ignore[override]
        """Async variant — delegates to ``before_model``."""
        return self.before_model(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime) -> dict | None:  # type: ignore[override]
        """Async variant — delegates to ``after_model``."""
        return self.after_model(state, runtime)
