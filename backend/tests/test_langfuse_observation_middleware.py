"""Tests for deerflow.agents.middlewares.langfuse_observation_middleware.

All tests are pure unit tests: ``deerflow.*`` and ``langfuse`` are mocked so
they run without the full harness package.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ------------------------------------------------------------------
# Bootstrap: import the middleware module with mocked dependencies
# ------------------------------------------------------------------

_MIDDLEWARE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "packages/harness/deerflow/agents/middlewares/langfuse_observation_middleware.py",
)
_MIDDLEWARE_PATH = os.path.normpath(os.path.abspath(_MIDDLEWARE_PATH))


@pytest.fixture(scope="session")
def _modules():
    """Build ``sys.modules`` stubs so the middleware module can be imported."""
    # --- deerflow stub ---
    deerflow = types.ModuleType("deerflow")
    deerflow.__path__ = []
    deerflow.__package__ = "deerflow"

    deerflow_config = types.ModuleType("deerflow.config")
    deerflow_config.__package__ = "deerflow.config"
    deerflow_config.__path__ = []

    deerflow_tracing_config = types.ModuleType("deerflow.config.tracing_config")
    deerflow_tracing_config.__package__ = "deerflow.config.tracing_config"

    def _get_enabled() -> list[str]:
        return _get_enabled._enabled  # type: ignore[attr-defined]

    _get_enabled._enabled = []  # type: ignore[attr-defined]
    deerflow_tracing_config.get_enabled_tracing_providers = _get_enabled  # type: ignore[attr-defined]

    sys.modules["deerflow"] = deerflow
    sys.modules["deerflow.config"] = deerflow_config
    sys.modules["deerflow.config.tracing_config"] = deerflow_tracing_config

    # --- langfuse stub ---
    fake_langfuse_module = types.ModuleType("langfuse")
    fake_langfuse_module.__package__ = "langfuse"
    fake_langfuse_module.__path__ = []
    sys.modules["langfuse"] = fake_langfuse_module

    # Load the actual middleware module
    spec = importlib.util.spec_from_file_location(
        "deerflow.agents.middlewares.langfuse_observation_middleware",
        _MIDDLEWARE_PATH,
        submodule_search_locations=[],
    )
    if spec is None:
        pytest.fail(f"Could not find middleware module at {_MIDDLEWARE_PATH}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    if spec.loader:
        spec.loader.exec_module(mod)

    return {
        "mod": mod,
        "set_providers": lambda providers: setattr(_get_enabled, "_enabled", providers),
    }


@pytest.fixture(autouse=True)
def _reset_providers(_modules):
    """Reset enabled-tracing-providers to empty before each test."""
    _modules["set_providers"]([])
    yield


class FakeSpan:
    def __init__(self) -> None:
        self.input: Any = None
        self.output: Any = None
        self.ended = False
        self.name: str | None = None

    def update(self, *, output: dict | None = None) -> None:
        self.output = output

    def end(self) -> None:
        self.ended = True


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_observation(self, *, name: str, input: dict | None = None, output: dict | None = None, **kwargs: Any) -> FakeSpan:
        s = FakeSpan()
        s.name = name
        s.input = input
        s.output = output
        self.spans.append(s)
        return s

    def get_current_trace_id(self) -> str | None:
        return "fake-trace-id"


@pytest.fixture
def _fake_langfuse(_modules, monkeypatch):
    """Stub ``langfuse.get_client`` to return a ``FakeLangfuseClient``."""
    client = FakeLangfuseClient()
    # Directly set the attribute on the stub module (created in _modules session fixture)
    sys.modules["langfuse"].get_client = lambda: client
    return client


# ------------------------------------------------------------------
# Tests: _is_langfuse_enabled
# ------------------------------------------------------------------


def test_disabled_when_langfuse_not_configured(_modules):
    assert _modules["mod"].LangfuseObservationMiddleware._is_langfuse_enabled() is False


def test_enabled_when_langfuse_configured(_modules):
    _modules["set_providers"](["langfuse"])
    assert _modules["mod"].LangfuseObservationMiddleware._is_langfuse_enabled() is True


# ------------------------------------------------------------------
# Tests: _get_last_user_question
# ------------------------------------------------------------------


class TestGetLastUserQuestion:
    def test_returns_last_human_message(self, _modules):
        MWClass = _modules["mod"].LangfuseObservationMiddleware
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="first question"),
            AIMessage(content="first answer"),
            HumanMessage(content="今天的天气是什么？"),
        ]
        assert MWClass._get_last_user_question(msgs) == "今天的天气是什么？"

    def test_returns_none_when_no_human_message(self, _modules):
        MWClass = _modules["mod"].LangfuseObservationMiddleware
        assert MWClass._get_last_user_question([SystemMessage(content="sys"), AIMessage(content="answer")]) is None

    def test_returns_none_when_empty_messages(self, _modules):
        assert _modules["mod"].LangfuseObservationMiddleware._get_last_user_question([]) is None

    def test_handles_multimodal_content_list(self, _modules):
        MWClass = _modules["mod"].LangfuseObservationMiddleware
        msgs = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ]
            )
        ]
        assert MWClass._get_last_user_question(msgs) == "Describe this image"


# ------------------------------------------------------------------
# Tests: _get_system_prompt
# ------------------------------------------------------------------


class TestGetSystemPrompt:
    def test_returns_first_system_message(self, _modules):
        MWClass = _modules["mod"].LangfuseObservationMiddleware
        assert MWClass._get_system_prompt([SystemMessage(content="You are helpful."), HumanMessage(content="hi")]) == "You are helpful."

    def test_returns_none_on_empty_system(self, _modules):
        MWClass = _modules["mod"].LangfuseObservationMiddleware
        assert MWClass._get_system_prompt([SystemMessage(content=""), HumanMessage(content="hi")]) is None

    def test_returns_none_when_no_system_message(self, _modules):
        assert _modules["mod"].LangfuseObservationMiddleware._get_system_prompt([]) is None


# ------------------------------------------------------------------
# Tests: _format_output
# ------------------------------------------------------------------


class TestFormatOutput:
    def test_text_only(self, _modules):
        msg = AIMessage(content="今日は晴れです。")
        output = _modules["mod"].LangfuseObservationMiddleware._format_output(msg)
        assert output == {"content": "今日は晴れです。"}

    def test_text_with_tool_calls(self, _modules):
        msg = AIMessage(
            content="Let me check the weather.",
            tool_calls=[{"name": "web_search", "args": {"query": "北京 天气 2024-01-01"}, "id": "call_123"}],
        )
        output = _modules["mod"].LangfuseObservationMiddleware._format_output(msg)
        assert output["content"] == "Let me check the weather."
        tcs = json.loads(output["tool_calls"])
        assert tcs == [{"name": "web_search", "args": {"query": "北京 天气 2024-01-01"}, "id": "call_123"}]

    def test_tool_calls_only_no_text(self, _modules):
        msg = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "北京 天气"}, "id": "call_456"}])
        output = _modules["mod"].LangfuseObservationMiddleware._format_output(msg)
        assert "content" not in output
        assert json.loads(output["tool_calls"]) == [{"name": "web_search", "args": {"query": "北京 天气"}, "id": "call_456"}]

    def test_empty_message(self, _modules):
        assert _modules["mod"].LangfuseObservationMiddleware._format_output(AIMessage(content="")) == {}

    def test_multiple_tool_calls(self, _modules):
        msg = AIMessage(
            content="Using multiple tools.",
            tool_calls=[
                {"name": "read_file", "args": {"path": "/mnt/user-data/data.txt"}, "id": "c1"},
                {"name": "web_search", "args": {"query": "天氣"}, "id": "c2"},
            ],
        )
        output = _modules["mod"].LangfuseObservationMiddleware._format_output(msg)
        tcs = json.loads(output["tool_calls"])
        assert len(tcs) == 2
        assert tcs[0]["name"] == "read_file"
        assert tcs[1]["name"] == "web_search"


# ------------------------------------------------------------------
# Tests: lifecycle
# ------------------------------------------------------------------


def test_before_model_noop_when_langfuse_disabled(_modules):
    MW = _modules["mod"].LangfuseObservationMiddleware()
    result = MW.before_model({"messages": [SystemMessage(content="sys"), HumanMessage(content="hi")]}, runtime=None)
    assert result is None
    assert MW._current_span is None


def test_before_model_noop_when_no_user_question(_modules):
    _modules["set_providers"](["langfuse"])
    MW = _modules["mod"].LangfuseObservationMiddleware()
    result = MW.before_model({"messages": [SystemMessage(content="sys"), AIMessage(content="answer")]}, runtime=None)
    assert result is None
    assert MW._current_span is None


def test_before_model_creates_span_with_input(_modules, _fake_langfuse):
    _modules["set_providers"](["langfuse"])
    MW = _modules["mod"].LangfuseObservationMiddleware()
    state = {
        "messages": [
            SystemMessage(content="You are a weather assistant with tools: web_search, read_file."),
            HumanMessage(content="今天的天气是什么？"),
        ]
    }
    result = MW.before_model(state, runtime=None)
    assert result is None

    assert len(_fake_langfuse.spans) == 1
    span = _fake_langfuse.spans[0]
    assert span.name == "llm-decider-check"
    assert span.input == {
        "system_prompt": "You are a weather assistant with tools: web_search, read_file.",
        "user_question": "今天的天气是什么？",
    }
    assert MW._current_span is span


def test_after_model_closes_span_with_output_text_only(_modules, _fake_langfuse):
    """Model returned only text (no tool calls)."""
    _modules["set_providers"](["langfuse"])
    MW = _modules["mod"].LangfuseObservationMiddleware()
    state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there! How can I help you today?"),
        ]
    }

    MW.before_model(state, runtime=None)
    assert MW._current_span is _fake_langfuse.spans[0]

    result = MW.after_model(state, runtime=None)
    assert result is None

    span = _fake_langfuse.spans[0]
    assert span.output == {"content": "Hi there! How can I help you today?"}
    assert span.ended is True
    assert MW._current_span is None


def test_after_model_closes_span_with_tool_calls(_modules, _fake_langfuse):
    """Model decided to call tools — the main evaluator scenario."""
    _modules["set_providers"](["langfuse"])
    MW = _modules["mod"].LangfuseObservationMiddleware()
    state = {
        "messages": [
            SystemMessage(content="You have tools: web_search, read_file."),
            HumanMessage(content="今天的天气是什么？"),
            AIMessage(
                content="Let me search for the weather.",
                tool_calls=[{"name": "web_search", "args": {"query": "北京 天气"}, "id": "call_1"}],
            ),
        ]
    }

    MW.before_model(state, runtime=None)
    assert MW._current_span is _fake_langfuse.spans[0]

    result = MW.after_model(state, runtime=None)
    assert result is None

    span = _fake_langfuse.spans[0]
    assert span.ended is True
    assert span.output["content"] == "Let me search for the weather."
    tcs = json.loads(span.output["tool_calls"])
    assert tcs == [{"name": "web_search", "args": {"query": "北京 天气"}, "id": "call_1"}]
    assert MW._current_span is None


def test_after_model_noop_when_no_span_exists(_modules):
    _modules["set_providers"](["langfuse"])
    MW = _modules["mod"].LangfuseObservationMiddleware()
    result = MW.after_model({"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}, runtime=None)
    assert result is None


def test_after_model_noop_when_no_aimessage_in_state(_modules, _fake_langfuse):
    """State without an AIMessage should close span with no output."""
    _modules["set_providers"](["langfuse"])
    MW = _modules["mod"].LangfuseObservationMiddleware()
    state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="hi")]}

    MW.before_model(state, runtime=None)
    assert MW._current_span is not None

    result = MW.after_model(state, runtime=None)
    assert result is None

    span = _fake_langfuse.spans[0]
    assert span.ended is True
    assert span.output is None


def test_full_lifecycle_with_multiple_spans(_modules, _fake_langfuse):
    """Simulate two turns to verify spans are independent."""
    _modules["set_providers"](["langfuse"])
    MW = _modules["mod"].LangfuseObservationMiddleware()

    # Turn 1: tool-calling
    state_1 = {
        "messages": [
            SystemMessage(content="Assistant with tools."),
            HumanMessage(content="北京的天气是什么？"),
            AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "北京 天气"}, "id": "call_1"}]),
        ]
    }
    MW.before_model(state_1, runtime=None)
    MW.after_model(state_1, runtime=None)
    assert _fake_langfuse.spans[0].ended is True
    assert json.loads(_fake_langfuse.spans[0].output["tool_calls"])[0]["name"] == "web_search"

    # Turn 2: no tool calls (just text)
    state_2 = {
        "messages": [
            SystemMessage(content="Same assistant."),
            HumanMessage(content="Tell me a joke."),
            AIMessage(content="Why did the chicken cross the road?"),
        ]
    }
    MW.before_model(state_2, runtime=None)
    MW.after_model(state_2, runtime=None)

    assert len(_fake_langfuse.spans) == 2
    span2 = _fake_langfuse.spans[1]
    assert span2.output == {"content": "Why did the chicken cross the road?"}
    assert "tool_calls" not in span2.output


def test_async_delegates_to_sync(_modules, _fake_langfuse):
    """Async variant should call the sync implementation."""
    import asyncio

    _modules["set_providers"](["langfuse"])
    MW = _modules["mod"].LangfuseObservationMiddleware()
    state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="hi"), AIMessage(content="hello")]}

    async def run():
        assert await MW.abefore_model(state, runtime=None) is None
        assert await MW.aafter_model(state, runtime=None) is None

    asyncio.run(run())

    assert len(_fake_langfuse.spans) == 1
    span = _fake_langfuse.spans[0]
    assert span.input["user_question"] == "hi"
    assert span.output == {"content": "hello"}
    assert span.ended is True
