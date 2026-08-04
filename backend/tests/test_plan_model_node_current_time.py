"""
current_time 注入逻辑单元测试。

覆盖 deerflow.agentsv2.nodes.current_time 的两个辅助函数：
  - extract_current_time_date
  - has_current_time_for_today

场景：
  - 消息中没有 <current_time> → 需要注入
  - 消息中有今天的 <current_time> → 不重复注入
  - 消息中只有昨天的 <current_time> → 重新注入
  - 消息中有格式不合法 / 无法解析的 <current_time> → 重新注入
  - 非 HumanMessage（AI / Tool）上的 <current_time> 不参与判断
"""

from datetime import date, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agentsv2.current_time import (
    extract_current_time_date,
    has_current_time_for_today,
)


def _d(offset: int) -> str:
    """今天偏移 offset 天的日期字符串（yyyy-MM-dd）。"""
    return (date.today() + timedelta(days=offset)).strftime("%Y-%m-%d")


@pytest.mark.parametrize(
    "content",
    [
        f"<current_time>{_d(0)}</current_time>",
        f"<current_time>{_d(0)} 14:30:00</current_time>",
        f"<current_time>{_d(0)}T14:30:00</current_time>",
        f"<current_time>{_d(-1)}</current_time>",
    ],
)
def test_extract_current_time_valid(content: str) -> None:
    parsed = extract_current_time_date(content)
    assert parsed is not None


def test_extract_current_time_exact_date() -> None:
    assert extract_current_time_date(f"<current_time>{_d(0)}</current_time>") == date.today()
    assert extract_current_time_date(f"<current_time>{_d(-1)}</current_time>") == date.today() - timedelta(days=1)


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
        "普通用户消息",
        "not a current_time",
        "<current_time>abc</current_time>",
        "<current_time></current_time>",
        "<current_time>2026-13-99</current_time>",
        f"前缀{_d(0)}",
    ],
)
def test_extract_current_time_invalid(content: str | None) -> None:
    assert extract_current_time_date(content) is None


def test_missing_current_time_needs_injection() -> None:
    msgs = [HumanMessage(content="今天天气怎么样？")]
    assert has_current_time_for_today(msgs) is False


def test_today_current_time_no_duplicate_injection() -> None:
    msgs = [
        HumanMessage(content="帮我安排今天的行程"),
        HumanMessage(content=f"<current_time>{_d(0)}</current_time>"),
    ]
    assert has_current_time_for_today(msgs) is True


def test_today_current_time_with_datetime_suffix_no_duplicate() -> None:
    msgs = [
        HumanMessage(content=f"<current_time>{_d(0)} 09:30:00</current_time>"),
    ]
    assert has_current_time_for_today(msgs) is True


def test_stale_current_time_requires_new_injection() -> None:
    msgs = [
        HumanMessage(content=f"<current_time>{_d(-2)}</current_time>"),
        HumanMessage(content="昨天的任务还挂着"),
    ]
    assert has_current_time_for_today(msgs) is False


def test_malformed_current_time_requires_new_injection() -> None:
    msgs = [
        HumanMessage(content="<current_time>not-a-date</current_time>"),
    ]
    assert has_current_time_for_today(msgs) is False


def test_non_human_messages_ignored() -> None:
    """AI / Tool 消息里的 <current_time> 不应被当作状态中已有的注入。"""
    msgs = [
        HumanMessage(content="帮我处理下"),
        AIMessage(content="好的，我来处理"),
        ToolMessage(content=f"<current_time>{_d(0)}</current_time>", tool_call_id="t1"),
        HumanMessage(content=f"<current_time>{_d(-1)}</current_time>"),
    ]
    # 唯一的 HumanMessage current_time 是昨天的 → 仍需要注入
    assert has_current_time_for_today(msgs) is False


def test_future_current_time_requires_new_injection() -> None:
    msgs = [
        HumanMessage(content=f"<current_time>{_d(1)}</current_time>"),
    ]
    assert has_current_time_for_today(msgs) is False
