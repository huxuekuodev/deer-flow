"""
current_time 注入辅助函数。

供规划节点（plan_model_node）在注入当前时间前，先检查 state.messages
中是否已存在今天的 <current_time> 消息，避免重复注入。

- extract_current_time_date: 从 <current_time> 消息内容中解析日期
- has_current_time_for_today: 判断消息列表中是否已有今天的 <current_time>

使用第三方库 python-dateutil 解析日期（兼容多种格式，如 yyyy-MM-dd、
yyyy-MM-dd HH:mm:ss、ISO8601 等）。

独立放在 agentsv2 包层级（而非 nodes/ 下），避免 import 时触发
nodes/__init__.py 的完整 harness 依赖链，方便单元测试。
"""

from datetime import date
from typing import Any

from dateutil import parser as _dt_parser
from langchain_core.messages import HumanMessage


def extract_current_time_date(content: Any) -> date | None:
    """从 <current_time>...</current_time> 消息内容中解析出日期。

    仅当 content 是字符串、以 <current_time> 开头且能解析出有效日期时
    返回该日期，否则返回 None（调用方会重新注入当天时间）。
    """
    if not isinstance(content, str) or not content.strip().startswith("<current_time>"):
        return None
    try:
        text = content.strip().replace("<current_time>", "").replace("</current_time>", "")
        return _dt_parser.parse(text).date()
    except Exception:
        return None


def has_current_time_for_today(messages: list[Any]) -> bool:
    """判断消息列表中是否已存在今天的 <current_time> 消息。

    遍历所有 HumanMessage，若存在 <current_time> 且其日期就是今天，
    返回 True；否则返回 False（调用方会追加一条新的当前时间消息）。
    """
    today = date.today()
    for m in messages:
        if not isinstance(m, HumanMessage):
            continue
        parsed = extract_current_time_date(getattr(m, "content", None))
        if parsed is not None and parsed == today:
            return True
    return False
