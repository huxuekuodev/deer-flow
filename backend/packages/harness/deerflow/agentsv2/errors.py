"""
v2 规划节点 LLM 错误处理（独立于 v1）。

规划节点用 LangGraph 的 ``retry_policy`` 做节点级重试，但只对**可恢复的 LLM 错误**
（服务超时、连接中断、5xx、429、服务繁忙）重试；对**不可恢复的错误**（欠费、认证失败）
直接返回友好中文提示，不重试。

本模块自包含，不复用 v1 的 ``LLMErrorHandlingMiddleware``，避免 v1/v2 耦合。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# 可重试的 HTTP 状态码（服务端暂时不可用）
_RETRIABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# 可重试的异常类名（langchain 各 provider 抛出的瞬态错误）
_RETRIABLE_EXCEPTION_NAMES = frozenset(
    {
        "APITimeoutError",  # 请求超时
        "APIConnectionError",  # 连接失败
        "InternalServerError",  # 服务端 5xx
        "ReadError",  # httpx 连接中断
        "RemoteProtocolError",  # 服务端异常关闭连接
        "StreamChunkTimeoutError",  # 流式块超时
        "TimeoutError",  # 通用超时
        "RateLimitError",  # 限流
    }
)

# 服务繁忙特征（从错误信息匹配）
_BUSY_PATTERNS = (
    "server busy",
    "temporarily unavailable",
    "try again later",
    "please retry",
    "please try again",
    "overloaded",
    "high demand",
    "rate limit",
    "负载较高",
    "服务繁忙",
    "稍后重试",
    "请稍后重试",
)

# 欠费 / 额度不足特征
_QUOTA_PATTERNS = (
    "insufficient_quota",
    "insufficient balance",
    "quota",
    "billing",
    "credit",
    "payment",
    "余额不足",
    "超出限额",
    "额度不足",
    "欠费",
)

# 认证失败特征
_AUTH_PATTERNS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "permission",
    "forbidden",
    "access denied",
    "无权",
    "未授权",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """大小写不敏感地匹配任意模式。

    detail 已被 lower()，而部分模式（如 "Insufficient Balance"）含大写，
    必须将模式也转小写后再比较，否则漏判欠费。
    """
    lowered = text.lower()
    return any(p.lower() in lowered for p in patterns)


def _extract_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _extract_error_detail(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    return exc.__class__.__name__


def _extract_error_code(exc: BaseException) -> Any:
    for attr in ("code", "error_code"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            return value
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("code", "type"):
                value = error.get(key)
                if value not in (None, ""):
                    return value
    return None


def classify_llm_error(exc: BaseException) -> tuple[bool, str]:
    """分类 LLM 错误。

    Returns:
        (is_retriable, reason):
            - (True, "transient"): 可恢复，应重试（超时/连接/5xx）
            - (True, "busy"):      可恢复，应重试（服务繁忙）
            - (False, "quota"):    欠费/额度不足，不重试
            - (False, "auth"):     认证失败，不重试
            - (False, "generic"):  其他未知错误，不重试
    """
    detail = _extract_error_detail(exc).lower()
    error_code = _extract_error_code(exc)
    code_text = str(error_code).lower() if error_code is not None else ""
    status_code = _extract_status_code(exc)

    # 欠费 / 额度不足：直接失败，不重试
    # 判定依据：错误信息命中欠费特征，或错误码命中，或 HTTP 402 Payment Required
    if _matches_any(detail, _QUOTA_PATTERNS) or _matches_any(code_text, _QUOTA_PATTERNS):
        return False, "quota"

    # 认证失败：直接失败，不重试
    if _matches_any(detail, _AUTH_PATTERNS):
        return False, "auth"

    # 已知瞬态异常：重试
    if exc.__class__.__name__ in _RETRIABLE_EXCEPTION_NAMES:
        return True, "transient"

    # 可重试状态码：重试
    if status_code in _RETRIABLE_STATUS_CODES:
        return True, "transient"

    # 服务繁忙特征：重试
    if _matches_any(detail, _BUSY_PATTERNS):
        return True, "busy"

    return False, "generic"


def should_retry(exc: Exception) -> bool:
    """LangGraph retry_policy 的 retry_on 回调：仅对可恢复错误返回 True。"""
    try:
        retriable, _ = classify_llm_error(exc)
        return retriable
    except Exception:
        logger.debug("Failed to classify LLM error for retry decision: %s", exc)
        return False


def build_error_fallback_message(exc: Exception) -> AIMessage:
    """根据错误类型构建用户友好的中文提示消息。"""
    _, reason = classify_llm_error(exc)
    if reason == "quota":
        return AIMessage(
            content="LLM 服务账户欠费或额度不足，暂时无法处理您的请求。请检查账户余额或额度，补充后重试。",
            additional_kwargs={"deerflow_error_fallback": True, "error_reason": reason},
        )
    if reason == "auth":
        return AIMessage(
            content="LLM 服务认证失败，请检查 API Key 或认证配置是否正确。",
            additional_kwargs={"deerflow_error_fallback": True, "error_reason": reason},
        )
    if reason in ("transient", "busy"):
        return AIMessage(
            content="LLM 服务暂时不可用，多次重试后仍未成功。请稍等片刻后继续对话。",
            additional_kwargs={"deerflow_error_fallback": True, "error_reason": reason},
        )
    return AIMessage(
        content="生成计划时发生错误，请重新描述您的需求。",
        additional_kwargs={"deerflow_error_fallback": True, "error_reason": reason},
    )
