"""
Plan 存储抽象层：支持内存 + 可选 Redis 后端。

Co-Sight 的 Plan 对象是共享的可变状态（全局 TaskManager dict），
我们将其抽象为 PlanStorage，提供同一接口的内存/Redis 实现，
通过配置开关切换。
"""

from abc import ABC, abstractmethod
from typing import Any

from deerflow.agentsv2.plan_document import PlanDocument


class PlanStorage(ABC):
    """Plan 存储抽象。"""

    @abstractmethod
    async def save(self, plan: PlanDocument) -> None:
        """持久化 plan。"""
        ...

    @abstractmethod
    async def load(self, plan_id: str) -> PlanDocument | None:
        """加载 plan。"""
        ...

    @abstractmethod
    async def delete(self, plan_id: str) -> None:
        """删除 plan。"""
        ...

    @abstractmethod
    async def update_status(self, plan_id: str, step_index: int, status: str, notes: str = "") -> None:
        """原地更新步骤状态（避免全量读写）。"""
        ...

    @abstractmethod
    async def get_ready_steps(self, plan_id: str) -> list[int]:
        """DAG 依赖解析。"""
        ...


class MemoryPlanStorage(PlanStorage):
    """进程内内存存储（默认）。"""

    def __init__(self):
        self._plans: dict[str, PlanDocument] = {}

    async def save(self, plan: PlanDocument) -> None:
        self._plans[plan.plan_id] = plan.model_copy(deep=True)

    async def load(self, plan_id: str) -> PlanDocument | None:
        raw = self._plans.get(plan_id)
        return raw.model_copy(deep=True) if raw else None

    async def delete(self, plan_id: str) -> None:
        self._plans.pop(plan_id, None)

    async def update_status(self, plan_id: str, step_index: int, status: str, notes: str = "") -> None:
        plan = self._plans.get(plan_id)
        if plan:
            plan.mark_step(step_index, status, notes)

    async def get_ready_steps(self, plan_id: str) -> list[int]:
        plan = self._plans.get(plan_id)
        return plan.get_ready_steps() if plan else []


class RedisPlanStorage(PlanStorage):
    """Redis 后端存储。"""

    PLAN_KEY_PREFIX = "planv2:{plan_id}"

    def __init__(self, redis_client: Any, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl

    def _key(self, plan_id: str) -> str:
        return self.PLAN_KEY_PREFIX.format(plan_id=plan_id)

    async def save(self, plan: PlanDocument) -> None:
        await self.redis.setex(self._key(plan.plan_id), self.ttl, plan.model_dump_json())

    async def load(self, plan_id: str) -> PlanDocument | None:
        raw = await self.redis.get(self._key(plan_id))
        if not raw:
            return None
        return PlanDocument.model_validate_json(raw)

    async def delete(self, plan_id: str) -> None:
        await self.redis.delete(self._key(plan_id))

    async def update_status(self, plan_id: str, step_index: int, status: str, notes: str = "") -> None:
        plan = await self.load(plan_id)
        if plan:
            plan.mark_step(step_index, status, notes)
            await self.save(plan)

    async def get_ready_steps(self, plan_id: str) -> list[int]:
        plan = await self.load(plan_id)
        return plan.get_ready_steps() if plan else []


# ---- 单例工厂 ----

_storage: PlanStorage | None = None


def get_plan_storage() -> PlanStorage:
    """获取 Plan 存储实例。根据配置决定使用内存还是 Redis。"""
    global _storage
    if _storage is None:
        try:
            from deerflow.config.app_config import get_app_config

            config = get_app_config()
            if config.agentsv2 and config.agentsv2.plan_redis_enabled and config.agentsv2.plan_redis_url:
                import redis.asyncio as aioredis

                r = aioredis.from_url(config.agentsv2.plan_redis_url)
                _storage = RedisPlanStorage(r, ttl=config.agentsv2.plan_redis_ttl or 86400)
            else:
                _storage = MemoryPlanStorage()
        except Exception:
            _storage = MemoryPlanStorage()
    return _storage


def reset_plan_storage() -> None:
    """重置存储（用于测试）。"""
    global _storage
    _storage = None
