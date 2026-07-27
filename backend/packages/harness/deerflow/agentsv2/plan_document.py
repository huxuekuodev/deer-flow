"""
DAG-based Plan 数据模型：支持 Redis 可选持久化。

Co-Sight 模式改造:
  - Plan = DAG (steps + dependencies)，而非顺序 phases
  - 计划通过工具 (create_plan / update_plan) 而非 Pydantic structured output 创建
  - 步骤执行器通过 mark_step 工具推进计划
  - Redis 持久化为可选项，默认内存
"""

import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class StepStatus:
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


def _normalize_dependencies(dependencies: dict[int, list[int]]) -> dict[int, list[int]]:
    """将可能 1 基编号的依赖转换为 0 基编号 (Co-Sight 兼容)."""
    if not dependencies:
        return {}
    keys = list(dependencies.keys())
    values = [d for v in dependencies.values() for d in v]
    if 0 in keys or any(d == 0 for d in values):
        return dependencies
    if min(keys) >= 1 and (not values or min(values) >= 1):
        return {k - 1: [d - 1 for d in v] for k, v in dependencies.items()}
    return dependencies


class PlanDocument(BaseModel):
    """
    DAG 计划文档。

    与 Co-Sight Plan 类对应：
      - steps: list[str] — 步骤描述列表
      - dependencies: { i: [j, k] } — 邻接表，step i 依赖 step j, k
      - step_statuses: { idx: str } — 每个步骤的状态
      - step_notes / step_tool_calls — 执行记录
    """

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""
    steps: list[str] = Field(default_factory=list)
    dependencies: dict[int, list[int]] = Field(default_factory=dict)
    step_statuses: dict[str, str] = Field(default_factory=dict)
    step_notes: dict[str, str] = Field(default_factory=dict)
    step_tool_calls: dict[str, list[dict]] = Field(default_factory=dict)
    result: str = ""
    created_at: float = Field(default_factory=time.time)

    def model_post_init(self, __context: Any) -> None:
        """初始化后自动补齐字段。"""
        if not self.step_statuses:
            for i in range(len(self.steps)):
                si = str(i)
                if si not in self.step_statuses:
                    self.step_statuses[si] = StepStatus.NOT_STARTED
        # 仅当 dependencies 未提供（None）时设为顺序依赖
        # 如果显式传了 {}（空 dict），表示各步骤间无依赖（DAG 并行）
        if self.dependencies is None and len(self.steps) > 1:
            self.dependencies = {i: [i - 1] for i in range(1, len(self.steps))}

    def get_ready_steps(self) -> list[int]:
        """获取所有前置依赖已完成的步骤索引（DAG 依赖解析）。"""
        ready = []
        for i in range(len(self.steps)):
            status = self.step_statuses.get(str(i), StepStatus.NOT_STARTED)
            if status != StepStatus.NOT_STARTED:
                continue
            deps = self.dependencies.get(i, [])
            all_done = all(self.step_statuses.get(str(d), StepStatus.NOT_STARTED) == StepStatus.COMPLETED for d in deps)
            if all_done:
                ready.append(i)
        return ready

    def mark_step(self, step_index: int, status: str, notes: str = "") -> None:
        """标记步骤状态，同 Co-Sight Plan.mark_step。"""
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}, steps count: {len(self.steps)}")
        self.step_statuses[str(step_index)] = status
        if notes:
            self.step_notes[str(step_index)] = notes

    def add_tool_call(self, step_index: int, tool_name: str, tool_args: str, tool_result: str = "") -> None:
        """记录工具调用。"""
        key = str(step_index)
        if key not in self.step_tool_calls:
            self.step_tool_calls[key] = []
        self.step_tool_calls[key].append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_result[:500],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def get_progress(self) -> dict:
        """进度统计。"""
        total = len(self.steps)
        completed = sum(1 for s in self.step_statuses.values() if s == StepStatus.COMPLETED)
        in_progress = sum(1 for s in self.step_statuses.values() if s == StepStatus.IN_PROGRESS)
        blocked = sum(1 for s in self.step_statuses.values() if s == StepStatus.BLOCKED)
        not_started = sum(1 for s in self.step_statuses.values() if s == StepStatus.NOT_STARTED)
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "not_started": not_started,
        }

    def format(self, with_detail: bool = False) -> str:
        """格式化计划为可读字符串（用于 prompt 注入）。"""
        progress = self.get_progress()
        pct = (progress["completed"] / progress["total"] * 100) if progress["total"] > 0 else 0
        lines = [
            f"📋 Plan: {self.title}",
            f"Progress: {progress['completed']}/{progress['total']} ({pct:.1f}%)",
            f"  ✅ {progress['completed']} completed, 🔄 {progress['in_progress']} in_progress,",
            f"  ⛔ {progress['blocked']} blocked, ⬜ {progress['not_started']} not_started",
            "",
        ]
        for i, step in enumerate(self.steps):
            sym = {
                StepStatus.NOT_STARTED: "⬜",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.BLOCKED: "⛔",
            }.get(self.step_statuses.get(str(i), StepStatus.NOT_STARTED), "⬜")
            deps = self.dependencies.get(i, [])
            dep_str = f" (depends on: {deps})" if deps else ""
            lines.append(f"Step{i}: {sym} {step}{dep_str}")
            if with_detail and self.step_notes.get(str(i)):
                lines.append(f"   Notes: {self.step_notes[str(i)]}")
        return "\n".join(lines)

    # ---- Redis 持久化 ----

    def to_redis(self, redis_client: Any, ttl: int = 86400) -> None:
        """将 plan 序列化后写入 Redis。"""
        key = f"plan:{self.plan_id}"
        redis_client.setex(key, ttl, self.model_dump_json())

    @classmethod
    def from_redis(cls, redis_client: Any, plan_id: str) -> Optional["PlanDocument"]:
        """从 Redis 恢复 plan。"""
        key = f"plan:{plan_id}"
        raw = redis_client.get(key)
        if not raw:
            return None
        return cls.model_validate_json(raw)
