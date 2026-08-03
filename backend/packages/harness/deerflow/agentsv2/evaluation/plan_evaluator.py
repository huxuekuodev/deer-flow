"""
规划节点评估器（实验性）。

对 plan_model_node 的规划输出做 LLM-as-Judge 评估，评估 3 个维度：

1. clarification_quality (0-1)：
   用户问题含糊/缺信息时，规划 agent 是否应调用 ask_clarification。
   评估器看到规划节点**实际收到的完整轨迹**（历史 + 当前计划状态），
   从而判断"此前的上下文是否已消歧"，避免事后上帝视角误判。
2. task_atomicity (0-1)：
   每个子任务是否是不可再分的原子步骤（无"先做A再做B"这种复合任务）。
3. agent_selection_validity (0-1)：
   每个子任务的 execution_agent 是否命中配置里真实存在的 agent。
   —— 这是**规则校验**，不需要 LLM。

设计说明（与 Langfuse 的衔接）：
  - 不使用 callback 创建 trace；用 `Langfuse().span(trace_id=..., name="plan_evaluation")`
    在**现有 trace 下**创建一个兄弟 observation（与 plan_agent span 并列，按时间排序）。
    这与 `LangfuseObservationMiddleware`（`llm-decider-check`）的模式一致。
  - 3 个维度用 `span.score(...)` 写为 Langfuse Scores，可在 Scores / Datasets 页聚合。
  - 评估发生在 plan_model_node 内部、拿到 plan_output 之后，此时 plan_agent 的 span 已
    结束，无法嵌套为其子节点；兄弟 observation 是实验期最稳妥的方式。
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# 内置的执行 agent（与 SubTask.execution_agent 默认值一致）
_BUILTIN_EXECUTION_AGENTS = {"general_agent"}


@dataclass
class PlanEvaluationConfig:
    """规划评估配置。"""

    enabled: bool = True
    """主开关。"""

    sample_rate: float = 1.0
    """采样率 0-1，实验期可设较小值。"""

    judge_model: str | None = None
    """Judge LLM 名称；None 时用规划用的 plan_llm。"""

    dimensions: dict[str, bool] = field(
        default_factory=lambda: {
            "clarification_quality": True,
            "task_atomicity": True,
            "agent_selection_validity": True,
        }
    )
    """各维度开关。"""

    def is_dimension_enabled(self, name: str) -> bool:
        return self.dimensions.get(name, True)


@dataclass
class EvaluationInput:
    """评估输入轨迹：规划节点实际看到的内容 + 规划输出。"""

    user_messages: list[str] = field(default_factory=list)
    """对话历史中的用户消息（用于澄清维度判断）。"""

    history: list[dict] = field(default_factory=list)
    """规划节点收到的完整消息轨迹（含 PlanStatus / current_time 注入消息）。"""

    plan_status: str = ""
    """注入到 agent 的 <PlanStatus> 内容（当前计划状态）。"""

    current_time: str = ""
    """注入的当前时间。"""

    plan_action: str = ""
    """create / update。"""

    tasks: list[dict] = field(default_factory=list)
    """规划输出的子任务列表（已序列化）。"""

    clarification_requested: bool = False
    """本次规划 agent 是否调用了 ask_clarification。"""


@dataclass
class PlanEvaluationResult:
    """评估结果。"""

    scores: dict[str, float] = field(default_factory=dict)
    """各维度得分 0-1。"""

    rationales: dict[str, str] = field(default_factory=dict)
    """各维度打分理由。"""

    passed: bool = True
    """是否全部维度通过（均 >= 0.8）。"""

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "rationales": self.rationales,
            "passed": self.passed,
        }


def _serialize_message(msg: BaseMessage) -> dict:
    """序列化一条 LangChain 消息为可读 dict。"""
    content = msg.content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        content = " ".join(parts)
    return {
        "type": type(msg).__name__,
        "content": str(content)[:2000],
        "name": getattr(msg, "name", None),
    }


class PlanEvaluator:
    """规划节点评估器。"""

    def __init__(
        self,
        *,
        langfuse: Any,
        enabled_agents: set[str],
        config: PlanEvaluationConfig,
        judge_llm: Any | None = None,
    ) -> None:
        self._lf = langfuse
        self._enabled_agents = enabled_agents | _BUILTIN_EXECUTION_AGENTS
        self._config = config
        self._judge_llm = judge_llm

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        *,
        trace_id: str,
        eval_input: EvaluationInput,
        messages: list[BaseMessage] | None = None,
        config: RunnableConfig | None = None,
    ) -> PlanEvaluationResult | None:
        """执行评估，返回结果；未启用或被采样跳过时返回 None。

        Args:
            trace_id: 关联到现有 Langfuse trace。
            eval_input: 评估输入轨迹。
            messages: 规划节点收到的原始消息列表（未序列化，用于补全 history）。
            config: RunnableConfig，用于 judge_llm 的 invoke。
        """
        if not self._config.enabled:
            return None
        if self._config.sample_rate < 1.0 and random.random() > self._config.sample_rate:
            return None

        # 补全 history（若调用方未传入已序列化的 history）
        if not eval_input.history and messages:
            eval_input.history = [_serialize_message(m) for m in messages]

        result = PlanEvaluationResult()

        # 输出类型判定：
        #   - clarification：agent 调用了 ask_clarification（中断等待用户）
        #   - plan：输出了有效计划（tasks 非空）
        #   - direct_reply：直接回复（未澄清也未产出计划 → 失败场景）
        has_plan = bool(eval_input.tasks)

        # 维度 3：agent 幻觉（规则校验，不依赖 LLM）
        if self._config.is_dimension_enabled("agent_selection_validity"):
            if has_plan:
                agent_score, hallucinated = self._check_agent_selection(eval_input.tasks)
                result.scores["agent_selection_validity"] = agent_score
                if hallucinated:
                    result.rationales["agent_selection_validity"] = f"检测到配置中不存在的 agent: {', '.join(hallucinated)}。允许的 agent: {sorted(self._enabled_agents)}"
                else:
                    result.rationales["agent_selection_validity"] = "所有 execution_agent 均在配置集合内。"
            else:
                # 无任务可评，跳过该维度（不设分数）
                result.rationales["agent_selection_validity"] = "无计划输出，跳过 agent 选择校验。"

        # 维度 1 & 2：LLM judge
        llm_scores, llm_rationales = await self._llm_judge(
            trace_id=trace_id,
            eval_input=eval_input,
            config=config,
        )
        if self._config.is_dimension_enabled("clarification_quality") and "clarification_quality" in llm_scores:
            result.scores["clarification_quality"] = llm_scores["clarification_quality"]
            result.rationales["clarification_quality"] = llm_rationales.get("clarification_quality", "")
        if self._config.is_dimension_enabled("task_atomicity") and has_plan and "task_atomicity" in llm_scores:
            result.scores["task_atomicity"] = llm_scores["task_atomicity"]
            result.rationales["task_atomicity"] = llm_rationales.get("task_atomicity", "")

        result.passed = all(score >= 0.8 for score in result.scores.values())

        # 写入 Langfuse observation + scores
        self._emit_langfuse(trace_id=trace_id, eval_input=eval_input, result=result)

        return result

    # ------------------------------------------------------------------
    # 维度 3：agent 幻觉规则校验
    # ------------------------------------------------------------------

    def _check_agent_selection(self, tasks: list[dict]) -> tuple[float, list[str]]:
        hallucinated: list[str] = []
        for t in tasks:
            agent = t.get("execution_agent", "general_agent")
            if agent not in self._enabled_agents:
                hallucinated.append(agent)
        if hallucinated:
            return 0.0, list(dict.fromkeys(hallucinated))
        return 1.0, []

    # ------------------------------------------------------------------
    # 维度 1 & 2：LLM judge
    # ------------------------------------------------------------------

    def _build_judge_prompt(self, eval_input: EvaluationInput) -> str:
        """构建 LLM judge 的评估 prompt。"""
        tasks_json = json.dumps(eval_input.tasks, ensure_ascii=False, indent=2)

        history_text = "\n".join(f"[{m.get('type')}] {m.get('content', '')}" for m in eval_input.history[-10:]) or "(无历史消息)"

        lines = [
            "你是规划质量评估器。评估规划 agent 的决策质量，输出严格 JSON。",
            "",
            "【输入】",
            "<用户问题历史>",
            history_text,
            "</用户问题历史>",
            "",
            "<当前计划状态>",
            eval_input.plan_status or "(无)",
            "</当前计划状态>",
            "",
            "<规划输出>",
            f"action: {eval_input.plan_action}",
            tasks_json,
            f"clarification_requested: {eval_input.clarification_requested}",
            "</规划输出>",
            "",
            "【规划 agent 可能的三种输出】",
            "1. 调用了 ask_clarification（clarification_requested=true）：需求含糊时澄清，是正确行为。",
            "2. 输出计划（tasks 非空）：需求清晰时拆解为原子任务。",
            "3. 直接回复（既未澄清也未产出计划）：若需求可执行却直接放弃，则是失败场景。",
            "",
            "【评估维度】",
            "1. clarification_quality (0-1)：",
            "   - 需求含糊/缺信息 + agent 调用澄清 → 1.0",
            "   - 需求含糊/缺信息 + agent 未澄清就强行规划或直接回复 → 0.0-0.3",
            "   - 需求清晰 + agent 未澄清（直接规划）→ 1.0（澄清反而多余）",
            "   - 需求清晰但 agent 仍反复澄清 → 0.0-0.5",
            "   - 判断依据必须结合<用户问题历史>：若历史中用户已提供足够信息（包括此前澄清的答案），则不应再澄清，此时直接规划给 1.0。",
            "2. task_atomicity (0-1)：",
            "   - 无任务输出（tasks 为空）→ 该维度给 0.0，但仅在输出类型为'直接回复'时才算失败",
            "   - 有任务时：每个任务是否不可再分的原子步骤；存在复合任务（如'先搜索再总结'）按占比扣分。",
            "",
            "【输出格式】严格 JSON，不要输出其他内容：",
            """{"clarification_quality": 0.0, "task_atomicity": 0.0, "rationale": "简要说明"}""",
        ]
        return "\n".join(lines)

    async def _llm_judge(
        self,
        *,
        trace_id: str,
        eval_input: EvaluationInput,
        config: RunnableConfig | None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        if self._judge_llm is None:
            logger.debug("No judge LLM configured; skip LLM-based dimensions")
            return {}, {}

        try:
            prompt = self._build_judge_prompt(eval_input)
            response = await self._judge_llm.ainvoke(prompt, config=config)
            content = getattr(response, "content", None)
            raw = str(content) if content else ""

            # 提取 JSON（兼容 ```json 包裹）
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)
            scores: dict[str, float] = {}
            rationales: dict[str, str] = {}
            for key in ("clarification_quality", "task_atomicity"):
                if key in data:
                    scores[key] = float(data[key])
            if "rationale" in data:
                rationales["clarification_quality"] = str(data["rationale"])
                rationales["task_atomicity"] = str(data["rationale"])
            return scores, rationales
        except Exception as exc:
            logger.warning("LLM judge failed: %s", exc)
            return {}, {}

    # ------------------------------------------------------------------
    # Langfuse 写入
    # ------------------------------------------------------------------

    def _emit_langfuse(self, *, trace_id: str, eval_input: EvaluationInput, result: PlanEvaluationResult) -> None:
        """在现有 trace 下创建 plan_evaluation observation，并写入 scores。

        注意：langfuse 4.x 没有 ``Langfuse().span()`` API（那是旧版本）。
        正确方式是 ``start_observation(as_type="span", trace_context={"trace_id": ...})``，
        返回带 ``.score()`` / ``.end()`` 的 LangfuseSpan 对象。
        """
        if self._lf is None:
            return
        try:
            span = self._lf.start_observation(
                name="plan_evaluation",
                as_type="span",
                trace_context={"trace_id": trace_id},
                input={
                    "user_messages": eval_input.user_messages,
                    "history": eval_input.history[-10:],
                    "plan_status": eval_input.plan_status,
                    "current_time": eval_input.current_time,
                    "plan": {
                        "action": eval_input.plan_action,
                        "tasks": eval_input.tasks,
                    },
                    "clarification_requested": eval_input.clarification_requested,
                },
                output=result.to_dict(),
            )
            for metric, value in result.scores.items():
                comment = result.rationales.get(metric, "")
                span.score(
                    name=f"plan_evaluation/{metric}",
                    value=float(value),
                    data_type="NUMERIC",
                    comment=comment,
                )
            span.end()
        except Exception as exc:
            logger.warning("Failed to emit Langfuse plan evaluation: %s", exc)
