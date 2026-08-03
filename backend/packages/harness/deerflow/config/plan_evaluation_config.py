"""规划节点评估配置（实验性）。

控制 plan_model_node 的 LLM-as-Judge 评估：
  - enabled: 主开关（默认关闭，实验性）
  - sample_rate: 采样率 0-1
  - judge_model: Judge LLM 名称；None 时用规划用的 plan_llm
  - dimensions: 各评估维度开关
"""

from pydantic import BaseModel, Field


class PlanEvaluationConfig(BaseModel):
    """规划节点评估配置。"""

    enabled: bool = Field(
        default=False,
        description="主开关：是否启用规划节点评估（实验性，默认关闭）。",
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="采样率 0-1；实验期可设较小值以控制成本。",
    )
    judge_model: str | None = Field(
        default=None,
        description="Judge LLM 名称；None 时用规划 agent 的 plan_llm。",
    )
    dimensions: dict[str, bool] = Field(
        default_factory=lambda: {
            "clarification_quality": True,
            "task_atomicity": True,
            "agent_selection_validity": True,
        },
        description="各评估维度开关。",
    )
