from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolGroupConfig(BaseModel):
    """Config section for a tool group"""

    name: str = Field(..., description="Unique name for the tool group")
    model_config = ConfigDict(extra="allow")


class ToolConfig(BaseModel):
    """Config section for a tool"""

    name: str = Field(..., description="Unique name for the tool")
    group: str = Field(..., description="Group name for the tool")
    use: str = Field(
        ...,
        description="Variable name of the tool provider(e.g. deerflow.sandbox.tools:bash_tool)",
    )
    stage: Literal["plan", "execute", "both"] = Field(
        default="both",
        description=(
            "Which agent stage this tool is available to:\n"
            '  - "plan": only available during plan creation (plan_model_node)\n'
            '  - "execute": only available during step execution (step subgraph)\n'
            '  - "both": available in both stages (default, v1 compatible)\n'
            "This field is optional. When absent, defaults to 'both'."
        ),
    )
    model_config = ConfigDict(extra="allow")
