from pydantic import BaseModel, Field


class LangfusePromptConfig(BaseModel):
    plan_system_prompt: str | None = Field(
        default=None,
        description="Name of the Langfuse prompt template for plan generation. When set, the prompt is fetched from Langfuse instead of using the built-in default.",
    )
