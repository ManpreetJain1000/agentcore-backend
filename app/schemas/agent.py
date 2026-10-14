from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.tools import available_tools, unknown_tools
from app.models.agent import RunStatus


class AgentSessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100, examples=["Research Assistant"])
    system_prompt: str = Field(
        default="",
        max_length=10_000,
        examples=["You are a helpful research assistant. Be concise and cite numbers."],
    )
    tools_enabled: list[str] = Field(
        default_factory=list,
        examples=[["web_search", "calculator", "remember_fact"]],
    )

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be blank")
        return v

    @field_validator("tools_enabled")
    @classmethod
    def tools_must_exist(cls, v: list[str]) -> list[str]:
        v = list(dict.fromkeys(v))  # remove duplicates, keep order
        unknown = unknown_tools(v)
        if unknown:
            raise ValueError(
                f"Unknown tool(s): {', '.join(unknown)}. "
                f"Available tools: {', '.join(available_tools())}"
            )
        return v


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_message: str
    status: RunStatus
    final_answer: str | None
    tokens_used: int
    created_at: datetime


class AgentSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    system_prompt: str
    tools_enabled: list[str]
    created_at: datetime


class AgentSessionDetail(AgentSessionOut):
    recent_runs: list[RunSummary] = Field(default_factory=list)