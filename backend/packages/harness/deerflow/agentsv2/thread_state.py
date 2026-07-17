import uuid

from langchain.agents import AgentState
from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_desc: str
    agent_name: str
    done: bool = False


class TodoItem(BaseModel):
    todo: list[Task]
    done: bool = False


class ThreadState(AgentState):
    todo_list: list[TodoItem]
    pass
