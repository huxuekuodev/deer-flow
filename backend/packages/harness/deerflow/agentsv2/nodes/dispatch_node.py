from langgraph.runtime import Runtime

from deerflow.agentsv2.lead_agent import GraphContext
from deerflow.agentsv2.thread_state import ThreadState


def dispatch_node(state: ThreadState, runtime: Runtime[GraphContext]) -> dict:
    """Dispatch the state to the next node."""
    # todo_list = state["todo_list"]
    return state
