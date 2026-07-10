import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# ── Resolve project root ────────────────────────────────────────────────
# ci.py lives at: deer-flow/backend/packages/harness/deerflow/ci.py
# Project root is 4 levels up from this file.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Load .env from the project root BEFORE any deerflow import so that
# LANGFUSE_TRACING / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
# LANGFUSE_BASE_URL (and any model api keys) are available when
# get_app_config() and build_tracing_callbacks() read the environment.
_load_path = _PROJECT_ROOT / ".env"
if _load_path.exists():
    load_dotenv(_load_path)
else:
    # Fallback: try cwd/.env (backward compatibility)
    load_dotenv()

# Ensure the project root and backend/ are on sys.path so deerflow
# and app imports resolve from the intended working tree.
_backend = _PROJECT_ROOT / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from langfuse import Langfuse  # noqa: E402

# Langfuse SDK picks up LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
# LANGFUSE_BASE_URL from the process environment automatically.
langfuse = Langfuse()


def run_ci_evaluation() -> None:
    """Run Langfuse dataset experiment against the DeerFlow agent.

    Each dataset item drives a fresh DeerFlow agent invocation.
    Scoring is handled server-side via LLM-as-a-Judge configured
    in the Langfuse platform — no local evaluators are attached here.
    """
    dataset_name = "长轮次对话助手"
    dataset = langfuse.get_dataset(dataset_name)

    def agent_task(*, item, **kwargs):
        input_text = item.input if hasattr(item, "input") else item.get("input", "")
        return run_deerflow_agent(input_text["user_question"])

    result = dataset.run_experiment(
        name=f"ci-regression-{time.strftime('%Y%m%d-%H%M%S')}",
        description="GitLab CI 自动回测",
        task=agent_task,
        max_concurrency=3,
    )

    print(f"实验完成，dataset_run_id = {result.dataset_run_id}")
    print(f"Langfuse: {result.dataset_run_url}")
    print(result.format())


def run_deerflow_agent(input_text: str) -> str:
    """Drive the DeerFlow agent and return final output text.

    Tracing alignment with the Gateway path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``DeerFlowClient.stream()`` internally:
    * calls ``build_tracing_callbacks()`` and appends the result to
      ``config["callbacks"]`` (at the graph invocation root, so the
      Langfuse CallbackHandler sees ``parent_run_id=None`` and lifts
      ``langfuse_session_id`` / ``langfuse_user_id`` onto the trace).
    * calls ``inject_langfuse_metadata()`` to stamp
      ``config["metadata"]`` with the reserved Langfuse keys.

    These two calls mirror exactly what ``worker.run_agent()`` does
    in the Gateway runtime — no custom tracing wiring is needed here.

    The only prerequisite is that the Langfuse env vars are present
    in the process environment, which is handled by ``load_dotenv()``
    at module top.
    """
    from deerflow.client import DeerFlowClient

    client = DeerFlowClient(
        # Pin the config so AppConfig / project_root() are deterministic
        # regardless of the caller's cwd.
        config_path=str(_PROJECT_ROOT / "config.yaml"),
        model_name=os.getenv("EXPERIMENT_MODEL"),
        subagent_enabled=True,
    )

    accumulated: list[str] = []
    thread_id = "ci-" + uuid.uuid4().hex[:8]

    try:
        for event in client.stream(
            input_text,
            thread_id=thread_id,
            recursion_limit=200,
        ):
            if event.type == "messages-tuple":
                data = event.data
                if data.get("type") == "ai":
                    accumulated.append(data.get("content", ""))
    except Exception as e:
        print(f"Agent execution error: {e}")
        return "".join(accumulated) + f"\n[Execution Error: {str(e)}]"

    return "".join(accumulated)


def fetch_scores_via_sdk(trace_id, max_retries=6, wait_interval=10):
    """
    通过 Langfuse SDK 轮询获取指定 trace 的异步评分
    """
    for attempt in range(max_retries):
        try:
            # 2. 调用 scores.list 接口，按 trace_id 过滤
            response = langfuse.api.scores.get_many(page=1, limit=2)
            print(response)

            # 检查是否已有评分数据返回
            if response.data:
                # 将对象转换为字典方便阅读或处理
                return [score.dict() for score in response.data]

            # 如果评分为空，说明线上 LLM 评估器还在异步执行中
            print(f"Trace {trace_id} 评分尚未生成，等待 {wait_interval} 秒后重试...")
            time.sleep(wait_interval)

        except Exception as e:
            print(f"获取评分时发生错误: {e}")
            time.sleep(wait_interval)

    return []


if __name__ == "__main__":
    fetch_scores_via_sdk(trace_id="83622714-cda1-426c-b5d7-23672d0c2362")
