import json

from langchain.tools import tool
from tavily import TavilyClient

from deerflow.config import get_app_config
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime


def _get_tavily_client() -> TavilyClient:
    config = get_app_config().get_tool_config("web_search")
    api_key = None
    if config is not None and "api_key" in config.model_extra:
        api_key = config.model_extra.get("api_key")
    return TavilyClient(api_key=api_key)


def _resolve_thread_id(runtime: Runtime) -> str:
    """Extract the current thread_id from the tool runtime."""
    if runtime.context is not None:
        tid = runtime.context.get("thread_id")
        if tid:
            return str(tid)
    if runtime.config is not None:
        tid = runtime.config.get("configurable", {}).get("thread_id")
        if tid:
            return str(tid)
    try:
        from langgraph.config import get_config

        return str(get_config().get("configurable", {}).get("thread_id", ""))
    except RuntimeError:
        return ""


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str, runtime: Runtime) -> str:
    """Search the web.

    Args:
        query: The query to search for.
    """
    config = get_app_config().get_tool_config("web_search")
    max_results = 5
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results")

    # ── Semantic cache lookup ──────────────────────────────────────────
    try:
        from deerflow.community.tavily.cache import get_search_cache

        cache = get_search_cache()
        if cache is not None:
            thread_id = _resolve_thread_id(runtime)
            user_id = resolve_runtime_user_id(runtime)
            hit = cache.get(query, user_id=user_id, thread_id=thread_id)
            if hit is not None:
                cached_json, score = hit
                return f"[缓存命中] 您已经搜索过相似度 {score:.2%} 的相关内容，请不要再搜索，直接根据以下已有信息回答。\n\n{cached_json}"
    except Exception:
        # Cache miss or cache unavailable — fall through to live API call.
        pass

    # ── Live Tavily API call ───────────────────────────────────────────
    client = _get_tavily_client()
    res = client.search(query, max_results=max_results)
    normalized_results = [
        {
            "title": result["title"],
            "url": result["url"],
            "snippet": result["content"],
        }
        for result in res["results"]
    ]
    json_results = json.dumps(normalized_results, indent=2, ensure_ascii=False)

    # ── Populate cache ─────────────────────────────────────────────────
    try:
        from deerflow.community.tavily.cache import get_search_cache

        cache = get_search_cache()
        if cache is not None:
            thread_id = _resolve_thread_id(runtime)
            user_id = resolve_runtime_user_id(runtime)
            cache.set(query, json_results, user_id=user_id, thread_id=thread_id)
    except Exception:
        pass

    return json_results


@tool("web_fetch", parse_docstring=True)
def web_fetch_tool(url: str) -> str:
    """Fetch the contents of a web page at a given URL.
    Only fetch EXACT URLs that have been provided directly by the user or have been returned in results from the web_search and web_fetch tools.
    This tool can NOT access content that requires authentication, such as private Google Docs or pages behind login walls.
    Do NOT add www. to URLs that do NOT have them.
    URLs must include the schema: https://example.com is a valid URL while example.com is an invalid URL.

    Args:
        url: The URL to fetch the contents of.
    """
    client = _get_tavily_client()
    res = client.extract([url])
    if "failed_results" in res and len(res["failed_results"]) > 0:
        return f"Error: {res['failed_results'][0]['error']}"
    elif "results" in res and len(res["results"]) > 0:
        result = res["results"][0]
        return f"# {result['title']}\n\n{result['raw_content'][:4096]}"
    else:
        return "Error: No results found"
