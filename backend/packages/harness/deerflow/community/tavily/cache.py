"""Semantic search cache backed by Redis Stack.

Module-level singleton — no FastAPI lifecycle dependency.  The first call
to :func:`get_search_cache` lazily initialises the Redis connection and
index.  Tools call :meth:`SearchCache.get` / :meth:`SearchCache.set`
directly; when Redis is unavailable the entire module degrades to a
silent no-op (every lookup returns ``None``).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

import redis

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REDIS_KEY_PREFIX = "tavily:cache"
_INDEX_NAME = "idx:tavily:cache"

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_cache: SearchCache | None = None
_cache_lock = threading.Lock()
_cache_failed: bool = False  # Avoid retrying Redis after a fatal init failure.


class SearchCache:
    """Semantic cache for web-search tool results.

    Uses Redis Stack to store search-results indexed by their embedding
    vectors.  A ``get()`` call vectorises the query, runs a KNN search
    within the caller's ``(user_id, thread_id)`` partition, and returns
    the highest-scoring result if its cosine similarity exceeds the
    configured threshold.
    """

    def __init__(self, config) -> None:
        self._config = config
        self._redis: redis.Redis | None = None
        self._index_ready: bool = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    @property
    def redis(self) -> redis.Redis | None:
        """Lazily connect (and create the vector index) on first access."""
        if self._redis is not None:
            return self._redis
        if not self._config.enabled:
            return None
        with self._init_lock:
            if self._redis is not None:
                return self._redis
            try:
                self._redis = redis.Redis.from_url(
                    self._config.redis_url,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                    decode_responses=False,
                )
                self._redis.ping()
                self._ensure_index()
                self._index_ready = True
                logger.info(
                    "SearchCache initialised: redis=%s, dim=%d, ttl=%ds, threshold=%.2f",
                    self._config.redis_url,
                    self._config.embedding_dimensions,
                    self._config.ttl_seconds,
                    self._config.similarity_threshold,
                )
            except Exception:
                # Redis unavailable → degrade silently.  The connection
                # object stays None so every get() returns None quickly.
                self._redis = None
                logger.warning("SearchCache: Redis unavailable; semantic cache disabled for this process", exc_info=True)
            return self._redis

    def _ensure_index(self) -> None:
        """Create the RediSearch vector index if it does not already exist."""
        try:
            self._redis.ft(_INDEX_NAME).info()
            self._index_ready = True
            return
        except Exception:
            # Index doesn't exist — create it.
            pass

        try:
            from redis.commands.search.field import TextField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            # NOTE: user_id / thread_id are NOT indexed as RediSearch fields.
            # UUIDs with hyphens cause query-syntax issues in both TagField
            # and TextField.  Instead we index only the vector + data, and
            # filter user/thread in the Python callback.
            schema = (
                TextField("results_json"),
                TextField("user_id"),  # 声明在 schema 中只是为了返回该字段
                TextField("thread_id"),  # 查询时不做文本过滤，仅在 Python 侧比对
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._config.embedding_dimensions,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            definition = IndexDefinition(prefix=[f"{_REDIS_KEY_PREFIX}:"], index_type=IndexType.HASH)
            self._redis.ft(_INDEX_NAME).create_index(
                fields=schema,
                definition=definition,
            )
            self._index_ready = True
        except Exception:
            logger.warning("SearchCache: failed to create Redis index; caching disabled", exc_info=True)
            self._index_ready = False

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text* via OpenAI-compatible API."""
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if self._config.embedding_api_key:
            kwargs["api_key"] = self._config.embedding_api_key
        if self._config.embedding_base_url:
            kwargs["base_url"] = self._config.embedding_base_url
        client = OpenAI(**kwargs)
        resp = client.embeddings.create(
            model=self._config.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    def _embedding_bytes(self, vector: list[float]) -> bytes:
        """Pack a float32 vector into raw bytes for Redis."""
        import struct

        return struct.pack(f"{len(vector)}f", *vector)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _redis_key(user_id: str, thread_id: str, query: str) -> str:
        """Build the per-entry Redis key."""
        qhash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        return f"{_REDIS_KEY_PREFIX}:{user_id}:{thread_id}:{qhash}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        query: str,
        *,
        user_id: str = "default",
        thread_id: str = "",
    ) -> tuple[str, float] | None:
        """Return cached search results for a semantically similar query.

        Returns a ``(results_json, similarity)`` tuple on cache hit, ``None``
        on cache miss, Redis failure, or config ``enabled: false``.
        *similarity* is in [0, 1] where 1.0 means identical (converted from
        the raw COSINE distance returned by RediSearch).
        """
        r = self.redis
        if r is None or not self._index_ready:
            return None

        try:
            vector = self._embed(query)
        except Exception:
            logger.debug("SearchCache: embedding failed for query %r", query[:80], exc_info=True)
            return None

        try:
            # Pure KNN vector search — no pre-filter for user_id/thread_id.
            # UUIDs with hyphens break both TagField {} and TextField ""
            # syntax in RediSearch queries, so we do the partition-filtering
            # in Python below instead.
            q = redis.commands.search.query.Query("*=>[KNN 20 @embedding $vec AS _score]").dialect(2)
            q_params = {"vec": self._embedding_bytes(vector)}
            result = r.ft(_INDEX_NAME).search(q, query_params=q_params)

            if not result.docs:
                logger.debug("SearchCache: query %r no results", query[:80])
                return None

            # Filter by user_id/thread_id in Python, then pick the best score
            for doc in result.docs:
                # Document is a dynamic-attribute class, not a dict — use getattr().
                doc_user_id = getattr(doc, "user_id", "") or ""
                doc_thread_id = getattr(doc, "thread_id", "") or ""
                if isinstance(doc_user_id, bytes):
                    doc_user_id = doc_user_id.decode("utf-8")
                if isinstance(doc_thread_id, bytes):
                    doc_thread_id = doc_thread_id.decode("utf-8")
                if doc_user_id != user_id or doc_thread_id != thread_id:
                    continue

                score = float(getattr(doc, "_score", 0))
                # RediSearch COSINE returns a distance in [0, 2].
                # Convert to similarity in [0, 1] before comparing against
                # the configured threshold (which is a similarity, not a distance).
                similarity = 1.0 - (score / 2.0)
                if similarity < self._config.similarity_threshold:
                    logger.debug("SearchCache: query %r similarity %.3f < threshold %.3f", query[:80], similarity, self._config.similarity_threshold)
                    continue

                raw = getattr(doc, "results_json", "")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                logger.debug("SearchCache: hit for query %r (similarity=%.3f)", query[:80], similarity)
                return raw, similarity

            logger.debug("SearchCache: query %r no matching result for user/thread", query[:80])
            return None
        except Exception:
            logger.debug("SearchCache: lookup failed for query %r", query[:80], exc_info=True)
            return None

    def set(
        self,
        query: str,
        results_json: str,
        *,
        user_id: str = "default",
        thread_id: str = "",
    ) -> None:
        """Store search results indexed by their semantic embedding.

        Idempotent — if the entry already exists the previous embedding is
        overwritten.  Failures are logged and swallowed so a write failure
        never surfaces as a tool error.
        """
        r = self.redis
        if r is None or not self._index_ready:
            return

        try:
            vector = self._embed(query)
        except Exception:
            logger.debug("SearchCache: embedding failed for set %r", query[:80], exc_info=True)
            return

        key = self._redis_key(user_id, thread_id, query)
        try:
            r.hset(
                key,
                mapping={
                    "results_json": results_json,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "embedding": self._embedding_bytes(vector),
                },
            )
            if self._config.ttl_seconds > 0:
                r.expire(key, self._config.ttl_seconds)
        except Exception:
            logger.debug("SearchCache: write failed for key %s", key, exc_info=True)

    def close(self) -> None:
        """Close the Redis connection pool.

        After calling this method the next access lazily reconnects.
        Primarily useful in test teardown.
        """
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
        self._redis = None
        self._index_ready = False


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


def get_search_cache() -> SearchCache | None:
    """Return the process-wide :class:`SearchCache` singleton.

    The first call lazy-initialises the Redis connection and index.
    Returns ``None`` when ``search_cache.enabled`` is ``false`` in
    ``config.yaml``.

    Thread-safe — the singleton is protected by a module-level lock.
    Once initialisation has failed fatally (e.g. missing dependency)
    the failure is cached and no further connection attempts are made
    for the lifetime of the process.
    """
    global _cache, _cache_failed
    if _cache_failed:
        return None
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache_failed:
            return None
        if _cache is not None:
            return _cache
        config = get_app_config().search_cache
        if not config.enabled:
            _cache_failed = True
            return None
        try:
            _cache = SearchCache(config)
        except Exception:
            _cache_failed = True
            logger.warning("SearchCache: initialisation failed", exc_info=True)
        return _cache


def close_search_cache() -> None:
    """Close the singleton cache connection and reset state.

    After calling this function the next :func:`get_search_cache` will
    re-initialise from scratch, re-reading ``config.yaml``.
    """
    global _cache, _cache_failed
    with _cache_lock:
        if _cache is not None:
            _cache.close()
        _cache = None
        _cache_failed = False
