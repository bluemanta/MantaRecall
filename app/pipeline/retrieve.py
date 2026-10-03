"""Search 阶段：混合检索。

扩展点：实现 RetrievalStrategy.search() 即可替换。
- HybridRRF：稠密（pgvector 余弦）+ 词法（tsvector/ts_rank）-> RRF 融合
- DenseOnly / LexicalOnly：单路基线（消融实验用）

铁律：只返回记忆证据，不生成答案；SQL 层面按 user_id 隔离。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import asyncpg

from app.config import Settings, get_cfg


@dataclass
class Retrieved:
    id: int
    content: str
    score: float
    session_id: str | None
    timestamp: int | None


class RetrievalStrategy(Protocol):
    async def search(
        self,
        query: str,
        options: list[str] | None,
        user_id: str,
        top_k: int,
        pool: asyncpg.Pool,
        embedder,
    ) -> list[Retrieved]: ...


def _query_text(query: str, options: list[str] | None) -> str:
    # 选择题场景：把选项并入查询文本，增强词法命中
    if options:
        return query + " " + " ".join(options)
    return query


def _rrf_fuse(
    ranked_lists: list[list[tuple[int, Retrieved]]], k: int = 60
) -> list[Retrieved]:
    scores: dict[int, float] = {}
    items: dict[int, Retrieved] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, item) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            items[doc_id] = item
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for doc_id, s in ordered:
        it = items[doc_id]
        it.score = s
        out.append(it)
    return out


def _meta_dict(row) -> dict:
    """asyncpg 默认把 jsonb 以 str 返回，这里统一转成 dict。"""
    meta = row["meta"]
    if isinstance(meta, str):
        try:
            import json as _json

            meta = _json.loads(meta)
        except Exception:
            meta = {}
    return meta or {}


async def _dense_search(
    pool: asyncpg.Pool, user_id: str, qvec: list[float], k: int
) -> list[tuple[int, Retrieved]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, session_id, meta,
                   1 - (embedding <=> $1) AS score
            FROM memories
            WHERE user_id = $2 AND status = 'active'
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            qvec, user_id, k,
        )
    out = []
    for r in rows:
        meta = _meta_dict(r)
        out.append((
            r["id"],
            Retrieved(
                id=r["id"], content=r["content"], score=float(r["score"]),
                session_id=r["session_id"], timestamp=meta.get("timestamp"),
            ),
        ))
    return out


async def _lexical_search(
    pool: asyncpg.Pool, user_id: str, qtext: str, k: int
) -> list[tuple[int, Retrieved]]:
    if not qtext.strip():
        return []
    # 词法检索用 OR 语义：自然语言问题（8-12 个词）AND 几乎不可能命中。
    # websearch_to_tsquery 对纯文本仍是 AND，必须在 Python 侧显式用 OR 连接；
    # 去双引号防止 websearch 语法解析异常。
    or_qtext = " OR ".join(t.replace('"', "") for t in qtext.split() if t)
    if not or_qtext:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, session_id, meta,
                   ts_rank_cd(content_tsv, q) AS score
            FROM memories, websearch_to_tsquery('english', $1) AS q
            WHERE user_id = $2 AND status = 'active'
              AND content_tsv @@ q
            ORDER BY score DESC
            LIMIT $3
            """,
            or_qtext, user_id, k,
        )
    out = []
    for r in rows:
        meta = _meta_dict(r)
        out.append((
            r["id"],
            Retrieved(
                id=r["id"], content=r["content"], score=float(r["score"]),
                session_id=r["session_id"], timestamp=meta.get("timestamp"),
            ),
        ))
    return out


async def _expand_context(
    pool: asyncpg.Pool, user_id: str, seeds: list[Retrieved], window: int
) -> list[Retrieved]:
    """对每个种子补充同会话前后各 window 条消息（去重）。window=0 时跳过。"""
    if window <= 0 or not seeds:
        return seeds
    seed_ids = [s.id for s in seeds]
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH seed AS (
              SELECT id, session_id, created_at FROM memories WHERE id = ANY($1)
            )
            SELECT DISTINCT m.id, m.content, m.session_id, m.meta
            FROM memories m
            JOIN seed s ON m.session_id = s.session_id
            WHERE m.user_id = $2 AND m.status = 'active'
              AND m.id <> ALL($1)
              AND abs(extract(epoch from (m.created_at - s.created_at))) <= 3600
            """,
            seed_ids, user_id,
        )
    seen = {s.id for s in seeds}
    extra = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        meta = _meta_dict(r)
        extra.append(Retrieved(
            id=r["id"], content=r["content"], score=0.0,
            session_id=r["session_id"], timestamp=meta.get("timestamp"),
        ))
    # 上下文补充不改变排序：追加在种子之后
    return seeds + extra


class HybridRRF:
    def __init__(self, settings: Settings):
        self.dense_k = get_cfg(settings, "retrieval", "dense_k", default=200)
        self.lexical_k = get_cfg(settings, "retrieval", "lexical_k", default=200)
        self.rrf_k = get_cfg(settings, "retrieval", "rrf_k", default=60)
        self.seed_k = get_cfg(settings, "retrieval", "seed_k", default=100)
        self.context_window = get_cfg(settings, "retrieval", "context_window", default=0)

    async def search(self, query, options, user_id, top_k, pool, embedder) -> list[Retrieved]:
        qtext = _query_text(query, options)
        qvec = (await embedder.embed([qtext]))[0]
        dense = await _dense_search(pool, user_id, qvec, self.dense_k)
        lexical = await _lexical_search(pool, user_id, qtext, self.lexical_k)
        fused = _rrf_fuse([dense, lexical], k=self.rrf_k)
        seeds = fused[: max(self.seed_k, top_k)]
        return await _expand_context(pool, user_id, seeds, self.context_window)


class DenseOnly:
    def __init__(self, settings: Settings):
        self.dense_k = get_cfg(settings, "retrieval", "dense_k", default=200)

    async def search(self, query, options, user_id, top_k, pool, embedder) -> list[Retrieved]:
        qvec = (await embedder.embed([_query_text(query, options)]))[0]
        dense = await _dense_search(pool, user_id, qvec, max(self.dense_k, top_k))
        return [it for _, it in dense]


class LexicalOnly:
    def __init__(self, settings: Settings):
        self.lexical_k = get_cfg(settings, "retrieval", "lexical_k", default=200)

    async def search(self, query, options, user_id, top_k, pool, embedder) -> list[Retrieved]:
        lexical = await _lexical_search(
            pool, user_id, _query_text(query, options), max(self.lexical_k, top_k)
        )
        return [it for _, it in lexical]
