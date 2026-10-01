"""AML 文本赛道参赛服务：GET /health, POST /add, POST /search。

契约（2026-10-01 对照官网 + 第三方对照文档核验）：
- /health 无需鉴权，2xx = 就绪
- /add 同步：HTTP 200 时全部消息已持久化且立即可搜；
  响应原样回传 request_id/user_id/session_id；
  相同 request_id + 相同内容 -> 幂等 200；相同 request_id + 不同内容 -> 409
- /search 返回 {"data": [...]}，相关性排序，不超过 top_k；
  只返回记忆证据，不生成答案；user_id 是唯一隔离边界
- 鉴权：X-Api-Key / Authorization: Bearer / Authorization: Token
"""
from __future__ import annotations

import hashlib
import json
import logging
from contextlib import asynccontextmanager

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.auth import require_api_key
from app.config import Settings, build_app_state, get_cfg
from app.db import check_embedding_identity, create_pool, init_schema
from app.models import AddRequest, AddResponse, SearchRequest, SearchResponse
from app.pipeline.conflict import Candidate

log = logging.getLogger("aml")


def _content_hash(messages) -> str:
    canonical = json.dumps(
        [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in messages],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    db_pool_max = get_cfg(settings, "server", "db_pool_max", default=50)
    pool: asyncpg.Pool = await create_pool(settings.database_url, max_size=db_pool_max)
    app.state.pool = pool
    await init_schema(pool, settings.embedding_dim)
    state = build_app_state(settings)
    app.state.embedder = state["embedder"]
    app.state.llm = state["llm"]
    app.state.extraction = state["extraction"]
    app.state.conflict = state["conflict"]
    app.state.retrieval = state["retrieval"]
    app.state.rerank = state["rerank"]
    await check_embedding_identity(pool, state["embedder"].identity)
    # reranker 预热（none 时是 no-op；cross-encoder 时加载模型+torch，避免首 query 冷启动）
    await state["rerank"].awarmup()
    log.info("ready: embedder=%s llm=%s rerank=%s",
             state["embedder"].identity, getattr(state["llm"], "model", "none"),
             state["rerank"].stats())
    yield
    await state["embedder"].aclose()
    await state["llm"].aclose()
    await pool.close()


app = FastAPI(title="AML Textual Memory Service", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
async def health(request: Request):
    pool: asyncpg.Pool = request.app.state.pool
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "detail": str(e)[:200]})
    out = {"status": "ok"}
    try:
        out["rerank"] = request.app.state.rerank.stats()
    except Exception:
        pass
    return out


async def _fetch_conflict_candidates(
    pool: asyncpg.Pool, user_id: str, vector: list[float], k: int, min_sim: float
) -> list[Candidate]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, 1 - (embedding <=> $1) AS sim
            FROM memories
            WHERE user_id = $2 AND status = 'active'
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            vector, user_id, k,
        )
    return [
        Candidate(id=r["id"], content=r["content"], similarity=float(r["sim"]))
        for r in rows
        if float(r["sim"]) >= min_sim
    ]


@app.post("/add", response_model=AddResponse)
async def add(req: AddRequest, request: Request, _key: str = Depends(require_api_key)):
    settings: Settings = request.app.state.settings
    pool: asyncpg.Pool = request.app.state.pool
    body_hash = _content_hash(req.messages)

    # 1) 幂等 / 409 判定（DB 主键保证跨进程正确）
    async with pool.acquire() as conn:
        inserted = await conn.fetchval(
            """
            INSERT INTO add_requests (request_id, user_id, session_id, content_hash)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (request_id) DO NOTHING
            RETURNING request_id
            """,
            req.request_id, req.user_id, req.session_id, body_hash,
        )
        if inserted is None:
            row = await conn.fetchrow(
                "SELECT content_hash FROM add_requests WHERE request_id = $1",
                req.request_id,
            )
            if row is None or row["content_hash"] != body_hash:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="request_id already used with different content",
                )
            # 相同内容重复提交：幂等成功
            return AddResponse(
                request_id=req.request_id, user_id=req.user_id, session_id=req.session_id
            )

    # 2) 抽取 -> 向量化 -> 冲突裁决（都在事务外做，不占用 DB 连接做 LLM 等待）
    try:
        facts = await request.app.state.extraction.extract(req.messages)
        vectors = (
            await request.app.state.embedder.embed([f.text for f in facts])
            if facts else []
        )
        plans = await request.app.state.conflict.plan(
            facts, vectors, req.user_id,
            lambda uid, vec, k, ms: _fetch_conflict_candidates(pool, uid, vec, k, ms),
        )

        # 3) 单事务落库：HTTP 200 返回前必须持久化完成
        async with pool.acquire() as conn:
            async with conn.transaction():
                for plan in plans:
                    if plan.action == "skip":
                        continue
                    if plan.supersede_ids:
                        await conn.execute(
                            "UPDATE memories SET status = 'superseded' "
                            "WHERE id = ANY($1) AND user_id = $2",
                            plan.supersede_ids, req.user_id,
                        )
                    meta = dict(plan.fact.meta or {})
                    meta["kind"] = plan.fact.kind
                    await conn.execute(
                        """
                        INSERT INTO memories
                          (user_id, session_id, request_id, kind, content, embedding, meta)
                        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                        """,
                        req.user_id, req.session_id, req.request_id,
                        plan.fact.kind, plan.fact.text, plan.vector,
                        json.dumps(meta, ensure_ascii=False),
                    )
    except HTTPException:
        raise
    except Exception as e:
        # 流水线失败：删掉占位行，允许客户端重试（否则永久 409）
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM add_requests WHERE request_id = $1", req.request_id
            )
        log.exception("add failed: %s", req.request_id)
        raise HTTPException(status_code=500, detail=f"add failed: {str(e)[:300]}")

    return AddResponse(
        request_id=req.request_id, user_id=req.user_id, session_id=req.session_id
    )


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request, _key: str = Depends(require_api_key)):
    pool: asyncpg.Pool = request.app.state.pool
    top_k = max(1, min(int(req.top_k or 10), 1000))
    try:
        items = await request.app.state.retrieval.search(
            query=req.query,
            options=req.options,
            user_id=req.user_id,
            top_k=top_k,
            pool=pool,
            embedder=request.app.state.embedder,
        )
        items = await request.app.state.rerank.rerank(req.query, items)
    except Exception as e:
        log.exception("search failed")
        raise HTTPException(status_code=500, detail=f"search failed: {str(e)[:300]}")
    # 铁律：只返回证据，不生成答案；数量不超过 top_k
    data = [
        {
            "content": it.content,
            "score": it.score,
            "session_id": it.session_id,
            "timestamp": it.timestamp,
        }
        for it in items[:top_k]
    ]
    return {"data": data}
