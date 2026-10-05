"""AML 文本赛道参赛服务：GET /health, POST /add, POST /search。

契约（2026-10-01 对照官网 + 第三方对照文档核验）：
- /health 无需鉴权，2xx = 就绪
- /add 同步：HTTP 200 时全部消息已持久化且立即可搜；
  响应 success=true，原样回传 request_id/user_id/session_id；
  相同 request_id + 相同内容 -> 幂等 200；相同 request_id + 不同内容 -> 409；
  幂等由 add_requests 状态机保证（processing/committed/failed），
  只有 committed 才能幂等返回成功
- /search 返回 {"data": [...]}，相关性排序，不超过 top_k；
  每条证据带生命周期内稳定的字符串 id；
  只返回记忆证据，不生成答案；user_id 是唯一隔离边界
- 鉴权：X-Api-Key / Authorization: Bearer / Authorization: Token
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
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


def _content_hash(user_id: str, session_id: str, messages) -> str:
    """幂等/409 判定用的内容指纹：user_id + session_id + messages。

    把 user_id/session_id 纳入指纹：相同 request_id + 相同消息但不同用户/会话
    必须判 409，不能误判为幂等成功。
    """
    canonical = json.dumps(
        {
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in messages
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- Add 幂等状态机 ----
# processing（处理中） -> committed（已落库，可幂等返回） / failed（失败，可认领重试）
# 只有 committed 才能幂等返回 200；worker_token 防止慢 worker 被误接管导致重复写入。
_POLL_TIMEOUT = 30.0        # 等 processing 转终态的最长秒数
_POLL_INTERVAL = 0.5
_STALE_MINUTES = 30         # processing 超过此时长未更新视为崩溃残留，可认领


class _Takeover(Exception):
    """提交时发现 request_id 已被其他 worker 接管。"""


async def _read_request_row(pool: asyncpg.Pool, request_id: str):
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT request_id, user_id, session_id, content_hash, status,"
            " worker_token, updated_at FROM add_requests WHERE request_id = $1",
            request_id,
        )


async def _claim_request(pool: asyncpg.Pool, request_id: str, token: str,
                         from_status: str, stale: bool = False) -> bool:
    """认领一行 failed（或崩溃残留的 processing）。返回是否认领成功（原子操作）。"""
    async with pool.acquire() as conn:
        if stale:
            row = await conn.fetchrow(
                """
                UPDATE add_requests
                SET status = 'processing', worker_token = $2, updated_at = now()
                WHERE request_id = $1 AND status = 'processing'
                  AND updated_at < now() - make_interval(mins => $3)
                RETURNING request_id
                """,
                request_id, token, _STALE_MINUTES,
            )
        else:
            row = await conn.fetchrow(
                """
                UPDATE add_requests
                SET status = 'processing', worker_token = $2, updated_at = now()
                WHERE request_id = $1 AND status = $3
                RETURNING request_id
                """,
                request_id, token, from_status,
            )
        return row is not None


async def _mark_failed(pool: asyncpg.Pool, request_id: str, token: str) -> None:
    """worker 失败时标记 failed（只敢动自己的 token，不覆盖接管者）。"""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE add_requests SET status = 'failed', updated_at = now()"
            " WHERE request_id = $1 AND worker_token = $2",
            request_id, token,
        )


async def _resolve_existing(pool: asyncpg.Pool, req: AddRequest,
                          body_hash: str, token: str) -> str:
    """request_id 已存在时的状态机裁决。

    返回 'committed'（幂等成功，由调用方直接返回 200）
    或 'claimed'（本请求已用 token 认领为 worker，继续走流水线）。
    否则 raise 409（内容不一致）/ 503（前序仍在处理中，客户端应重试）。
    """
    deadline = time.monotonic() + _POLL_TIMEOUT
    while True:
        row = await _read_request_row(pool, req.request_id)
        if row is None:  # 防御性：正常流程下占位行永不删除，不应发生
            raise HTTPException(status_code=500, detail="idempotency record lost")
        if row["content_hash"] != body_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="request_id already used with different content",
            )
        st = row["status"]
        if st == "committed":
            return "committed"
        if st == "failed":
            if await _claim_request(pool, req.request_id, token, "failed"):
                return "claimed"
            continue  # 被别人认领了，重读
        # st == "processing"：等前序完成；崩溃残留（超时未更新）可认领
        if await _claim_request(pool, req.request_id, token, "processing", stale=True):
            return "claimed"
        if time.monotonic() > deadline:
            raise HTTPException(
                status_code=503,
                detail="duplicate request still processing, retry later",
            )
        await asyncio.sleep(_POLL_INTERVAL)


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
        # C1 修复：同 _dense_search，MATERIALIZED CTE 强制精确检索，
        # 避免 HNSW + user_id 过滤导致候选静默截断。
        rows = await conn.fetch(
            """
            WITH u AS MATERIALIZED (
              SELECT id, content, embedding
              FROM memories
              WHERE user_id = $2 AND status = 'active'
            )
            SELECT id, content, 1 - (embedding <=> $1) AS sim
            FROM u
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
    body_hash = _content_hash(req.user_id, req.session_id, req.messages)
    token = uuid.uuid4().hex  # 本次请求的 worker 身份

    # 1) 幂等 / 409 / 状态机判定（INSERT 原子抢占；冲突走状态机裁决）
    async with pool.acquire() as conn:
        inserted = await conn.fetchval(
            """
            INSERT INTO add_requests
              (request_id, user_id, session_id, content_hash, status, worker_token)
            VALUES ($1, $2, $3, $4, 'processing', $5)
            ON CONFLICT (request_id) DO NOTHING
            RETURNING request_id
            """,
            req.request_id, req.user_id, req.session_id, body_hash, token,
        )
    if inserted is None:
        action = await _resolve_existing(pool, req, body_hash, token)
        if action == "committed":
            # 相同内容重复提交：幂等成功（前序已落库，无需再做）
            return AddResponse(
                success=True,
                request_id=req.request_id, user_id=req.user_id, session_id=req.session_id,
            )
        # action == "claimed"：本请求已认领为 worker，继续走流水线

    # 2) 抽取 -> 向量化 -> 冲突裁决（都在事务外做，不占用 DB 连接做网络等待）
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

        # 3) 单事务落库 + 状态提交：条件更新防并发接管（token 对不上说明已被接管，
        #    此时回滚避免重复写入，由接管者完成）
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
                committed = await conn.fetchval(
                    "UPDATE add_requests SET status = 'committed', updated_at = now()"
                    " WHERE request_id = $1 AND status = 'processing' AND worker_token = $2"
                    " RETURNING request_id",
                    req.request_id, token,
                )
                if committed is None:
                    raise _Takeover()
    except _Takeover:
        # 被其他 worker 接管：以最新状态为准
        row = await _read_request_row(pool, req.request_id)
        if row is not None and row["status"] == "committed" \
                and row["content_hash"] == body_hash:
            return AddResponse(
                success=True,
                request_id=req.request_id, user_id=req.user_id, session_id=req.session_id,
            )
        raise HTTPException(status_code=503, detail="add raced with another worker, retry")
    except Exception as e:
        # 流水线失败：标记 failed（保留审计痕迹），允许认领重试
        await _mark_failed(pool, req.request_id, token)
        log.exception("add failed: %s", req.request_id)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"add failed: {str(e)[:300]}")

    return AddResponse(
        success=True,
        request_id=req.request_id, user_id=req.user_id, session_id=req.session_id,
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
            "id": str(it.id),
            "content": it.content,
            "score": it.score,
            "session_id": it.session_id,
            "timestamp": it.timestamp,
        }
        for it in items[:top_k]
    ]
    return {"data": data}
