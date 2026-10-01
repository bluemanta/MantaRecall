"""Postgres + pgvector：连接池、表结构、embedding 身份 guard。"""
from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector


async def create_pool(dsn: str, min_size: int = 2, max_size: int = 50) -> asyncpg.Pool:
    async def _init(conn: asyncpg.Connection):
        await register_vector(conn)

    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=_init)


def ddl(dim: int) -> str:
    return f"""
-- pgvector extension 由 init_schema() 单独先建，此处不再重复

CREATE TABLE IF NOT EXISTS kv (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);

-- Add 幂等 / 409 判定：request_id 主键
CREATE TABLE IF NOT EXISTS add_requests (
  request_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memories (
  id BIGSERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'fact',          -- fact | event | preference | message
  content TEXT NOT NULL,
  content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
  embedding vector({dim}) NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',      -- active | superseded | deleted
  meta JSONB NOT NULL DEFAULT '{{}}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user_status ON memories (user_id, status);
CREATE INDEX IF NOT EXISTS idx_memories_user_session ON memories (user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_memories_tsv ON memories USING gin (content_tsv);
-- HNSW：空表建索引没问题；大数据量建议 Full 前做 VACUUM ANALYZE
CREATE INDEX IF NOT EXISTS idx_memories_vec ON memories USING hnsw (embedding vector_cosine_ops);
"""


async def init_schema(pool: asyncpg.Pool, dim: int) -> None:
    async with pool.acquire() as conn:
        # extension 单独建：失败时报错信息明确，且不依赖多语句 execute 的行为
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            raise RuntimeError(
                "创建 pgvector extension 失败，数据库用户需要 CREATE 权限 "
                f"(通常用 superuser)：{e}"
            ) from e
        await conn.execute(ddl(dim))


async def check_embedding_identity(pool: asyncpg.Pool, identity: str) -> None:
    """防止不同 embedding 模型/维度的向量静默混用：首次写入定身份，后续不一致直接拒绝启动。"""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT v FROM kv WHERE k = 'embedding_identity'")
        if row is None:
            await conn.execute(
                "INSERT INTO kv (k, v) VALUES ('embedding_identity', $1)", identity
            )
        elif row["v"] != identity:
            raise RuntimeError(
                f"embedding 身份不一致：库内是 {row['v']!r}，当前配置是 {identity!r}。"
                "换模型/维度必须用新库重建，不要混用。"
            )
