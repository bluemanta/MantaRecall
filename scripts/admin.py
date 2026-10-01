#!/usr/bin/env python3
"""数据库管理：查看统计、按 user_id 前缀清理某次运行的数据。

注意：评测数据按规则只能用于当次运行，跑后 30 天内删除；不要保留评测原文副本。

用法：
  python scripts/admin.py --database-url postgresql://aml:amlpass@localhost:5432/aml stats
  python scripts/admin.py --database-url ... purge-prefix --user-prefix 'smoke:abc123' --confirm
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import asyncpg
from pgvector.asyncpg import register_vector


async def _pool(dsn: str):
    async def _init(conn):
        await register_vector(conn)

    return await asyncpg.create_pool(dsn, init=_init)


async def stats(dsn: str):
    pool = await _pool(dsn)
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT count(*) FROM memories")
        active = await conn.fetchval("SELECT count(*) FROM memories WHERE status='active'")
        users = await conn.fetchval("SELECT count(DISTINCT user_id) FROM memories")
        adds = await conn.fetchval("SELECT count(*) FROM add_requests")
        ident = await conn.fetchval("SELECT v FROM kv WHERE k='embedding_identity'")
    await pool.close()
    print(f"memories 总数: {total}（active: {active}）")
    print(f"user 数: {users}，add_requests: {adds}")
    print(f"embedding 身份: {ident}")


async def purge_prefix(dsn: str, prefix: str, confirm: bool):
    if not confirm:
        print("危险操作：必须加 --confirm 才会执行")
        sys.exit(2)
    pool = await _pool(dsn)
    async with pool.acquire() as conn:
        async with conn.transaction():
            # asyncpg 的 execute 返回 "DELETE n"，用 CTE 取 count 更稳
            n_mem = await conn.fetchval(
                "WITH d AS (DELETE FROM memories WHERE user_id LIKE $1 RETURNING 1) "
                "SELECT count(*) FROM d", prefix + "%")
            n_add = await conn.fetchval(
                "WITH d AS (DELETE FROM add_requests WHERE user_id LIKE $1 RETURNING 1) "
                "SELECT count(*) FROM d", prefix + "%")
    await pool.close()
    print(f"已删除 memories {n_mem} 条，add_requests {n_add} 条（前缀 {prefix!r}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL",
                    "postgresql://aml:amlpass@localhost:5432/aml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    p = sub.add_parser("purge-prefix")
    p.add_argument("--user-prefix", required=True)
    p.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    if args.cmd == "stats":
        asyncio.run(stats(args.database_url))
    else:
        asyncio.run(purge_prefix(args.database_url, args.user_prefix, args.confirm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
