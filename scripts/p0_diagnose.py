#!/usr/bin/env python3
"""Full 前 P0 只读诊断（对应 eval/reports/full_readiness_review_20261005.md 的 P0 清单）。

只读保证：
- 连接级 default_transaction_read_only=on，任何写操作都会被数据库拒绝；
- statement_timeout=30s，单连接串行执行，不与线上抢连接池；
- 不打印、不导出任何记忆原文（只输出计数、长度、id、user_id）；
- dense 检查用库内已有向量作查询向量，不调用 DashScope（零成本）。

检查项：
  env      版本 / max_connections / 连接占用 / pgvector 参数
  config   WORKERS × db_pool_max 与 max_connections 的连接预算（C6）
  schema   content_tsv 生成列是否为 simple（I1）、索引清单
  share    各 user 占表比例（HNSW 截断风险区，C0/I2）
  orphans  add_requests 状态分布 + 超时 processing 孤儿（I6）
  length   content 长度分布（reranker 512 token 窗口，C8）
  ts       meta.timestamp 单位/范围（秒级会被渲染成 1970 年，C3）
  dense    用生产 _dense_search 在新连接上连调 8 次，对比精确检索（C0/I2）
  code     app/ 代码指纹，与本地仓库比对确认镜像 = 提交（I7）
  health   （可选）/health 中 reranker 状态

用法（生产，容器内跑；容器名按实际调整）：
  docker cp scripts/p0_diagnose.py aml-api-1:/tmp/p0_diagnose.py
  docker exec -w /app aml-api-1 python /tmp/p0_diagnose.py \\
      --health-url http://127.0.0.1:8000/health --json-out /tmp/p0_report.json
  docker cp aml-api-1:/tmp/p0_report.json ./p0_report.json

本地只算代码指纹（与容器内输出的 code digest 比对）：
  python scripts/p0_diagnose.py --hash-only

退出码：0 = 无 FAIL；1 = 有 FAIL；2 = 运行错误。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# 让 `from app...` 在仓库内、容器 /app、或脚本被拷到 /tmp 时都能导入
for _p in (Path(__file__).resolve().parents[1], Path.cwd(), Path("/app")):
    if (_p / "app" / "__init__.py").exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS: list[dict] = []


def record(check: str, level: str, msg: str, data=None) -> None:
    """level: PASS | WARN | FAIL | INFO"""
    RESULTS.append({"check": check, "level": level, "msg": msg, "data": data})
    print(f"[{level:4}] {check:8} {msg}", flush=True)


def _app_root() -> Path | None:
    for p in sys.path:
        if (Path(p) / "app" / "__init__.py").exists():
            return Path(p)
    return None


def code_fingerprint() -> dict:
    root = _app_root()
    if root is None:
        return {"error": "找不到 app/ 目录"}
    files = sorted(
        [p for p in (root / "app").rglob("*") if p.is_file() and p.suffix in (".py", ".txt")]
        + [root / "config.yaml"]
    )
    per_file, h = {}, hashlib.sha256()
    for p in files:
        if not p.exists():
            continue
        rel = str(p.relative_to(root))
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        per_file[rel] = d[:12]
        h.update(f"{rel}:{d}\n".encode())
    return {"root": str(root), "digest": h.hexdigest()[:16], "files": per_file}


async def _connect(dsn: str):
    import asyncpg
    from pgvector.asyncpg import register_vector

    async def _init(conn):
        await register_vector(conn)

    # 与生产 create_pool 相同的 init（register_vector），额外加只读与超时保护
    return await asyncpg.create_pool(
        dsn, min_size=1, max_size=1, init=_init,
        server_settings={
            "default_transaction_read_only": "on",
            "statement_timeout": "30000",
            "application_name": "p0_diagnose",
        },
    )


def _ns(user_id: str) -> str:
    """user_id 命名空间：eval:locomo:conv-30 -> eval:locomo；smoke:xxx -> smoke"""
    parts = user_id.split(":")
    return ":".join(parts[:2]) if parts[0] == "eval" else parts[0]


async def check_env(pool) -> dict:
    async with pool.acquire() as c:
        ver = await c.fetchval("SHOW server_version")
        pgv = await c.fetchval("SELECT extversion FROM pg_extension WHERE extname='vector'")
        maxc = int(await c.fetchval("SHOW max_connections"))
        reserved = int(await c.fetchval("SHOW superuser_reserved_connections"))
        pcm = await c.fetchval("SHOW plan_cache_mode")
        # 先触发 vector 库加载，GUC 才会注册（之前"不识别 hnsw.ef_search"多半是没加载库就 SHOW）
        await c.fetchval("SELECT '[1,2]'::vector <=> '[1,2]'::vector")
        ef = await c.fetchval("SELECT current_setting('hnsw.ef_search', true)")
        it = await c.fetchval("SELECT current_setting('hnsw.iterative_scan', true)")
        acts = await c.fetch(
            "SELECT coalesce(nullif(application_name,''),'(none)') AS app, state, count(*) AS n"
            " FROM pg_stat_activity WHERE datname = current_database() GROUP BY 1,2 ORDER BY 3 DESC"
        )
    info = {
        "server_version": ver, "pgvector": pgv, "max_connections": maxc,
        "superuser_reserved_connections": reserved, "plan_cache_mode": pcm,
        "hnsw.ef_search": ef, "hnsw.iterative_scan": it,
        "connections_now": [dict(r) for r in acts],
    }
    record("env", "INFO",
           f"PG {ver} / pgvector {pgv} / max_connections={maxc} / plan_cache_mode={pcm} / "
           f"hnsw.ef_search={ef} / iterative_scan={it or 'n/a(<0.8)'}", info)
    return info


def check_config(env_info: dict) -> None:
    workers = int(os.environ.get("WORKERS", "2"))
    pool_max = 50
    try:
        from app.config import Settings, get_cfg

        pool_max = int(get_cfg(Settings(), "server", "db_pool_max", default=50))
    except Exception as e:  # noqa: BLE001
        record("config", "WARN", f"无法读取 config.yaml（按默认 50 计算）：{e}")
    budget = workers * pool_max
    maxc = env_info["max_connections"]
    data = {
        "WORKERS": workers, "db_pool_max": pool_max, "budget": budget, "max_connections": maxc,
        "RERANKER_MODEL": os.environ.get("RERANKER_MODEL"),
        "RERANK_TOP_N": os.environ.get("RERANK_TOP_N"),
        "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL"),
    }
    # 留 10 个给 psql 排查 / admin / 本脚本
    level = "FAIL" if budget > maxc - 10 else "PASS"
    record("config", level,
           f"连接预算 WORKERS({workers}) × db_pool_max({pool_max}) = {budget}，"
           f"max_connections={maxc}（需 ≤ {maxc - 10}）", data)
    if (data["RERANKER_MODEL"] or "none").lower() == "none":
        record("config", "WARN", "RERANKER_MODEL=none：生产预期应开启 cross-encoder", data)


async def check_schema(pool) -> None:
    async with pool.acquire() as c:
        expr = await c.fetchval(
            """
            SELECT pg_get_expr(d.adbin, d.adrelid)
            FROM pg_attrdef d
            JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
            WHERE d.adrelid = 'memories'::regclass AND a.attname = 'content_tsv'
            """
        )
        idx = await c.fetch(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'memories' ORDER BY 1"
        )
    data = {"content_tsv_expr": expr, "indexes": [dict(r) for r in idx]}
    if expr and "'simple'" in expr:
        record("schema", "PASS", f"content_tsv = {expr}（与查询端 simple 一致）", data)
    else:
        record("schema", "FAIL",
               f"content_tsv = {expr!r}；查询端用 websearch_to_tsquery('simple')，"
               "不一致会让 lexical 通道静默失效（english 回滚后未重建列？）", data)
    if not any("hnsw" in (r["indexdef"] or "") for r in idx):
        record("schema", "INFO", "memories 上没有 HNSW 索引（dense 必为精确检索）", data)


async def check_share(pool) -> list[dict]:
    async with pool.acquire() as c:
        total = await c.fetchval("SELECT count(*) FROM memories")
        rows = await c.fetch(
            "SELECT user_id, count(*) AS n,"
            " count(*) FILTER (WHERE status='active') AS n_active,"
            " max(created_at) AS last_at"
            " FROM memories GROUP BY user_id ORDER BY n DESC"
        )
    users = [
        {"user_id": r["user_id"], "n": r["n"], "n_active": r["n_active"],
         "share": (r["n"] / total) if total else 0.0, "last_at": r["last_at"].isoformat()}
        for r in rows
    ]
    by_ns: dict[str, int] = {}
    for u in users:
        by_ns[_ns(u["user_id"])] = by_ns.get(_ns(u["user_id"]), 0) + u["n"]
    risky = [u for u in users if u["share"] >= 0.02]
    data = {"total": total, "n_users": len(users), "by_namespace": by_ns,
            "top": users[:15], "share_ge_2pct": [u["user_id"] for u in risky]}
    # 本地复现（PG16.2 + pgvector 0.6.2）：占比 1.2–1.6% 走精确，3.9% 起出现 HNSW 截断
    level = "WARN" if risky else "PASS"
    record("share", level,
           f"memories={total} 行 / {len(users)} 个 user；占比 ≥2% 的 user {len(risky)} 个"
           f"（截断风险区，详见 dense 检查）；命名空间：{by_ns}", data)
    return users


async def check_orphans(pool, stale_minutes: int) -> None:
    async with pool.acquire() as c:
        st = await c.fetch(
            "SELECT status, count(*) AS n, min(updated_at) AS oldest, max(updated_at) AS newest"
            " FROM add_requests GROUP BY status ORDER BY status"
        )
        orphans = await c.fetch(
            "SELECT request_id, user_id, updated_at FROM add_requests"
            " WHERE status = 'processing' AND updated_at < now() - make_interval(mins => $1)"
            " ORDER BY updated_at LIMIT 20",
            stale_minutes,
        )
    data = {
        "by_status": [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in dict(r).items()} for r in st],
        "stale_processing_sample": [
            {"request_id": r["request_id"], "user_id": r["user_id"], "updated_at": r["updated_at"].isoformat()}
            for r in orphans
        ],
    }
    if orphans:
        record("orphans", "FAIL",
               f"{len(orphans)}+ 条 processing 超过 {stale_minutes} 分钟未更新（孤儿；"
               "同 request_id 重试会先等 30s 再 503，直到 30 分钟过期）", data)
    else:
        record("orphans", "PASS", f"无超过 {stale_minutes} 分钟的 processing；状态分布见 JSON", data)


async def check_length(pool) -> None:
    async with pool.acquire() as c:
        rows = await c.fetch(
            """
            SELECT CASE WHEN user_id LIKE 'eval:%' THEN 'eval'
                        WHEN user_id LIKE 'load:%' THEN 'load'
                        ELSE 'other(smoke 等)' END AS grp,
                   count(*) AS n,
                   percentile_cont(0.5)  WITHIN GROUP (ORDER BY char_length(content)) AS p50,
                   percentile_cont(0.9)  WITHIN GROUP (ORDER BY char_length(content)) AS p90,
                   percentile_cont(0.99) WITHIN GROUP (ORDER BY char_length(content)) AS p99,
                   max(char_length(content)) AS max_len,
                   count(*) FILTER (WHERE char_length(content) > 2000) AS over_2000
            FROM memories GROUP BY 1 ORDER BY 1
            """
        )
    data = [dict(r) for r in rows]
    for r in data:
        frac = r["over_2000"] / r["n"] if r["n"] else 0
        # 约 2000 英文字符 ≈ 450+ token，超出部分 cross-encoder 看不到
        level = "WARN" if r["grp"].startswith("other") and frac >= 0.05 else "INFO"
        record("length", level,
               f"{r['grp']}: n={r['n']} p50={r['p50']:.0f} p90={r['p90']:.0f} p99={r['p99']:.0f} "
               f"max={r['max_len']} >2000字符={r['over_2000']}（{frac:.1%}）", r)


async def check_timestamps(pool) -> None:
    async with pool.acquire() as c:
        r = await c.fetchrow(
            """
            WITH t AS (
              SELECT (meta->>'timestamp')::numeric AS ts FROM memories
              WHERE meta ? 'timestamp' AND jsonb_typeof(meta->'timestamp') = 'number'
            )
            SELECT count(*) AS n,
                   min(ts) AS min_ts, max(ts) AS max_ts,
                   count(*) FILTER (WHERE ts < 100000000000) AS looks_like_seconds
            FROM t
            """
        )
        bad_render = await c.fetchval(
            "SELECT count(*) FROM memories WHERE content ~ '\\[\\d{2} \\w+ 19[67]\\d\\]'"
        )
    data = {k: (float(v) if v is not None and not isinstance(v, int) else v) for k, v in dict(r).items()}
    data["content_with_1960s_1970s_date"] = bad_render
    if r["looks_like_seconds"] or bad_render:
        record("ts", "FAIL",
               f"疑似秒级时间戳 {r['looks_like_seconds']} 条；content 含 196x/197x 日期 {bad_render} 条", data)
    else:
        record("ts", "PASS",
               f"带 timestamp 的记忆 {r['n']} 条，未见秒级/1970 渲染（Full 若带时间戳仍需加防护）", data)


DENSE_SQL = """
    SELECT id FROM memories
    WHERE user_id = $2 AND status = 'active'
    ORDER BY embedding <=> $1
    LIMIT $3
"""

EXACT_SQL = """
    WITH u AS MATERIALIZED (
      SELECT id, embedding FROM memories WHERE user_id = $2 AND status = 'active'
    )
    SELECT id FROM u ORDER BY embedding <=> $1 LIMIT $3
"""


async def check_dense(dsn: str, users: list[dict], dense_k: int, calls: int) -> None:
    try:
        from app.pipeline.retrieve import _dense_search
        prod_fn = "app.pipeline.retrieve._dense_search"
    except Exception as e:  # noqa: BLE001
        _dense_search = None
        prod_fn = f"内置等价 SQL（导入失败：{e}）"
    record("dense", "INFO", f"生产函数：{prod_fn}；每个 user 新建连接后连调 {calls} 次，k={dense_k}")
    any_fail = False
    for u in users:
        uid, n_active = u["user_id"], u["n_active"]
        if n_active == 0:
            continue
        expected = min(dense_k, n_active)
        pool = await _connect(dsn)  # 新连接 = 新的预处理语句缓存，复现前 5 次 custom plan
        try:
            async with pool.acquire() as c:
                qvec = await c.fetchval(
                    "SELECT embedding FROM memories WHERE user_id=$1 AND status='active'"
                    " ORDER BY id LIMIT 1 OFFSET $2",
                    uid, min(5, n_active - 1),
                )
            counts, first_ids, t0 = [], None, time.perf_counter()
            for _ in range(calls):
                if _dense_search is not None:
                    res = await _dense_search(pool, uid, qvec, dense_k)
                    ids = [rid for rid, _ in res]
                else:
                    async with pool.acquire() as c:
                        ids = [r["id"] for r in await c.fetch(DENSE_SQL, qvec, uid, dense_k)]
                counts.append(len(ids))
                first_ids = first_ids if first_ids is not None else ids
            ms = (time.perf_counter() - t0) * 1000 / calls
        finally:
            await pool.close()
        pool = await _connect(dsn)
        try:
            async with pool.acquire() as c:
                plan = " | ".join(r[0].strip() for r in await c.fetch("EXPLAIN " + DENSE_SQL, qvec, uid, dense_k))
                t1 = time.perf_counter()
                exact_ids = [r["id"] for r in await c.fetch(EXACT_SQL, qvec, uid, dense_k)]
                exact_ms = (time.perf_counter() - t1) * 1000
        finally:
            await pool.close()
        overlap = len(set(first_ids or []) & set(exact_ids)) / max(1, len(exact_ids))
        uses_hnsw = "idx_memories_vec" in plan
        data = {"user_id": uid, "share": round(u["share"], 4), "n_active": n_active,
                "expected": expected, "per_call": counts, "custom_plan_uses_hnsw": uses_hnsw,
                "overlap_first_call_vs_exact": round(overlap, 4),
                "avg_ms_prod": round(ms, 1), "exact_ms": round(exact_ms, 1), "plan": plan}
        short = min(counts) < expected
        any_fail |= short
        record("dense", "FAIL" if short else "PASS",
               f"{uid} share={u['share']:.1%} active={n_active} 期望={expected} 每次返回={counts} "
               f"plan={'HNSW' if uses_hnsw else 'Sort(精确)'} 与精确重合={overlap:.0%} "
               f"耗时 prod≈{ms:.0f}ms / exact={exact_ms:.0f}ms", data)
    if not any_fail:
        record("dense", "INFO",
               "本次抽样未截断；注意截断取决于 user 占表比例，清理 eval 数据（表变小）后需重跑")


def pick_users(users: list[dict], top: int, recent: int, extra: list[str]) -> list[dict]:
    by_id = {u["user_id"]: u for u in users}
    chosen: dict[str, dict] = {}
    for u in users[:top]:
        chosen[u["user_id"]] = u
    non_eval = sorted(
        [u for u in users if not u["user_id"].startswith(("eval:", "load:"))],
        key=lambda u: u["last_at"], reverse=True,
    )
    for u in non_eval[:recent]:
        chosen[u["user_id"]] = u
    for uid in extra:
        if uid in by_id:
            chosen[uid] = by_id[uid]
        else:
            record("dense", "WARN", f"--user {uid} 不存在")
    return list(chosen.values())


def check_health(url: str) -> None:
    try:
        import httpx

        r = httpx.get(url, timeout=10, trust_env=False)
        body = r.json()
    except Exception as e:  # noqa: BLE001
        record("health", "FAIL", f"{url} 请求失败：{e}")
        return
    rr = body.get("rerank") or {}
    level = "PASS" if r.status_code == 200 and rr.get("enabled") else "WARN"
    record("health", level,
           f"HTTP {r.status_code} status={body.get('status')} rerank.enabled={rr.get('enabled')} "
           f"model={rr.get('model')} top_n={rr.get('top_n')} p95_ms={rr.get('p95_ms')}", body)


async def run(args) -> int:
    fp = code_fingerprint()
    record("code", "INFO", f"app/ 代码指纹 digest={fp.get('digest')}（与本地 --hash-only 输出比对）", fp)
    if args.hash_only:
        return 0
    dsn = args.database_url or os.environ.get("DATABASE_URL")
    if not dsn:
        print("缺少 --database-url 且环境变量 DATABASE_URL 为空", file=sys.stderr)
        return 2
    pool = await _connect(dsn)
    try:
        env_info = await check_env(pool)
        check_config(env_info)
        await check_schema(pool)
        users = await check_share(pool)
        await check_orphans(pool, args.stale_minutes)
        await check_length(pool)
        await check_timestamps(pool)
    finally:
        await pool.close()
    dense_k = args.dense_k
    if dense_k is None:
        try:
            from app.config import Settings, get_cfg

            dense_k = int(get_cfg(Settings(), "retrieval", "dense_k", default=200))
        except Exception:  # noqa: BLE001
            dense_k = 200
    await check_dense(dsn, pick_users(users, args.top_users, args.recent_users, args.user or []),
                      dense_k, args.calls)
    if args.health_url:
        check_health(args.health_url)
    return 1 if any(r["level"] == "FAIL" for r in RESULTS) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MantaRecall Full 前 P0 只读诊断")
    ap.add_argument("--database-url", default=None, help="默认读环境变量 DATABASE_URL")
    ap.add_argument("--health-url", default=None, help="如 http://127.0.0.1:8000/health")
    ap.add_argument("--json-out", default=None, help="完整报告写入 JSON（不含记忆原文）")
    ap.add_argument("--top-users", type=int, default=5, help="dense 检查：按行数取前 N 个 user")
    ap.add_argument("--recent-users", type=int, default=3, help="dense 检查：最近写入的非 eval/load user 数")
    ap.add_argument("--user", action="append", help="dense 检查：额外指定 user_id（可重复）")
    ap.add_argument("--dense-k", type=int, default=None, help="默认取 config.yaml retrieval.dense_k")
    ap.add_argument("--calls", type=int, default=8, help="每个 user 在新连接上连调次数（>5 才能覆盖 generic plan 切换）")
    ap.add_argument("--stale-minutes", type=int, default=5, help="processing 超过多少分钟视为孤儿")
    ap.add_argument("--hash-only", action="store_true", help="只输出代码指纹，不连数据库")
    args = ap.parse_args()
    try:
        code = asyncio.run(run(args))
    except Exception as e:  # noqa: BLE001
        record("run", "FAIL", f"诊断中断：{type(e).__name__}: {e}")
        code = 2
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": RESULTS},
                       ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"JSON 报告：{args.json_out}")
    summary = {lv: sum(1 for r in RESULTS if r["level"] == lv) for lv in ("FAIL", "WARN", "PASS")}
    print(f"汇总：{summary}，退出码 {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
