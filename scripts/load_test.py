#!/usr/bin/env python3
"""并发压测：验证服务扛住 Add 64 / Search 256 的评测并发。

注意：Add 压测会产生真实写入（走 embedding/LLM，有成本），先用小量级试。
用法：
  python scripts/load_test.py --base-url http://127.0.0.1:8000 --api-key KEY \
      --adds 64 --searches 256 --add-concurrency 64 --search-concurrency 64
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import uuid

import httpx


async def _one_add(client: httpx.AsyncClient, headers: dict, i: int, run: str):
    t0 = time.perf_counter()
    try:
        r = await client.post("/add", headers=headers, json={
            "request_id": f"load:{run}:add-{i}",
            "messages": [{"role": "user",
                          "content": f"Load test fact number {i}, run {run}."}],
            "user_id": f"load:{run}:user-{i % 8}",
            "session_id": f"load:{run}:sess-{i % 8}",
        })
        ok = r.status_code == 200
        err = "" if ok else f"status={r.status_code}"
    except Exception as e:  # noqa: BLE001
        ok, err = False, repr(e)[:120]
    return ok, time.perf_counter() - t0, err


async def _one_search(client: httpx.AsyncClient, headers: dict, i: int, run: str):
    t0 = time.perf_counter()
    try:
        r = await client.post("/search", headers=headers, json={
            "query": f"Load test fact number {i % 64}",
            "user_id": f"load:{run}:user-{i % 8}",
            "top_k": 100,
        })
        ok = r.status_code == 200 and isinstance(r.json().get("data"), list)
        err = "" if ok else f"status={r.status_code}"
    except Exception as e:  # noqa: BLE001
        ok, err = False, repr(e)[:120]
    return ok, time.perf_counter() - t0, err


async def _run_phase(client, headers, total, concurrency, fn, run, label):
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(i):
        async with sem:
            return await fn(client, headers, i, run)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[_guarded(i) for i in range(total)])
    dt = time.perf_counter() - t0
    oks = [lat for ok, lat, _ in results if ok]
    fails = [(i, e) for i, (ok, _, e) in enumerate(results) if not ok]
    print(f"--- {label}: {total} 请求 / 并发 {concurrency} / 用时 {dt:.1f}s ---")
    print(f"成功 {len(oks)}/{total}，失败 {len(fails)}")
    if oks:
        print(f"延迟 p50={statistics.median(oks):.3f}s "
              f"p95={sorted(oks)[int(len(oks) * 0.95)]:.3f}s "
              f"max={max(oks):.3f}s")
    for i, e in fails[:5]:
        print(f"  失败样例 #{i}: {e}")
    return not fails


async def main_async(args) -> int:
    run = uuid.uuid4().hex[:8]
    headers = {"X-Api-Key": args.api_key}
    limits = httpx.Limits(max_connections=300, max_keepalive_connections=100)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), timeout=180.0, limits=limits,
        trust_env=False,  # 忽略代理环境变量（沙箱 no_proxy 格式会触发 httpx 解析 bug）
    ) as client:
        ok1 = await _run_phase(client, headers, args.adds, args.add_concurrency,
                               _one_add, run, "ADD")
        ok2 = await _run_phase(client, headers, args.searches, args.search_concurrency,
                               _one_search, run, "SEARCH")
    print(f"\n本次压测 user 前缀：load:{run}（可用 scripts/admin.py 按前缀清理）")
    return 0 if (ok1 and ok2) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--adds", type=int, default=64)
    ap.add_argument("--searches", type=int, default=256)
    ap.add_argument("--add-concurrency", type=int, default=64)
    ap.add_argument("--search-concurrency", type=int, default=64)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
