#!/usr/bin/env python3
"""P2 真实压测：对已有 eval:locomo:* user 用 LoCoMo 风格问题做 Search 压测。

合规：只用 LoCoMo 公开题，不用官方 Smoke 题目。
用法：
  python scripts/p2_load_test.py --base-url https://aml-api.imalltrix.com --api-key KEY
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time

import httpx

# LoCoMo 风格的真实问题池（来自公开数据集，非官方 Smoke 题）
QUESTIONS = [
    "When did Caroline go to the LGBTQ support group?",
    "When did Melanie paint a sunrise?",
    "What did Sam say about the birthday party?",
    "Where did Alex and Jordan first meet?",
    "What was the name of the restaurant they went to last Friday?",
    "Did Maria mention her mother's health condition?",
    "What time did the meeting start on March 15th?",
    "Who recommended the book about mindfulness?",
    "What did Tom say about his new job interview?",
    "When is Sarah's dentist appointment?",
    "What was discussed about the quarterly planning?",
    "Did anyone mention traveling to Japan next year?",
    "What is the name of John's dog?",
    "When did they decide to move to a new apartment?",
    "What did Lisa cook for dinner last Tuesday?",
]

# 生产上已有的 eval user（600+ 行 each）
USERS = [
    "eval:locomo:conv-47",
    "eval:locomo:conv-48",
    "eval:locomo-enr:conv-47",
    "eval:locomo-enr:conv-48",
]


async def _one_search(client: httpx.AsyncClient, headers: dict, i: int):
    q = QUESTIONS[i % len(QUESTIONS)]
    u = USERS[i % len(USERS)]
    t0 = time.perf_counter()
    try:
        r = await client.post("/search", headers=headers, json={
            "query": q,
            "user_id": u,
            "top_k": 100,
        })
        body = r.json()
        ok = r.status_code == 200 and isinstance(body.get("data"), list)
        err = "" if ok else f"status={r.status_code} body={r.text[:100]}"
        n_results = len(body.get("data", [])) if ok else 0
    except Exception as e:  # noqa: BLE001
        ok, err, n_results = False, repr(e)[:120], 0
    return ok, time.perf_counter() - t0, err, n_results


async def _run_level(client, headers, concurrency, duration_s, label):
    sem = asyncio.Semaphore(concurrency)
    results = []
    stop = False

    async def _worker(wid):
        i = wid
        while not stop:
            async with sem:
                r = await _one_search(client, headers, i)
                results.append(r)
            i += len(workers)

    workers = [asyncio.create_task(_worker(w)) for w in range(concurrency)]
    t0 = time.perf_counter()
    await asyncio.sleep(duration_s)
    stop = True
    await asyncio.gather(*workers, return_exceptions=True)
    dt = time.perf_counter() - t0

    oks = [lat for ok, lat, _, _ in results if ok]
    fails = [(e) for ok, _, e, _ in results if not ok]
    n_res = [n for ok, _, _, n in results if ok]

    print(f"\n=== {label}: 并发 {concurrency} / {duration_s}s / 共 {len(results)} 请求 ===")
    print(f"成功 {len(oks)}/{len(results)} ({len(oks)/max(1,len(results)):.1%})，失败 {len(fails)}")
    if oks:
        s = sorted(oks)
        print(f"延迟 p50={statistics.median(s):.3f}s "
              f"p95={s[int(len(s)*0.95)]:.3f}s "
              f"p99={s[int(len(s)*0.99)]:.3f}s "
              f"max={max(s):.3f}s")
        print(f"吞吐 {len(results)/dt:.1f} req/s，平均返回 {statistics.mean(n_res):.1f} 条")
    for e in fails[:3]:
        print(f"  失败: {e}")
    return len(fails) == 0


async def main_async(args) -> int:
    headers = {"X-Api-Key": args.api_key}
    limits = httpx.Limits(max_connections=400, max_keepalive_connections=100)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), timeout=120.0, limits=limits,
        trust_env=False,
    ) as client:
        all_ok = True
        for conc in [64, 128, 256]:
            ok = await _run_level(client, headers, conc, args.duration, f"P2-SEARCH")
            all_ok = all_ok and ok
            await asyncio.sleep(5)  # 级间冷却
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://aml-api.imalltrix.com")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--duration", type=int, default=120,
                    help="每档并发持续秒数（默认120s）")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
