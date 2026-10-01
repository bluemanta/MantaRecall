#!/usr/bin/env python3
"""全量 LoCoMo 回放：10 个样本全部灌入 + 全部有效 QA 评测，产出 MantaRecall baseline。

关键设计（正确性优先）：
- 每个样本用独立 user_id（"eval:locomo:<sample>"），避免不同样本的
  session_id（D1:3 等）跨样本污染检索结果。
- 唯一例外 conv-30 沿用 user_id="eval:locomo"：最小样本已灌入其前 50 条
  turn，request_id 方案确定（"eval:locomo:conv-30:<dia>" + 相同内容），
  /add 幂等路径（ON CONFLICT DO NOTHING -> 相同内容直接 200）不会重复
  调用 embedding，因此重放这 50 条零重复计费。
- session_id = 原始 dia_id（turn 级对齐），在各自 user 命名空间内唯一。

用法：
  python3 eval/replay_full.py [--concurrency 4] [--top-k 100]

只调公网 Add/Search API；key 默认读 ~/workspace/aml/deploy/MEMORY_API_KEY.txt。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from replay import (  # noqa: E402
    CATEGORIES,
    RECALL_KS,
    iter_turns,
    load_api_key,
    post_json,
    score_one,
)

BASE_USER = "eval:locomo"
CONV30_USER = "eval:locomo"  # 沿用最小样本的 user_id，复用已灌入的 50 条


def user_id_for(sample_id: str) -> str:
    return CONV30_USER if sample_id == "conv-30" else f"{BASE_USER}:{sample_id}"


def request_id_for(user_id: str, sample_id: str, dia: str) -> str:
    return f"{user_id}:{sample_id}:{dia}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Full LoCoMo replay for MantaRecall baseline")
    ap.add_argument("--base-url", default="https://aml-api.imalltrix.com")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--data", default=os.path.expanduser("~/workspace/aml/eval/data/locomo10.json"))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    api_key = load_api_key(args.api_key)
    base = args.base_url.rstrip("/")
    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    dataset.sort(key=lambda s: s.get("sample_id", ""))

    metric_keys = [f"recall@{k}" for k in RECALL_KS] + ["mrr", "ndcg@100"]

    def agg(rows):
        return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in metric_keys} if rows else {}

    t_all = time.time()
    all_valid, per_sample, ingest_stats = [], {}, {}
    total_add_calls = 0
    total_ingest_chars = 0
    total_q_chars = 0
    ingest_time = 0.0
    search_time = 0.0

    for sample in dataset:
        sid = sample["sample_id"]
        uid = user_id_for(sid)
        turns = [(d, t) for d, sn, t in iter_turns(sample) if t]
        ingested_ids = [d for d, _ in turns]
        print(f"[{sid}] user_id={uid} turns={len(turns)}", flush=True)

        # ---- 灌入 ----
        def do_add(item):
            dia, text = item
            payload = {
                "request_id": request_id_for(uid, sid, dia),
                "messages": [{"role": "user", "content": text}],
                "user_id": uid,
                "session_id": dia,
            }
            post_json(f"{base}/add", payload, api_key)
            return len(text)

        t0 = time.time()
        ok, failed, chars_in = 0, [], 0
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(do_add, it): it[0] for it in turns}
            for fut in cf.as_completed(futs):
                dia = futs[fut]
                try:
                    chars_in += fut.result()
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    failed.append({"dia_id": dia, "error": str(e)[:200]})
        dt = time.time() - t0
        ingest_time += dt
        total_add_calls += len(turns)
        total_ingest_chars += chars_in
        print(f"[{sid}] ingest ok={ok}/{len(turns)} failed={len(failed)} {dt:.1f}s", flush=True)

        ingested_set = set(ingested_ids)
        # ---- 选 QA ----
        qa_pool = [q for q in sample["qa"]
                   if q.get("category") != 5 and q.get("evidence")
                   and all(e in ingested_set for e in (q.get("evidence") or []))]

        # ---- 查询 + 算分 ----
        def do_search(q):
            res = post_json(f"{base}/search",
                            {"query": q["question"], "user_id": uid,
                             "top_k": args.top_k}, api_key)
            ranked = [it.get("session_id") for it in res.get("data", [])]
            return q, ranked, len(q["question"])

        t1 = time.time()
        per_q, q_chars = [], 0
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(do_search, q) for q in qa_pool]
            for fut in cf.as_completed(futs):
                try:
                    q, ranked, nch = fut.result()
                    s = score_one(set(q["evidence"]), ranked)
                    s.update({"sample": sid,
                              "category": CATEGORIES.get(q["category"], str(q["category"])),
                              "question": q["question"][:120]})
                    per_q.append(s)
                    q_chars += nch
                except Exception as e:  # noqa: BLE001
                    per_q.append({"sample": sid, "error": str(e)[:200]})
        dt = time.time() - t1
        search_time += dt
        total_q_chars += q_chars
        valid = [s for s in per_q if "error" not in s]
        all_valid.extend(valid)
        per_sample[sid] = {"n": len(valid),
                           "n_errors": len(per_q) - len(valid),
                           "turns": len(turns),
                           "ingest_ok": ok,
                           "metrics": agg(valid)}
        ingest_stats[sid] = {"user_id": uid, "turns": len(turns), "ok": ok,
                             "failed": failed}
        print(f"[{sid}] scored={len(valid)}/{len(qa_pool)} {dt:.1f}s", flush=True)

    by_cat = defaultdict(list)
    for s in all_valid:
        by_cat[s["category"]].append(s)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": base,
            "top_k": args.top_k,
            "concurrency": args.concurrency,
            "user_id_scheme": ("conv-30 -> 'eval:locomo'（复用最小样本已灌入的 50 条，"
                               "幂等 replay 无重复 embedding）；其余样本 -> 'eval:locomo:<sample>'"),
            "request_id_scheme": "deterministic '<user_id>:<sample>:<dia>'（相同内容重复提交幂等 200）",
            "retrieval": "hybrid_rrf", "extraction": "passthrough",
            "conflict": "none", "reranker": "none",
            "embedding": "text-embedding-v4",
            "dataset": "LoCoMo locomo10.json（CC BY-NC 4.0），adversarial 类不计分",
        },
        "ingest": {
            "total_turns": sum(v["turns"] for v in ingest_stats.values()),
            "add_api_calls": total_add_calls,
            "idempotent_replays_no_embedding": 50,
            "failed": sum(len(v["failed"]) for v in ingest_stats.values()),
            "ingest_chars": total_ingest_chars,
            "per_sample": ingest_stats,
        },
        "eval": {
            "n_questions": len(all_valid),
            "overall": agg(all_valid),
            "per_category": {c: {"n": len(r), **agg(r)} for c, r in sorted(by_cat.items())},
            "per_sample": per_sample,
            "per_question": all_valid,
        },
        "cost": {
            "ingest_chars": total_ingest_chars,
            "query_chars": total_q_chars,
            # passthrough: 1 turn = 1 fact = 1 embedding；search 每次 1 次 query embedding
            "est_embedding_calls": (total_add_calls - 50) + len(all_valid),
            "note": "token≈chars/4；DashScope text-embedding-v4 约 ¥0.7/1M tokens",
        },
        "timing": {
            "ingest_s": round(ingest_time, 1),
            "search_s": round(search_time, 1),
            "total_s": round(time.time() - t_all, 1),
        },
    }

    out = args.out or os.path.expanduser(
        f"~/workspace/aml/eval/reports/replay_full_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==== FULL BASELINE ====")
    print(f"questions scored: {len(all_valid)}")
    for k, v in report["eval"]["overall"].items():
        print(f"  {k}: {v}")
    print("per category:")
    for c, m in report["eval"]["per_category"].items():
        n = m.pop("n")
        print(f"  {c} (n={n}): " + ", ".join(f"{k}={v}" for k, v in m.items()))
    print(f"timing: ingest {ingest_time:.0f}s + search {search_time:.0f}s")
    print(f"est embedding calls: {report['cost']['est_embedding_calls']}")
    print(f"report: {out}")
    return 0 if not any(v["failed"] for v in ingest_stats.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
