#!/usr/bin/env python3
"""PRF 可行性验证：两轮检索 + 客户端 RRF 融合（multi_hop 召回攻坚）。

只调公网 Search API，不动服务端。数据复用 eval:locomo 已灌入（幂等，不重灌）。

流程（每题）：
  1. 第一轮：POST /search(top_k=100)，原 query
  2. q2 = 原 query + " " + 第一轮 top5 的 content（每条截断 200 字符）
  3. 第二轮：POST /search(top_k=100)，q2
  4. 客户端 RRF 融合两轮结果（k=60，与服务端一致；按 session_id 融合，
     本评测 1 turn = 1 memory，session_id 即 dia_id，与按 id 融合等价）
  5. 融合 top100 按 evidence 算分（复用 replay.score_one）

对照组：reports/replay_ablation2_hybrid_norank_20261002T001040.json
（hybrid_rrf，reranker 关；总体 R@1 0.164 / MRR 0.268；
 multi_hop R@1 0.0212 / R@100 0.5034）。

用法：
  python3 eval/replay_prf.py [--concurrency 4] [--top-k 100]

只调公网 Search API；key 默认读 ~/workspace/aml/deploy/MEMORY_API_KEY.txt。
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
CONV30_USER = "eval:locomo"

RRF_K = 60
PRF_TOP_N = 5       # 取第一轮 topN 做反馈
PRF_TRUNC = 200    # 每条 content 截断字符数


def user_id_for(sample_id: str) -> str:
    return CONV30_USER if sample_id == "conv-30" else f"{BASE_USER}:{sample_id}"


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """客户端 RRF：与服务端 _rrf_fuse 同语义，按 session_id 融合。"""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, sid in enumerate(ranked, start=1):
            if not sid:
                continue
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
    return [sid for sid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def main() -> int:
    ap = argparse.ArgumentParser(description="PRF two-round retrieval feasibility check")
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

    def do_search(query_text: str, uid: str):
        res = post_json(f"{base}/search",
                        {"query": query_text, "user_id": uid,
                         "top_k": args.top_k}, api_key)
        items = res.get("data", []) or []
        return items

    def do_question(q, uid: str):
        qtext = q["question"]
        # 第一轮：原 query
        r1 = do_search(qtext, uid)
        # 构造 q2：原 query + 第一轮 topN 的 content（截断）
        feedback = " ".join((it.get("content") or "")[:PRF_TRUNC] for it in r1[:PRF_TOP_N])
        q2 = (qtext + " " + feedback).strip()
        # 第二轮：扩展 query
        r2 = do_search(q2, uid)
        # 客户端 RRF 融合
        ranked1 = [it.get("session_id") for it in r1]
        ranked2 = [it.get("session_id") for it in r2]
        fused = rrf_fuse([ranked1, ranked2])[: args.top_k]
        s = score_one(set(q["evidence"]), fused)
        s.update({"sample": q["_sample"],
                  "category": CATEGORIES.get(q["category"], str(q["category"])),
                  "question": qtext[:120]})
        return s, len(qtext), len(q2)

    t_all = time.time()
    all_valid, per_sample = [], {}
    total_q1_chars = 0
    total_q2_chars = 0
    search_time = 0.0

    for sample in dataset:
        sid = sample["sample_id"]
        uid = user_id_for(sid)
        turns = [(d, t) for d, sn, t in iter_turns(sample) if t]
        ingested_set = {d for d, _ in turns}
        print(f"[{sid}] user_id={uid} turns={len(turns)}", flush=True)

        qa_pool = [q for q in sample["qa"]
                   if q.get("category") != 5 and q.get("evidence")
                   and all(e in ingested_set for e in (q.get("evidence") or []))]
        for q in qa_pool:
            q["_sample"] = sid

        t1 = time.time()
        per_q = []
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(do_question, q, uid) for q in qa_pool]
            for fut in cf.as_completed(futs):
                try:
                    s, n1, n2 = fut.result()
                    per_q.append(s)
                    total_q1_chars += n1
                    total_q2_chars += n2
                except Exception as e:  # noqa: BLE001
                    per_q.append({"sample": sid, "error": str(e)[:200]})
        dt = time.time() - t1
        search_time += dt
        valid = [s for s in per_q if "error" not in s]
        all_valid.extend(valid)
        per_sample[sid] = {"n": len(valid),
                           "n_errors": len(per_q) - len(valid),
                           "turns": len(turns),
                           "metrics": agg(valid)}
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
            "user_id_scheme": "conv-30 -> 'eval:locomo'；其余 -> 'eval:locomo:<sample>'（复用已灌入数据）",
            "retrieval": "hybrid_rrf（服务端）+ 客户端 PRF 两轮 RRF 融合",
            "prf": {
                "rounds": 2,
                "round1": "原 query，top_k=100",
                "feedback": f"第一轮 top{PRF_TOP_N} 的 content，每条截断 {PRF_TRUNC} 字符",
                "round2": "q2 = 原 query + 反馈文本，top_k=100",
                "fuse": f"客户端 RRF（k={RRF_K}）按 session_id 融合两轮，取 top{args.top_k}",
            },
            "extraction": "passthrough",
            "conflict": "none", "reranker": "none（服务端关闭，与对照组一致）",
            "embedding": "text-embedding-v4",
            "dataset": "LoCoMo locomo10.json（CC BY-NC 4.0），adversarial 类不计分",
            "control": "reports/replay_ablation2_hybrid_norank_20261002T001040.json",
        },
        "eval": {
            "n_questions": len(all_valid),
            "overall": agg(all_valid),
            "per_category": {c: {"n": len(r), **agg(r)} for c, r in sorted(by_cat.items())},
            "per_sample": per_sample,
            "per_question": all_valid,
        },
        "cost": {
            "round1_query_chars": total_q1_chars,
            "round2_query_chars": total_q2_chars,
            "est_embedding_calls": len(all_valid) * 2,
            "note": "token≈chars/4；DashScope text-embedding-v4 约 ¥0.7/1M tokens",
        },
        "timing": {
            "search_s": round(search_time, 1),
            "total_s": round(time.time() - t_all, 1),
        },
    }

    out = args.out or os.path.expanduser(
        f"~/workspace/aml/eval/reports/replay_prf_norank_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==== PRF RESULT ====")
    print(f"questions scored: {len(all_valid)}")
    for k, v in report["eval"]["overall"].items():
        print(f"  {k}: {v}")
    print("per category:")
    for c, m in report["eval"]["per_category"].items():
        n = m.pop("n")
        print(f"  {c} (n={n}): " + ", ".join(f"{k}={v}" for k, v in m.items()))
    print(f"timing: search {search_time:.0f}s")
    print(f"est embedding calls: {report['cost']['est_embedding_calls']}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
