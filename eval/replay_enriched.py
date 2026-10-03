#!/usr/bin/env python3
"""Enriched LoCoMo 回放：Phase 1 数据地基消融。

与 replay_full.py 的唯一区别是 Add 时喂入完整数据：
- content: "{speaker}: {text}"（speaker 名进可检索文本）
- role: speaker_a -> user, speaker_b -> assistant（不再全填 user）
- timestamp: session date_time 解析为 Unix 毫秒（API 契约允许的可选字段）
- session_id: 保持 dia_id（沿用现有计分 plumbing；session 分组的影响单独测）

用独立 user 命名空间 "eval:locomo-enr:<sample>"，不与 baseline 数据混杂。
与 baseline 对比，量化"把丢掉的数据喂回去"的 uplift（主要看 temporal 类别）。

注意：生产端目前不用 timestamp 做检索，所以本实验测的是 speaker 前缀 +
role 映射的效果；timestamp 的价值在"生产端日期增强"步骤单独测。

用法：
  python3 eval/replay_enriched.py [--concurrency 4] [--top-k 100]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from replay import (  # noqa: E402
    CATEGORIES,
    RECALL_KS,
    load_api_key,
    post_json,
    score_one,
)

BASE_USER = "eval:locomo-enr"


def user_id_for(sample_id: str, namespace: str | None = None) -> str:
    return f"{namespace or BASE_USER}:{sample_id}"


def request_id_for(user_id: str, sample_id: str, dia: str) -> str:
    return f"{user_id}:{sample_id}:{dia}"


def parse_session_time(s: str) -> int | None:
    """'1:56 pm on 8 May, 2023' -> Unix 毫秒。解析失败返回 None。"""
    if not s:
        return None
    try:
        dt = datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError:
        return None


def iter_enriched_turns(sample: dict):
    """产出 (dia_id, speaker, role, text, timestamp_ms)。"""
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "")
    speaker_b = conv.get("speaker_b", "")

    def skey(k: str):
        m = re.match(r"session_(\d+)$", k)
        return (0, int(m.group(1))) if m else (1, 0)

    for k in sorted(conv.keys(), key=skey):
        m = re.match(r"session_(\d+)$", k)
        if not m:
            continue
        ts = parse_session_time(conv.get(f"session_{m.group(1)}_date_time", ""))
        for i, t in enumerate(conv[k]):
            dia = f"D{m.group(1)}:{i + 1}"
            text = (t.get("text") or "").strip()
            if not text:
                continue
            speaker = t.get("speaker") or ""
            role = "user" if speaker == speaker_a else "assistant"
            yield dia, speaker, role, text, ts


def main() -> int:
    ap = argparse.ArgumentParser(description="Enriched LoCoMo replay (Phase 1)")
    ap.add_argument("--base-url", default="https://aml-api.imalltrix.com")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--data", default=os.path.expanduser("~/workspace/aml/eval/data/locomo10.json"))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--namespace", default=None,
                    help="覆盖 user 命名空间（默认 eval:locomo-enr），用于内容变化后的重测")
    ap.add_argument("--samples", default=None,
                    help="只跑指定样本，逗号分隔（如 conv-48,conv-50）")
    args = ap.parse_args()

    api_key = load_api_key(args.api_key)
    base = args.base_url.rstrip("/")
    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    dataset.sort(key=lambda s: s.get("sample_id", ""))
    if args.samples:
        wanted = {s.strip() for s in args.samples.split(",")}
        dataset = [s for s in dataset if s.get("sample_id") in wanted]
        print(f"filtered samples: {[s.get('sample_id') for s in dataset]}", flush=True)

    metric_keys = [f"recall@{k}" for k in RECALL_KS] + ["mrr", "ndcg@100"]

    def agg(rows):
        return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in metric_keys} if rows else {}

    t_all = time.time()
    all_valid, per_sample = [], {}
    total_add_calls = 0

    for sample in dataset:
        sid = sample["sample_id"]
        uid = user_id_for(sid, args.namespace)
        turns = list(iter_enriched_turns(sample))
        ingested_ids = [d for d, _, _, _, _ in turns]
        print(f"[{sid}] user_id={uid} turns={len(turns)}", flush=True)

        # ---- 灌入（enriched） ----
        def do_add(item):
            dia, speaker, role, text, ts = item
            content = f"{speaker}: {text}" if speaker else text
            msg = {"role": role, "content": content}
            if ts:
                msg["timestamp"] = ts
            payload = {
                "request_id": request_id_for(uid, sid, dia),
                "messages": [msg],
                "user_id": uid,
                "session_id": dia,
            }
            post_json(f"{base}/add", payload, api_key)
            return len(content)

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
        total_add_calls += len(turns)
        print(f"[{sid}] ingest ok={ok}/{len(turns)} failed={len(failed)} {dt:.1f}s", flush=True)
        if failed:
            print(f"  failed sample: {failed[:3]}", flush=True)

        ingested_set = set(ingested_ids)
        qa_pool = [q for q in sample["qa"]
                   if q.get("category") != 5 and q.get("evidence")
                   and all(e in ingested_set for e in (q.get("evidence") or []))]

        # ---- 查询 + 算分（与 baseline 完全一致） ----
        def do_search(q):
            res = post_json(f"{base}/search",
                            {"query": q["question"], "user_id": uid,
                             "top_k": args.top_k}, api_key)
            ranked = [it.get("session_id") for it in res.get("data", [])]
            return q, ranked, len(q["question"])

        t1 = time.time()
        per_q = []
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(do_search, q) for q in qa_pool]
            for fut in cf.as_completed(futs):
                try:
                    q, ranked, _ = fut.result()
                    s = score_one(set(q["evidence"]), ranked)
                    s.update({"sample": sid,
                              "category": CATEGORIES.get(q["category"], str(q["category"])),
                              "question": q["question"][:120]})
                    per_q.append(s)
                except Exception as e:  # noqa: BLE001
                    print(f"[{sid}] search failed: {str(e)[:150]}", flush=True)
        search_dt = time.time() - t1

        all_valid.extend(per_q)
        per_sample[sid] = {"overall": agg(per_q), "n": len(per_q),
                           "search_s": round(search_dt, 1)}
        by_cat = defaultdict(list)
        for r in per_q:
            by_cat[r["category"]].append(r)
        print(f"[{sid}] n={len(per_q)} overall={per_sample[sid]['overall']}", flush=True)
        for cat, rows in sorted(by_cat.items()):
            print(f"  [{sid}] {cat}: {agg(rows)} (n={len(rows)})", flush=True)

    total_dt = time.time() - t_all
    overall = agg(all_valid)
    by_cat_all = defaultdict(list)
    for r in all_valid:
        by_cat_all[r["category"]].append(r)

    report = {
        "variant": f"enriched:{args.namespace or BASE_USER}",
        "samples": args.samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(all_valid),
        "total_add_calls": total_add_calls,
        "total_time_s": round(total_dt, 1),
        "overall": overall,
        "by_category": {c: agg(rs) for c, rs in sorted(by_cat_all.items())},
        "per_sample": per_sample,
    }
    print("=" * 60)
    print(f"ENRICHED OVERALL (n={len(all_valid)}): {overall}")
    for c, rs in sorted(by_cat_all.items()):
        print(f"  {c}: {agg(rs)} (n={len(rs)})")

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "reports",
        f"replay_enriched_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
