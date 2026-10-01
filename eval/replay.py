#!/usr/bin/env python3
"""LoCoMo 回放评测 harness（MantaRecall / AML 文本赛道）。

流程：读 locomo10.json -> POST /add 逐条灌入对话（1 turn = 1 次 /add，
session_id=dia_id 以便 turn 级对齐）-> 对每道 QA 题 POST /search(top_k=100)
-> 按 evidence(dia_id) 计算 recall@k / MRR / nDCG -> 输出 JSON 报告。

只用标准库 + requests（沙箱已装 2.31.0；必须用 Session keep-alive，
urllib 每次新建 CONNECT 隧道会被 egress proxy 随机掐断）。测试数据 user_id 统一为 "eval:locomo"，事后可用
scripts/admin.py purge-prefix --user-prefix 'eval:locomo' 清理。

用法示例（最小验证）：
  python3 eval/replay.py --sample conv-30 --max-turns 50 --max-qa 20
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import requests  # 2.31.0 已确认安装；用 Session keep-alive 走 egress proxy 才稳定

# category id -> 名称（与论文一致）
CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
SCORED_CATEGORIES = {1, 2, 3, 4}  # adversarial 期望拒答，不计入检索指标
RECALL_KS = (1, 5, 10, 20, 100)

DEFAULT_KEY_FILE = os.path.expanduser("~/workspace/aml/deploy/MEMORY_API_KEY.txt")


def load_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    env = os.environ.get("MEMORY_API_KEY")
    if env:
        return env.strip()
    with open(DEFAULT_KEY_FILE, encoding="utf-8") as f:
        return f.read().strip()


_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()  # trust_env 默认 True：走沙箱 egress proxy；keep-alive 复用隧道
        _tls.s = s
    return s


def post_json(url: str, payload: dict, api_key: str, timeout: int = 90,
               retries: int = 4) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            r = _session().post(
                url, json=payload, timeout=timeout,
                headers={"X-Api-Key": api_key},
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException,) as e:
            last_err = f"{type(e).__name__}: {str(e)[:150]}"
            # 连接被中途掐断时换一条新隧道重试
            _tls.s = None
            time.sleep(1 + attempt)
    raise RuntimeError(f"POST {url} 失败({retries}次重试): {last_err}")


def load_sample(path: str, sample_id: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        if s.get("sample_id") == sample_id:
            return s
    raise ValueError(f"sample {sample_id} 不在 {path} 中")


def iter_turns(sample: dict):
    """按会话顺序产出 (dia_id, session_no, text)。"""
    conv = sample["conversation"]

    def skey(k: str):
        m = re.match(r"session_(\d+)$", k)
        return (0, int(m.group(1))) if m else (1, 0)

    for k in sorted(conv.keys(), key=skey):
        m = re.match(r"session_(\d+)$", k)
        if not m:
            continue
        sn = int(m.group(1))
        for i, t in enumerate(conv[k]):
            dia = f"D{sn}:{i + 1}"
            assert t.get("dia_id", dia) == dia, (t.get("dia_id"), dia)
            yield dia, sn, (t.get("text") or "").strip()


def score_one(gold: set[str], ranked: list[str]) -> dict:
    """gold: 证据 dia_id 集合；ranked: 检索返回的 session_id 序列（已按排名）。"""
    seen = set()
    dedup = [d for d in ranked if d and not (d in seen or seen.add(d))]
    hits = [d for d in dedup if d in gold]
    out = {}
    for k in RECALL_KS:
        out[f"recall@{k}"] = len(set(dedup[:k]) & gold) / len(gold)
    out["mrr"] = 0.0
    for i, d in enumerate(dedup):
        if d in gold:
            out["mrr"] = 1.0 / (i + 1)
            break
    # nDCG@100, binary relevance
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(dedup[:100]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 100)))
    out["ndcg@100"] = dcg / idcg if idcg else 0.0
    out["n_gold"] = len(gold)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LoCoMo replay harness for MantaRecall")
    ap.add_argument("--base-url", default="https://aml-api.imalltrix.com")
    ap.add_argument("--api-key", default=None, help="默认读 ~/workspace/aml/deploy/MEMORY_API_KEY.txt")
    ap.add_argument("--data", default=os.path.expanduser("~/workspace/aml/eval/data/locomo10.json"))
    ap.add_argument("--sample", default="conv-30")
    ap.add_argument("--max-turns", type=int, default=50, help="只灌入前 N 条 turn")
    ap.add_argument("--max-qa", type=int, default=20, help="最多评测 N 道题")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--user-id", default="eval:locomo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    api_key = load_api_key(args.api_key)
    base = args.base_url.rstrip("/")
    sample = load_sample(args.data, args.sample)

    # ---- 1. 灌入 ----
    turns = [(d, t) for d, sn, t in iter_turns(sample) if t][: args.max_turns]
    ingested_ids = [d for d, _ in turns]
    print(f"[ingest] sample={args.sample} turns={len(turns)}", flush=True)

    def do_add(item):
        dia, text = item
        payload = {
            "request_id": f"{args.user_id}:{args.sample}:{dia}",
            "messages": [{"role": "user", "content": text}],
            "user_id": args.user_id,
            "session_id": dia,  # turn 级对齐：检索结果的 session_id 即 dia_id
        }
        post_json(f"{base}/add", payload, api_key)
        return dia, len(text)

    t0 = time.time()
    ok, failed, chars_in = 0, [], 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(do_add, it): it[0] for it in turns}
        for fut in cf.as_completed(futs):
            dia = futs[fut]
            try:
                _, nch = fut.result()
                ok += 1
                chars_in += nch
            except Exception as e:  # noqa: BLE001
                failed.append({"dia_id": dia, "error": str(e)[:200]})
    print(f"[ingest] ok={ok} failed={len(failed)} time={time.time()-t0:.1f}s", flush=True)

    ingested_set = set(ingested_ids)
    # ---- 2. 选 QA（证据全部在已灌入范围内、非对抗类）----
    qa_pool = []
    skipped_adv, skipped_ev = 0, 0
    for q in sample["qa"]:
        ev = q.get("evidence") or []
        if q.get("category") == 5:
            skipped_adv += 1
            continue
        if not ev or not all(e in ingested_set for e in ev):
            skipped_ev += 1
            continue
        qa_pool.append(q)
    qa_pool = qa_pool[: args.max_qa]
    print(f"[eval] questions={len(qa_pool)} (skipped adversarial={skipped_adv}, "
          f"evidence-out-of-range={skipped_ev})", flush=True)

    # ---- 3. 查询 + 算分 ----
    def do_search(q):
        res = post_json(f"{base}/search",
                        {"query": q["question"], "user_id": args.user_id,
                         "top_k": args.top_k}, api_key)
        ranked = [it.get("session_id") for it in res.get("data", [])]
        return q, ranked, len(q["question"])

    per_q, q_chars = [], 0
    t1 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(do_search, q) for q in qa_pool]
        for fut in cf.as_completed(futs):
            try:
                q, ranked, nch = fut.result()
                s = score_one(set(q["evidence"]), ranked)
                s.update({"category": CATEGORIES.get(q["category"], str(q["category"])),
                          "question": q["question"][:120]})
                per_q.append(s)
                q_chars += nch
            except Exception as e:  # noqa: BLE001
                per_q.append({"error": str(e)[:200]})
    print(f"[eval] done time={time.time()-t1:.1f}s", flush=True)

    # ---- 4. 汇总 ----
    valid = [s for s in per_q if "error" not in s]
    metric_keys = [f"recall@{k}" for k in RECALL_KS] + ["mrr", "ndcg@100"]

    def agg(rows):
        return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in metric_keys} if rows else {}

    by_cat = defaultdict(list)
    for s in valid:
        by_cat[s["category"]].append(s)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": base,
            "sample": args.sample,
            "max_turns": args.max_turns,
            "max_qa": args.max_qa,
            "top_k": args.top_k,
            "concurrency": args.concurrency,
            "user_id": args.user_id,
            # 服务端检索配置（v0.1.0 当前值，供 A/B 对照记录）
            "retrieval": "hybrid_rrf", "extraction": "passthrough",
            "conflict": "none", "reranker": "none",
            "embedding": "text-embedding-v4",
        },
        "ingest": {"turns": len(turns), "ok": ok, "failed": failed,
                   "ingest_chars": chars_in},
        "eval": {
            "n_questions": len(valid),
            "skipped": {"adversarial": skipped_adv, "evidence_out_of_range": skipped_ev,
                        "errors": len(per_q) - len(valid)},
            "overall": agg(valid),
            "per_category": {c: {"n": len(r), **agg(r)} for c, r in sorted(by_cat.items())},
            "per_question": per_q,
        },
        "cost": {
            # 本次运行实际字符数；token 按英文 ~4 chars/token 估算
            "ingest_chars": chars_in,
            "query_chars": q_chars,
            "note": "token≈chars/4；DashScope text-embedding-v4 约 ¥0.7/1M tokens（README 参考价 ¥210/300M）",
        },
    }

    out = args.out or os.path.expanduser(
        f"~/workspace/aml/eval/reports/replay_{args.sample}_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 人类可读摘要
    print("\n==== summary ====")
    print(f"ingest: {ok}/{len(turns)} ok, failed={len(failed)}")
    print(f"questions scored: {len(valid)}")
    for k, v in report["eval"]["overall"].items():
        print(f"  {k}: {v}")
    print("per category:")
    for c, m in report["eval"]["per_category"].items():
        print(f"  {c} (n={m.pop('n')}): " + ", ".join(f"{k}={v}" for k, v in m.items()))
    print(f"report: {out}")
    return 0 if not failed and valid else 1


if __name__ == "__main__":
    sys.exit(main())
