#!/usr/bin/env python3
"""本地 Smoke 自检：对照 AML 公开契约验证 Add/Search 全链路。

不联系赛事官网，不消耗官方 Smoke 配额。
建议用 EMBEDDING_PROVIDER=stub（零成本）先跑通契约，再换真实 embedding 跑。

用法：
  python scripts/smoke.py --base-url http://127.0.0.1:8000 --api-key <MEMORY_API_KEY>
"""
from __future__ import annotations

import argparse
import sys
import uuid

import httpx

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    CHECKS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", required=True)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    key = args.api_key
    run = uuid.uuid4().hex[:8]
    user_a = f"smoke:{run}:user-a"
    user_b = f"smoke:{run}:user-b"
    token = f"SMOKETOKEN-{run}"  #  planted unique token

    # 代理处理：沙箱出口必须走代理（https_proxy），但 no_proxy 里的 [::1] 写法会触发
    # httpx 解析 bug，先清掉；VPS 上无代理变量，trust_env=True 与 False 行为一致。
    import os
    for _k in ("no_proxy", "NO_PROXY"):
        os.environ.pop(_k, None)
    c = httpx.Client(base_url=base, timeout=60.0, trust_env=True)

    # 1. health 无需鉴权
    r = c.get("/health")
    check("health 无鉴权 2xx", 200 <= r.status_code < 300, f"status={r.status_code}")

    # 2. add 无鉴权 -> 401
    r = c.post("/add", json={"request_id": "x", "messages": [], "user_id": "x", "session_id": "x"})
    check("add 无鉴权 401", r.status_code == 401, f"status={r.status_code}")

    # 3. 三种鉴权方式都可用
    auth_variants = {
        "X-Api-Key": {"X-Api-Key": key},
        "Bearer": {"Authorization": f"Bearer {key}"},
        "Token": {"Authorization": f"Token {key}"},
    }
    for name, headers in auth_variants.items():
        r = c.get("/health", headers=headers)  # health 本来就不需要鉴权，换用 search 探
        r = c.post("/search", headers=headers,
                   json={"query": "ping", "user_id": f"smoke:{run}:nobody", "top_k": 1})
        check(f"鉴权方式 {name} 可用", r.status_code == 200, f"status={r.status_code}")
    H = {"X-Api-Key": key}

    # 4. add 回传 request_id/user_id/session_id
    req_id = f"smoke:{run}:req-1"
    payload = {
        "request_id": req_id,
        "messages": [
            {"role": "user", "timestamp": 1704067200000,
             "content": f"Ada Lovelace wrote notes about the Analytical Engine. {token}"}
        ],
        "user_id": user_a,
        "session_id": f"smoke:{run}:session-1",
    }
    r = c.post("/add", headers=H, json=payload)
    ok = r.status_code == 200
    if ok:
        body = r.json()
        ok = (body.get("success") is True
              and body.get("request_id") == req_id
              and body.get("user_id") == user_a
              and body.get("session_id") == f"smoke:{run}:session-1")
    check("add 200、success=true 且原样回传三字段", ok, f"status={r.status_code} body={r.text[:200]}")

    # 5. 相同 request_id + 相同内容 -> 幂等 200（不产生重复）
    r = c.post("/add", headers=H, json=payload)
    check("重复提交幂等 200", r.status_code == 200, f"status={r.status_code}")

    # 6. 相同 request_id + 不同内容 -> 409
    bad = dict(payload)
    bad["messages"] = [{"role": "user", "content": "different content"}]
    r = c.post("/add", headers=H, json=bad)
    check("request_id 复用不同内容 409", r.status_code == 409, f"status={r.status_code}")

    # 7. search 立即可见；schema 为 {"data": [...]}
    r = c.post("/search", headers=H, json={
        "query": "Who wrote notes about the Analytical Engine?",
        "user_id": user_a, "top_k": 100,
    })
    ok = r.status_code == 200
    data = r.json().get("data") if ok else None
    ok = ok and isinstance(data, list) and len(data) > 0
    hit = ok and any(token in (d.get("content") or "") for d in data)
    check("search 立即可见且含 planted 证据", bool(ok and hit),
          f"status={r.status_code} n={len(data) if isinstance(data, list) else '?'}")
    ids_ok = bool(ok) and all(isinstance(d.get("id"), str) and d.get("id") for d in data)
    check("search 每条证据有非空字符串 id", ids_ok,
          f"sample={str(data[0])[:160] if ok and data else '?'}")
    uniq_ok = bool(ok) and len({d.get("id") for d in data}) == len(data)
    check("search 同一响应内 id 唯一", uniq_ok)

    # 8. 不超过 top_k
    r = c.post("/search", headers=H, json={
        "query": "Analytical Engine", "user_id": user_a, "top_k": 3})
    data = r.json().get("data", [])
    check("search 结果不超过 top_k", r.status_code == 200 and len(data) <= 3,
          f"n={len(data)}")

    # 9. options 字段可传
    r = c.post("/search", headers=H, json={
        "query": "Who wrote the notes?", "options": ["A. Ada Lovelace", "B. Grace Hopper"],
        "user_id": user_a, "top_k": 10})
    check("search options 字段可用", r.status_code == 200 and isinstance(r.json().get("data"), list),
          f"status={r.status_code}")

    # 10. 未知用户 -> 空 data
    r = c.post("/search", headers=H, json={
        "query": "anything", "user_id": f"smoke:{run}:nobody", "top_k": 10})
    check("未知用户返回空 data", r.status_code == 200 and r.json().get("data") == [],
          f"body={r.text[:200]}")

    # 11. 跨用户隔离
    r = c.post("/add", headers=H, json={
        "request_id": f"smoke:{run}:req-b",
        "messages": [{"role": "user", "content": f"User B secret fact. {token}-B"}],
        "user_id": user_b, "session_id": f"smoke:{run}:session-b",
    })
    check("用户 B 写入成功", r.status_code == 200, f"status={r.status_code}")
    r = c.post("/search", headers=H, json={
        "query": f"{token}-B", "user_id": user_a, "top_k": 100})
    leaked = any(f"{token}-B" in (d.get("content") or "") for d in r.json().get("data", []))
    check("跨用户隔离（A 搜不到 B 的记忆）", r.status_code == 200 and not leaked)
    r = c.post("/search", headers=H, json={
        "query": f"{token}-B", "user_id": user_b, "top_k": 100})
    found = any(f"{token}-B" in (d.get("content") or "") for d in r.json().get("data", []))
    check("同用户可搜到自己的记忆", r.status_code == 200 and found)

    # 12. 幂等未产生重复：A 的证据条数应为 1（含 token 的）
    r = c.post("/search", headers=H, json={
        "query": token, "user_id": user_a, "top_k": 100})
    n = sum(1 for d in r.json().get("data", []) if token in (d.get("content") or ""))
    check("幂等未产生重复写入", n == 1, f"n={n}")

    # 13. 相同 request_id + 相同内容但不同 user_id -> 409（指纹已含 user_id）
    cross = dict(payload)
    cross["user_id"] = user_b
    r = c.post("/add", headers=H, json=cross)
    check("同 request_id 换用户复用 409", r.status_code == 409, f"status={r.status_code}")

    # 14. 证据 id 跨请求稳定（同一记忆两次搜到的是同一个 id）
    r1 = c.post("/search", headers=H, json={"query": token, "user_id": user_a, "top_k": 100})
    r2 = c.post("/search", headers=H, json={"query": token, "user_id": user_a, "top_k": 100})
    def _ids(resp):
        return sorted(d.get("id") for d in resp.json().get("data", [])
                      if token in (d.get("content") or ""))
    ids1, ids2 = _ids(r1), _ids(r2)
    check("证据 id 跨请求稳定", r1.status_code == 200 and r2.status_code == 200
          and len(ids1) > 0 and ids1 == ids2, f"ids1={ids1} ids2={ids2}")

    print()
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"共 {len(CHECKS)} 项，通过 {len(CHECKS) - len(failed)} 项"
          + (f"，失败：{failed}" if failed else "，全部通过"))
    print(f"本次 smoke 使用的 user 前缀：smoke:{run}（可用 scripts/admin.py 按前缀清理）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
