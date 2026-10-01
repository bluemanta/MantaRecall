"""Add 阶段：冲突消解（记忆治理：更新、冲突消解、删除与遗忘）。

扩展点：实现 ConflictStrategy.plan() 即可替换。
- NoConflict：全部直接入库（最快最便宜）
- LLMConflictJudge：新事实 vs 库内相似候选，用 gpt-4o-mini 裁决
  duplicate（跳过）/ supersedes+contradicts（新入库、旧标 superseded）/ unrelated（入库）

注意：裁决是软删除（status='superseded'），保留审计痕迹，不做物理删除。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from app.pipeline.extract import ExtractedFact

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass
class Candidate:
    id: int
    content: str
    similarity: float


@dataclass
class FactPlan:
    fact: ExtractedFact
    vector: list[float]
    action: str = "insert"  # insert | skip
    supersede_ids: list[int] = field(default_factory=list)


# (user_id, vector, k, min_sim) -> candidates
FetchCandidates = Callable[[str, list[float], int, float], Awaitable[list[Candidate]]]


class ConflictStrategy(Protocol):
    async def plan(
        self,
        facts: list[ExtractedFact],
        vectors: list[list[float]],
        user_id: str,
        fetch_candidates: FetchCandidates,
    ) -> list[FactPlan]: ...


class NoConflict:
    async def plan(
        self,
        facts: list[ExtractedFact],
        vectors: list[list[float]],
        user_id: str,
        fetch_candidates: FetchCandidates,
    ) -> list[FactPlan]:
        return [FactPlan(fact=f, vector=v) for f, v in zip(facts, vectors)]


class LLMConflictJudge:
    def __init__(self, llm, settings):
        self.llm = llm
        self.settings = settings
        self.system_prompt = (PROMPTS_DIR / "judge_conflict.txt").read_text(encoding="utf-8")
        from app.config import get_cfg

        self.candidate_k: int = get_cfg(settings, "conflict", "candidate_k", default=5)
        self.min_similarity: float = get_cfg(
            settings, "conflict", "min_similarity", default=0.75
        )
        max_c: int = get_cfg(settings, "server", "add_llm_concurrency", default=16)
        self._sem = asyncio.Semaphore(max_c)

    async def _judge_one(
        self,
        fact: ExtractedFact,
        vector: list[float],
        user_id: str,
        fetch_candidates: FetchCandidates,
    ) -> FactPlan:
        candidates = await fetch_candidates(
            user_id, vector, self.candidate_k, self.min_similarity
        )
        if not candidates:
            return FactPlan(fact=fact, vector=vector)
        cand_text = "\n".join(
            f"[{c.id}] (sim={c.similarity:.3f}): {c.content}" for c in candidates
        )
        user_msg = "【新记忆】\n" + fact.text + "\n\n【已有记忆】\n" + cand_text
        try:
            async with self._sem:
                result = await self.llm.chat_json(
                    system=self.system_prompt,
                    user=user_msg,
                    max_tokens=1500,
                    temperature=0.0,
                )
        except Exception:
            # 裁决失败：保守策略——新记忆入库，不动旧记忆
            return FactPlan(fact=fact, vector=vector)
        decisions = result.get("decisions") or []
        by_id = {c.id: c for c in candidates}
        plan = FactPlan(fact=fact, vector=vector)
        for d in decisions:
            if not isinstance(d, dict):
                continue
            cid = d.get("candidate_id")
            rel = d.get("relation")
            if cid not in by_id:
                continue
            if rel == "duplicate":
                plan.action = "skip"
                plan.supersede_ids = []
                break
            if rel in ("supersedes", "contradicts"):
                plan.supersede_ids.append(cid)
        return plan

    async def plan(
        self,
        facts: list[ExtractedFact],
        vectors: list[list[float]],
        user_id: str,
        fetch_candidates: FetchCandidates,
    ) -> list[FactPlan]:
        tasks = [
            self._judge_one(f, v, user_id, fetch_candidates)
            for f, v in zip(facts, vectors)
        ]
        return await asyncio.gather(*tasks)
