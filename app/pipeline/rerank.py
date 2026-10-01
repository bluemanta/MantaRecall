"""Search 精排（rerank）扩展点。

学术榜：reranker 不限。
- NoneReranker（默认）：不做精排，直接用融合排序。最快、零成本。
- 可选实现方向（按需二选一，在 config.yaml 里把 rerank.strategy 改掉并实现类）：
  1. CrossEncoderRerank：本地 cross-encoder（如 bge-reranker 系列），需 GPU/内存，
     但不消耗 LLM 配额，学术榜合规（reranker 不限）。
  2. LLMJudgeRerank：用 gpt-4o-mini 对候选逐个打分排序，学术榜合规（LLM 必须用
     gpt-4o-mini），但 72 小时 Full 全程调用成本极高，慎用。

实现新 reranker 时实现下面协议即可，main.py 会自动装配（需在 build_app_state 里注册）。
"""
from __future__ import annotations

from typing import Protocol

from app.pipeline.retrieve import Retrieved


class Reranker(Protocol):
    async def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]: ...


class NoneReranker:
    async def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]:
        return items


# --- 下面是可选实现的骨架，按需取消注释并补全 ---
#
# class CrossEncoderRerank:
#     """本地 cross-encoder 精排骨架。
#     依赖：pip install sentence-transformers；模型如 BAAI/bge-reranker-v2-m3。
#     """
#     def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
#         from sentence_transformers import CrossEncoder
#         self.model = CrossEncoder(model_name)
#
#     async def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]:
#         import asyncio
#         pairs = [(query, it.content) for it in items]
#         loop = asyncio.get_running_loop()
#         scores = await loop.run_in_executor(None, self.model.predict, pairs)
#         for it, s in zip(items, scores):
#             it.score = float(s)
#         return sorted(items, key=lambda x: x.score, reverse=True)
#
#
# class LLMJudgeRerank:
#     """gpt-4o-mini judge 精排骨架：让模型给每条候选打 0-10 相关性分数。
#     注意成本：Full 评测问题量大时费用可观。
#     """
#     def __init__(self, llm):
#         self.llm = llm
#
#     async def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]:
#         ...逐个或批量打分，按分排序...
#         return items
