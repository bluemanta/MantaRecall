"""Search 精排（rerank）。

学术榜：reranker 不限。
- NoneReranker（默认）：不做精排，直接用融合排序。最快、零成本。
- CrossEncoderRerank：本地 cross-encoder 精排。RRF 融合后取前 top_n 候选，
  用 cross-encoder 对 (query, candidate) 打分重排，剩余候选保持原序缀在后面。
  不消耗 LLM 配额，学术榜合规。

选型由环境变量控制（见 app/config.py Settings）：
- RERANKER_MODEL=none（默认）→ NoneReranker
- RERANKER_MODEL=<模型名或本地路径> → CrossEncoderRerank
- RERANK_TOP_N：精排候选数，默认 50

实现新 reranker 时实现下面协议即可，main.py 会自动装配
（在 build_app_state 里按 RERANKER_MODEL 注册）。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Protocol

from app.pipeline.retrieve import Retrieved

log = logging.getLogger(__name__)


class Reranker(Protocol):
    async def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]: ...

    async def awarmup(self) -> None: ...

    def stats(self) -> dict: ...


class NoneReranker:
    async def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]:
        return items

    async def awarmup(self) -> None:
        return None

    def stats(self) -> dict:
        return {"enabled": False}


class CrossEncoderRerank:
    """本地 cross-encoder 精排。

    模型在首次调用时懒加载（线程安全单例），Docker 构建阶段已把模型
    baked 进镜像（/app/models/...），运行时不依赖外网。
    CPU 推理在 executor 里跑，不阻塞事件循环。
    """

    def __init__(self, model_name: str, top_n: int = 50):
        self.model_name = model_name
        self.top_n = max(1, int(top_n))
        self._model = None
        self._lock = threading.Lock()
        self._lat_ms: deque[float] = deque(maxlen=1024)
        self._calls = 0

    @property
    def _model_dir(self) -> str | None:
        p = Path(self.model_name)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        return None

    def _load(self):
        # 双重检查锁：多 worker 进程各自加载一次，进程内只加载一次
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import CrossEncoder

            src = self._model_dir or self.model_name
            log.info("loading cross-encoder reranker: %s", src)
            # trust_remote_code=False：只用标准 transformer 结构
            self._model = CrossEncoder(src, trust_remote_code=False)
            log.info("cross-encoder reranker ready: %s", src)
            return self._model

    async def awarmup(self) -> None:
        """启动时预热：把模型和 torch 都加载好，首个真实 query 不冷启动。"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load)
        # 跑一次 dummy 打分，触发 torch 线程池 / oneDNN 初始化
        await loop.run_in_executor(None, self._model.predict, [("warmup", "warmup")])

    async def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]:
        if not items:
            return items
        model = self._load()
        head, tail = items[: self.top_n], items[self.top_n :]
        pairs = [(query, it.content) for it in head]
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        scores = await loop.run_in_executor(None, model.predict, pairs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._lat_ms.append(dt_ms)
        self._calls += 1
        for it, s in zip(head, scores):
            it.score = float(s)
        head.sort(key=lambda x: x.score, reverse=True)
        return head + tail

    def stats(self) -> dict:
        lat = sorted(self._lat_ms)

        def pct(p: float) -> float | None:
            if not lat:
                return None
            i = min(len(lat) - 1, int(p * len(lat)))
            return round(lat[i], 1)

        return {
            "enabled": True,
            "model": self.model_name,
            "top_n": self.top_n,
            "calls": self._calls,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "max_ms": round(lat[-1], 1) if lat else None,
        }
