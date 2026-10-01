"""Embedding provider。

学术榜硬约束：模型必须是 text-embedding-v4（1024 维）。
默认走阿里 DashScope 的 OpenAI 兼容接口：
    POST {base_url}/embeddings  {"model": "text-embedding-v4", "input": [...], "dimensions": 1024}
"""
from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Protocol

import httpx


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> str: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def aclose(self) -> None: ...


class OpenAICompatibleEmbedding:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        send_dimensions: bool = True,
        batch_size: int = 32,
        timeout: float = 60.0,
        max_concurrency: int = 8,
    ):
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY 为空，无法调用 embedding 服务")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.send_dimensions = send_dimensions
        self.batch_size = batch_size
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}"},
            # 直连 DashScope/OpenAI 兼容接口，不读代理环境变量
            # （沙箱 no_proxy 里的 [::1] 写法会触发 httpx 解析 bug）
            trust_env=False,
        )

    @property
    def identity(self) -> str:
        return f"openai_compat:{self.model}:{self.dim}"

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": batch}
        if self.send_dimensions:
            payload["dimensions"] = self.dim
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                async with self._sem:
                    r = await self._client.post("/embeddings", json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"embedding {r.status_code}: {r.text[:200]}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                data = r.json()["data"]
                # 按 index 排序，保证与输入顺序一致
                ordered = sorted(data, key=lambda d: d["index"])
                vecs = [d["embedding"] for d in ordered]
                for v in vecs:
                    if len(v) != self.dim:
                        raise RuntimeError(
                            f"embedding 维度 {len(v)} 与配置 {self.dim} 不一致"
                        )
                return vecs
            except httpx.HTTPError as e:
                last_err = e
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"embedding 调用失败（重试耗尽）: {last_err}")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        batches = [
            texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
        ]
        # 顺序执行 batch，避免一次打爆配额；batch 内部已是批量调用
        for b in batches:
            out.extend(await self._embed_batch(b))
        return out

    async def aclose(self) -> None:
        await self._client.aclose()


class StubEmbedding:
    """确定性伪向量：仅用于本地无成本联调 / Smoke。绝不能用于正式评测。"""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    @property
    def identity(self) -> str:
        return f"stub:{self.dim}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            v = [rng.gauss(0, 1) for _ in range(self.dim)]
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])
        return vecs

    async def aclose(self) -> None:
        return None
