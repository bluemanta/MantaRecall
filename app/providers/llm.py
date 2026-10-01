"""LLM client。学术榜硬约束：LLM 相关组件必须用 gpt-4o-mini。"""
from __future__ import annotations

import asyncio
import json
from typing import Protocol

import httpx


class LLMClient(Protocol):
    async def chat_json(
        self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.0
    ) -> dict: ...
    async def aclose(self) -> None: ...


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        # 去掉首尾 ``` 行
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        s = "\n".join(lines)
    return s.strip()


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_concurrency: int = 16,
    ):
        if not api_key:
            raise RuntimeError("LLM_API_KEY 为空，无法调用 LLM")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}"},
            # 直连，不读代理环境变量（沙箱 no_proxy 格式会触发 httpx 解析 bug）
            trust_env=False,
        )

    async def chat_json(
        self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.0
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                async with self._sem:
                    r = await self._client.post("/chat/completions", json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"llm {r.status_code}: {r.text[:200]}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"] or "{}"
                return json.loads(_strip_fences(content))
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
                last_err = e
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用失败（重试耗尽）: {last_err}")

    async def aclose(self) -> None:
        await self._client.aclose()


class NullLLM:
    """extraction 关闭时的占位：被调用说明配置有误，直接报错。"""

    async def chat_json(self, system: str, user: str, max_tokens: int = 2000,
                        temperature: float = 0.0) -> dict:
        raise RuntimeError("LLM 未配置（extraction.enabled=false），不应调用 LLM")

    async def aclose(self) -> None:
        return None
