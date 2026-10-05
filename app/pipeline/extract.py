"""Add 阶段：消息 -> 记忆条目抽取。

扩展点：实现 ExtractionStrategy 协议即可替换。
- PassthroughExtraction：原文直存（零 LLM 成本，Smoke/联调用）
- LLMExtract：用 gpt-4o-mini 抽事实/事件/偏好（正式评测用）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.models import Message

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _format_date_prefix(timestamp_ms: int | None) -> str:
    """时间地基：timestamp（毫秒）存在则返回 ' [08 May 2023]'，否则空字符串。

    日期进 content 文本，便于 lexical 匹配（"May 2023"）、dense 感知、
    以及下游答案推断（如 "yesterday" 的消解）。无 timestamp 时优雅降级。

    C4：年份不在 2000–2100 则不拼日期（不猜单位）。秒级时间戳
    （如 1683556560）会被渲染成 1970 年，污染 dense/lexical/证据。
    """
    if not timestamp_ms:
        return ""
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        if not 2000 <= dt.year <= 2100:
            return ""
        return f" [{dt.strftime('%d %B %Y')}]"
    except (ValueError, OSError, OverflowError):
        return ""


@dataclass
class ExtractedFact:
    text: str
    kind: str = "fact"  # fact | event | preference | message
    meta: dict = field(default_factory=dict)


class ExtractionStrategy(Protocol):
    async def extract(self, messages: list[Message]) -> list[ExtractedFact]: ...


class PassthroughExtraction:
    """每条非空消息原样存一条 kind=message 的记录。"""

    async def extract(self, messages: list[Message]) -> list[ExtractedFact]:
        facts = []
        for m in messages:
            content = (m.content or "").strip()
            if not content:
                continue
            date_prefix = _format_date_prefix(m.timestamp)
            facts.append(
                ExtractedFact(
                    text=f"{m.role}{date_prefix}: {content}",
                    kind="message",
                    meta={"timestamp": m.timestamp, "role": m.role},
                )
            )
        return facts


class LLMExtract:
    """用 gpt-4o-mini 把消息批量抽成原子事实/事件/偏好。"""

    def __init__(self, llm, settings):
        self.llm = llm
        self.settings = settings
        self.system_prompt = (PROMPTS_DIR / "extract_facts.txt").read_text(encoding="utf-8")
        from app.config import get_cfg

        self.batch_messages: int = get_cfg(settings, "extraction", "batch_messages", default=10)
        self.max_tokens: int = get_cfg(settings, "extraction", "max_tokens", default=2000)

    def _format_batch(self, messages: list[Message]) -> str:
        lines = []
        for i, m in enumerate(messages):
            ts = f" [ts={m.timestamp}]" if m.timestamp else ""
            lines.append(f"[{i}] ({m.role}){ts}: {m.content}")
        return "\n".join(lines)

    async def extract(self, messages: list[Message]) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        batches = [
            messages[i : i + self.batch_messages]
            for i in range(0, len(messages), self.batch_messages)
        ]
        for batch in batches:
            if not any((m.content or "").strip() for m in batch):
                continue
            try:
                result = await self.llm.chat_json(
                    system=self.system_prompt,
                    user=self._format_batch(batch),
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
            except Exception:
                # LLM 抽取失败时降级为直存，保证 Add 不丢数据
                facts.extend(await PassthroughExtraction().extract(batch))
                continue
            items = result.get("facts") or []
            if not isinstance(items, list):
                facts.extend(await PassthroughExtraction().extract(batch))
                continue
            ts0 = next((m.timestamp for m in batch if m.timestamp), None)
            for it in items:
                text = (it.get("text") or "").strip() if isinstance(it, dict) else ""
                if not text:
                    continue
                kind = it.get("kind", "fact") if isinstance(it, dict) else "fact"
                if kind not in ("fact", "event", "preference"):
                    kind = "fact"
                facts.append(
                    ExtractedFact(text=text, kind=kind, meta={"timestamp": ts0})
                )
            if not items:
                facts.extend(await PassthroughExtraction().extract(batch))
        return facts
