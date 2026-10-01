"""配置：环境变量（.env）+ config.yaml（策略选型）。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    memory_api_key: str = Field(default="change-me", description="Add/Search 鉴权 Key")
    database_url: str = "postgresql://aml:amlpass@localhost:5432/aml"

    embedding_provider: str = "openai_compatible"  # openai_compatible | stub
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-v4"     # 学术榜硬约束：必须是 text-embedding-v4
    embedding_dim: int = 1024                      # text-embedding-v4 = 1024 维
    embedding_send_dimensions: bool = True
    embedding_batch_size: int = 32

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"                 # 学术榜硬约束：LLM 必须用 gpt-4o-mini

    config_path: str = "./config.yaml"
    log_level: str = "info"


def load_yaml_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_cfg(settings: Settings, *keys: str, default=None):
    """从 config.yaml 按路径取值，如 get_cfg(s, 'retrieval', 'strategy')."""
    node = load_yaml_config(settings.config_path)
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def build_app_state(settings: Settings):
    """组装 provider + 策略。延迟 import，避免循环依赖。"""
    from app.providers.embeddings import (
        OpenAICompatibleEmbedding,
        StubEmbedding,
    )
    from app.providers.llm import NullLLM, OpenAICompatibleLLM
    from app.pipeline.extract import LLMExtract, PassthroughExtraction
    from app.pipeline.conflict import LLMConflictJudge, NoConflict
    from app.pipeline.retrieve import DenseOnly, HybridRRF, LexicalOnly
    from app.pipeline.rerank import NoneReranker

    if settings.embedding_provider == "stub":
        embedder = StubEmbedding(dim=settings.embedding_dim)
    else:
        embedder = OpenAICompatibleEmbedding(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            send_dimensions=settings.embedding_send_dimensions,
            batch_size=settings.embedding_batch_size,
        )

    llm = None
    extraction_name = get_cfg(settings, "extraction", "strategy", default="passthrough")
    extraction_enabled = bool(get_cfg(settings, "extraction", "enabled", default=False))
    if extraction_enabled and extraction_name == "llm_extract":
        if not settings.llm_api_key:
            raise RuntimeError("extraction.enabled=true 但 LLM_API_KEY 为空")
        llm = OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        extraction = LLMExtract(llm=llm, settings=settings)
    else:
        extraction = PassthroughExtraction()

    conflict_name = get_cfg(settings, "conflict", "strategy", default="llm_judge")
    if conflict_name == "none" or llm is None and conflict_name == "llm_judge":
        # 没有 LLM 可用时自动降级为 none，避免 Add 流程崩掉
        conflict = NoConflict()
    else:
        if llm is None:
            llm = OpenAICompatibleLLM(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
            )
        conflict = LLMConflictJudge(llm=llm, settings=settings)

    retrieval_name = get_cfg(settings, "retrieval", "strategy", default="hybrid_rrf")
    retrieval = {
        "hybrid_rrf": HybridRRF(settings=settings),
        "dense": DenseOnly(settings=settings),
        "lexical": LexicalOnly(settings=settings),
    }[retrieval_name]

    rerank = NoneReranker()  # 扩展点：见 app/pipeline/rerank.py

    return {
        "embedder": embedder,
        "llm": llm or NullLLM(),
        "extraction": extraction,
        "conflict": conflict,
        "retrieval": retrieval,
        "rerank": rerank,
    }
