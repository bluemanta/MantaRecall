"""AML 接口的 Pydantic schema。

契约来源（2026-10-01 核验）：
- GET /health 无需鉴权，2xx 即就绪
- POST /add 同步；HTTP 200 = 全部消息已持久化且立即可搜；
  响应必须原样回传 request_id / user_id / session_id；
  相同 request_id + 相同内容幂等，相同 request_id + 不同内容 -> 409
- POST /search 返回 {"data": [...]}，按相关性排序，数量不超过 top_k；
  只返回记忆证据，不生成答案；user_id 是唯一的隔离边界
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = "user"
    content: str
    timestamp: int | None = None  # 毫秒时间戳（可选）


class AddRequest(BaseModel):
    request_id: str
    messages: list[Message] = Field(default_factory=list)
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    options: list[str] | None = None  # 选择题选项（可选）
    user_id: str
    top_k: int = 10


class EvidenceItem(BaseModel):
    content: str
    score: float
    session_id: str | None = None
    timestamp: int | None = None


class SearchResponse(BaseModel):
    data: list[EvidenceItem] = Field(default_factory=list)
