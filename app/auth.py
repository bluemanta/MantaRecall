"""鉴权：支持 X-Api-Key / Authorization: Bearer / Authorization: Token 三种。"""
from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings


def _extract_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) == 2 and parts[0].lower() in ("bearer", "token"):
        return parts[1].strip()
    return auth.strip() or None


def require_api_key(request: Request) -> str:
    settings: Settings = request.app.state.settings
    provided = _extract_key(request)
    expected = settings.memory_api_key
    if not provided or not expected or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing api key",
        )
    return provided
