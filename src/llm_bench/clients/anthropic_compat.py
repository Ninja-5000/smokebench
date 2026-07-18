"""Anthropic-compatible chat client (``POST /v1/messages``)."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from llm_bench.clients._http import make_client
from llm_bench.clients.base import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    LLMClient,
    ModelInfo,
    TokenUsage,
)


class AnthropicCompatClient(LLMClient):
    protocol = "anthropic"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "llm-bench/0.1",
        }

    def _client(self, timeout: float = 60.0) -> httpx.AsyncClient:
        # httpx is fine without auth header for Anthropic, we set x-api-key manually.
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def list_models(self) -> list[ModelInfo]:
        async with self._client() as http:
            resp = await http.get("/models")
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
        models: list[ModelInfo] = []
        for entry in data.get("data", []):
            mid = entry.get("id")
            if not mid:
                continue
            models.append(
                ModelInfo(
                    id=str(mid),
                    context_window=(
                        int(entry["context_window"])
                        if isinstance(entry.get("context_window"), int)
                        else None
                    ),
                    raw=entry,
                )
            )
        return models

    async def probe(self) -> bool:
        try:
            async with self._client(timeout=10.0) as http:
                resp = await http.get("/models")
                if resp.status_code in (200, 404):
                    return True
                return resp.status_code < 500
        except httpx.HTTPError:
            return False

    def _payload(self, req: ChatRequest) -> dict[str, Any]:
        msgs: list[dict[str, Any]] = []
        for m in req.messages:
            msgs.append({"role": m.role, "content": m.content})
        payload: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": msgs,
        }
        if req.system:
            payload["system"] = req.system
        if req.extra:
            payload.update(req.extra)
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._payload(request)
        start = time.perf_counter()
        async with self._client(timeout=max(60.0, request.max_tokens / 20.0)) as http:
            resp = await http.post("/messages", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency = time.perf_counter() - start
        text = _extract_text(data.get("content") or [])
        usage_data = data.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_data.get("input_tokens", 0) or 0),
            output_tokens=int(usage_data.get("output_tokens", 0) or 0),
        )
        return ChatResponse(
            text=text,
            usage=usage,
            finish_reason=data.get("stop_reason"),
            latency_s=latency,
            raw=data,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        payload = self._payload(request)
        payload["stream"] = True
        start = time.perf_counter()
        first = True
        text_parts: list[str] = []
        usage = TokenUsage()
        finish_reason: str | None = None
        async with self._client(timeout=max(60.0, request.max_tokens / 20.0)) as http:
            async with http.stream("POST", "/messages", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if not payload_str:
                        continue
                    try:
                        evt: dict[str, Any] = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    etype = evt.get("type")
                    if etype == "content_block_delta":
                        delta = evt.get("delta") or {}
                        piece = delta.get("text") or ""
                        if piece:
                            text_parts.append(piece)
                        ttft = first
                        first = False
                        yield ChatChunk(delta=piece, ttft=ttft)
                    elif etype == "message_start":
                        msg = evt.get("message") or {}
                        u = msg.get("usage") or {}
                        usage = TokenUsage(
                            input_tokens=int(u.get("input_tokens", 0) or 0),
                            output_tokens=int(u.get("output_tokens", 0) or 0),
                        )
                    elif etype == "message_delta":
                        u = evt.get("usage") or {}
                        usage = TokenUsage(
                            input_tokens=int(u.get("input_tokens", usage.input_tokens) or 0),
                            output_tokens=int(
                                u.get("output_tokens", usage.output_tokens) or 0
                            ),
                        )
                        finish_reason = evt.get("delta", {}).get("stop_reason") or finish_reason
                    elif etype == "message_stop":
                        break
        yield ChatChunk(delta="", ttft=False, usage=usage, finish_reason=finish_reason)
        del start, text_parts, first


def _extract_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
