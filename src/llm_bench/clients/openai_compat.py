"""OpenAI-compatible chat client (also covers OpenRouter, vLLM, Ollama, etc.)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

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


class OpenAICompatClient(LLMClient):
    protocol = "openai"

    def _client(self, timeout: float = 60.0) -> httpx.AsyncClient:
        return make_client(self.base_url, self.api_key, timeout=timeout)

    async def list_models(self) -> list[ModelInfo]:
        async with self._client() as http:
            resp = await http.get("/models")
            resp.raise_for_status()
            data = resp.json()
        models: list[ModelInfo] = []
        for entry in data.get("data", []):
            mid = entry.get("id") or entry.get("model")
            if not mid:
                continue
            info = ModelInfo(
                id=str(mid),
                context_window=_extract_context(entry),
                modalities=_extract_modalities(entry),
                features=_extract_features(entry),
                raw=entry,
            )
            models.append(info)
        return models

    async def probe(self) -> bool:
        try:
            async with self._client(timeout=10.0) as http:
                resp = await http.get("/models")
                if resp.status_code != 200:
                    return False
                data = resp.json()
                return bool(data.get("data"))
        except httpx.HTTPError:
            return False

    def _payload(self, req: ChatRequest) -> dict[str, Any]:
        msgs: list[dict[str, Any]] = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        for m in req.messages:
            msgs.append({"role": m.role, "content": m.content})
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": msgs,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": False,
        }
        if req.json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(req.extra)
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._payload(request)
        start = time.perf_counter()
        async with self._client(timeout=max(60.0, request.max_tokens / 20.0)) as http:
            resp = await http.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency = time.perf_counter() - start
        choice = (data.get("choices") or [{}])[0]
        text = _extract_text(choice.get("message") or {})
        usage_data = data.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage_data.get("completion_tokens", 0) or 0),
        )
        return ChatResponse(
            text=text,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            latency_s=latency,
            raw=data,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        payload = self._payload(request)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        start = time.perf_counter()
        first = True
        text_parts: list[str] = []
        usage = TokenUsage()
        finish_reason: str | None = None
        async with self._client(timeout=max(60.0, request.max_tokens / 20.0)) as http:
            async with http.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        evt: dict[str, Any] = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    choice = (evt.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        text_parts.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    if "usage" in evt and evt["usage"]:
                        usage = TokenUsage(
                            input_tokens=int(evt["usage"].get("prompt_tokens", 0) or 0),
                            output_tokens=int(
                                evt["usage"].get("completion_tokens", 0) or 0
                            ),
                        )
                    ttft = first
                    first = False
                    yield ChatChunk(
                        delta=piece,
                        ttft=ttft,
                        usage=usage if (piece == "" and not first) else None,
                        finish_reason=finish_reason,
                    )
        # Final flush chunk carrying the full usage & latency info.
        yield ChatChunk(
            delta="",
            ttft=False,
            usage=usage,
            finish_reason=finish_reason,
        )
        del start, text_parts, first


def _extract_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def _extract_context(entry: dict[str, Any]) -> int | None:
    for key in ("context_window", "max_context_tokens", "context_length"):
        val = entry.get(key)
        if isinstance(val, int):
            return val
    meta = entry.get("meta") or {}
    for key in ("context_window", "max_context_tokens", "context_length"):
        val = meta.get(key)
        if isinstance(val, int):
            return val
    return None


def _extract_modalities(entry: dict[str, Any]) -> list[str]:
    arch = entry.get("architecture") or {}
    modalities = arch.get("input_modalities") or entry.get("modalities") or []
    if isinstance(modalities, list):
        return [str(m) for m in modalities]
    return []


def _extract_features(entry: dict[str, Any]) -> list[str]:
    feats: list[str] = []
    arch = entry.get("architecture") or {}
    out = arch.get("output_modalities") or []
    if isinstance(out, list):
        for m in out:
            feats.append(f"out:{m}")
    if entry.get("supports_json_mode") is True or "json_mode" in entry:
        feats.append("json_mode")
    if entry.get("supports_tools") is True or "tools" in entry:
        feats.append("tools")
    return feats
