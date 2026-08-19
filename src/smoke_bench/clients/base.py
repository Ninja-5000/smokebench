"""LLM client abstraction."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from smoke_bench.clients._http import abort_client

if TYPE_CHECKING:
    import httpx


@dataclass
class ModelInfo:
    """Metadata about a model exposed by an endpoint."""

    id: str
    context_window: int | None = None
    modalities: list[str] = field(default_factory=list)  # e.g. ["text", "image"]
    features: list[str] = field(default_factory=list)  # e.g. ["json_mode", "tools"]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatRequest:
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.0
    system: str | None = None
    json_mode: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatChunk:
    """A streamed chunk."""

    delta: str = ""
    ttft: bool = False  # marks the first chunk
    usage: TokenUsage | None = None
    finish_reason: str | None = None


@dataclass
class ChatResponse:
    text: str
    usage: TokenUsage
    finish_reason: str | None = None
    latency_s: float = 0.0
    ttft_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract LLM client. Implementations are stateless w.r.t. config."""

    protocol: str = "abstract"

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._active_http: dict[asyncio.Task, httpx.AsyncClient] = {}

    async def abort_active(self) -> None:
        """Abort the in-flight HTTP request owned by the calling task, if any.

        Clients register the per-request httpx client before sending, so a
        retry loop can force-close the socket of the exact request that timed
        out even when multiple requests share this client concurrently.
        """
        http = self._active_http.pop(asyncio.current_task(), None)
        if http is not None:
            await abort_client(http)

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...

    @abstractmethod
    async def probe(self) -> bool:
        """Lightweight reachability check. Returns True on success."""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Async iterator over streamed chunks. Yields at least one chunk."""
        yield ChatChunk()  # pragma: no cover
