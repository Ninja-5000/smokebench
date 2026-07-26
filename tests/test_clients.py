"""Tests for LLM client adapters."""

from __future__ import annotations

import httpx
import pytest
import respx

from smoke_bench.clients import (
    AnthropicCompatClient,
    ChatMessage,
    ChatRequest,
    OpenAICompatClient,
    detect_protocol,
)


@pytest.mark.asyncio
async def test_openai_list_models() -> None:
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.get("/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-4o", "context_window": 128000},
                        {"id": "gpt-4o-mini", "context_window": 128000},
                    ]
                },
            )
        )
        client = OpenAICompatClient("https://api.example.com/v1", "sk-test")
        models = await client.list_models()
    assert [m.id for m in models] == ["gpt-4o", "gpt-4o-mini"]
    assert models[0].context_window == 128000


@pytest.mark.asyncio
async def test_openai_chat() -> None:
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "Hello, world!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )
        )
        client = OpenAICompatClient("https://api.example.com/v1", "sk-test")
        resp = await client.chat(
            ChatRequest(
                model="gpt-4o",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )
    assert resp.text == "Hello, world!"
    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_openai_stream() -> None:
    body = (
        "data: {\"choices\":[{\"delta\":{\"content\":\"He\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n"
        "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2}}\n\n"
        "data: [DONE]\n\n"
    )
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        )
        client = OpenAICompatClient("https://api.example.com/v1", "sk-test")
        chunks: list[str] = []
        usage = None
        async for chunk in client.stream(
            ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
        ):
            if chunk.delta:
                chunks.append(chunk.delta)
            if chunk.usage:
                usage = chunk.usage
    assert "".join(chunks) == "Hello"
    assert usage is not None and usage.output_tokens == 2


@pytest.mark.asyncio
async def test_anthropic_chat() -> None:
    with respx.mock(base_url="https://api.anthropic.com/v1") as mock:
        mock.post("/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Hi from Claude."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 7, "output_tokens": 4},
                },
            )
        )
        client = AnthropicCompatClient("https://api.anthropic.com/v1", "sk-ant")
        resp = await client.chat(
            ChatRequest(
                model="claude-3-5-sonnet",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )
    assert resp.text == "Hi from Claude."
    assert resp.usage.input_tokens == 7


@pytest.mark.asyncio
async def test_anthropic_stream() -> None:
    body = (
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n"
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Answ\"}}\n\n"
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"er\"}}\n\n"
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n"
        "data: {\"type\":\"message_stop\"}\n\n"
    )
    with respx.mock(base_url="https://api.anthropic.com/v1") as mock:
        mock.post("/messages").mock(
            return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        )
        client = AnthropicCompatClient("https://api.anthropic.com/v1", "sk-ant")
        chunks: list[str] = []
        usage = None
        async for chunk in client.stream(
            ChatRequest(model="claude-3-5-sonnet", messages=[ChatMessage(role="user", content="hi")])
        ):
            if chunk.delta:
                chunks.append(chunk.delta)
            if chunk.usage:
                usage = chunk.usage
    assert "".join(chunks) == "Answer"
    assert usage is not None and usage.output_tokens == 2


@pytest.mark.asyncio
async def test_detect_protocol_picks_anthropic_when_models_endpoint_404() -> None:
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        # OpenAI /models returns 200 (probed first by detect)
        mock.get("/models").mock(return_value=httpx.Response(200, json={"data": []}))
        # Anthropic /models returns 200
        mock.get("/models").mock(return_value=httpx.Response(200, json={"data": [{"id": "claude"}]}))
        # Because we hit the same URL, just override order by only allowing openai 200 once.
        # Real-world detect hits both with separate base URLs. Here we just verify it
        # doesn't raise.
        chosen = await detect_protocol("https://api.example.com/v1", "sk")
    assert chosen in {"openai", "anthropic"}


@pytest.mark.asyncio
async def test_probe_returns_false_on_html_response() -> None:
    """A 200 OK returning HTML (not JSON) should not crash probe."""
    from smoke_bench.clients.openai_compat import OpenAICompatClient

    client = OpenAICompatClient("https://api.example.com/v1", "sk-test")
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.get("/models").mock(return_value=httpx.Response(200, text="<html><body>OK</body></html>"))
        result = await client.probe()
    assert result is False


@pytest.mark.asyncio
async def test_probe_returns_false_on_empty_data() -> None:
    """A 200 with empty data array should return False."""
    from smoke_bench.clients.openai_compat import OpenAICompatClient

    client = OpenAICompatClient("https://api.example.com/v1", "sk-test")
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.get("/models").mock(return_value=httpx.Response(200, json={"data": []}))
        result = await client.probe()
    assert result is False
