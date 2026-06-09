"""Tests for the Apple Foundation Models client facade (agent/apple_fm_client.py).

The ``fm serve`` subprocess is never spawned — an injected fake server with a
fake httpx client stands in for it.
"""

from __future__ import annotations

import json

import pytest

from agent.apple_fm_client import (
    AppleFMAPIError,
    AppleFMClient,
    _fm_model_name,
    _inject_tool_instructions,
    _sanitize_messages_for_fm,
    translate_apple_response,
)
from agent.error_classifier import FailoverReason, classify_api_error

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


class _FakeStream:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return ""

    def json(self):
        return {}

    @property
    def text(self):
        return ""

    def iter_text(self):
        yield from self._chunks


class FakeHTTP:
    def __init__(self, response=None, stream_chunks=None):
        self.response = response
        self.stream_chunks = stream_chunks or []
        self.last_post = None

    def post(self, path, json=None, timeout=None):
        self.last_post = {"path": path, "json": json}
        return self.response

    def stream(self, method, path, json=None, timeout=None):
        self.last_post = {"path": path, "json": json, "method": method}
        return _FakeStream(self.stream_chunks)


class FakeServer:
    def __init__(self, http=None, available=None, reasons=None):
        self.http = http or FakeHTTP()
        self._available = (
            available if available is not None else {"system": True, "pcc": True}
        )
        self._reasons = reasons or {}

    def ensure_started(self, timeout=None):
        return None

    def available_models(self, refresh=False):
        return dict(self._available)

    def unavailable_reason(self, model, refresh=False):
        return self._reasons.get(model, "")


def _make_client(http=None, available=None, reasons=None):
    server = FakeServer(http=http, available=available, reasons=reasons)
    return AppleFMClient(server=server), server


def _ok_payload(content="Hi there", model="system"):
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


# ── model id mapping ──────────────────────────────────────────────────────

def test_fm_model_name_strips_prefix():
    assert _fm_model_name("apple/system") == "system"
    assert _fm_model_name("apple/pcc") == "pcc"
    assert _fm_model_name("system") == "system"
    assert _fm_model_name(None) == "system"


# ── tools: strip / shim / native ──────────────────────────────────────────

def test_tools_never_sent_to_wire_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_APPLE_FM_NATIVE_TOOLS", raising=False)
    http = FakeHTTP(response=DummyResponse(payload=_ok_payload("ok")))
    client, _ = _make_client(http=http)
    client.chat.completions.create(
        model="apple/system",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    body = http.last_post["json"]
    assert "tools" not in body
    assert "tool_choice" not in body
    assert body["model"] == "system"


def test_tool_shim_injects_instructions(monkeypatch):
    monkeypatch.delenv("HERMES_APPLE_FM_NATIVE_TOOLS", raising=False)
    monkeypatch.setenv("HERMES_APPLE_FM_TOOL_SHIM", "1")
    http = FakeHTTP(response=DummyResponse(payload=_ok_payload("ok")))
    client, _ = _make_client(http=http)
    client.chat.completions.create(
        model="apple/system",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in Paris?"},
        ],
        tools=[WEATHER_TOOL],
    )
    body = http.last_post["json"]
    assert "tools" not in body
    injected = "\n".join(
        m["content"] for m in body["messages"] if m.get("role") == "system"
    )
    assert "<tool_call>" in injected
    assert "get_weather" in injected
    # Primary system prompt preserved at index 0.
    assert body["messages"][0]["content"] == "You are helpful."


def test_native_tools_passthrough_when_enabled(monkeypatch):
    monkeypatch.setenv("HERMES_APPLE_FM_NATIVE_TOOLS", "1")
    http = FakeHTTP(response=DummyResponse(payload=_ok_payload("ok")))
    client, _ = _make_client(http=http)
    client.chat.completions.create(
        model="apple/pcc",
        messages=[{"role": "user", "content": "hi"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    body = http.last_post["json"]
    assert body["tools"] == [WEATHER_TOOL]
    assert body["tool_choice"] == "auto"


# ── response translation ──────────────────────────────────────────────────

def test_non_streaming_response_normalized():
    http = FakeHTTP(response=DummyResponse(payload=_ok_payload("Hello!")))
    client, _ = _make_client(http=http)
    resp = client.chat.completions.create(
        model="apple/system", messages=[{"role": "user", "content": "hi"}]
    )
    msg = resp.choices[0].message
    assert msg.content == "Hello!"
    assert resp.choices[0].finish_reason == "stop"
    assert msg.tool_calls is None
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 8


def test_shim_parses_tool_calls_from_text(monkeypatch):
    monkeypatch.delenv("HERMES_APPLE_FM_NATIVE_TOOLS", raising=False)
    monkeypatch.setenv("HERMES_APPLE_FM_TOOL_SHIM", "1")
    block = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }
    content = f"Let me check.\n<tool_call>{json.dumps(block)}</tool_call>"
    http = FakeHTTP(response=DummyResponse(payload=_ok_payload(content)))
    client, _ = _make_client(http=http)
    resp = client.chat.completions.create(
        model="apple/system",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[WEATHER_TOOL],
    )
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "tool_calls"
    assert msg.tool_calls and msg.tool_calls[0].function.name == "get_weather"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert "<tool_call>" not in (msg.content or "")


def test_translate_apple_response_without_shim_keeps_text():
    payload = _ok_payload("<tool_call>{}</tool_call> not parsed")
    resp = translate_apple_response(payload, "system", shim_active=False)
    # No shim → the text is returned verbatim, no tool extraction.
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.tool_calls is None


# ── message sanitization ──────────────────────────────────────────────────

def test_sanitize_rewrites_tool_protocol_messages():
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        },
        {"role": "tool", "name": "read_file", "tool_call_id": "call_1", "content": "file body"},
    ]
    out = _sanitize_messages_for_fm(messages)
    assert all(m["role"] in {"user", "assistant", "system"} for m in out)
    assistant = out[1]
    assert assistant["role"] == "assistant"
    assert "<tool_call>" in assistant["content"] and "read_file" in assistant["content"]
    tool_result = out[2]
    assert tool_result["role"] == "user"
    assert "file body" in tool_result["content"]
    assert "read_file" in tool_result["content"]


# ── error mapping → classifier ────────────────────────────────────────────

def test_pcc_unavailable_precheck_raises_model_not_found():
    client, _ = _make_client(
        available={"system": True, "pcc": False},
        reasons={"pcc": "PCC inference is not available in this context."},
    )
    with pytest.raises(AppleFMAPIError) as ei:
        client.chat.completions.create(
            model="apple/pcc", messages=[{"role": "user", "content": "hi"}]
        )
    err = ei.value
    assert err.status_code == 503
    assert err.body["error"]["type"] == "service_unavailable"
    assert classify_api_error(err, provider="apple", model="apple/pcc").reason == (
        FailoverReason.model_not_found
    )


def test_unknown_model_400_maps_to_model_not_found():
    payload = {
        "error": {
            "type": "invalid_request_error",
            "message": "Unknown model 'apple/foo'. Available models: system, pcc",
            "code": "400",
        }
    }
    http = FakeHTTP(response=DummyResponse(status_code=400, payload=payload))
    client, _ = _make_client(http=http)
    with pytest.raises(AppleFMAPIError) as ei:
        client.chat.completions.create(
            model="apple/foo", messages=[{"role": "user", "content": "hi"}]
        )
    assert ei.value.status_code == 400
    assert classify_api_error(ei.value, provider="apple").reason == (
        FailoverReason.model_not_found
    )


def test_quota_error_maps_to_rate_limit():
    payload = {
        "error": {
            "type": "service_unavailable",
            "message": "PCC quota exhausted; resets daily",
            "code": "503",
        }
    }
    http = FakeHTTP(response=DummyResponse(status_code=503, payload=payload))
    client, _ = _make_client(http=http)  # pcc available so we POST and get quota err
    with pytest.raises(AppleFMAPIError) as ei:
        client.chat.completions.create(
            model="apple/pcc", messages=[{"role": "user", "content": "hi"}]
        )
    assert ei.value.status_code == 429
    assert classify_api_error(ei.value, provider="apple", model="apple/pcc").reason == (
        FailoverReason.rate_limit
    )


def test_context_overflow_message_rewritten_for_classifier():
    payload = {
        "error": {
            "type": "invalid_request_error",
            "message": "The input is too long for this session.",
            "code": "400",
        }
    }
    http = FakeHTTP(response=DummyResponse(status_code=400, payload=payload))
    client, _ = _make_client(http=http)
    with pytest.raises(AppleFMAPIError) as ei:
        client.chat.completions.create(
            model="apple/system", messages=[{"role": "user", "content": "x" * 100}]
        )
    assert "maximum context length is 8192" in str(ei.value)
    assert classify_api_error(
        ei.value, provider="apple", model="apple/system"
    ).reason == FailoverReason.context_overflow


# ── streaming (no-tools chat) ─────────────────────────────────────────────

def test_streaming_yields_content_and_finish():
    sse = [
        'data: {"id":"c1","choices":[{"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"c1","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        'data: {"id":"c1","choices":[{"delta":{"content":"lo"}}]}\n\n',
        'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n',
        "data: [DONE]\n\n",
    ]
    http = FakeHTTP(stream_chunks=sse)
    client, _ = _make_client(http=http)
    chunks = list(
        client.chat.completions.create(
            model="apple/system",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    )
    text = "".join(
        c.choices[0].delta.content
        for c in chunks
        if getattr(c.choices[0].delta, "content", None)
    )
    assert text == "Hello"
    assert any(c.choices[0].finish_reason == "stop" for c in chunks)


# ── client/server wiring ──────────────────────────────────────────────────

def test_create_openai_client_returns_apple_client(monkeypatch):
    from types import SimpleNamespace

    import agent.agent_runtime_helpers as helpers
    from agent.apple_fm_client import AppleFMClient as _Client

    agent = SimpleNamespace(
        provider="apple",
        _client_log_context=lambda: "ctx",
    )
    client = helpers.create_openai_client(
        agent,
        {"base_url": "applefm://local", "api_key": "apple-fm"},
        reason="test",
        shared=False,
    )
    assert isinstance(client, _Client)
    # Constructing it must not spawn fm serve.
    assert client.base_url == "applefm://local"
