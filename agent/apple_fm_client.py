"""OpenAI-compatible client facade for Apple Foundation Models (``fm serve``).

Hermes talks to Apple's on-device (``system``) and Private Cloud Compute
(``pcc``) models through the OpenAI-compatible server started by ``fm serve``
(managed by :mod:`agent.apple_fm_server`).  This module is a thin facade — it
exposes ``client.chat.completions.create(**kwargs)`` returning OpenAI-shaped
``SimpleNamespace`` objects, exactly like ``GeminiNativeClient`` and
``CopilotACPClient``, so the existing ``chat_completions`` transport and the
agent loop consume it unchanged.

Two Apple-specific quirks are handled here:

* **No native tool calling.** ``fm serve`` (macOS 27 beta) rejects a populated
  OpenAI ``tools`` array with HTTP 400.  We therefore (a) never put ``tools``
  on the wire by default, (b) implement a *prompt-injection shim* — inject the
  tool schema as a system message instructing the model to emit
  ``<tool_call>{...}</tool_call>`` blocks, then parse them back into
  ``message.tool_calls`` (reusing the parser from ``copilot_acp_client``).
  Set ``HERMES_APPLE_FM_NATIVE_TOOLS=1`` to forward ``tools`` verbatim instead
  (auto-works if/when Apple fixes the endpoint).

* **Classifiable errors.** :class:`AppleFMAPIError` carries ``.status_code``
  and ``.body`` so :mod:`agent.error_classifier` routes PCC-unavailable,
  quota, context-overflow, and bad-request failures to the right recovery
  path with no special-casing in the agent loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import httpx

from agent.apple_fm_server import AppleFMServer, get_shared_server
from agent.copilot_acp_client import (
    _extract_tool_calls_from_text,
    _render_message_content,
)

logger = logging.getLogger(__name__)

APPLE_FM_MARKER_BASE_URL = "applefm://local"
_CHAT_PATH = "/v1/chat/completions"

# Context windows (tokens) per WWDC26 — used only for the context-overflow
# error-message safety net (the authoritative copy lives in model_metadata).
_CONTEXT_WINDOW = {"system": 8192, "pcc": 32768}

_TOOL_SHIM_INSTRUCTIONS = (
    "You can call tools to help answer. To call a tool, output a line "
    "containing ONLY a tool-call block in this exact form:\n"
    '<tool_call>{"id": "call_1", "type": "function", "function": '
    '{"name": "<tool_name>", "arguments": "<json-encoded-args-string>"}}</tool_call>\n'
    "Rules: `arguments` MUST be a JSON-encoded STRING (e.g. \"{\\\"path\\\": "
    "\\\"a.txt\\\"}\"). You may emit multiple <tool_call> blocks. Use a unique "
    "id per call. If no tool is needed, just answer normally without a "
    "<tool_call> block.\n"
    "Available tools (OpenAI function schema):\n"
)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _native_tools_enabled() -> bool:
    return _env_truthy("HERMES_APPLE_FM_NATIVE_TOOLS", default=False)


def _tool_shim_enabled() -> bool:
    return _env_truthy("HERMES_APPLE_FM_TOOL_SHIM", default=True)


class AppleFMAPIError(Exception):
    """Error shape compatible with Hermes' error classifier / retry loop.

    ``status_code`` and ``body`` are read by ``agent.error_classifier`` to pick
    the recovery action (fallback / backoff / compress).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "apple_fm_error",
        status_code: Optional[int] = None,
        body: Optional[dict] = None,
        response: Optional[httpx.Response] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.body = body or {}
        self.response = response
        self.retry_after = retry_after


# ── model id ─────────────────────────────────────────────────────────────

def _fm_model_name(model: Optional[str]) -> str:
    """Map a Hermes model id (``apple/system``) to the ``fm serve`` name."""
    m = (model or "system").strip()
    if "/" in m:
        m = m.split("/", 1)[1]
    return m or "system"


# ── tool shim: message transformation ─────────────────────────────────────

def _tool_specs(tools: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        specs.append(
            {
                "name": name.strip(),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return specs


def _build_tool_instructions(
    tools: Optional[list[dict[str, Any]]], tool_choice: Any
) -> str:
    specs = _tool_specs(tools)
    if not specs:
        return ""
    text = _TOOL_SHIM_INSTRUCTIONS + json.dumps(specs, ensure_ascii=False)
    if isinstance(tool_choice, dict):
        fn = (tool_choice.get("function") or {}).get("name")
        if fn:
            text += f"\nYou MUST call the tool named '{fn}' on this turn."
    elif tool_choice == "required":
        text += "\nYou MUST call at least one tool on this turn."
    return text


def _sanitize_messages_for_fm(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rewrite OpenAI tool-protocol messages into plain text roles.

    ``fm serve`` accepts ``system``/``user``/``assistant`` text messages but
    rejects a populated ``tools`` array (HTTP 400) and is not known to accept
    ``role:"tool"`` messages or ``assistant.tool_calls``. To keep multi-turn
    tool conversations working through the prompt-injection shim we render:

    * ``assistant`` messages carrying ``tool_calls`` → assistant text with the
      original ``<tool_call>{...}</tool_call>`` blocks appended, so the model
      sees its own prior calls.
    * ``role:"tool"`` results → a ``user`` message labelled with the tool name.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "tool":
            name = msg.get("name") or msg.get("tool_name") or ""
            call_id = msg.get("tool_call_id") or ""
            label = f"tool '{name}'" if name else "tool"
            if call_id:
                label += f" (id {call_id})"
            out.append(
                {
                    "role": "user",
                    "content": f"[Result of {label}]:\n{_render_message_content(msg.get('content'))}",
                }
            )
            continue

        tool_calls = msg.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            rendered = _render_message_content(msg.get("content"))
            blocks: list[str] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    try:
                        args = json.dumps(args or {}, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args = "{}"
                block = {
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": (fn.get("name") or ""), "arguments": args},
                }
                blocks.append(f"<tool_call>{json.dumps(block, ensure_ascii=False)}</tool_call>")
            content = "\n".join(filter(None, [rendered, *blocks])) or "(called tools)"
            out.append({"role": "assistant", "content": content})
            continue

        # Plain message — coerce list/multimodal content to text so a small
        # on-device model isn't handed structures fm serve may not render.
        content = msg.get("content")
        if isinstance(content, list):
            content = _render_message_content(content)
        new_msg = {"role": role or "user", "content": content if content is not None else ""}
        out.append(new_msg)
    return out


def _inject_tool_instructions(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    tool_choice: Any,
) -> list[dict[str, Any]]:
    """Insert the tool-shim instructions as a system message."""
    instructions = _build_tool_instructions(tools, tool_choice)
    if not instructions:
        return messages
    sys_msg = {"role": "system", "content": instructions}
    # Insert right after a leading system message (preserve the primary
    # system prompt at index 0), else at the front.
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        return [messages[0], sys_msg, *messages[1:]]
    return [sys_msg, *messages]


# ── response translation ───────────────────────────────────────────────────

def _usage_ns(usage: Any) -> SimpleNamespace:
    u = usage if isinstance(usage, dict) else {}
    return SimpleNamespace(
        prompt_tokens=int(u.get("prompt_tokens") or 0),
        completion_tokens=int(u.get("completion_tokens") or 0),
        total_tokens=int(u.get("total_tokens") or 0),
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


def translate_apple_response(
    payload: dict[str, Any], model: str, *, shim_active: bool
) -> SimpleNamespace:
    """Translate an ``fm serve`` chat completion to an OpenAI-shaped object."""
    choices = payload.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    msg = choice.get("message") or {}
    content = msg.get("content")
    finish_reason = choice.get("finish_reason") or "stop"
    tool_calls = None

    if shim_active and isinstance(content, str) and content.strip():
        extracted, cleaned = _extract_tool_calls_from_text(content)
        if extracted:
            tool_calls = extracted
            content = cleaned or None
            finish_reason = "tool_calls"

    message_ns = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice_ns = SimpleNamespace(index=0, message=message_ns, finish_reason=finish_reason)
    return SimpleNamespace(
        id=payload.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=payload.get("created") or int(time.time()),
        model=model,
        choices=[choice_ns],
        usage=_usage_ns(payload.get("usage")),
    )


# ── streaming ──────────────────────────────────────────────────────────────

class _AppleStreamChunk(SimpleNamespace):
    pass


def _make_stream_chunk(
    *,
    model: str,
    content: str = "",
    role: bool = False,
    tool_call: Optional[dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
    usage: Any = None,
    chunk_id: str,
) -> _AppleStreamChunk:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = "assistant"
    if content:
        delta["content"] = content
    if tool_call is not None:
        delta["tool_calls"] = [
            SimpleNamespace(
                index=tool_call.get("index", 0),
                id=tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=SimpleNamespace(
                    name=tool_call.get("name") or "",
                    arguments=tool_call.get("arguments") or "",
                ),
            )
        ]
    choice = SimpleNamespace(
        index=0, delta=SimpleNamespace(**delta), finish_reason=finish_reason
    )
    chunk = _AppleStreamChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=None,
    )
    if usage is not None:
        chunk.usage = _usage_ns(usage)
    return chunk


# ── client facade ──────────────────────────────────────────────────────────

class _AppleFMChatCompletions:
    def __init__(self, client: "AppleFMClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _AppleFMChatNamespace:
    def __init__(self, client: "AppleFMClient"):
        self.completions = _AppleFMChatCompletions(client)


class AppleFMClient:
    """Minimal OpenAI-client-compatible facade over ``fm serve``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[dict[str, str]] = None,
        timeout: Any = None,
        server: Optional[AppleFMServer] = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or "apple-fm"
        self.base_url = base_url or APPLE_FM_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._default_timeout = timeout
        # Shared, process-global server (one warm `fm serve` for all clients).
        self._server = server or get_shared_server()
        self.chat = _AppleFMChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        # Do NOT stop the shared server here — other clients may still use it.
        # The server is torn down via its own atexit hook.
        self.is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ── request assembly ─────────────────────────────────────────────
    def _create_chat_completion(
        self,
        *,
        model: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        stream: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Any = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        self._server.ensure_started()
        fm_model = _fm_model_name(model)
        messages = list(messages or [])

        # Proactive PCC availability gate (avoid a wasted round-trip).
        if fm_model == "pcc":
            avail = self._server.available_models()
            if avail.get("pcc") is False:
                reason = self._server.unavailable_reason("pcc") or (
                    "PCC inference is not available in this context."
                )
                raise self._pcc_unavailable_error(reason)

        native = _native_tools_enabled()
        has_tools = bool(tools)
        shim_active = False

        if native:
            wire_messages = messages
        else:
            wire_messages = _sanitize_messages_for_fm(messages)
            if has_tools and _tool_shim_enabled():
                wire_messages = _inject_tool_instructions(
                    wire_messages, tools, tool_choice
                )
                shim_active = True

        body: dict[str, Any] = {
            "model": fm_model,
            "messages": wire_messages,
            "stream": bool(stream),
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        effective_max = max_tokens if max_tokens is not None else max_completion_tokens
        if effective_max is not None:
            body["max_tokens"] = effective_max
        if response_format is not None:
            body["response_format"] = response_format
        if native and has_tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice

        req_timeout = timeout if timeout is not None else self._default_timeout

        if stream and not shim_active:
            return self._stream_completion(body, fm_model, req_timeout)

        result = self._post_once(body, fm_model, shim_active, req_timeout)
        if stream:
            # Caller asked for a stream but the tool-shim needs the full text
            # to parse <tool_call> blocks — replay the buffered result as a
            # one-shot chunk sequence so the stream contract still holds.
            return iter(self._chunks_from_response(result, fm_model))
        return result

    def _post_once(
        self, body: dict[str, Any], fm_model: str, shim_active: bool, timeout: Any
    ) -> SimpleNamespace:
        try:
            resp = self._server.http.post(_CHAT_PATH, json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            raise AppleFMAPIError(
                f"Apple FM request failed: {exc}",
                code="apple_fm_transport_error",
            ) from exc
        if resp.status_code != 200:
            raise self._http_error(resp, fm_model)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise AppleFMAPIError(
                f"Invalid JSON from fm serve: {exc}",
                code="apple_fm_invalid_json",
                status_code=resp.status_code,
                response=resp,
            ) from exc
        return translate_apple_response(payload, fm_model, shim_active=shim_active)

    def _stream_completion(
        self, body: dict[str, Any], fm_model: str, timeout: Any
    ) -> Iterator[_AppleStreamChunk]:
        def _gen() -> Iterator[_AppleStreamChunk]:
            try:
                with self._server.http.stream(
                    "POST", _CHAT_PATH, json=body, timeout=timeout
                ) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        raise self._http_error(resp, fm_model)
                    buffer = ""
                    for text in resp.iter_text():
                        if not text:
                            continue
                        buffer += text
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.rstrip("\r")
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                return
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            for chunk in self._translate_stream_event(event, fm_model):
                                yield chunk
            except httpx.HTTPError as exc:
                raise AppleFMAPIError(
                    f"Apple FM streaming request failed: {exc}",
                    code="apple_fm_stream_error",
                ) from exc

        return _gen()

    @staticmethod
    def _translate_stream_event(
        event: dict[str, Any], model: str
    ) -> list[_AppleStreamChunk]:
        choices = event.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return []
        choice = choices[0]
        delta = choice.get("delta") or {}
        chunk_id = str(event.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}")
        out: list[_AppleStreamChunk] = []
        if delta.get("role"):
            out.append(_make_stream_chunk(model=model, role=True, chunk_id=chunk_id))
        text = delta.get("content")
        if isinstance(text, str) and text:
            out.append(_make_stream_chunk(model=model, content=text, chunk_id=chunk_id))
        finish = choice.get("finish_reason")
        if finish:
            out.append(
                _make_stream_chunk(
                    model=model,
                    finish_reason=finish,
                    usage=event.get("usage"),
                    chunk_id=chunk_id,
                )
            )
        return out

    @staticmethod
    def _chunks_from_response(
        result: SimpleNamespace, model: str
    ) -> list[_AppleStreamChunk]:
        chunk_id = getattr(result, "id", None) or f"chatcmpl-{uuid.uuid4().hex[:12]}"
        msg = result.choices[0].message
        chunks = [_make_stream_chunk(model=model, role=True, chunk_id=chunk_id)]
        if isinstance(getattr(msg, "content", None), str) and msg.content:
            chunks.append(
                _make_stream_chunk(model=model, content=msg.content, chunk_id=chunk_id)
            )
        for idx, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            chunks.append(
                _make_stream_chunk(
                    model=model,
                    tool_call={
                        "index": idx,
                        "id": getattr(tc, "id", None),
                        "name": getattr(tc.function, "name", ""),
                        "arguments": getattr(tc.function, "arguments", ""),
                    },
                    chunk_id=chunk_id,
                )
            )
        chunks.append(
            _make_stream_chunk(
                model=model,
                finish_reason=result.choices[0].finish_reason or "stop",
                usage={
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
                chunk_id=chunk_id,
            )
        )
        return chunks

    # ── errors ───────────────────────────────────────────────────────
    def _pcc_unavailable_error(self, reason: str) -> AppleFMAPIError:
        body = {
            "error": {
                "type": "service_unavailable",
                "message": f"Model 'pcc' is unavailable: {reason}",
                "code": "503",
            }
        }
        return AppleFMAPIError(
            "Apple Private Cloud Compute is unavailable in this context: "
            f"{reason} Use model 'apple/system' (on-device) or sign in to "
            "Apple Intelligence.",
            code="service_unavailable",
            status_code=503,
            body=body,
        )

    def _http_error(self, resp: httpx.Response, fm_model: str) -> AppleFMAPIError:
        status = resp.status_code
        body: dict[str, Any] = {}
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            body = {}
        err_obj = body.get("error")
        err = err_obj if isinstance(err_obj, dict) else {}
        raw_msg = str(err.get("message") or "").strip()
        if not raw_msg:
            try:
                raw_msg = (resp.text or "")[:300]
            except Exception:
                raw_msg = ""
        etype = str(err.get("type") or "").strip()
        low = raw_msg.lower()

        mapped_status = status
        code = etype or f"apple_fm_http_{status}"

        # PCC quota exhaustion → present as a transient rate limit so the loop
        # backs off / fails over rather than aborting.
        if "quota" in low or "usage limit" in low or "rate limit" in low:
            mapped_status = 429
            message = (
                f"Apple PCC quota/usage limit reached: {raw_msg} "
                "It resets periodically — use model 'apple/system' (on-device) "
                "or try again later."
            )
            return AppleFMAPIError(
                message, code=code, status_code=mapped_status, body=body, response=resp
            )

        # PCC unavailable in this context → message carries the phrase the
        # classifier maps to model_not_found (fail over, don't retry-loop).
        if "not available in this context" in low or (
            "pcc" in low and "unavailable" in low
        ):
            message = (
                f"Apple Private Cloud Compute is unavailable in this context: {raw_msg} "
                "Use model 'apple/system' (on-device) or sign in to Apple Intelligence."
            )
            return AppleFMAPIError(
                message, code=code, status_code=status, body=body, response=resp
            )

        # Context-window overflow safety net: ensure the message carries a
        # phrase the classifier recognizes as context_overflow so the agent
        # compresses + retries instead of aborting.
        ctx_like = any(
            kw in low for kw in ("context", "too long", "token", "exceed")
        )
        already_parseable = "maximum context" in low or "context length" in low
        if ctx_like and not already_parseable:
            ctx = _CONTEXT_WINDOW.get(fm_model, 8192)
            message = f"{raw_msg} (maximum context length is {ctx} tokens)"
            return AppleFMAPIError(
                message, code=code, status_code=status, body=body, response=resp
            )

        message = f"Apple FM HTTP {status}: {raw_msg}" if raw_msg else (
            f"Apple FM returned HTTP {status}"
        )
        return AppleFMAPIError(
            message, code=code, status_code=status, body=body, response=resp
        )
