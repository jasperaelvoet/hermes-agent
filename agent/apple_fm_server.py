"""Managed ``fm serve`` lifecycle for the Apple Foundation Models provider.

macOS 27 ships ``/usr/bin/fm`` (Apple Foundation Models CLI).  ``fm serve``
starts an OpenAI-compatible HTTP server exposing the on-device ``system``
model and the Private Cloud Compute ``pcc`` model:

    POST /v1/chat/completions   (streaming + non-streaming)
    GET  /v1/models
    GET  /health

This module owns a single, *process-global* ``fm serve`` subprocess shared
across every :class:`~agent.apple_fm_client.AppleFMClient` (primary,
auxiliary, and any client rebuilds) so the warm model process is started
once and reused — spawning one ``fm serve`` per client would load the model
N times.  The lifecycle mirrors ``agent.copilot_acp_client.CopilotACPClient``
(Popen + lock + ``close()``), and the subprocess inherits the real login
``HOME`` so PCC entitlements resolve.

Environment overrides:
    HERMES_APPLE_FM_COMMAND       path to the ``fm`` binary (default: fm)
    HERMES_APPLE_FM_SOCKET        Unix-domain-socket path (default: loopback TCP)
    HERMES_APPLE_FM_SERVE_TIMEOUT seconds to wait for /health (default: 30)
    HERMES_APPLE_FM_SERVE_ARGS    extra args appended to ``fm serve``
"""

from __future__ import annotations

import atexit
import logging
import os
import shlex
import socket
import subprocess
import threading
import time
from collections import deque
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_SERVE_TIMEOUT = 30.0
# Health endpoint advertises readiness with this status string.
_HEALTH_READY_MARKER = "running"


class AppleFMServerError(RuntimeError):
    """Raised when ``fm serve`` cannot be started or fails its health check."""


def _resolve_fm_command() -> str:
    """Resolve the ``fm`` binary path."""
    override = os.getenv("HERMES_APPLE_FM_COMMAND", "").strip()
    if override:
        return override
    # Apple ships the CLI at /usr/bin/fm on macOS 27+. Prefer the absolute
    # path so a stale PATH shadow can't hijack it, but fall back to PATH
    # lookup (e.g. a beta installed elsewhere).
    if os.path.exists("/usr/bin/fm"):
        return "/usr/bin/fm"
    return "fm"


def _resolve_serve_extra_args() -> list[str]:
    raw = os.getenv("HERMES_APPLE_FM_SERVE_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def _resolve_serve_timeout() -> float:
    raw = os.getenv("HERMES_APPLE_FM_SERVE_TIMEOUT", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return _DEFAULT_SERVE_TIMEOUT


def _build_subprocess_env() -> dict[str, str]:
    """Environment for the ``fm serve`` child.

    Inherits the parent environment but pins ``HOME`` to the profile home
    when one is configured (so the child sees the real login session that
    carries the Apple Intelligence / PCC entitlement). Mirrors
    ``copilot_acp_client._build_subprocess_env``.
    """
    env = os.environ.copy()
    try:
        from hermes_constants import get_subprocess_home

        profile_home = get_subprocess_home()
        if profile_home:
            env["HOME"] = profile_home
    except Exception:
        pass
    return env


def _free_loopback_port() -> int:
    """Reserve an ephemeral loopback port and return it.

    Standard bind-to-0 trick: a tiny race exists between releasing the
    socket and ``fm serve`` binding it, but the kernel does not immediately
    recycle the port, so in practice it is free when the child binds.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


class AppleFMServer:
    """Manages one ``fm serve`` subprocess and HTTP access to it."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._http: Optional[httpx.Client] = None
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._health: Optional[dict[str, Any]] = None
        self._atexit_registered = False

        self.command = _resolve_fm_command()
        self.socket_path: Optional[str] = (
            os.getenv("HERMES_APPLE_FM_SOCKET", "").strip() or None
        )
        # Populated when started in TCP mode.
        self.port: Optional[int] = None
        # Base URL used by the httpx client. For UDS the host portion is a
        # placeholder; routing happens through the socket transport.
        self.base_url: str = ""

    # ── lifecycle ────────────────────────────────────────────────────
    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def ensure_started(self, timeout: Optional[float] = None) -> None:
        """Start ``fm serve`` (if not already running) and wait for health."""
        if self.is_running() and self._http is not None:
            return
        with self._lock:
            if self.is_running() and self._http is not None:
                return
            self._start_locked(timeout if timeout is not None else _resolve_serve_timeout())

    def _serve_args(self) -> list[str]:
        if self.socket_path:
            args = ["serve", "--socket", self.socket_path]
        else:
            self.port = _free_loopback_port()
            args = ["serve", "--host", "127.0.0.1", "--port", str(self.port)]
        return args + _resolve_serve_extra_args()

    def _build_http_client(self) -> httpx.Client:
        timeout = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=30.0)
        if self.socket_path:
            transport = httpx.HTTPTransport(uds=self.socket_path)
            return httpx.Client(
                transport=transport, base_url="http://fm.local", timeout=timeout
            )
        self.base_url = f"http://127.0.0.1:{self.port}"
        return httpx.Client(base_url=self.base_url, timeout=timeout)

    def _start_locked(self, timeout: float) -> None:
        # Clean up any dead remnants first.
        self._teardown_locked()

        args = self._serve_args()
        try:
            proc = subprocess.Popen(
                [self.command, *args],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise AppleFMServerError(
                f"Could not start Apple Foundation Models CLI ('{self.command} serve'). "
                "This provider requires macOS 26+/27 with Apple Intelligence and the "
                "`fm` CLI on PATH. Verify with `fm available`, or set "
                "HERMES_APPLE_FM_COMMAND to the binary path."
            ) from exc

        self._proc = proc
        # Drain stdout/stderr (merged) into a bounded tail so a full pipe can
        # never deadlock the child, and we can surface diagnostics on failure.
        threading.Thread(
            target=self._drain_output, args=(proc,), daemon=True
        ).start()

        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

        self._http = self._build_http_client()

        # Poll /health until ready or the process dies / we time out.
        deadline = time.monotonic() + timeout
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = "\n".join(self._stderr_tail).strip()
                raise AppleFMServerError(
                    f"`{self.command} serve` exited during startup "
                    f"(code {proc.returncode}).\n{tail}".strip()
                )
            try:
                resp = self._http.get("/health", timeout=2.0)
                if resp.status_code == 200:
                    body = resp.json()
                    status = str(body.get("status") or "")
                    if _HEALTH_READY_MARKER in status.lower():
                        self._health = body
                        logger.info(
                            "Apple FM `fm serve` ready (%s)",
                            self.socket_path or self.base_url,
                        )
                        return
            except Exception as exc:  # not yet listening — keep polling
                last_err = exc
            time.sleep(0.15)

        # Timed out.
        tail = "\n".join(self._stderr_tail).strip()
        self._teardown_locked()
        raise AppleFMServerError(
            f"`{self.command} serve` did not become healthy within {timeout:.0f}s. "
            f"Last error: {last_err}. Output:\n{tail}".strip()
        )

    def _drain_output(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in stream:
                self._stderr_tail.append(line.rstrip("\n"))
        except Exception:
            pass

    # ── access ───────────────────────────────────────────────────────
    @property
    def http(self) -> httpx.Client:
        self.ensure_started()
        assert self._http is not None
        return self._http

    def health(self, refresh: bool = False) -> dict[str, Any]:
        self.ensure_started()
        if refresh or self._health is None:
            try:
                resp = self._http.get("/health", timeout=5.0)  # type: ignore[union-attr]
                if resp.status_code == 200:
                    self._health = resp.json()
            except Exception as exc:
                logger.debug("Apple FM /health refresh failed: %s", exc)
        return self._health or {}

    def available_models(self, refresh: bool = False) -> dict[str, bool]:
        """Return ``{model_name: available}`` from /health."""
        models = self.health(refresh=refresh).get("models")
        result: dict[str, bool] = {}
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    result[entry["name"]] = bool(entry.get("available"))
        return result

    def unavailable_reason(self, model: str, refresh: bool = False) -> str:
        """Return the /health ``reason`` for an unavailable model, if any."""
        models = self.health(refresh=refresh).get("models")
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict) and entry.get("name") == model:
                    return str(entry.get("reason") or "")
        return ""

    # ── teardown ───────────────────────────────────────────────────────
    def _teardown_locked(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            try:
                http.close()
            except Exception:
                pass
        proc, self._proc = self._proc, None
        self._health = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def close(self) -> None:
        with self._lock:
            self._teardown_locked()


# ── process-global singleton ─────────────────────────────────────────────

_SERVER_SINGLETON: Optional[AppleFMServer] = None
_SINGLETON_LOCK = threading.Lock()


def get_shared_server() -> AppleFMServer:
    """Return the process-global :class:`AppleFMServer` (created on first use)."""
    global _SERVER_SINGLETON
    if _SERVER_SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SERVER_SINGLETON is None:
                _SERVER_SINGLETON = AppleFMServer()
    return _SERVER_SINGLETON
