"""Tests for the managed ``fm serve`` lifecycle (agent/apple_fm_server.py).

``fm`` is never actually executed — ``subprocess.Popen`` and the health HTTP
client are faked.
"""

from __future__ import annotations

import io

import pytest

import agent.apple_fm_server as srv
from agent.apple_fm_server import (
    AppleFMServer,
    AppleFMServerError,
    get_shared_server,
)

HEALTH = {
    "status": "fm serve is running",
    "models": [
        {"name": "system", "available": True},
        {
            "name": "pcc",
            "available": False,
            "reason": "PCC inference is not available in this context.",
        },
    ],
}


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeHealthHTTP:
    def __init__(self, payload):
        self._payload = payload
        self.closed = False

    def get(self, path, timeout=None):
        return DummyResponse(200, self._payload)

    def close(self):
        self.closed = True


class FakeProc:
    def __init__(self, alive=True):
        self._alive = alive
        self.returncode = None if alive else 1
        self.stdout = io.StringIO("")
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


class _FakeSubprocess:
    """Stand-in for the ``subprocess`` module used by apple_fm_server."""

    DEVNULL = -3
    PIPE = -1
    STDOUT = -2

    def __init__(self, popen_factory):
        self._popen_factory = popen_factory

    def Popen(self, *args, **kwargs):
        return self._popen_factory(*args, **kwargs)


@pytest.fixture(autouse=True)
def _socket_mode(monkeypatch):
    # Use socket mode so _serve_args() does not bind a real loopback port.
    monkeypatch.setenv("HERMES_APPLE_FM_SOCKET", "/tmp/test-fm-probe.sock")


def _patch_popen(monkeypatch, proc_factory):
    monkeypatch.setattr(srv, "subprocess", _FakeSubprocess(proc_factory))


def test_singleton_identity():
    assert get_shared_server() is get_shared_server()


def test_ensure_started_parses_health(monkeypatch):
    proc = FakeProc(alive=True)
    _patch_popen(monkeypatch, lambda *a, **k: proc)
    s = AppleFMServer()
    monkeypatch.setattr(s, "_build_http_client", lambda: FakeHealthHTTP(HEALTH))

    s.ensure_started(timeout=5)
    assert s.is_running()
    assert s.available_models() == {"system": True, "pcc": False}
    assert "not available" in s.unavailable_reason("pcc").lower()
    s.close()
    assert proc.terminated
    assert not s.is_running()


def test_startup_exit_raises(monkeypatch):
    proc = FakeProc(alive=False)  # process exits immediately
    _patch_popen(monkeypatch, lambda *a, **k: proc)
    s = AppleFMServer()
    monkeypatch.setattr(s, "_build_http_client", lambda: FakeHealthHTTP(HEALTH))
    with pytest.raises(AppleFMServerError):
        s.ensure_started(timeout=2)


def test_missing_fm_binary_raises_guidance(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("fm not found")

    _patch_popen(monkeypatch, _boom)
    s = AppleFMServer()
    with pytest.raises(AppleFMServerError) as ei:
        s.ensure_started(timeout=2)
    assert "fm" in str(ei.value).lower()


def test_close_is_idempotent(monkeypatch):
    proc = FakeProc(alive=True)
    _patch_popen(monkeypatch, lambda *a, **k: proc)
    s = AppleFMServer()
    monkeypatch.setattr(s, "_build_http_client", lambda: FakeHealthHTTP(HEALTH))
    s.ensure_started(timeout=5)
    s.close()
    s.close()  # second close must not raise
    assert proc.terminated
