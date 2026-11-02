"""Own-process page trimming does not alter the existing memory safety gate."""
from types import SimpleNamespace

import ctypes
import psutil

import ml.foundation_runtime as runtime


def test_non_windows_trim_is_explicit_noop(monkeypatch):
    monkeypatch.setattr(runtime.sys, 'platform', 'linux')
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=600_000_000))
    result = runtime.trim_cpu_working_set()
    assert result['supported'] is False
    assert result['trimmed'] is False
    assert result['after_available_bytes'] == 600_000_000


class NativeCall:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def test_windows_trim_uses_only_current_process_handle(monkeypatch):
    monkeypatch.setattr(runtime.sys, 'platform', 'win32')
    before_after = iter([600_000_000, 4_000_000_000])
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=next(before_after)))
    current = NativeCall(12345)
    empty = NativeCall(1)
    kernel = SimpleNamespace(GetCurrentProcess=current)
    psapi = SimpleNamespace(EmptyWorkingSet=empty)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda name, **kwargs: kernel if name == 'kernel32' else psapi,
                        raising=False)
    result = runtime.trim_cpu_working_set()
    assert current.calls == [()]
    assert empty.calls == [(12345,)]
    assert result['supported'] is True and result['trimmed'] is True
    assert result['before_available_bytes'] == 600_000_000
    assert result['after_available_bytes'] == 4_000_000_000


def test_native_trim_failure_is_reported_without_bypassing_memory_guard(monkeypatch):
    monkeypatch.setattr(runtime.sys, 'platform', 'win32')
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=200_000_000))
    def denied(*args, **kwargs):
        raise OSError('access denied')
    monkeypatch.setattr(ctypes, 'WinDLL', denied, raising=False)
    result = runtime.trim_cpu_working_set()
    assert result['trimmed'] is False
    assert result['after_available_bytes'] == 200_000_000
    assert 'access denied' in result['reason']
