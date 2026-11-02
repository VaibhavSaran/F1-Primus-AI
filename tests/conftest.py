"""Offline is the test-suite default; paid provider evaluations use explicit CLIs."""
import pytest


@pytest.fixture(autouse=True)
def disable_implicit_paid_provider(monkeypatch):
    monkeypatch.setenv('F1_PRIMUS_MODE','offline')
