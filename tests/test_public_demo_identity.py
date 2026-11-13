"""Public visitors must never inherit the owner's conversation identity."""
from types import SimpleNamespace
import sys

from streamlit_app import services


def test_public_demo_identity_is_session_scoped_and_not_operator(monkeypatch):
    monkeypatch.setenv('F1_PUBLIC_DEMO', '1')
    state = {}
    fake = SimpleNamespace(session_state=state, get_option=lambda _: '127.0.0.1')
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    alice = services.local_identity()
    assert alice.identity.startswith('visitor-')
    assert alice.role == 'ai_user'
    assert services.local_identity() == alice
    state.clear()
    bob = services.local_identity()
    assert bob.identity != alice.identity
    assert bob.identity != 'local'


def test_public_chat_policy_has_tenfold_request_allowances(monkeypatch,tmp_path):
    from agents.claude_provider import ClaudeProvider
    monkeypatch.setenv('F1_PUBLIC_DEMO','1')
    monkeypatch.setattr(ClaudeProvider,'from_env',classmethod(lambda cls:SimpleNamespace()))
    monkeypatch.setattr(services,'LOCAL',tmp_path)
    monkeypatch.setattr(services,'archive_context',lambda:None)
    policy=services.chat_service().limiter.policy
    assert (policy.per_minute,policy.per_day,policy.global_active)==(20,200,20)
    assert policy.searches==2
