"""Trusted server identities. Never construct principals from browser fields/headers.

Python callers are part of the trusted application boundary. The OIDC verifier
must verify signature, issuer, audience, expiry and the configured role claim.
Local configuration must be operator-controlled and the server loopback-bound.
"""
from dataclasses import dataclass
from typing import Callable, Mapping, Optional


class AccessDenied(PermissionError):
    def __init__(self, code: str = 'forbidden'):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Principal:
    identity: str
    role: str
    source: str = 'local'

    def __post_init__(self):
        if (not isinstance(self.identity, str) or not self.identity.strip()
                or len(self.identity) > 256 or any(ord(c) < 32 for c in self.identity)
                or self.role not in ('viewer', 'ai_user', 'operator')
                or self.source not in ('local', 'oidc')):
            raise ValueError('invalid_principal')


def local_principal(*, enabled: bool, loopback_only: bool,
                    identity: str = 'local-operator', role: str = 'operator') -> Principal:
    if enabled is not True or loopback_only is not True:
        raise AccessDenied('local_mode_disabled')
    return Principal(identity, role, 'local')


def oidc_principal(token: str, *, verifier: Optional[Callable[[str], Mapping]]) -> Principal:
    if not callable(verifier) or not isinstance(token, str) or not token or len(token) > 16384:
        raise AccessDenied('authentication_required')
    try:
        claims = verifier(token)
        return Principal(claims['sub'], claims['role'], 'oidc')
    except Exception:
        raise AccessDenied('authentication_failed') from None


def authorize(principal: Optional[Principal], action: str, resource_owner: Optional[str] = None,
              is_public: bool = False) -> None:
    if not isinstance(principal, Principal):
        raise AccessDenied('authentication_required')
    if action == 'read':
        if principal.role == 'operator' or is_public is True or resource_owner == principal.identity:
            return
    elif action in ('forecast', 'chat', 'scenario'):
        if principal.role in ('ai_user', 'operator') and (
                resource_owner is None or resource_owner == principal.identity or principal.role == 'operator'):
            return
    elif action in ('raw_ops', 'train', 'promote', 'operate') and principal.role == 'operator':
        return
    raise AccessDenied('forbidden')
