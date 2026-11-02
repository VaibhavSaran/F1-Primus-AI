from dataclasses import FrozenInstanceError

import pytest

from security.access import AccessDenied, Principal, authorize, local_principal, oidc_principal


def test_identity_is_immutable_and_local_requires_explicit_trusted_configuration():
    with pytest.raises(AccessDenied):
        local_principal(enabled=False, loopback_only=True)
    with pytest.raises(AccessDenied):
        local_principal(enabled=True, loopback_only=False)
    principal = local_principal(enabled=True, loopback_only=True)
    with pytest.raises(FrozenInstanceError):
        principal.role = 'viewer'


def test_oidc_uses_only_injected_verified_claims():
    with pytest.raises(AccessDenied):
        oidc_principal('unverified', verifier=None)
    assert oidc_principal('opaque', verifier=lambda token: {'sub': 'u1', 'role': 'ai_user'}).identity == 'u1'
    for claims in ({'sub': 'u1'}, {'sub': '', 'role': 'operator'}, {'sub': 'u1', 'role': 'admin'}):
        with pytest.raises(AccessDenied):
            oidc_principal('opaque', verifier=lambda token: claims)
    with pytest.raises(AccessDenied, match='authentication_failed'):
        oidc_principal('opaque', verifier=lambda token: 1 / 0)


def test_roles_owner_boundaries_and_fail_closed():
    viewer = Principal('alice', 'viewer')
    ai = Principal('alice', 'ai_user')
    operator = Principal('ops', 'operator')
    authorize(viewer, 'read', resource_owner='alice')
    authorize(viewer, 'read', resource_owner='bob', is_public=True)
    for principal, action, owner in ((None, 'read', 'alice'), (viewer, 'read', 'bob'),
                                     (viewer, 'forecast', None), (ai, 'train', None),
                                     (ai, 'read', 'bob'), (operator, 'unknown', None)):
        with pytest.raises(AccessDenied):
            authorize(principal, action, resource_owner=owner)
    for action in ('forecast', 'chat', 'scenario'):
        authorize(ai, action)
    for action in ('raw_ops', 'train', 'promote', 'operate'):
        authorize(operator, action)


@pytest.mark.parametrize('identity,role', [('', 'viewer'), ('x', 'admin'), ('x\nsecret', 'viewer')])
def test_malformed_principals_rejected(identity, role):
    with pytest.raises(ValueError):
        Principal(identity, role)
