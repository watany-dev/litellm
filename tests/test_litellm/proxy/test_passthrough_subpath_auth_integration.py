"""
Integration test for passthrough subpath auth fix.

Verifies that non-admin virtual keys can access sub-paths of registered
pass-through endpoints with include_subpath=True, using:
  1. Direct call to RouteChecks.non_proxy_admin_allowed_routes_check (the RBAC
     layer that was failing before the fix)
  2. Full HTTP requests via FastAPI TestClient through the real proxy stack

Bug scenario:
  - A pass-through endpoint is registered with include_subpath=True
  - A non-admin virtual key requests a dynamic sub-path like /myapi/sub1
  - Before the fix: is_llm_api_route() did NOT recognise the sub-path as a
    registered route, so non_proxy_admin_allowed_routes_check raised 403
  - After the fix: is_llm_api_route() calls is_registered_pass_through_route()
    early, so the sub-path is allowed
"""

import os
import sys
from unittest.mock import Mock, patch, MagicMock

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import litellm
import litellm.proxy.proxy_server
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.proxy_server import (
    hash_token,
    ProxyLogging,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    initialize_pass_through_endpoints,
    _registered_pass_through_routes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _mock_httpx_request(*args, **kwargs):
    """Return a fake 200 response so we never hit a real backend."""
    mock_response = httpx.Response(200, json={"ok": True, "echo": "mocked"})
    mock_response.request = Mock(spec=httpx.Request)
    return mock_response


def _make_mock_request(path: str) -> MagicMock:
    """Create a mock FastAPI Request with the given path."""
    mock_req = MagicMock(spec=Request)
    mock_req.query_params = {}
    mock_req.url = MagicMock()
    mock_req.url.path = path
    return mock_req


@pytest.fixture(autouse=True)
def _clean_proxy_state():
    """Reset proxy-level globals before/after each test."""
    orig_master_key = getattr(litellm.proxy.proxy_server, "master_key", None)
    orig_prisma = getattr(litellm.proxy.proxy_server, "prisma_client", None)
    orig_logging = getattr(litellm.proxy.proxy_server, "proxy_logging_obj", None)
    orig_general = getattr(litellm.proxy.proxy_server, "general_settings", None) or {}

    yield

    # Restore
    setattr(litellm.proxy.proxy_server, "master_key", orig_master_key)
    setattr(litellm.proxy.proxy_server, "prisma_client", orig_prisma)
    setattr(litellm.proxy.proxy_server, "proxy_logging_obj", orig_logging)
    setattr(litellm.proxy.proxy_server, "general_settings", orig_general)
    _registered_pass_through_routes.clear()


async def _register_passthrough_endpoint(
    path: str,
    target: str = "http://backend.example.com",
    include_subpath: bool = True,
):
    """Helper to register a pass-through endpoint with the proxy."""
    endpoints = [
        {
            "path": path,
            "target": target,
            "include_subpath": include_subpath,
            "headers": {},
        }
    ]
    # premium_user is imported inside initialize_pass_through_endpoints
    # from litellm.proxy.proxy_server; monkeypatch it there.
    with patch.object(
        litellm.proxy.proxy_server, "premium_user", True, create=True
    ):
        await initialize_pass_through_endpoints(endpoints)

    general_settings: dict = (
        getattr(litellm.proxy.proxy_server, "general_settings", {}) or {}
    )
    general_settings.update({"pass_through_endpoints": endpoints})
    setattr(litellm.proxy.proxy_server, "general_settings", general_settings)


# ---------------------------------------------------------------------------
# Test 1: Direct RBAC layer test (the core of the bug)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_llm_api_route_recognises_registered_subpath():
    """
    is_llm_api_route must return True for sub-paths of a registered
    pass-through endpoint (include_subpath=True).
    This is the exact check that was failing before the fix.
    """
    await _register_passthrough_endpoint("/myapi")

    # Exact path
    assert RouteChecks.is_llm_api_route(route="/myapi") is True

    # Sub-paths - these must be True after the fix
    assert RouteChecks.is_llm_api_route(route="/myapi/v1/resource") is True
    assert RouteChecks.is_llm_api_route(route="/myapi/v2/nested/deep") is True

    # Unrelated routes must NOT be recognised
    assert RouteChecks.is_llm_api_route(route="/totally-different") is False
    assert RouteChecks.is_llm_api_route(route="/myapi-imposter/resource") is False


@pytest.mark.asyncio
async def test_non_proxy_admin_allowed_routes_check_allows_subpath():
    """
    Directly invoke non_proxy_admin_allowed_routes_check with a non-admin
    token accessing a subpath. Before the fix this raised an Exception.
    """
    await _register_passthrough_endpoint("/myapi")

    user_obj = None  # non-admin, no user table entry
    valid_token = UserAPIKeyAuth(
        token="hashed-fake-token",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    mock_request = _make_mock_request("/myapi/v1/some-resource")

    # This should NOT raise after the fix
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER.value,
        route="/myapi/v1/some-resource",
        request=mock_request,
        valid_token=valid_token,
        request_data={},
    )


@pytest.mark.asyncio
async def test_non_proxy_admin_allowed_routes_check_blocks_unregistered():
    """
    Negative test: non_proxy_admin_allowed_routes_check still rejects
    routes that are NOT registered as pass-through endpoints.
    """
    await _register_passthrough_endpoint("/myapi")

    valid_token = UserAPIKeyAuth(
        token="hashed-fake-token",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    mock_request = _make_mock_request("/totally-different/resource")

    with pytest.raises(Exception) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/totally-different/resource",
            request=mock_request,
            valid_token=valid_token,
            request_data={},
        )

    assert "Only proxy admin" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2: Full HTTP integration via TestClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_testclient_subpath_request_succeeds(monkeypatch):
    """
    Full integration: register /myapi with include_subpath=True, set up auth,
    and fire actual HTTP requests through the proxy via TestClient.

    Uses the LIVE app object from litellm.proxy.proxy_server (not a stale
    module-level import) so that routes added by initialize_pass_through_endpoints
    are visible.
    """
    monkeypatch.setattr("httpx.AsyncClient.request", _mock_httpx_request)

    # Access the live module-level objects (survives conftest module reload)
    uakc = litellm.proxy.proxy_server.user_api_key_cache

    master_key = "sk-master-test-key"
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    setattr(litellm.proxy.proxy_server, "prisma_client", "FAKE-VAR")
    setattr(litellm.proxy.proxy_server, "user_api_key_cache", uakc)

    proxy_logging_obj = ProxyLogging(user_api_key_cache=uakc)
    proxy_logging_obj._init_litellm_callbacks()
    setattr(litellm.proxy.proxy_server, "proxy_logging_obj", proxy_logging_obj)

    await _register_passthrough_endpoint("/myapi")

    # Create a non-admin virtual key in the cache
    virtual_key = "sk-non-admin-user-key-12345"
    hashed = hash_token(virtual_key)
    cache_value = UserAPIKeyAuth(
        token=hashed,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    uakc.set_cache(key=hashed, value=cache_value)

    # Use the CURRENT app object from the module (not a stale import)
    current_app = litellm.proxy.proxy_server.app
    client = TestClient(current_app)

    # Exact path
    resp_exact = client.post(
        "/myapi",
        json={"test": "exact-path"},
        headers={"Authorization": f"Bearer {virtual_key}"},
    )
    print(f"[exact]  /myapi -> {resp_exact.status_code}  body={resp_exact.text[:200]}")

    # Sub-path (this was 403 before the fix)
    resp_sub = client.post(
        "/myapi/v1/some-resource",
        json={"test": "subpath"},
        headers={"Authorization": f"Bearer {virtual_key}"},
    )
    print(
        f"[sub]    /myapi/v1/some-resource -> {resp_sub.status_code}  body={resp_sub.text[:200]}"
    )

    # Deep sub-path
    resp_deep = client.get(
        "/myapi/v2/nested/deep/resource",
        headers={"Authorization": f"Bearer {virtual_key}"},
    )
    print(
        f"[deep]   /myapi/v2/nested/deep/resource -> {resp_deep.status_code}  body={resp_deep.text[:200]}"
    )

    # Key assertion: sub-paths must NOT return 401 or 403
    assert resp_exact.status_code == 200, (
        f"Exact path /myapi should be 200 but got {resp_exact.status_code}: {resp_exact.text}"
    )
    assert resp_sub.status_code == 200, (
        f"Sub-path /myapi/v1/some-resource should be 200 but got {resp_sub.status_code}: {resp_sub.text}"
    )
    assert resp_deep.status_code == 200, (
        f"Deep sub-path /myapi/v2/nested/deep/resource should be 200 but got {resp_deep.status_code}: {resp_deep.text}"
    )
