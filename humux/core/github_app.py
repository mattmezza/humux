"""GitHub App authentication (issue #111): mint short-lived installation tokens.

A GitHub App authenticates by signing a short JWT with its RSA private key,
exchanging that JWT for an *installation access token* (valid ~1h), and using
that token as ``GH_TOKEN``.  The result: ``gh``/``git`` act as the App's bot
identity (``<app>[bot]``), with the App's own fine-grained permissions and
rate-limit pool instead of the owner's.

The private key never leaves this process; tokens are cached in memory and
refreshed shortly before expiry, never persisted.  A stolen token expires fast.
"""

from __future__ import annotations

import base64
import json
import logging
import time

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

log = logging.getLogger(__name__)

_API = "https://api.github.com"
_API_VERSION = "2022-11-28"
# (app_id, installation_id) -> (token, expires_epoch)
_cache: dict[tuple[str, str], tuple[str, float]] = {}
# Refresh this many seconds before GitHub's stated expiry (clock-skew cushion).
_REFRESH_SKEW = 300


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _app_jwt(app_id: str, private_key_pem: str, now: int) -> str:
    """RS256-signed JWT proving the App's identity (GitHub caps its life at 10m)."""
    header = {"alg": "RS256", "typ": "JWT"}
    # iat backdated 60s for clock drift; exp kept under GitHub's 10-minute ceiling.
    payload = {"iat": now - 60, "exp": now + 540, "iss": str(app_id)}
    signing_input = (
        f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}."
        f"{_b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    )
    key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    sig = key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_b64url(sig)}"


def _jwt_headers(jwt: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {jwt}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": _API_VERSION,
    }


def _token_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": _API_VERSION,
    }


def _mint(app_id: str, installation_id: str, private_key_pem: str, now: int) -> tuple[str, float]:
    """Exchange a fresh App JWT for an installation token. Returns (token, expiry_epoch)."""
    jwt = _app_jwt(app_id, private_key_pem, now)
    # Short timeout: this runs sync on the async loop (the injection point is
    # sync), so a hanging GitHub API must not stall the loop for long. Rare —
    # only on a cache miss/refresh, ~once/hour per installation.
    resp = httpx.post(
        f"{_API}/app/installations/{installation_id}/access_tokens",
        headers=_jwt_headers(jwt),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    # expires_at is ISO-8601 (e.g. "2026-07-01T12:00:00Z"), ~1h out.
    import datetime

    exp = datetime.datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")).timestamp()
    return data["token"], exp


def installation_token(app_id: str, installation_id: str, private_key_pem: str) -> str | None:
    """Cached installation token as ``GH_TOKEN``; refreshed ~5m before expiry.

    Returns ``None`` if minting fails and no still-valid token is cached — the
    caller then falls back to the PAT.  A blocking HTTP call happens at most once
    per hour per installation; every other call is an in-memory dict hit.
    ponytail: sync httpx (the injection point is sync) — one ~1s stall/hour is
    fine; make it async only if it ever shows up in a latency trace.
    """
    key = (str(app_id), str(installation_id))
    now = time.time()
    hit = _cache.get(key)
    if hit and hit[1] - _REFRESH_SKEW > now:
        return hit[0]
    try:
        token, exp = _mint(str(app_id), str(installation_id), private_key_pem, int(now))
    except Exception as exc:  # noqa: BLE001 — never let a mint failure crash a tool call
        log.warning("GitHub App token mint failed: %s", exc)
        return hit[0] if hit else None  # serve the (still-valid) cached token if we have one
    _cache[key] = (token, exp)
    return token


def test_installation(app_id: str, installation_id: str, private_key_pem: str) -> dict:
    """Verify App credentials for the admin "Test connection" button.

    Returns ``{"login": "<app>[bot]", "repos": [...]}``. Raises on any failure so
    the caller can surface the error verbatim.
    """
    now = int(time.time())
    jwt = _app_jwt(app_id, private_key_pem, now)
    with httpx.Client(timeout=15) as client:
        app_resp = client.get(f"{_API}/app", headers=_jwt_headers(jwt))
        app_resp.raise_for_status()
        slug = app_resp.json().get("slug", "")
        tok_resp = client.post(
            f"{_API}/app/installations/{installation_id}/access_tokens",
            headers=_jwt_headers(jwt),
        )
        tok_resp.raise_for_status()
        token = tok_resp.json()["token"]
        repos_resp = client.get(f"{_API}/installation/repositories", headers=_token_headers(token))
        repos_resp.raise_for_status()
        repos = [r["full_name"] for r in repos_resp.json().get("repositories", [])]
    return {"login": f"{slug}[bot]" if slug else "", "repos": repos}


if __name__ == "__main__":
    # ponytail: self-check the crypto (RS256 JWT) + the cache/refresh logic, the
    # only non-trivial parts. No network: _mint is stubbed.
    from cryptography.hazmat.primitives.asymmetric import rsa

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()

    # JWT is well-formed and its signature verifies against the public key.
    jwt = _app_jwt("42", pem, 1_000_000)
    h_b64, p_b64, s_b64 = jwt.split(".")
    payload = json.loads(base64.urlsafe_b64decode(p_b64 + "=="))
    assert payload["iss"] == "42" and payload["exp"] - payload["iat"] <= 600, payload
    pad = lambda s: s + "=" * (-len(s) % 4)  # noqa: E731
    sig = base64.urlsafe_b64decode(pad(s_b64))
    priv.public_key().verify(
        sig, f"{h_b64}.{p_b64}".encode(), padding.PKCS1v15(), hashes.SHA256()
    )  # raises if the signature is wrong

    # Cache: first call mints, second is a hit, expiry forces a re-mint.
    calls = {"n": 0}

    def fake_mint(app_id, inst, key, now):
        calls["n"] += 1
        return f"tok{calls['n']}", now + 3600  # expires in 1h

    real_mint, globals()["_mint"] = _mint, fake_mint
    try:
        _cache.clear()
        assert installation_token("42", "7", pem) == "tok1"
        assert installation_token("42", "7", pem) == "tok1"  # cached, no re-mint
        assert calls["n"] == 1
        # Force expiry → re-mint.
        _cache[("42", "7")] = ("stale", time.time() + 10)  # inside the refresh skew
        assert installation_token("42", "7", pem) == "tok2"
        assert calls["n"] == 2
    finally:
        globals()["_mint"] = real_mint
        _cache.clear()

    print("github_app.py self-check OK")
