"""GitHub App auth (#111): token minting/caching, repo-scope gate, admin wiring."""

from __future__ import annotations

import base64
import json
from typing import Any, cast

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core import github_app
from core.agent import _narrow_gh_repos
from core.agents import Agent
from core.config import Config
from core.config_store import ConfigStore
from core.tools import _gh_env, effective_tool_env, github_repo_violation

AUTH = {"Authorization": "Bearer secret"}


def _pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def test_gh_numeric_ids_coerced_to_str() -> None:
    # GitHub hands app_id / installation_id over as ints (via YAML or the
    # App-install API); Config must coerce them, not reject them (#111).
    cfg = Config.model_validate(
        {"tools": {"gh": {"app_id": 4191143, "installation_id": 143760902}}}
    )
    assert cfg.tools.gh.app_id == "4191143"
    assert cfg.tools.gh.installation_id == "143760902"


def test_app_jwt_is_valid_rs256() -> None:
    pem = _pem()
    pub = serialization.load_pem_private_key(pem.encode(), password=None).public_key()
    jwt = github_app._app_jwt("99", pem, 1_700_000_000)
    h, p, s = jwt.split(".")
    payload = json.loads(base64.urlsafe_b64decode(p + "=="))
    assert payload["iss"] == "99"
    assert payload["exp"] - payload["iat"] <= 600  # GitHub caps App JWT life at 10m
    sig = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    pub.verify(sig, f"{h}.{p}".encode(), padding.PKCS1v15(), hashes.SHA256())  # raises if bad


def test_installation_token_caches_and_refreshes(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_mint(app_id, inst, key, now):
        calls["n"] += 1
        return f"tok{calls['n']}", now + 3600

    monkeypatch.setattr(github_app, "_mint", fake_mint)
    github_app._cache.clear()
    assert github_app.installation_token("1", "2", "pem") == "tok1"
    assert github_app.installation_token("1", "2", "pem") == "tok1"  # cache hit
    assert calls["n"] == 1
    # Expiry inside the refresh skew forces a re-mint.
    github_app._cache[("1", "2")] = ("stale", github_app.time.time() + 10)
    assert github_app.installation_token("1", "2", "pem") == "tok2"
    assert calls["n"] == 2
    github_app._cache.clear()


def test_installation_token_serves_stale_on_mint_failure(monkeypatch) -> None:
    github_app._cache.clear()
    github_app._cache[("1", "2")] = ("cached", github_app.time.time() + 10)

    def boom(*_a):
        raise RuntimeError("network down")

    monkeypatch.setattr(github_app, "_mint", boom)
    assert github_app.installation_token("1", "2", "pem") == "cached"
    github_app._cache.clear()


def test_gh_env_prefers_app_then_falls_back_to_pat(monkeypatch) -> None:
    cfg = Config()
    cfg.tools.gh.enabled = True
    cfg.tools.gh.token = "pat-token"
    cfg.tools.gh.app_id = "1"
    cfg.tools.gh.installation_id = "2"
    cfg.tools.gh.private_key = "PEM"

    monkeypatch.setattr(github_app, "installation_token", lambda *_a: "app-token")
    assert _gh_env(cfg) == {"GH_TOKEN": "app-token"}  # App wins

    monkeypatch.setattr(github_app, "installation_token", lambda *_a: None)
    assert _gh_env(cfg) == {"GH_TOKEN": "pat-token"}  # mint failed → PAT fallback


def test_agent_without_pat_uses_app_bot_identity(monkeypatch) -> None:
    cfg = Config()
    cfg.tools.gh.enabled = True
    cfg.tools.gh.app_id = "1"
    cfg.tools.gh.installation_id = "2"
    cfg.tools.gh.private_key = "PEM"
    monkeypatch.setattr(github_app, "installation_token", lambda *_a: "bot-token")

    # gh enabled, no own vault token → borrows the App bot (not the owner).
    coder = Agent(name="coder", tool_config={"gh": {"enabled": True}})
    env = effective_tool_env(cfg, coder, lambda _n: None)
    assert env["GH_TOKEN"] == "bot-token"


def _app_cfg() -> Config:
    cfg = Config()
    cfg.tools.gh.enabled = True
    cfg.tools.gh.app_id = "42"
    cfg.tools.gh.installation_id = "7"
    cfg.tools.gh.private_key = "PEM"
    return cfg


def test_agent_own_app_beats_system_app(monkeypatch) -> None:
    # Tag the minted token by app_id so we can tell own-App from system-App.
    monkeypatch.setattr(github_app, "installation_token", lambda app_id, *_a: f"app:{app_id}")
    cfg = _app_cfg()
    vault = {"CODER_KEY": "coder-pem"}
    coder = Agent(
        name="coder",
        tool_config={
            "gh": {
                "enabled": True,
                "auth": "app",
                "app_id": "500",
                "installation_id": "9",
                "private_key_secret": "CODER_KEY",
            }
        },
    )
    # Agent's OWN app (500) is used, not the system app (42) — multiple apps.
    assert effective_tool_env(cfg, coder, vault.get)["GH_TOKEN"] == "app:500"
    # auth="app" but no own creds → falls back to the shared system app bot.
    shared = Agent(name="w", tool_config={"gh": {"enabled": True, "auth": "app"}})
    assert effective_tool_env(cfg, shared, vault.get)["GH_TOKEN"] == "app:42"


def test_agent_own_app_mint_failure_does_not_cross_to_system(monkeypatch) -> None:
    # Own-app mint fails transiently → NO token, never the system bot (no silent
    # escalation to a possibly-broader identity).
    monkeypatch.setattr(github_app, "installation_token", lambda *_a: None)
    cfg = _app_cfg()
    coder = Agent(
        name="coder",
        tool_config={
            "gh": {
                "enabled": True,
                "auth": "app",
                "app_id": "500",
                "installation_id": "9",
                "private_key_secret": "CODER_KEY",
            }
        },
    )
    assert "GH_TOKEN" not in effective_tool_env(cfg, coder, {"CODER_KEY": "pem"}.get)


def test_legacy_auth_own_app_mint_failure_does_not_cross_to_system(monkeypatch) -> None:
    # Same no-crossover rule for the legacy auth="" path (reachable via raw YAML):
    # own-app configured + mint fails → no token, never the system bot.
    monkeypatch.setattr(github_app, "installation_token", lambda *_a: None)
    cfg = _app_cfg()
    coder = Agent(
        name="coder",
        tool_config={
            "gh": {
                "enabled": True,
                "app_id": "500",
                "installation_id": "9",
                "private_key_secret": "CODER_KEY",
            }
        },
    )
    assert "GH_TOKEN" not in effective_tool_env(cfg, coder, {"CODER_KEY": "pem"}.get)


def test_agent_auth_pat_never_uses_app(monkeypatch) -> None:
    monkeypatch.setattr(github_app, "installation_token", lambda app_id, *_a: f"app:{app_id}")
    cfg = _app_cfg()
    vault = {"GH_TOKEN_p": "pat-tok"}
    # Explicit PAT mode: own PAT is used even though App fields linger.
    p = Agent(name="p", tool_config={"gh": {"enabled": True, "auth": "pat", "app_id": "500"}})
    assert effective_tool_env(cfg, p, vault.get)["GH_TOKEN"] == "pat-tok"
    # Explicit PAT with no token → nothing (never the App bot, never the owner).
    none = Agent(name="none", tool_config={"gh": {"enabled": True, "auth": "pat"}})
    assert "GH_TOKEN" not in effective_tool_env(cfg, none, lambda _n: None)


def test_system_auth_pat_disables_app(monkeypatch) -> None:
    monkeypatch.setattr(github_app, "installation_token", lambda app_id, *_a: f"app:{app_id}")
    cfg = _app_cfg()
    cfg.tools.gh.auth = "pat"
    cfg.tools.gh.token = "sys-pat"
    # _gh_env forces the PAT even though App fields are set.
    assert _gh_env(cfg) == {"GH_TOKEN": "sys-pat"}
    # And the App bot is not offered to agents.
    atlas = Agent(name="atlas", tool_config={"gh": {"enabled": True}})
    assert "GH_TOKEN" not in effective_tool_env(cfg, atlas, lambda _n: None)


def test_gh_env_auth_app_uses_app(monkeypatch) -> None:
    monkeypatch.setattr(github_app, "installation_token", lambda app_id, *_a: f"app:{app_id}")
    cfg = _app_cfg()
    cfg.tools.gh.auth = "app"
    cfg.tools.gh.token = "sys-pat"
    assert _gh_env(cfg) == {"GH_TOKEN": "app:42"}


def test_repo_gate_blocks_only_disallowed_repo_flags() -> None:
    coder = Agent(name="coder", tool_config={"gh": {"enabled": True, "repos": ["me/humux"]}})
    assert github_repo_violation(coder, "gh pr list --repo me/humux") is None
    assert github_repo_violation(coder, "gh pr view 1 --repo me/secret") == "me/secret"
    assert github_repo_violation(coder, "gh api user") is None  # no --repo → cannot tell → allow
    open_p = Agent(name="open", tool_config={"gh": {"enabled": True}})
    assert github_repo_violation(open_p, "gh pr view 1 --repo any/thing") is None


def test_repo_gate_is_quote_glue_tolerant_and_gh_scoped() -> None:
    coder = Agent(name="coder", tool_config={"gh": {"enabled": True, "repos": ["me/humux"]}})
    # Quoting / `=` / glued `-R` must not bypass the gate.
    assert github_repo_violation(coder, 'gh pr view 1 --repo "me/evil"') == "me/evil"
    assert github_repo_violation(coder, "gh pr view 1 --repo='me/evil'") == "me/evil"
    assert github_repo_violation(coder, "gh pr view 1 -Rme/evil") == "me/evil"
    assert github_repo_violation(coder, "gh pr view 1 --repository me/evil") == "me/evil"
    # Non-gh commands aren't repo-gated (no false-block on recursive flags).
    assert github_repo_violation(coder, "grep -R foo/bar .") is None


def test_repo_gate_empty_allowlist_blocks_everything() -> None:
    # A present-but-empty allowlist (disjoint-narrow result) = allow nothing.
    sub = Agent(name="sub", tool_config={"gh": {"enabled": True, "repos": []}})
    assert github_repo_violation(sub, "gh pr view 1 --repo me/humux") == "me/humux"
    # Absent repos key = unrestricted (distinct from empty list).
    plain = Agent(name="plain", tool_config={"gh": {"enabled": True}})
    assert github_repo_violation(plain, "gh pr view 1 --repo me/humux") is None


def test_narrow_gh_repos_never_widens() -> None:
    parent = Agent(name="p", tool_config={"gh": {"enabled": True, "repos": ["a/x", "a/y"]}})
    # Child asks for a repo the parent lacks → intersection drops it.
    child_tc = {"gh": {"enabled": True, "repos": ["a/y", "a/z"]}}
    assert _narrow_gh_repos(parent, child_tc)["gh"]["repos"] == ["a/y"]
    # Child unspecified → inherits the parent's restriction.
    assert _narrow_gh_repos(parent, {"gh": {"enabled": True}})["gh"]["repos"] == ["a/x", "a/y"]
    # Parent unrestricted → child keeps its own list (self-narrowing is fine).
    open_parent = Agent(name="o", tool_config={"gh": {"enabled": True}})
    assert _narrow_gh_repos(open_parent, child_tc)["gh"]["repos"] == ["a/y", "a/z"]


def test_narrow_gh_repos_disjoint_blocks_not_widens() -> None:
    # The escalation bug: disjoint child ∩ parent = [] must mean "block all",
    # NOT "unrestricted". Narrowing yields an empty list, and the gate blocks.
    parent = Agent(name="p", tool_config={"gh": {"enabled": True, "repos": ["me/a"]}})
    narrowed = _narrow_gh_repos(parent, {"gh": {"enabled": True, "repos": ["me/evil"]}})
    assert narrowed["gh"]["repos"] == []
    child = Agent(name="c", tool_config=narrowed)
    assert github_repo_violation(child, "gh issue create --repo other/xyz") == "other/xyz"


# -- Admin UI wiring --------------------------------------------------------


class _Store:
    def __init__(self, tmp_path):
        self._data = {
            "agent.agents_db_path": str(tmp_path / "agents.db"),
            "history.db_path": str(tmp_path / "history.db"),
            "memory.db_path": str(tmp_path / "memory.db"),
            "tools.gh.enabled": "true",
        }

    async def is_setup_complete(self) -> bool:
        return True

    async def get(self, key: str):
        if key == "admin.password_hash":
            return "hash"
        if key == "admin.password_salt":
            return "salt"
        return self._data.get(key)

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"

    async def get_all_redacted(self) -> dict:
        return {}


class _Permissions:
    rules: dict = {}


class _AgentStub:
    def __init__(self):
        self.config = Config()
        self.channels = {}
        self.scheduler = None
        self.permissions = _Permissions()


def _client(tmp_path):
    agent = _AgentStub()
    app, _ = create_admin_app(
        AgentState(agent=cast(Any, agent)), cast(ConfigStore, _Store(tmp_path))
    )
    return TestClient(app)


def test_tools_partial_shows_github_app_fields(tmp_path) -> None:
    client = _client(tmp_path)
    html = client.get("/partials/tools", headers=AUTH).text
    assert "GitHub App" in html
    assert "Installation ID" in html
    assert "github.com/settings/apps/new" in html


def test_gh_test_app_mode_requires_all_fields(tmp_path) -> None:
    client = _client(tmp_path)
    r = client.post("/tools/gh/test", headers=AUTH, json={"mode": "app", "app_id": "1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and "required" in body["error"].lower()
