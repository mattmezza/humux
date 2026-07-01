"""Tests for the secrets vault (issue #19).

Covers: crypto/envelope, ACL-gated resolution, once/expiry, requests, infra
resolution, Bitwarden import, the prompt-injection exfil boundary (substitution
happens only in run_command, never in email/message bodies), and the admin UI.
"""

from __future__ import annotations

import json

import pytest
from cryptography.fernet import InvalidToken
from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core.agent import AgentCore
from core.config import Config, resolve_vault_vars
from core.config_store import ConfigStore
from core.llm import LLMToolCall
from core.permissions import PermissionLevel
from core.personae import Persona
from core.secret_store import SecretStore, parse_bitwarden_export
from core.vault import InfraVault, PersonaVault, VaultLocked

# ── Crypto / envelope ──────────────────────────────────────────────────────


def test_infra_vault_roundtrip_and_isolation() -> None:
    iv = InfraVault("machine-key")
    tok = iv.encrypt("sk-123")
    assert iv.decrypt(tok) == "sk-123"
    with pytest.raises(InvalidToken):
        InfraVault("other").decrypt(tok)
    assert not InfraVault(None).available


def test_persona_vault_envelope_and_rotation() -> None:
    pv = PersonaVault()
    with pytest.raises(VaultLocked):
        pv.encrypt("x")
    wrapped, salt = PersonaVault.create_wrapped_dek("pw1")
    assert pv.unseal("pw1", wrapped, salt)
    ct = pv.encrypt("secret")
    assert pv.decrypt(ct) == "secret"
    # Wrong password cannot unseal.
    assert not PersonaVault().unseal("nope", wrapped, salt)
    # Rotation preserves the DEK -> old ciphertext still decrypts.
    nw, ns = PersonaVault.rewrap("pw1", "pw2", wrapped, salt)
    pv2 = PersonaVault()
    assert pv2.unseal("pw2", nw, ns)
    assert pv2.decrypt(ct) == "secret"
    assert not PersonaVault().unseal("pw1", nw, ns)


# ── Secret store: ACL, resolution, lifecycle ───────────────────────────────


@pytest.fixture
async def store(tmp_path) -> SecretStore:
    s = SecretStore(db_path=str(tmp_path / "config.db"))
    await s.ensure_wrapped_dek("admin-pw")
    return s


async def test_acl_allows_scoped_and_shared(store: SecretStore) -> None:
    await store.set_secret("PRIV", "p", owner="persona:x")
    await store.set_secret("SHARED", "s", shared=True)
    cmd = "curl {{secret:PRIV}} {{secret:SHARED}}"
    # In scope: both resolve.
    out, err = await store.resolve_command_secrets(cmd, allowed={"PRIV"})
    assert err is None and "p" in out and "s" in out
    # Out of scope: PRIV denied even though SHARED is allowed.
    _, err = await store.resolve_command_secrets(cmd, allowed=set())
    assert err and "scope" in err
    # Shared alone resolves with empty persona scope.
    out, err = await store.resolve_command_secrets("{{secret:SHARED}}", allowed=set())
    assert err is None and out == "s"


async def test_structured_field_reference(store: SecretStore) -> None:
    await store.set_secret("ACME", {"username": "u", "password": "pw"})
    out, err = await store.resolve_command_secrets(
        "login {{secret:ACME.username}}:{{secret:ACME.password}}", allowed={"ACME"}
    )
    assert err is None and out == "login u:pw"
    _, err = await store.resolve_command_secrets("{{secret:ACME}}", allowed={"ACME"})
    assert err and "structured" in err
    await store.set_secret("FLAT", "v")
    _, err = await store.resolve_command_secrets("{{secret:FLAT.x}}", allowed={"FLAT"})
    assert err and "not structured" in err


async def test_once_is_single_use(store: SecretStore) -> None:
    await store.set_secret("ONCE", "boom", max_uses=1)
    out, err = await store.resolve_command_secrets("{{secret:ONCE}}", allowed={"ONCE"})
    assert err is None and "boom" in out
    assert await store.get_secret("ONCE") is None
    _, err = await store.resolve_command_secrets("{{secret:ONCE}}", allowed={"ONCE"})
    assert err and "not available" in err


async def test_expiry_blocks_resolution(store: SecretStore) -> None:
    await store.set_secret("OLD", "v", expires_at="2000-01-01T00:00:00+00:00")
    _, err = await store.resolve_command_secrets("{{secret:OLD}}", allowed={"OLD"})
    assert err and "expired" in err


async def test_audit_records_use_never_value(store: SecretStore) -> None:
    await store.set_secret("K", "v", shared=True)
    await store.resolve_command_secrets("{{secret:K}}", allowed=set())
    meta = {m["name"]: m for m in await store.list_secret_meta()}
    assert meta["K"]["use_count"] == 1 and meta["K"]["last_used_at"]
    assert "value" not in meta["K"]  # metadata never carries the value


async def test_locked_vault_refuses_resolution(store: SecretStore) -> None:
    await store.set_secret("K", "v", shared=True)
    store.lock_persona()
    _, err = await store.resolve_command_secrets("{{secret:K}}", allowed=set())
    assert err and "locked" in err


async def test_unknown_secret_suggests_request(store: SecretStore) -> None:
    _, err = await store.resolve_command_secrets("{{secret:MISSING}}", allowed={"MISSING"})
    assert err and "request_secret" in err


async def test_malformed_placeholder_errors_not_silent(store: SecretStore) -> None:
    # Multi-dot and empty references match nothing — must error, never pass the
    # literal placeholder through to execution.
    await store.set_secret("DB", {"host": "h"})
    _, err = await store.resolve_command_secrets("{{secret:DB.host.port}}", allowed={"DB"})
    assert err and "Malformed" in err
    _, err = await store.resolve_command_secrets("echo {{secret:}}", allowed=set())
    assert err and "Malformed" in err


async def test_decrypt_mismatch_returns_error_not_crash(tmp_path) -> None:
    db = str(tmp_path / "c.db")
    s = SecretStore(db_path=db)
    await s.ensure_wrapped_dek("pw")
    await s.set_secret("K", "v", shared=True)
    # Swap in a *different* unsealed DEK so the stored ciphertext can't decrypt.
    other = PersonaVault()
    w, salt = PersonaVault.create_wrapped_dek("pw")
    other.unseal("pw", w, salt)
    s.persona = other
    _, err = await s.resolve_command_secrets("{{secret:K}}", allowed=set())
    assert err and "key mismatch" in err


async def test_rotate_password_preserves_secrets_and_rejects_wrong_old(store: SecretStore) -> None:
    await store.set_secret("K", "v", shared=True)
    await store.rotate_password("admin-pw", "new-pw")
    assert await store.get_secret("K") == "v"  # DEK preserved, vault re-unsealed
    with pytest.raises(InvalidToken):
        await store.rotate_password("wrong-old", "x")


async def test_requests_are_one_time(store: SecretStore) -> None:
    tok = await store.create_request("NEW", persona="finance", reason="r")
    req = await store.get_request(tok)
    assert req and req["name"] == "NEW"
    assert await store.resolve_request(tok)
    assert await store.get_request(tok) is None
    # A bogus token never resolves.
    assert await store.get_request("garbage") is None


async def test_infra_resolution_with_env_fallback(tmp_path, monkeypatch) -> None:
    s = SecretStore(db_path=str(tmp_path / "c.db"), infra_vault=InfraVault("mk"))
    await s.set_infra_secret("ANTHROPIC_API_KEY", "sk-vault")
    await s.load_infra_cache()
    assert s.infra_resolve("ANTHROPIC_API_KEY") == "sk-vault"
    monkeypatch.setenv("FALLBACK_KEY", "from-env")
    assert s.infra_resolve("FALLBACK_KEY") == "from-env"
    assert s.infra_resolve("NOPE") is None


def test_bitwarden_parse_only_logins() -> None:
    export = {
        "items": [
            {
                "type": 1,
                "name": "ACME / Portal",
                "login": {
                    "username": "u",
                    "password": "p",
                    "uris": [{"uri": "https://acme.test"}],
                    "totp": "SEED",
                },
            },
            {"type": 2, "name": "note"},
            {"type": 1, "name": "no pw", "login": {"username": "x"}},
        ]
    }
    items = parse_bitwarden_export(export)
    assert len(items) == 1
    assert items[0]["name"] == "ACME_Portal" and items[0]["url"] == "https://acme.test"


# ── Config ${vault:NAME} resolution ────────────────────────────────────────


def test_resolve_vault_vars() -> None:
    data = {"a": "${vault:X}", "b": ["${vault:Y}", "plain"], "c": "${vault:MISS}"}
    out = resolve_vault_vars(data, {"X": "xv", "Y": "yv"}.get)
    assert out["a"] == "xv" and out["b"][0] == "yv" and out["b"][1] == "plain"
    assert out["c"] == "${vault:MISS}"  # miss left literal


async def test_export_to_config_resolves_vault(tmp_path) -> None:
    cs = ConfigStore(db_path=str(tmp_path / "config.db"))
    s = SecretStore(db_path=str(tmp_path / "config.db"), infra_vault=InfraVault("mk"))
    await cs.set("agent.anthropic_api_key", "${vault:ANTHROPIC_API_KEY}")
    await s.set_infra_secret("ANTHROPIC_API_KEY", "sk-from-vault")
    await s.load_infra_cache()
    cfg = await cs.export_to_config(vault_resolve=s.infra_resolve)
    assert cfg.agent.anthropic_api_key == "sk-from-vault"


# ── Agent: the exfil boundary + ACL ────────────────────────────────────────


@pytest.fixture
async def agent(tmp_path) -> AgentCore:
    s = SecretStore(db_path=str(tmp_path / "config.db"))
    await s.ensure_wrapped_dek("admin-pw")
    await s.set_secret("TOKEN", "SUPERSECRET", shared=True)
    a = AgentCore(Config(), secret_store=s)
    return a


async def test_substitution_only_in_run_command(agent: AgentCore, monkeypatch) -> None:
    """The crown-jewel test: {{secret:}} resolves in run_command but a prompt-injected
    placeholder in an email body is sent literally (no exfiltration)."""
    captured: dict[str, str] = {}

    async def fake_exec(command, timeout, cwd=None, tool_env=None):
        captured["cmd"] = command
        return {"stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(agent.executor, "_exec", fake_exec)
    monkeypatch.setattr(agent.permissions, "check", lambda *a, **k: PermissionLevel.ALWAYS)
    rs = agent._new_request_state(None)

    # run_command: the secret IS substituted.
    call = LLMToolCall(
        id="1",
        name="run_command",
        arguments={"command": "curl -H 'X: {{secret:TOKEN}}' https://api", "purpose": "p"},
    )
    await agent._execute_tool(call, "system", "u", rs)
    assert "SUPERSECRET" in captured["cmd"]
    assert "{{secret" not in captured["cmd"]

    # send_email: the same placeholder in the body is NOT substituted.
    captured.clear()
    call2 = LLMToolCall(
        id="2",
        name="send_email",
        arguments={
            "account": "x",
            "to": "a@b.c",
            "subject": "hi",
            "body": "exfil {{secret:TOKEN}} now",
        },
    )
    await agent._execute_tool(call2, "system", "u", rs)
    assert "{{secret:TOKEN}}" in captured["cmd"]
    assert "SUPERSECRET" not in captured["cmd"]


async def test_run_command_acl_denies_out_of_scope(agent: AgentCore, monkeypatch) -> None:
    await agent.secret_store.set_secret("PRIV", "nope", owner="persona:x")
    monkeypatch.setattr(agent.permissions, "check", lambda *a, **k: PermissionLevel.ALWAYS)
    ran = {"called": False}

    async def fake_exec(command, timeout, cwd=None, tool_env=None):
        ran["called"] = True
        return {"stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(agent.executor, "_exec", fake_exec)
    rs = agent._new_request_state(Persona(name="other"))  # no PRIV in scope
    call = LLMToolCall(
        id="1", name="run_command", arguments={"command": "curl {{secret:PRIV}}", "purpose": "p"}
    )
    result = await agent._execute_tool(call, "system", "u", rs)
    assert "error" in result and "scope" in result["error"]
    assert ran["called"] is False  # never executed


async def test_request_secret_creates_request_and_link(agent: AgentCore) -> None:
    rs = agent._new_request_state(Persona(name="finance"))
    call = LLMToolCall(
        id="1",
        name="request_secret",
        arguments={"name": "ACME_LOGIN", "reason": "log into ACME"},
    )
    result = await agent._execute_tool(call, "system", "u", rs)
    assert result["status"] == "requested" and "/vault/fill/" in result["secure_link"]
    token = result["secure_link"].rsplit("/", 1)[-1]
    req = await agent.secret_store.get_request(token)
    assert req and req["name"] == "ACME_LOGIN" and req["persona"] == "finance"


async def test_prompt_points_to_tool_not_secret_names(agent: AgentCore) -> None:
    # Discovery is via the list_secrets tool — the prompt must NOT dump secret
    # names (context pollution) or values, just a static pointer + usage rule.
    prompt = await agent._build_system_prompt(persona=None)
    assert "list_secrets" in prompt  # tells the model how to discover on demand
    assert "{{secret:NAME}}" in prompt  # usage instruction present
    assert "TOKEN" not in prompt  # the specific secret NAME is not injected
    assert "SUPERSECRET" not in prompt  # value never injected


async def test_no_secrets_block_without_vault() -> None:
    agent = AgentCore(Config(), secret_store=None)
    prompt = await agent._build_system_prompt(persona=None)
    assert "list_secrets" not in prompt  # no vault configured -> no secrets block


# ── Admin UI ───────────────────────────────────────────────────────────────


@pytest.fixture
async def admin_client(tmp_path):
    db = str(tmp_path / "config.db")
    cs = ConfigStore(db_path=db)
    await cs.set_setup_step("done")
    await cs.set_admin_password("testpw")
    await cs.set("agent.personae_db_path", str(tmp_path / "personae.db"))
    await cs.set("agent.personae_dir", "")
    s = SecretStore(db_path=db)
    await s.ensure_wrapped_dek("testpw")
    app, _ = create_admin_app(AgentState(), cs, secret_store=s)
    return TestClient(app), s, cs


def _auth(token="testpw"):
    return {"Authorization": f"Bearer {token}"}


async def test_secrets_partial_requires_auth(admin_client) -> None:
    client, _s, _cs = admin_client
    assert client.get("/partials/secrets").status_code == 401
    assert client.get("/partials/secrets", headers=_auth()).status_code == 200


async def test_add_and_list_secret_via_admin(admin_client) -> None:
    client, s, _cs = admin_client
    resp = client.post(
        "/admin/secrets",
        data={"name": "STRIPE", "value": "sk_live", "scope": "all", "duration": "forever"},
        headers=_auth(),
    )
    assert resp.status_code == 200 and "STRIPE" in resp.text
    assert "sk_live" not in resp.text  # value never rendered
    assert await s.get_secret("STRIPE") == "sk_live"


async def test_delete_secret_via_admin(admin_client) -> None:
    client, s, _cs = admin_client
    await s.set_secret("DOOMED", "v", shared=True)
    resp = client.post("/admin/secrets/delete", data={"name": "DOOMED"}, headers=_auth())
    assert resp.status_code == 200 and "DOOMED" not in resp.text
    assert await s.get_secret("DOOMED") is None


async def test_delete_infra_secret_via_admin(tmp_path) -> None:
    db = str(tmp_path / "config.db")
    cs = ConfigStore(db_path=db)
    await cs.set_setup_step("done")
    await cs.set_admin_password("testpw")
    s = SecretStore(db_path=db, infra_vault=InfraVault("test-machine-key"))
    app, _ = create_admin_app(AgentState(), cs, secret_store=s)
    client = TestClient(app)
    await s.set_infra_secret("OPENAI_API_KEY", "sk-old")
    resp = client.post(
        "/admin/secrets/infra/delete", data={"name": "OPENAI_API_KEY"}, headers=_auth()
    )
    assert resp.status_code == 200 and "OPENAI_API_KEY" not in resp.text
    assert await s.get_infra_secret("OPENAI_API_KEY") is None


async def test_vault_fill_flow(admin_client) -> None:
    client, s, _cs = admin_client
    token = await s.create_request("NEWKEY", persona="", reason="need")
    # Page shell is public.
    assert client.get(f"/vault/fill/{token}").status_code == 200
    # Detail + submit require auth.
    assert client.get(f"/vault/request/{token}").status_code == 401
    assert client.get(f"/vault/request/{token}", headers=_auth()).json()["name"] == "NEWKEY"
    resp = client.post(
        f"/vault/fill/{token}",
        data={"value": "filled-value", "scope": "all", "duration": "forever"},
        headers=_auth(),
    )
    assert resp.json()["ok"] is True
    assert await s.get_secret("NEWKEY") == "filled-value"
    # Request consumed.
    assert await s.get_request(token) is None


async def test_bitwarden_import_parse_and_commit(admin_client) -> None:
    client, s, _cs = admin_client
    export = json.dumps(
        {"items": [{"type": 1, "name": "Site", "login": {"username": "u", "password": "pw"}}]}
    )
    resp = client.post(
        "/admin/secrets/import/parse",
        files={"file": ("bw.json", export, "application/json")},
        headers=_auth(),
    )
    assert resp.status_code == 200 and "Site" in resp.text
    commit = client.post(
        "/admin/secrets/import/commit",
        data={"selected": "Site", "username__Site": "u", "password__Site": "pw", "scope": "all"},
        headers=_auth(),
    )
    assert commit.status_code == 200
    assert await s.get_secret("Site") == {"username": "u", "password": "pw"}


async def test_change_password_rotates_and_keeps_secrets(admin_client) -> None:
    client, s, _cs = admin_client
    await s.set_secret("K", "v", shared=True)
    resp = client.post(
        "/admin/password",
        json={"current_password": "testpw", "new_password": "newpw"},
        headers=_auth("testpw"),
    )
    assert resp.json()["ok"] is True
    assert await s.get_secret("K") == "v"  # secret survives the rotation


async def test_wizard_secrets_blocked_after_setup(admin_client) -> None:
    client, _s, _cs = admin_client  # fixture marks setup complete
    r = client.post("/setup/step/secrets", data={"action": "generate"})
    assert r.status_code == 403


async def test_wizard_secrets_step_generates_key_and_imports(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # so data/master.key lands in the tmp dir
    db = str(tmp_path / "config.db")
    cs = ConfigStore(db_path=db)
    await cs.set_setup_step("admin")
    await cs.set("agent.anthropic_api_key", "sk-plaintext")
    s = SecretStore(db_path=db)
    app, _ = create_admin_app(AgentState(), cs, secret_store=s)
    client = TestClient(app)
    # Generate the machine key.
    r = client.post("/setup/step/secrets", data={"action": "generate"})
    assert r.status_code == 200
    assert s.infra.available
    # Import the detected plaintext infra secret into the vault.
    r = client.post("/setup/step/secrets", data={"action": "import"})
    assert r.status_code == 200
    assert await cs.get("agent.anthropic_api_key") == "${vault:ANTHROPIC_API_KEY}"
    assert await s.get_infra_secret("ANTHROPIC_API_KEY") == "sk-plaintext"


# ── Post-setup migration (issue #35) ───────────────────────────────────────


async def _migrate_app(tmp_path, *, machine_key="mk"):
    db = str(tmp_path / "config.db")
    cs = ConfigStore(db_path=db)
    await cs.set_setup_step("done")
    await cs.set_admin_password("testpw")
    await cs.set("agent.personae_db_path", str(tmp_path / "personae.db"))
    await cs.set("agent.personae_dir", "")
    s = SecretStore(db_path=db, infra_vault=InfraVault(machine_key))
    await s.ensure_wrapped_dek("testpw")
    app, _ = create_admin_app(AgentState(), cs, secret_store=s)
    return TestClient(app), s, cs


async def test_post_setup_migrate_endpoint(tmp_path) -> None:
    client, s, cs = await _migrate_app(tmp_path)
    await cs.set("agent.anthropic_api_key", "sk-plaintext")
    await cs.set("search.api_key", "tvly-plaintext")
    await cs.set("agent.openai_api_key", "${vault:OPENAI_API_KEY}")  # already a ref → skip
    resp = client.post("/admin/secrets/migrate", headers=_auth())
    assert resp.status_code == 200 and "Migrated 2 credential(s)" in resp.text
    # Plaintext values never echoed back into the rendered partial.
    assert "sk-plaintext" not in resp.text and "tvly-plaintext" not in resp.text
    assert await cs.get("agent.anthropic_api_key") == "${vault:ANTHROPIC_API_KEY}"
    assert await cs.get("search.api_key") == "${vault:TAVILY_API_KEY}"
    assert await s.get_infra_secret("ANTHROPIC_API_KEY") == "sk-plaintext"
    assert await s.get_infra_secret("TAVILY_API_KEY") == "tvly-plaintext"
    # Idempotent: a second run finds nothing left to move.
    resp2 = client.post("/admin/secrets/migrate", headers=_auth())
    assert "Nothing to migrate" in resp2.text


async def test_migrate_endpoint_requires_machine_key(tmp_path) -> None:
    client, _s, cs = await _migrate_app(tmp_path, machine_key=None)
    await cs.set("agent.anthropic_api_key", "sk-plaintext")
    resp = client.post("/admin/secrets/migrate", headers=_auth())
    assert resp.status_code == 200 and "No machine key configured" in resp.text
    # Untouched — no half-migration without a key.
    assert await cs.get("agent.anthropic_api_key") == "sk-plaintext"


async def test_patch_config_preserves_vault_ref(tmp_path) -> None:
    # A collapsed (vault-managed) field submits empty; the ref must survive.
    client, s, cs = await _migrate_app(tmp_path)
    await s.set_infra_secret("ANTHROPIC_API_KEY", "sk-real")
    await cs.set("agent.anthropic_api_key", "${vault:ANTHROPIC_API_KEY}")
    resp = client.patch(
        "/config",
        json={"values": {"agent.anthropic_api_key": "", "agent.model": "claude-x"}},
        headers=_auth(),
    )
    assert resp.status_code == 200
    assert "agent.anthropic_api_key" not in resp.json()["updated"]  # dropped, not written
    assert await cs.get("agent.anthropic_api_key") == "${vault:ANTHROPIC_API_KEY}"
    assert await cs.get("agent.model") == "claude-x"  # non-secret still saved
    # Re-entering a real key now auto-vaults it (#114): the config keeps only the
    # reference and the vault value is updated in place — plaintext never lands
    # in config.
    client.patch(
        "/config",
        json={"values": {"agent.anthropic_api_key": "sk-new-plaintext"}},
        headers=_auth(),
    )
    assert await cs.get("agent.anthropic_api_key") == "${vault:ANTHROPIC_API_KEY}"
    assert await s.get_infra_secret("ANTHROPIC_API_KEY") == "sk-new-plaintext"


# ── Safe UI collapse (issue #35) ───────────────────────────────────────────


async def test_llm_tab_collapses_vaulted_key(admin_client) -> None:
    client, _s, cs = admin_client
    await cs.set("agent.anthropic_api_key", "${vault:ANTHROPIC_API_KEY}")
    body = client.get("/partials/llm", headers=_auth()).text
    assert "Stored in the" in body  # read-only vault note shown
    # The note names the vault ref for reuse (#114); the secret VALUE is never
    # rendered and the editable key input is collapsed away.
    assert "${vault:ANTHROPIC_API_KEY}" in body
    assert 'x-model="apiKey"' not in body  # anthropic input replaced by the note
    assert 'x-model="openaiKey"' in body  # other providers still editable


async def test_tools_and_search_collapse_vaulted(admin_client) -> None:
    client, _s, cs = admin_client
    await cs.set("tools.gh.token", "${vault:GH_TOKEN}")
    await cs.set("search.api_key", "${vault:TAVILY_API_KEY}")
    tools = client.get("/partials/tools", headers=_auth()).text
    assert "Stored in the" in tools and "${vault:GH_TOKEN}" not in tools
    assert 'x-model="ghToken"' not in tools  # input replaced by the note
    search = client.get("/partials/search", headers=_auth()).text
    assert "Stored in the" in search and "${vault:TAVILY_API_KEY}" not in search


async def test_wizard_context_skips_vault_refs(tmp_path) -> None:
    # The wizard must not pre-fill a form field with a ${vault:} reference.
    from api.admin import _wizard_step_context

    cs = ConfigStore(db_path=str(tmp_path / "config.db"))
    await cs.set("agent.anthropic_api_key", "${vault:ANTHROPIC_API_KEY}")
    await cs.set("agent.openai_api_key", "sk-plain")
    await cs.set("channels.telegram.bot_token", "${vault:TELEGRAM_BOT_TOKEN}")
    await cs.set("search.api_key", "${vault:TAVILY_API_KEY}")
    llm = await _wizard_step_context("llm", cs)
    assert "anthropic_api_key" not in llm  # vault ref filtered out
    assert llm.get("openai_api_key") == "sk-plain"  # plaintext still pre-filled
    assert "bot_token" not in await _wizard_step_context("telegram", cs)
    assert "tavily_key" not in await _wizard_step_context("search", cs)


async def test_telegram_editor_and_save_preserve_vaulted_token(admin_client) -> None:
    client, _s, cs = admin_client
    await cs.set("channels.telegram.bot_token", "${vault:TELEGRAM_BOT_TOKEN}")
    editor = client.get("/channels/wizard?channel=telegram", headers=_auth()).text
    assert "Secrets" in editor and "${vault:TELEGRAM_BOT_TOKEN}" not in editor
    # Saving with an empty token field must NOT 400 and must keep the ref.
    resp = client.post(
        "/channels/telegram",
        json={"bot_token": "", "user_ids": "42", "enabled": "true"},
        headers=_auth(),
    )
    assert resp.status_code == 200
    assert await cs.get("channels.telegram.bot_token") == "${vault:TELEGRAM_BOT_TOKEN}"
    assert await cs.get("channels.telegram.allowed_user_ids") == "42"


async def test_delete_channel_preserves_vaulted_token(admin_client) -> None:
    # Disabling a channel must not orphan its vault-managed token.
    client, _s, cs = admin_client
    await cs.set("channels.telegram.bot_token", "${vault:TELEGRAM_BOT_TOKEN}")
    await cs.set("channels.telegram.enabled", "true")
    resp = client.delete("/channels/telegram", headers=_auth())
    assert resp.status_code == 200
    assert await cs.get("channels.telegram.bot_token") == "${vault:TELEGRAM_BOT_TOKEN}"
    assert await cs.get("channels.telegram.enabled") == "false"  # still disabled


async def test_llm_provider_key_autovaults_on_save(tmp_path) -> None:
    """A freshly-typed LLM provider key is vaulted, never stored plaintext (#114)."""
    db = str(tmp_path / "config.db")
    cs = ConfigStore(db_path=db)
    await cs.set_setup_step("done")
    await cs.set_admin_password("testpw")
    s = SecretStore(db_path=db, infra_vault=InfraVault("test-machine-key"))
    app, _ = create_admin_app(AgentState(), cs, secret_store=s)
    client = TestClient(app)

    resp = client.patch(
        "/config",
        json={"values": {"agent.anthropic_api_key": "sk-ant-secret"}},
        headers=_auth(),
    )
    assert resp.status_code == 200
    # Config keeps only the reference; the raw key lives in the vault.
    assert await cs.get("agent.anthropic_api_key") == "${vault:ANTHROPIC_API_KEY}"
    assert await s.get_infra_secret("ANTHROPIC_API_KEY") == "sk-ant-secret"
    # A later blank save (the vaulted field submits empty) must not orphan the ref.
    client.patch("/config", json={"values": {"agent.anthropic_api_key": ""}}, headers=_auth())
    assert await cs.get("agent.anthropic_api_key") == "${vault:ANTHROPIC_API_KEY}"


async def test_llm_provider_key_plaintext_without_machine_key(tmp_path) -> None:
    """No machine key -> fall back to plaintext config instead of crashing (#114)."""
    db = str(tmp_path / "config.db")
    cs = ConfigStore(db_path=db)
    await cs.set_setup_step("done")
    await cs.set_admin_password("testpw")
    s = SecretStore(db_path=db)  # no infra vault -> not available
    app, _ = create_admin_app(AgentState(), cs, secret_store=s)
    client = TestClient(app)
    resp = client.patch(
        "/config", json={"values": {"agent.openai_api_key": "sk-plain"}}, headers=_auth()
    )
    assert resp.status_code == 200
    assert await cs.get("agent.openai_api_key") == "sk-plain"
