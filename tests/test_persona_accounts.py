"""Per-persona email/calendar account bindings + access enforcement (issue #110)."""

from __future__ import annotations

import pytest

from core.agent import AgentCore
from core.personae import Persona, PersonaStore, bind_existing_accounts
from core.subagents import narrow_accounts


# ── Persona model + access helpers ────────────────────────────────────────
def _persona() -> Persona:
    return Persona(
        name="fitness",
        email_accounts=[
            {"account": "fitness-agent", "access_level": "read_write", "is_sender_identity": True},
            {"account": "personal", "access_level": "read"},
        ],
        calendar_accounts=[{"account": "fitness-agent", "access_level": "read_write"}],
    )


def test_access_helpers() -> None:
    p = _persona()
    assert p.email_access("fitness-agent") == "read_write"
    assert p.email_access("personal") == "read"
    assert p.email_access("work") is None
    assert p.sender_identity() == "fitness-agent"
    assert p.calendar_access("fitness-agent") == "read_write"
    assert p.calendar_access("personal") is None
    # No bindings = no access (safe default).
    blank = Persona(name="blank")
    assert blank.email_access("personal") is None and blank.sender_identity() is None


# ── Email send routing + enforcement ──────────────────────────────────────
def test_send_defaults_to_sender_identity() -> None:
    account, err = AgentCore._resolve_email_send(_persona(), {})
    assert err is None and account == "fitness-agent"


def test_send_from_read_only_account_denied() -> None:
    account, err = AgentCore._resolve_email_send(_persona(), {"account": "personal"})
    assert account is None and err and "read-only" in err["error"]


def test_send_from_unbound_account_denied() -> None:
    account, err = AgentCore._resolve_email_send(_persona(), {"account": "work"})
    assert account is None and err and "not allowed" in err["error"]


def test_send_from_writable_account_allowed() -> None:
    account, err = AgentCore._resolve_email_send(_persona(), {"account": "fitness-agent"})
    assert err is None and account == "fitness-agent"


def test_persona_without_sender_identity_cannot_send() -> None:
    p = Persona(name="reader", email_accounts=[{"account": "personal", "access_level": "read"}])
    account, err = AgentCore._resolve_email_send(p, {})
    assert account is None and err and "sender" in err["error"]


def test_no_persona_is_legacy_requires_account() -> None:
    # Unscoped owner agent: account required, used verbatim (backward compatible).
    account, err = AgentCore._resolve_email_send(None, {"account": "personal"})
    assert err is None and account == "personal"
    account, err = AgentCore._resolve_email_send(None, {})
    assert account is None and err and "required" in err["error"]


# ── Calendar write routing + enforcement ──────────────────────────────────
def test_calendar_defaults_to_writable() -> None:
    cal, err = AgentCore._resolve_calendar_write(_persona(), {})
    assert err is None and cal == "fitness-agent"


def test_calendar_read_only_denied() -> None:
    p = Persona(name="ro", calendar_accounts=[{"account": "shared", "access_level": "read"}])
    cal, err = AgentCore._resolve_calendar_write(p, {"calendar": "shared"})
    assert cal is None and err and "read-only" in err["error"]


def test_calendar_unbound_denied() -> None:
    cal, err = AgentCore._resolve_calendar_write(_persona(), {"calendar": "work"})
    assert cal is None and err and "not allowed" in err["error"]


# ── Subagent inherit-never-widen ──────────────────────────────────────────
def test_narrow_accounts_downgrades_and_drops() -> None:
    parent = [{"account": "personal", "access_level": "read"}]
    child = [
        {"account": "personal", "access_level": "read_write", "is_sender_identity": True},
        {"account": "work", "access_level": "read_write"},
    ]
    out = narrow_accounts(parent, child)
    # 'work' dropped (parent lacks it); 'personal' downgraded to read and its send
    # identity stripped (parent can't write it).
    assert out == [{"account": "personal", "access_level": "read", "is_sender_identity": False}]


def test_account_note_lists_bindings_not_credentials() -> None:
    note = AgentCore._account_note(_persona())
    assert "fitness-agent (read_write, sender)" in note
    assert "personal (read)" in note
    assert "password" not in note.lower()


# ── Compat migration + DB round-trip ──────────────────────────────────────
@pytest.mark.asyncio
async def test_binding_migration_and_store_roundtrip(tmp_path) -> None:
    store = PersonaStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path / "missing")
    await store.upsert(Persona(name="coach"))
    await store.upsert(
        Persona(name="bound", email_accounts=[{"account": "x", "access_level": "read"}])
    )

    n = await bind_existing_accounts(store, ["work", "personal"], ["google"])
    assert n == 1  # only the persona with no bindings is touched

    coach = await store.get("coach")
    # Full read_write on all accounts; first email = sender identity (behaviour kept).
    assert coach.email_access("work") == "read_write"
    assert coach.email_access("personal") == "read_write"
    assert coach.sender_identity() == "work"
    assert coach.calendar_access("google") == "read_write"

    # Idempotent: an already-bound persona is left untouched (its read stays read).
    bound = await store.get("bound")
    assert bound.email_access("x") == "read" and bound.email_access("work") is None
    assert await bind_existing_accounts(store, ["work"], []) == 0
