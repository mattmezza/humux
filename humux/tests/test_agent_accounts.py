"""Per-agent email/calendar account bindings + access enforcement (issue #110)."""

from __future__ import annotations

import pytest

from core.agent import AgentCore
from core.agents import Agent, AgentStore, bind_existing_accounts
from core.subagents import narrow_accounts


# ── Agent model + access helpers ────────────────────────────────────────
def _agent() -> Agent:
    return Agent(
        name="fitness",
        email_accounts=[
            {"account": "fitness-agent", "access_level": "read_write", "is_sender_identity": True},
            {"account": "personal", "access_level": "read"},
        ],
        calendar_accounts=[{"account": "fitness-agent", "access_level": "read_write"}],
    )


def test_access_helpers() -> None:
    p = _agent()
    assert p.email_access("fitness-agent") == "read_write"
    assert p.email_access("personal") == "read"
    assert p.email_access("work") is None
    assert p.sender_identity() == "fitness-agent"
    assert p.calendar_access("fitness-agent") == "read_write"
    assert p.calendar_access("personal") is None
    # No bindings = no access (safe default).
    blank = Agent(name="blank")
    assert blank.email_access("personal") is None and blank.sender_identity() is None


# ── Email send routing + enforcement ──────────────────────────────────────
def test_send_defaults_to_sender_identity() -> None:
    account, err = AgentCore._resolve_email_send(_agent(), {})
    assert err is None and account == "fitness-agent"


def test_send_from_read_only_account_denied() -> None:
    account, err = AgentCore._resolve_email_send(_agent(), {"account": "personal"})
    assert account is None and err and "read-only" in err["error"].lower()


def test_send_from_unbound_account_denied() -> None:
    account, err = AgentCore._resolve_email_send(_agent(), {"account": "work"})
    assert account is None and err and "not allowed" in err["error"].lower()


def test_send_from_writable_account_allowed() -> None:
    account, err = AgentCore._resolve_email_send(_agent(), {"account": "fitness-agent"})
    assert err is None and account == "fitness-agent"


def test_agent_without_sender_identity_cannot_send() -> None:
    p = Agent(name="reader", email_accounts=[{"account": "personal", "access_level": "read"}])
    account, err = AgentCore._resolve_email_send(p, {})
    assert account is None and err and "sender" in err["error"]


def test_no_agent_is_legacy_requires_account() -> None:
    # Unscoped owner agent: account required, used verbatim (backward compatible).
    account, err = AgentCore._resolve_email_send(None, {"account": "personal"})
    assert err is None and account == "personal"
    account, err = AgentCore._resolve_email_send(None, {})
    assert account is None and err and "required" in err["error"]


# ── Calendar write routing + enforcement ──────────────────────────────────
def test_calendar_defaults_to_writable() -> None:
    cal, err = AgentCore._resolve_calendar_write(_agent(), {})
    assert err is None and cal == "fitness-agent"


def test_calendar_read_only_denied() -> None:
    p = Agent(name="ro", calendar_accounts=[{"account": "shared", "access_level": "read"}])
    cal, err = AgentCore._resolve_calendar_write(p, {"calendar": "shared"})
    assert cal is None and err and "read-only" in err["error"].lower()


def test_calendar_unbound_denied() -> None:
    cal, err = AgentCore._resolve_calendar_write(_agent(), {"calendar": "work"})
    assert cal is None and err and "not allowed" in err["error"].lower()


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
    note = AgentCore._account_note(_agent())
    assert "fitness-agent (read_write, sender)" in note
    assert "personal (read)" in note
    assert "password" not in note.lower()


# ── Contacts routing + enforcement (#110 follow-up) ───────────────────────
def _contacts_agent() -> Agent:
    return Agent(
        name="c",
        contacts_accounts=[
            {"account": "agent-book", "access_level": "read_write"},
            {"account": "shared", "access_level": "read"},
        ],
    )


def test_contacts_read_defaults_to_first_bound() -> None:
    acc, err = AgentCore._resolve_contacts_access(_contacts_agent(), {}, need_write=False)
    assert err is None and acc == "agent-book"


def test_contacts_write_defaults_to_first_writable() -> None:
    p = Agent(
        name="c",
        contacts_accounts=[
            {"account": "shared", "access_level": "read"},
            {"account": "agent-book", "access_level": "read_write"},
        ],
    )
    acc, err = AgentCore._resolve_contacts_access(p, {}, need_write=True)
    assert err is None and acc == "agent-book"  # skips the read-only one


def test_contacts_write_on_read_only_denied() -> None:
    acc, err = AgentCore._resolve_contacts_access(
        _contacts_agent(), {"account": "shared"}, need_write=True
    )
    assert acc is None and err and "read-only" in err["error"].lower()


def test_contacts_unbound_denied() -> None:
    acc, err = AgentCore._resolve_contacts_access(
        _contacts_agent(), {"account": "work"}, need_write=False
    )
    assert acc is None and err and "not allowed" in err["error"].lower()


def test_contacts_no_agent_legacy_requires_account() -> None:
    acc, err = AgentCore._resolve_contacts_access(None, {"account": "x"}, need_write=True)
    assert err is None and acc == "x"
    acc, err = AgentCore._resolve_contacts_access(None, {}, need_write=False)
    assert acc is None and err and "required" in err["error"].lower()


def test_account_note_includes_contacts() -> None:
    note = AgentCore._account_note(_contacts_agent())
    assert "Contacts accounts: agent-book (read_write), shared (read)" in note


def test_narrow_accounts_contacts_downgrades() -> None:
    parent = [{"account": "shared", "access_level": "read"}]
    child = [
        {"account": "shared", "access_level": "read_write"},
        {"account": "private", "access_level": "read_write"},
    ]
    assert narrow_accounts(parent, child) == [{"account": "shared", "access_level": "read"}]


# ── Default-agent bindings (#110 follow-up) ────────────────────────────────
def _config_with_defaults() -> object:
    from core.config import Config

    cfg = Config()
    cfg.agent.email_accounts = [
        {"account": "me", "access_level": "read_write", "is_sender_identity": True}
    ]
    cfg.agent.calendar_accounts = [{"account": "gcal", "access_level": "read"}]
    cfg.agent.contacts_accounts = [{"account": "book", "access_level": "read_write"}]
    return cfg


def test_default_accounts_none_when_unset() -> None:
    from core.config import Config

    assert AgentCore._build_default_accounts(Config()) is None  # unscoped legacy


def test_default_accounts_built_and_enforced() -> None:
    identity = AgentCore._build_default_accounts(_config_with_defaults())
    assert identity is not None
    # Default agent routes send to its bound sender identity…
    acc, err = AgentCore._resolve_email_send(identity, {})
    assert err is None and acc == "me"
    # …and is blocked from writing a read-only calendar.
    cal, err = AgentCore._resolve_calendar_write(identity, {"calendar": "gcal"})
    assert cal is None and err and "read-only" in err["error"].lower()
    # …and its contacts write routes to the writable book.
    cacc, err = AgentCore._resolve_contacts_access(identity, {}, need_write=True)
    assert err is None and cacc == "book"


@pytest.mark.asyncio
async def test_migration_binds_contacts_too(tmp_path) -> None:
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path / "missing")
    await store.upsert(Agent(name="coach"))
    n = await bind_existing_accounts(store, ["mail"], [], ["book"])
    assert n == 1
    coach = await store.get("coach")
    assert coach.contacts_access("book") == "read_write"


# ── Compat migration + DB round-trip ──────────────────────────────────────
@pytest.mark.asyncio
async def test_binding_migration_and_store_roundtrip(tmp_path) -> None:
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path / "missing")
    await store.upsert(Agent(name="coach"))
    await store.upsert(
        Agent(name="bound", email_accounts=[{"account": "x", "access_level": "read"}])
    )

    n = await bind_existing_accounts(store, ["work", "personal"], ["google"])
    assert n == 1  # only the agent with no bindings is touched

    coach = await store.get("coach")
    # Full read_write on all accounts; first email = sender identity (behaviour kept).
    assert coach.email_access("work") == "read_write"
    assert coach.email_access("personal") == "read_write"
    assert coach.sender_identity() == "work"
    assert coach.calendar_access("google") == "read_write"

    # Idempotent: an already-bound agent is left untouched (its read stays read).
    bound = await store.get("bound")
    assert bound.email_access("x") == "read" and bound.email_access("work") is None
    assert await bind_existing_accounts(store, ["work"], []) == 0
