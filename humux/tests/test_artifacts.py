"""Tests for web-artifact serving (core/artifacts.py + the /artifacts route).

Issue #82: artifacts are plain files the agent writes under
``{workspace}/artifacts/<slug>/`` with the coding-harness ``write_file`` tool.
There is no ArtifactStore, no TTL, no write_artifact tool — this module only
*serves* those files, with the same traversal/dotfile/symlink/hardlink hardening
the old store had (the route is public and the slug is guessable).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core.artifacts import ARTIFACTS_SUBDIR, resolve, serving_base, valid_id
from core.config_store import ConfigStore


def _make(ws: Path, slug: str, files: dict[str, str]) -> Path:
    """Lay out an artifact the way the agent's write_file would."""
    base = ws / ARTIFACTS_SUBDIR / slug
    base.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return ws / ARTIFACTS_SUBDIR


# -- resolve(): the serve-time path guard -------------------------------------


def test_resolve_index_default_and_subfiles(tmp_path) -> None:
    base = _make(
        tmp_path, "dash", {"index.html": "<h1>x</h1>", "data.json": "{}", "js/app.js": "1"}
    )
    assert resolve(base, "dash").read_text() == "<h1>x</h1>"  # empty path → index.html
    assert resolve(base, "dash", "data.json").read_text() == "{}"
    assert resolve(base, "dash", "js/app.js").read_text() == "1"


def test_resolve_rejects_traversal_and_dotfiles(tmp_path) -> None:
    base = _make(tmp_path, "a", {"index.html": "x", ".secret": "nope"})
    assert resolve(base, "a", "../../etc/passwd") is None
    assert resolve(base, "a", ".secret") is None
    assert resolve(base, "a", "../a/index.html") is None  # climbs out then back


def test_resolve_rejects_bad_slug_and_missing(tmp_path) -> None:
    _make(tmp_path, "a", {"index.html": "x"})
    base = tmp_path / ARTIFACTS_SUBDIR
    assert not valid_id("bad!id") and resolve(base, "bad!id") is None
    assert not valid_id("../../etc") and resolve(base, "../../etc") is None
    assert resolve(base, "missing") is None


def test_resolve_rejects_symlink_escape(tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("topsecret")
    base = _make(tmp_path, "a", {"index.html": "x"})
    os.symlink(secret, base / "a" / "leak.html")
    assert resolve(base, "a", "leak.html") is None


def test_resolve_rejects_hardlink_escape(tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("topsecret")
    base = _make(tmp_path, "a", {"index.html": "x"})
    os.link(secret, base / "a" / "hard.html")  # hardlink → st_nlink > 1
    assert resolve(base, "a", "hard.html") is None


def test_resolve_rejects_null_byte(tmp_path) -> None:
    base = _make(tmp_path, "a", {"index.html": "x"})
    assert resolve(base, "a", "index.html\x00.png") is None


# -- serving_base(): config gating --------------------------------------------


class _Cfg:
    """Minimal ConfigStore stub exposing only what the artifacts route reads."""

    def __init__(self, *, directory="", ws_enabled=True, artifacts_enabled=True):
        self._vals: dict[str, str] = {}
        if ws_enabled:
            self._vals["workspace.enabled"] = "true"
        if directory:
            self._vals["workspace.directory"] = str(directory)
        self._vals["artifacts.enabled"] = "true" if artifacts_enabled else "false"

    async def get(self, key, default=None):
        return self._vals.get(key, default)

    async def get_many(self, prefix: str = "") -> dict:
        return {k: v for k, v in self._vals.items() if k.startswith(prefix)}

    async def is_setup_complete(self) -> bool:
        return True


async def test_serving_base_gating(tmp_path) -> None:
    want = (tmp_path / ARTIFACTS_SUBDIR).resolve()
    assert await serving_base(_Cfg(directory=tmp_path)) == want
    assert await serving_base(_Cfg(directory=tmp_path, ws_enabled=False)) is None
    assert await serving_base(_Cfg(directory="")) is None  # no workspace dir
    assert await serving_base(_Cfg(directory=tmp_path, artifacts_enabled=False)) is None


# -- the /artifacts route -----------------------------------------------------


def _client(cfg: _Cfg) -> TestClient:
    app, _auth = create_admin_app(AgentState(agent=None), cast(ConfigStore, cfg))
    return TestClient(app)


def test_serve_index_and_subfile_with_content_types(tmp_path) -> None:
    _make(tmp_path, "dash", {"index.html": "<h1>dash</h1>", "data.json": "{}"})
    client = _client(_Cfg(directory=tmp_path))

    root = client.get("/artifacts/dash/")
    assert root.status_code == 200
    assert "<h1>dash</h1>" in root.text
    assert "text/html" in root.headers["content-type"]
    # CSP sandbox + nosniff on every artifact response; absence of allow-same-origin
    # is the actual protection for the admin localStorage.
    assert "sandbox" in root.headers["content-security-policy"]
    assert "allow-same-origin" not in root.headers["content-security-policy"]
    assert root.headers["x-content-type-options"] == "nosniff"

    sub = client.get("/artifacts/dash/data.json")
    assert sub.status_code == 200
    assert "json" in sub.headers["content-type"]


def test_no_slash_redirects_to_slash(tmp_path) -> None:
    _make(tmp_path, "rep", {"index.html": "<h1>x</h1>"})
    client = _client(_Cfg(directory=tmp_path))
    r = client.get("/artifacts/rep", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/artifacts/rep/"
    assert client.get("/artifacts/rep", follow_redirects=True).status_code == 200
    # A malformed slug is 404'd, not echoed into the Location header.
    assert client.get("/artifacts/bad!id", follow_redirects=False).status_code == 404


def test_dotfile_and_traversal_not_served(tmp_path) -> None:
    _make(tmp_path, "a", {"index.html": "<h1>x</h1>", ".env": "SECRET=1"})
    client = _client(_Cfg(directory=tmp_path))
    assert client.get("/artifacts/a/.env").status_code == 404
    assert client.get("/artifacts/anything/..%2f..%2fetc%2fpasswd").status_code == 404


def test_serve_missing_and_disabled(tmp_path) -> None:
    _make(tmp_path, "a", {"index.html": "<h1>x</h1>"})
    assert _client(_Cfg(directory=tmp_path)).get("/artifacts/nope/").status_code == 404
    off = _client(_Cfg(directory=tmp_path, artifacts_enabled=False))
    assert off.get("/artifacts/a/").status_code == 404
    no_ws = _client(_Cfg(directory=tmp_path, ws_enabled=False))
    assert no_ws.get("/artifacts/a/").status_code == 404


def test_serve_route_is_public(tmp_path) -> None:
    # No Authorization header — artifacts are public by design (the slug is the handle).
    _make(tmp_path, "a", {"index.html": "<h1>x</h1>"})
    assert _client(_Cfg(directory=tmp_path)).get("/artifacts/a/").status_code == 200


# -- the write_artifact tool is gone (replaced by write_file + a skill) --------


def test_write_artifact_tool_removed() -> None:
    from core.agent import TOOLS

    assert "write_artifact" not in {t["name"] for t in TOOLS}


def test_write_artifact_gone_from_agent_scope() -> None:
    from api.admin import GATEABLE_TOOLS, gateable_tools_for

    assert "write_artifact" not in GATEABLE_TOOLS
    assert set(gateable_tools_for()) == set(GATEABLE_TOOLS)


def test_artifact_hosting_skill_seeded() -> None:
    # The capability now lives in a skill, not a tool.
    assert Path("skills/artifact-hosting.md").is_file()
