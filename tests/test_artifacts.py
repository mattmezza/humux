"""Tests for agent-crafted web artifacts (core/artifacts.py + /artifacts route)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

import core.artifacts as artifacts_mod
from api.admin import AgentState, create_admin_app
from core.artifacts import ArtifactStore, cleanup_loop
from core.config_store import ConfigStore
from core.permissions import PermissionEngine, PermissionLevel

# -- ArtifactStore: create + resolve ------------------------------------------


def test_create_html_single_page(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    art_id = store.create(files={"index.html": "<h1>hello</h1>"}, title="greeting")
    served = store.resolve(art_id)  # empty path → entrypoint
    assert served is not None
    assert served.read_text(encoding="utf-8") == "<h1>hello</h1>"
    assert served.name == "index.html"


def test_create_multi_file_and_resolve_each(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    art_id = store.create(
        files={"index.html": "<link href='style.css'>", "style.css": "body{}", "js/app.js": "1"}
    )
    assert store.resolve(art_id, "style.css").read_text() == "body{}"
    assert store.resolve(art_id, "js/app.js").read_text() == "1"
    assert store.resolve(art_id).name == "index.html"  # default entrypoint


def test_create_source_path_file_sets_entrypoint(tmp_path) -> None:
    src = tmp_path / "report.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    store = ArtifactStore(tmp_path / "store")
    art_id = store.create(source_path=str(src))
    served = store.resolve(art_id)  # entrypoint became report.pdf
    assert served is not None and served.name == "report.pdf"
    assert served.read_bytes() == b"%PDF-1.4 fake"


def test_create_source_path_directory(tmp_path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>site</h1>")
    (site / "a.css").write_text("x{}")
    store = ArtifactStore(tmp_path / "store")
    art_id = store.create(source_path=str(site))
    assert store.resolve(art_id).read_text() == "<h1>site</h1>"
    assert store.resolve(art_id, "a.css").read_text() == "x{}"


def test_create_requires_exactly_one_source(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.create()  # neither
    with pytest.raises(ValueError):
        store.create(files={"index.html": "x"}, source_path=str(tmp_path))  # both


def test_create_rejects_oversized_and_cleans_up(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts_mod, "MAX_ARTIFACT_BYTES", 16)
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="too large"):
        store.create(files={"index.html": "x" * 64})
    # The partial directory must not linger.
    assert list(tmp_path.iterdir()) == []


# -- resolve: containment / safety --------------------------------------------


def test_resolve_rejects_traversal_and_dotfiles(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    art_id = store.create(files={"index.html": "<h1>x</h1>"})
    for bad in ["../../etc/passwd", "..", ".meta.json", "a/../../b", "x" * 80]:
        assert store.resolve(art_id, bad) is None, bad
    assert store.resolve("bad id!", "index.html") is None  # malformed id
    assert store.resolve(art_id, "missing.css") is None  # absent file


def test_resolve_rejects_symlink_escape(tmp_path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")
    store = ArtifactStore(tmp_path / "store")
    art_id = store.create(files={"index.html": "<h1>x</h1>"})
    (store.dir / art_id / "leak.html").symlink_to(outside)
    assert store.resolve(art_id, "leak.html") is None


def test_resolve_rejects_hardlink_escape(tmp_path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")
    store = ArtifactStore(tmp_path / "store")
    art_id = store.create(files={"index.html": "<h1>x</h1>"})
    os.link(outside, store.dir / art_id / "leak.html")
    assert store.resolve(art_id, "leak.html") is None


def test_resolve_rejects_null_byte(tmp_path) -> None:
    # A null byte makes Path.resolve raise ValueError; must degrade to None, not 500.
    store = ArtifactStore(tmp_path)
    art_id = store.create(files={"index.html": "<h1>x</h1>"})
    assert store.resolve(art_id, "foo\x00.png") is None


def test_route_null_byte_is_404_not_500(tmp_path) -> None:
    art_id = ArtifactStore(tmp_path).create(files={"index.html": "<h1>x</h1>"})
    r = _client(_Store(tmp_path)).get(f"/artifacts/{art_id}/foo%00.png")
    assert r.status_code == 404


# -- entrypoint resolution for source_path ------------------------------------


def test_source_dir_autopicks_single_file_entrypoint(tmp_path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "report.pdf").write_bytes(b"%PDF fake")  # no index.html, one file
    art_id = ArtifactStore(tmp_path / "store").create(source_path=str(site))
    served = ArtifactStore(tmp_path / "store").resolve(art_id)
    assert served is not None and served.name == "report.pdf"


def test_source_dir_without_index_and_multiple_files_errors(tmp_path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "a.html").write_text("a")
    (site / "b.html").write_text("b")
    with pytest.raises(ValueError, match="entrypoint"):
        ArtifactStore(tmp_path / "store").create(source_path=str(site))


def test_single_file_source_ignores_explicit_entrypoint(tmp_path) -> None:
    src = tmp_path / "r.pdf"
    src.write_bytes(b"%PDF fake")
    store = ArtifactStore(tmp_path / "store")
    art_id = store.create(source_path=str(src), entrypoint="index.html")
    assert store.resolve(art_id).name == "r.pdf"


# -- cleanup: per-artifact TTL ------------------------------------------------


def test_cleanup_respects_per_artifact_ttl(tmp_path) -> None:
    store = ArtifactStore(tmp_path, ttl_hours=168)
    keep = store.create(files={"index.html": "a"}, ttl_hours=0)  # forever
    expire = store.create(files={"index.html": "b"}, ttl_hours=1)
    # Backdate the expiring artifact's stored expiry into the past.
    meta_path = tmp_path / expire / ".meta.json"
    meta = json.loads(meta_path.read_text())
    meta["expires_at"] = time.time() - 10
    meta_path.write_text(json.dumps(meta))

    assert store.cleanup() == 1
    assert (tmp_path / keep).exists()
    assert not (tmp_path / expire).exists()


def test_cleanup_falls_back_to_mtime_without_meta(tmp_path) -> None:
    store = ArtifactStore(tmp_path, ttl_hours=1)
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "index.html").write_text("x")  # no .meta.json
    old = time.time() - 2 * 3600
    os.utime(orphan, (old, old))
    assert store.cleanup() == 1
    assert not orphan.exists()


# -- /artifacts route ---------------------------------------------------------


class _Store:
    """Minimal ConfigStore stub exposing only what the artifacts route reads."""

    def __init__(self, directory, *, enabled: bool = True, ttl_hours: int = 168):
        self._vals = {
            "artifacts.directory": str(directory),
            "artifacts.enabled": "true" if enabled else "false",
            "artifacts.ttl_hours": str(ttl_hours),
        }

    async def get_many(self, prefix: str = "") -> dict:
        return {k: v for k, v in self._vals.items() if k.startswith(prefix)}

    async def is_setup_complete(self) -> bool:
        return True


def _client(store: _Store) -> TestClient:
    app, _auth = create_admin_app(AgentState(agent=None), cast(ConfigStore, store))
    return TestClient(app)


def test_serve_index_and_subfile_with_content_types(tmp_path) -> None:
    art_id = ArtifactStore(tmp_path).create(
        files={"index.html": "<h1>dash</h1>", "data.json": "{}"}
    )
    client = _client(_Store(tmp_path))

    root = client.get(f"/artifacts/{art_id}/")
    assert root.status_code == 200
    assert "<h1>dash</h1>" in root.text
    assert "text/html" in root.headers["content-type"]
    # CSP sandbox + nosniff on every artifact response; absence of allow-same-origin
    # is the actual protection for the admin localStorage.
    assert "sandbox" in root.headers["content-security-policy"]
    assert "allow-same-origin" not in root.headers["content-security-policy"]
    assert root.headers["x-content-type-options"] == "nosniff"

    sub = client.get(f"/artifacts/{art_id}/data.json")
    assert sub.status_code == 200
    assert "json" in sub.headers["content-type"]


def test_no_slash_redirects_to_slash(tmp_path) -> None:
    art_id = ArtifactStore(tmp_path).create(files={"index.html": "<h1>x</h1>"})
    client = _client(_Store(tmp_path))
    r = client.get(f"/artifacts/{art_id}", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == f"/artifacts/{art_id}/"
    assert client.get(f"/artifacts/{art_id}", follow_redirects=True).status_code == 200
    # A malformed id is 404'd, not echoed into the Location header.
    assert client.get("/artifacts/bad!id", follow_redirects=False).status_code == 404


def test_meta_and_traversal_not_served(tmp_path) -> None:
    art_id = ArtifactStore(tmp_path).create(files={"index.html": "<h1>x</h1>"})
    client = _client(_Store(tmp_path))
    assert client.get(f"/artifacts/{art_id}/.meta.json").status_code == 404
    assert client.get("/artifacts/anything/..%2f..%2fetc%2fpasswd").status_code == 404


def test_serve_missing_and_disabled(tmp_path) -> None:
    art_id = ArtifactStore(tmp_path).create(files={"index.html": "<h1>x</h1>"})
    assert _client(_Store(tmp_path)).get("/artifacts/nope/").status_code == 404
    assert _client(_Store(tmp_path, enabled=False)).get(f"/artifacts/{art_id}/").status_code == 404


def test_serve_route_is_public(tmp_path) -> None:
    # No Authorization header — artifacts are gated by the unguessable id, not auth.
    art_id = ArtifactStore(tmp_path).create(files={"index.html": "<h1>x</h1>"})
    assert _client(_Store(tmp_path)).get(f"/artifacts/{art_id}/").status_code == 200


# -- cleanup_loop (the scheduled async task) ----------------------------------


async def test_cleanup_loop_sweeps_then_cancels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts_mod, "_CLEANUP_INTERVAL_S", 0.01)
    store = ArtifactStore(tmp_path, ttl_hours=1)
    art_id = store.create(files={"index.html": "old"}, ttl_hours=1)
    meta_path = tmp_path / art_id / ".meta.json"
    meta = json.loads(meta_path.read_text())
    meta["expires_at"] = time.time() - 10
    meta_path.write_text(json.dumps(meta))

    task = asyncio.create_task(cleanup_loop(_Store(tmp_path, ttl_hours=1)))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if not (tmp_path / art_id).exists():
            break
    assert not (tmp_path / art_id).exists()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# -- write_artifact tool handler ----------------------------------------------


def _fake_agent(tmp_path):
    from core.config import ArtifactsConfig

    return SimpleNamespace(
        config=SimpleNamespace(artifacts=ArtifactsConfig(directory=str(tmp_path))),
        _base_url=lambda: "http://host:8000",
    )


def test_tool_write_artifact_html_and_files(tmp_path) -> None:
    from core.agent import AgentCore

    fake = _fake_agent(tmp_path)
    res = AgentCore._tool_write_artifact(fake, {"html": "<h1>hi</h1>", "title": "t"})
    assert res["ok"] is True
    assert res["url"].startswith("http://host:8000/artifacts/")
    assert res["url"].endswith("/")  # trailing slash so relative links resolve

    multi = AgentCore._tool_write_artifact(
        fake, {"files": {"index.html": "<h1>m</h1>", "a.css": "x"}}
    )
    assert multi["ok"] is True


def test_tool_write_artifact_validation(tmp_path) -> None:
    from core.agent import AgentCore

    fake = _fake_agent(tmp_path)
    assert "error" in AgentCore._tool_write_artifact(fake, {})  # nothing
    assert "error" in AgentCore._tool_write_artifact(fake, {"files": "notadict"})
    assert "error" in AgentCore._tool_write_artifact(
        fake, {"html": "<p>x</p>", "source_path": str(tmp_path)}
    )  # both
    assert "error" in AgentCore._tool_write_artifact(fake, {"html": "x", "ttl_hours": "soon"})

    fake.config.artifacts.enabled = False
    assert "error" in AgentCore._tool_write_artifact(fake, {"html": "<p>x</p>"})


# -- permissions: inline ALWAYS, source_path ASK ------------------------------


def test_publishing_file_requires_approval(tmp_path) -> None:
    engine = PermissionEngine(db_path=str(tmp_path / "config.db"))

    inline = {"html": "<h1>x</h1>"}
    publish = {"source_path": "/tmp/report.pdf"}

    assert engine.check("write_artifact", inline) == PermissionLevel.ALWAYS
    assert engine.is_write_action("write_artifact", inline) is False

    assert engine.check("write_artifact", publish) == PermissionLevel.ASK
    assert engine.is_write_action("write_artifact", publish) is True
