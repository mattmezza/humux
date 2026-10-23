"""Release validation and version updates, with all external writes stubbed."""

import importlib.util
import io
import subprocess
import urllib.error
from pathlib import Path

import pytest


@pytest.fixture
def release(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "whistle_release", Path(__file__).parents[1] / "release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.sys, "argv", ["release.py", "v3.2.9"])
    (tmp_path / "VERSION").write_text("3.2.8\n")
    (tmp_path / "requirements.txt").write_text("cactus-needle==3.2.8\nfastapi==0.162.0\n")
    commands = []
    replies = {
        ("git", "branch", "--show-current"): "main\n",
        ("git", "rev-parse", "HEAD"): "tested-commit\n",
    }

    def run(*args, capture=False):
        commands.append(args)
        text = "main\n" if args[:3] == ("gh", "repo", "view") else replies.get(args, "")
        if args[:3] == ("git", "diff", "--name-only"):
            text = "VERSION\nrequirements.txt\n"
        return subprocess.CompletedProcess(args, 0, stdout=text)

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(b'{"info":{"version":"3.2.9"}}'),
    )
    return module, commands, replies


def test_release_updates_only_component_and_publishes_namespaced_tag(release):
    module, commands, _ = release
    module.main()
    assert (module.ROOT / "VERSION").read_text() == "3.2.9\n"
    assert (
        module.ROOT / "requirements.txt"
    ).read_text() == "cactus-needle==3.2.9\nfastapi==0.162.0\n"
    assert commands[-1] == (
        "gh",
        "release",
        "create",
        "whistle-v3.2.9",
        "--target",
        "tested-commit",
        "--title",
        "Whistle v3.2.9",
        "--generate-notes",
        "--latest=false",
    )


@pytest.mark.parametrize("name", ["", "v3.2", "3.2.9", "v3.2.9;exit"])
def test_invalid_version_does_not_touch_git(release, name, monkeypatch):
    module, commands, _ = release
    monkeypatch.setattr(module.sys, "argv", ["release.py", name])
    with pytest.raises(SystemExit, match="Usage"):
        module.main()
    assert commands == []


def test_reject_dirty_tree(release):
    module, commands, replies = release
    replies[("git", "status", "--porcelain")] = " M server.py"
    with pytest.raises(SystemExit, match="Commit or stash"):
        module.main()
    assert len(commands) == 1


def test_reject_feature_branch(release):
    module, commands, replies = release
    replies[("git", "branch", "--show-current")] = "feature\n"
    with pytest.raises(SystemExit, match="Release from main"):
        module.main()
    assert not any(command[:2] == ("git", "pull") for command in commands)


def test_unpublished_runtime_does_not_change_version(release, monkeypatch):
    module, commands, _ = release

    def missing(*args, **kwargs):
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr(module.urllib.request, "urlopen", missing)
    with pytest.raises(SystemExit, match="has not been published"):
        module.main()
    assert (module.ROOT / "VERSION").read_text() == "3.2.8\n"
    assert not any(command[:3] == ("gh", "release", "create") for command in commands)
