"""Release Whistle independently, with an image version matching cactus-needle."""

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args, capture=False):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=capture)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if not re.fullmatch(r"v\d+\.\d+\.\d+", name):
        raise SystemExit(
            "Usage: make release-whistle name=v3.2.9 (published cactus-needle version)"
        )
    version = name[1:]
    tag = f"whistle-{name}"
    if run("git", "status", "--porcelain", capture=True).stdout.strip():
        raise SystemExit("Commit or stash working-tree changes before releasing.")
    default_branch = run(
        "gh",
        "repo",
        "view",
        "--json",
        "defaultBranchRef",
        "--jq",
        ".defaultBranchRef.name",
        capture=True,
    ).stdout.strip()
    branch = run("git", "branch", "--show-current", capture=True).stdout.strip()
    if branch != default_branch:
        raise SystemExit(f"Release from {default_branch}, after the Whistle changes are merged.")
    run("git", "pull", "--ff-only", "origin", default_branch)
    if run("git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}", capture=True).stdout:
        raise SystemExit(f"Tag {tag} already exists.")
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/cactus-needle/{version}/json", timeout=30
        ) as response:
            metadata = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise SystemExit(f"cactus-needle {version} has not been published on PyPI.") from exc
        raise
    if metadata["info"]["version"] != version:
        raise SystemExit("PyPI version does not match the requested release.")
    requirements = ROOT / "requirements.txt"
    requirements.write_text(
        re.sub(
            r"^cactus-needle==.+$",
            f"cactus-needle=={version}",
            requirements.read_text(),
            flags=re.MULTILINE,
        )
    )
    (ROOT / "VERSION").write_text(version + "\n")
    changed = run("git", "diff", "--name-only", "--", "VERSION", "requirements.txt", capture=True)
    if changed.stdout.strip():
        run("git", "add", "VERSION", "requirements.txt")
        run("git", "commit", "-m", f"chore(whistle): release {name}")
    run("git", "push", "origin", default_branch)
    commit = run("git", "rev-parse", "HEAD", capture=True).stdout.strip()
    run(
        "gh",
        "release",
        "create",
        tag,
        "--target",
        commit,
        "--title",
        f"Whistle {name}",
        "--generate-notes",
        "--latest=false",
    )
    print(f"Created {tag}; Whistle CI will test and publish the image after verification.")


if __name__ == "__main__":
    main()
