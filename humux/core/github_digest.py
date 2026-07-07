"""GitHub API helpers for inbound webhook turns: thread digest (#237) and the
👀 start-of-work ack (#241).

A webhook-woken agent has only seen the turns *it* was woken for: comments by
other agents/humans, label changes, and state flips in between are invisible
to its chat history. GitHub holds the complete record, so each turn gets a
fresh, bounded snapshot of the thread prepended to its task — the agent sees
the full state before acting instead of (maybe) remembering to look it up.

Failure is always soft: any error returns ``None``/``False`` and the turn
runs exactly as it would have without the helper.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

_API = "https://api.github.com"
_COMMENTS = 10  # most recent comments included
_BODY_CAP = 1500  # issue body chars
_COMMENT_CAP = 500  # per-comment chars
_TOTAL_CAP = 6000  # whole digest chars


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _clip(text: str, cap: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= cap else text[: cap - 1] + "…"


async def ack_reaction(target: str, token: str, content: str = "eyes") -> bool:
    """React on the triggering item so "an agent is on it" shows in the GitHub
    UI (#241). ``target`` is an API path like ``repos/o/r/issues/7/reactions``.
    Best-effort: any failure returns ``False`` and is only logged."""
    try:
        async with httpx.AsyncClient(timeout=8, headers=_headers(token)) as client:
            resp = await client.post(f"{_API}/{target}", json={"content": content})
            resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 — the ack must never sink a turn
        log.warning("ack reaction failed for %s: %s", target, exc)
        return False


async def thread_digest(repo: str, number: int, token: str) -> str | None:
    """Current state of ``repo#number`` as compact text, or ``None`` on any failure.

    Two API calls: the issue (works for PRs too — every PR is an issue) and its
    comments. Issue-comment threads of a PR and the PR itself share the same
    number, so the one digest covers both faces of the thread.
    """
    try:
        async with httpx.AsyncClient(timeout=8, headers=_headers(token)) as client:
            issue_resp = await client.get(f"{_API}/repos/{repo}/issues/{number}")
            issue_resp.raise_for_status()
            issue = issue_resp.json()
            # The comments API only sorts ascending; one max-size page then the
            # tail. >100-comment threads lose the middle, which is the part
            # that matters least.
            comments_resp = await client.get(
                f"{_API}/repos/{repo}/issues/{number}/comments", params={"per_page": 100}
            )
            comments_resp.raise_for_status()
            comments = comments_resp.json()[-_COMMENTS:]
    except Exception as exc:  # noqa: BLE001 — digest is best-effort by contract
        log.warning("thread digest failed for %s#%s: %s", repo, number, exc)
        return None
    if not isinstance(issue, dict) or not isinstance(comments, list):
        return None
    labels = ", ".join(str((lb or {}).get("name") or "") for lb in issue.get("labels") or [] if lb)
    milestone = ((issue.get("milestone") or {}).get("title")) or ""
    assignees = ", ".join(
        "@" + str((a or {}).get("login") or "") for a in issue.get("assignees") or [] if a
    )
    head = f"[Thread state] {repo}#{number} ({issue.get('state')}): {issue.get('title') or ''}"
    meta = "; ".join(
        part
        for part in (
            f"labels: {labels}" if labels else "",
            f"milestone: {milestone}" if milestone else "",
            f"assignees: {assignees}" if assignees else "",
        )
        if part
    )
    lines = [head]
    if meta:
        lines.append(meta)
    body = _clip(str(issue.get("body") or ""), _BODY_CAP)
    if body:
        lines.append(f"Opened by @{(issue.get('user') or {}).get('login') or '?'}: {body}")
    if comments:
        lines.append(f"Last {len(comments)} comment(s):")
        for c in comments:
            if not isinstance(c, dict):
                continue
            author = (c.get("user") or {}).get("login") or "?"
            lines.append(f"@{author}: {_clip(str(c.get('body') or ''), _COMMENT_CAP)}")
    return _clip("\n".join(lines), _TOTAL_CAP)


if __name__ == "__main__":
    # ponytail: self-check the formatting/clipping logic; no network.
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    issue = {
        "state": "open",
        "title": "It breaks",
        "body": "b" * 2000,
        "user": {"login": "alice"},
        "labels": [{"name": "bug"}],
        "milestone": {"title": "M1"},
        "assignees": [{"login": "bob"}],
    }
    comments = [{"user": {"login": f"u{i}"}, "body": f"c{i}"} for i in range(15)]

    def _resp(data):
        r = MagicMock()
        r.json.return_value = data
        r.raise_for_status.return_value = None
        return r

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(side_effect=[_resp(issue), _resp(comments)])
    with patch("httpx.AsyncClient", return_value=client):
        digest = asyncio.run(thread_digest("acme/w", 7, "tok"))
    assert digest is not None and digest.startswith("[Thread state] acme/w#7 (open): It breaks")
    assert "labels: bug" in digest and "milestone: M1" in digest and "@bob" in digest
    assert "@u5:" in digest and "@u14:" in digest and "@u4:" not in digest  # last 10 only
    assert "b" * 1500 not in digest  # body clipped
    assert len(digest) <= 6000

    # Any failure → None, never a raise.
    client.get = AsyncMock(side_effect=httpx.ConnectError("down"))
    with patch("httpx.AsyncClient", return_value=client):
        assert asyncio.run(thread_digest("acme/w", 7, "tok")) is None

    print("github_digest.py self-check OK")
