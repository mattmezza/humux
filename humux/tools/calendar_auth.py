"""Shared OAuth 2.0 helper for CalDAV calendar tools.

Loads Google OAuth token from the ConfigStore DB, refreshes the access
token as needed, and provides a `caldav.DAVClient` with Bearer auth.

Non-Google CalDAV providers still use Basic Auth (username/password).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time

import caldav
import requests

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
CONFIG_DB_PATH = "data/config.db"
TOKEN_DB_KEY = "calendar.google_oauth_token"

# In-memory cache: (access_token, expiry_timestamp)
_token_cache: tuple[str, float] | None = None


def _load_token_from_db(db_path: str = CONFIG_DB_PATH) -> dict | None:
    """Read the OAuth token JSON from the config store (sync)."""
    try:
        db = sqlite3.connect(db_path)
        row = db.execute("SELECT value FROM config WHERE key = ?", (TOKEN_DB_KEY,)).fetchone()
        db.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _refresh_access_token(token_data: dict) -> tuple[str, int]:
    """Use the refresh token to get a fresh access token."""
    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": token_data["client_id"],
            "client_secret": token_data["client_secret"],
            "refresh_token": token_data["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"], data.get("expires_in", 3600)


def get_access_token(db_path: str = CONFIG_DB_PATH) -> str:
    """Return a valid access token, refreshing from the DB token if expired."""
    global _token_cache
    if _token_cache:
        token, expiry = _token_cache
        if time.time() < expiry - 60:  # 60s safety margin
            return token

    token_data = _load_token_from_db(db_path)
    if not token_data:
        print(
            f"Error: no Google OAuth token found in DB (key: {TOKEN_DB_KEY}).",
            file=sys.stderr,
        )
        print(
            "Run: uv run python tools/google_oauth.py --client-id ... --client-secret ...",
            file=sys.stderr,
        )
        sys.exit(1)

    access_token, expires_in = _refresh_access_token(token_data)
    _token_cache = (access_token, time.time() + expires_in)
    return access_token


def connect_google(provider: dict) -> caldav.Calendar:
    """Connect to Google CalDAV using OAuth 2.0 Bearer token."""
    access_token = get_access_token()

    client = caldav.DAVClient(
        url=provider["url"],
        headers={"Authorization": f"Bearer {access_token}"},
    )
    principal = client.principal()
    calendars = principal.calendars()
    if not calendars:
        print("Error: no calendars found for this account", file=sys.stderr)
        sys.exit(1)
    return calendars[0]


def connect_basic(provider: dict) -> caldav.Calendar:
    """Connect to a CalDAV server using Basic Auth (for non-Google providers)."""
    client = caldav.DAVClient(
        url=provider["url"],
        username=provider.get("username", ""),
        password=provider.get("password", ""),
    )
    principal = client.principal()
    calendars = principal.calendars()
    if not calendars:
        print("Error: no calendars found for this account", file=sys.stderr)
        sys.exit(1)
    return calendars[0]


def connect(provider: dict) -> caldav.Calendar:
    """Connect to a CalDAV server, auto-detecting auth method.

    Uses OAuth 2.0 if `auth_type` is "oauth" or if the URL contains
    "google", otherwise falls back to Basic Auth.
    """
    auth_type = provider.get("auth_type", "")
    url = provider.get("url", "")

    if auth_type == "oauth" or (not auth_type and "google" in url.lower()):
        return connect_google(provider)
    return connect_basic(provider)
