#!/usr/bin/env python3
"""Contacts CLI — query providers directly (Google + CardDAV).

Usage examples:
  python3 /app/tools/contacts.py list --provider google --output json
  python3 /app/tools/contacts.py search --provider icloud --query "Alice" --output json
  python3 /app/tools/contacts.py get --provider google --id people/c123 --output json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import requests
import vobject
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import _resolve_env_vars  # noqa: E402
from tools.contacts_auth import get_google_access_token  # noqa: E402


def _load_contacts_providers_from_db(db_path: str) -> dict[str, dict]:
    try:
        db = sqlite3.connect(db_path)
        row = db.execute(
            "SELECT value FROM config WHERE key = ?", ("contacts.providers",)
        ).fetchone()
        db.close()
    except Exception:
        return {}
    if not row:
        return {}
    try:
        providers = json.loads(row[0])
    except Exception:
        return {}
    if not isinstance(providers, list):
        return {}
    return {p.get("name", ""): p for p in providers if isinstance(p, dict)}


def _load_contacts_providers(
    config_path: str = "config.yml", db_path: str = "data/config.db"
) -> dict[str, dict]:
    providers = _load_contacts_providers_from_db(db_path)
    if providers:
        return providers
    load_dotenv()
    path = Path(config_path)
    if not path.exists():
        print("Error: config.yml not found", file=sys.stderr)
        sys.exit(1)
    raw = yaml.safe_load(path.read_text()) or {}
    resolved = _resolve_env_vars(raw)
    providers = resolved.get("contacts", {}).get("providers", [])
    return {p.get("name", ""): p for p in providers if isinstance(p, dict)}


def _flatten_vcard(card: vobject.base.Component) -> dict[str, object]:
    fn = ""
    if hasattr(card, "fn"):
        fn = str(card.fn.value)
    names = []
    if hasattr(card, "n"):
        n = card.n.value
        parts = [n.given, n.additional, n.family]
        names = [p for p in parts if p]
    phones = []
    if hasattr(card, "tel"):
        tel_items = card.tel if isinstance(card.tel, list) else [card.tel]
        for tel in tel_items:
            phones.append(str(tel.value))
    emails = []
    if hasattr(card, "email"):
        email_items = card.email if isinstance(card.email, list) else [card.email]
        for em in email_items:
            emails.append(str(em.value))
    return {
        "full_name": fn,
        "name_parts": names,
        "phones": phones,
        "emails": emails,
    }


# CardDAV over plain WebDAV (PROPFIND / GET / PUT). The `caldav` library shipped
# here has no CardDAV support, so we speak the protocol directly — which also works
# against simple WebDAV address books such as Purelymail's.
_DAV_NS = "{DAV:}"


def _carddav_session(provider: dict) -> requests.Session:
    s = requests.Session()
    s.auth = (provider.get("username", ""), provider.get("password", ""))
    return s


def _carddav_base(provider: dict) -> str:
    return provider.get("url", "").rstrip("/") + "/"


def _same_origin_url(base: str, href: str) -> str | None:
    """Resolve ``href`` against ``base`` only if it stays on the same origin.

    A CardDAV server response controls the hrefs we then GET. Because the request
    session carries the account's basic-auth on *every* request, an absolute href
    to another host (or an https→http downgrade) would leak those credentials to
    an attacker-controlled endpoint. So we reject anything whose scheme+host does
    not match the configured base, and only follow same-origin resources.
    """
    from urllib.parse import urljoin, urlsplit

    full = urljoin(base, href)
    b, f = urlsplit(base), urlsplit(full)
    if (f.scheme, f.netloc) != (b.scheme, b.netloc):
        return None
    return full


def _parse_propfind_hrefs(xml_text: str) -> list[str]:
    """Pull member hrefs out of a PROPFIND multistatus response."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    return [
        href.text
        for resp in root.findall(f"{_DAV_NS}response")
        for href in resp.findall(f"{_DAV_NS}href")
        if href.text
    ]


def _carddav_list(provider: dict) -> list[dict[str, object]]:
    base = _carddav_base(provider)
    s = _carddav_session(provider)
    body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<d:propfind xmlns:d="DAV:"><d:prop><d:getcontenttype/></d:prop></d:propfind>'
    )
    resp = s.request(
        "PROPFIND",
        base,
        data=body,
        headers={"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
        timeout=30,
    )
    resp.raise_for_status()
    contacts: list[dict[str, object]] = []
    for href in _parse_propfind_hrefs(resp.text):
        if not href.lower().endswith(".vcf"):
            continue
        full = _same_origin_url(base, href)  # never follow a cross-origin href (leaks creds)
        if not full:
            continue
        got = s.get(full, timeout=30)
        if got.status_code != 200:
            continue
        try:
            card = vobject.readOne(got.text)
        except Exception:
            continue
        data = _flatten_vcard(card)
        data["id"] = href.rsplit("/", 1)[-1]
        data["source"] = provider.get("name", "")
        contacts.append(data)
    return contacts


def _carddav_search(provider: dict, query: str) -> list[dict[str, object]]:
    query_lower = query.lower()
    return [
        c
        for c in _carddav_list(provider)
        if query_lower in (c.get("full_name") or "").lower()
        or any(query_lower in str(p).lower() for p in c.get("phones", []))
        or any(query_lower in str(e).lower() for e in c.get("emails", []))
    ]


def _carddav_get(provider: dict, contact_id: str) -> dict[str, object] | None:
    full = _same_origin_url(_carddav_base(provider), contact_id)
    if not full:
        return None
    s = _carddav_session(provider)
    got = s.get(full, timeout=30)
    if got.status_code != 200:
        return None
    try:
        card = vobject.readOne(got.text)
    except Exception:
        return None
    data = _flatten_vcard(card)
    data["id"] = contact_id.rsplit("/", 1)[-1]
    data["source"] = provider.get("name", "")
    return data


def _build_vcard(uid: str, name: str, email: str, phone: str, org: str) -> str:
    """Serialise a minimal vCard (FN/N + optional EMAIL/TEL/ORG)."""
    card = vobject.vCard()
    card.add("fn").value = name
    parts = name.split()
    n = card.add("n")
    n.value = vobject.vcard.Name(
        family=" ".join(parts[1:]) if len(parts) > 1 else "", given=parts[0] if parts else name
    )
    card.add("uid").value = uid
    if email:
        e = card.add("email")
        e.value = email
        e.type_param = "INTERNET"
    if phone:
        t = card.add("tel")
        t.value = phone
        t.type_param = "CELL"
    if org:
        card.add("org").value = [org]
    return card.serialize()


def _carddav_add(provider: dict, name: str, email: str, phone: str, org: str) -> dict[str, object]:
    import uuid
    from urllib.parse import urljoin

    uid = str(uuid.uuid4())
    resource = f"{uid}.vcf"
    body = _build_vcard(uid, name, email, phone, org)
    s = _carddav_session(provider)
    full = urljoin(_carddav_base(provider), resource)
    put = s.put(
        full,
        data=body.encode("utf-8"),
        headers={"Content-Type": "text/vcard; charset=utf-8"},
        timeout=30,
    )
    if put.status_code not in (200, 201, 204):
        return {"error": f"CardDAV PUT failed: HTTP {put.status_code} {put.text[:200]}"}
    return {"ok": True, "id": resource, "full_name": name, "source": provider.get("name", "")}


def _google_headers(db_path: str) -> dict[str, str]:
    token = get_google_access_token(db_path=db_path)
    return {"Authorization": f"Bearer {token}"}


def _google_list(provider: dict, db_path: str) -> list[dict[str, object]]:
    url = "https://people.googleapis.com/v1/people/me/connections"
    params = {
        "personFields": "names,emailAddresses,phoneNumbers",
        "pageSize": 1000,
    }
    contacts: list[dict[str, object]] = []
    page_token = None
    while True:
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(url, headers=_google_headers(db_path), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for person in data.get("connections", []) or []:
            contacts.append(_google_person_to_contact(person, provider))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return contacts


def _google_person_to_contact(person: dict, provider: dict) -> dict[str, object]:
    names = person.get("names") or []
    full_name = names[0].get("displayName") if names else ""
    phones = [p.get("value") for p in (person.get("phoneNumbers") or []) if p.get("value")]
    emails = [e.get("value") for e in (person.get("emailAddresses") or []) if e.get("value")]
    return {
        "id": person.get("resourceName", ""),
        "full_name": full_name or "",
        "phones": phones,
        "emails": emails,
        "source": provider.get("name", ""),
    }


def _google_search(provider: dict, query: str, db_path: str) -> list[dict[str, object]]:
    url = "https://people.googleapis.com/v1/people:searchContacts"
    params = {
        "query": query,
        "readMask": "names,emailAddresses,phoneNumbers",
        "pageSize": 30,
    }
    resp = requests.get(url, headers=_google_headers(db_path), params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    contacts = []
    for result in results:
        person = result.get("person") or {}
        contacts.append(_google_person_to_contact(person, provider))
    return contacts


def _google_get(provider: dict, contact_id: str, db_path: str) -> dict[str, object] | None:
    url = f"https://people.googleapis.com/v1/{contact_id}"
    params = {"personFields": "names,emailAddresses,phoneNumbers"}
    resp = requests.get(url, headers=_google_headers(db_path), params=params, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return _google_person_to_contact(resp.json(), provider)


def _select_provider(providers: dict[str, dict], name: str) -> dict:
    if name not in providers:
        available = ", ".join(sorted(providers.keys())) if providers else "(none configured)"
        print(f"Error: provider '{name}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)
    return providers[name]


def _is_google(provider: dict) -> bool:
    return provider.get("type") == "google_contacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Contacts CLI")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    parser.add_argument("--db", default="data/config.db", help="Path to config DB")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_cmd = sub.add_parser("list", help="List all contacts")
    list_cmd.add_argument("--provider", "-p", required=True)
    list_cmd.add_argument("--output", "-o", choices=["json", "text"], default="json")

    search_cmd = sub.add_parser("search", help="Search contacts")
    search_cmd.add_argument("--provider", "-p", required=True)
    search_cmd.add_argument("--query", "-q", required=True)
    search_cmd.add_argument("--output", "-o", choices=["json", "text"], default="json")

    get_cmd = sub.add_parser("get", help="Get contact details")
    get_cmd.add_argument("--provider", "-p", required=True)
    get_cmd.add_argument("--id", required=True)
    get_cmd.add_argument("--output", "-o", choices=["json", "text"], default="json")

    add_cmd = sub.add_parser("add", help="Create a contact (CardDAV only)")
    add_cmd.add_argument("--provider", "-p", required=True)
    add_cmd.add_argument("--name", required=True)
    add_cmd.add_argument("--email", default="")
    add_cmd.add_argument("--phone", default="")
    add_cmd.add_argument("--org", default="")
    add_cmd.add_argument("--output", "-o", choices=["json", "text"], default="json")

    args = parser.parse_args()
    providers = _load_contacts_providers(args.config, args.db)
    provider = _select_provider(providers, args.provider)

    if args.cmd == "list":
        if _is_google(provider):
            results = _google_list(provider, db_path=args.db)
        else:
            results = _carddav_list(provider)
    elif args.cmd == "search":
        if _is_google(provider):
            results = _google_search(provider, args.query, db_path=args.db)
        else:
            results = _carddav_search(provider, args.query)
    elif args.cmd == "add":
        if _is_google(provider):
            print(
                json.dumps({"error": "Google Contacts is read-only (OAuth scope)."}),
                file=sys.stderr,
            )
            sys.exit(1)
        results = _carddav_add(provider, args.name, args.email, args.phone, args.org)
        if isinstance(results, dict) and results.get("error"):
            print(json.dumps(results, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)
    else:
        if _is_google(provider):
            results = _google_get(provider, args.id, db_path=args.db)
        else:
            results = _carddav_get(provider, args.id)

    if args.output == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        if isinstance(results, list):
            for item in results:
                name = item.get("full_name") or "(no name)"
                phones = ", ".join(item.get("phones", []))
                emails = ", ".join(item.get("emails", []))
                print(f"{name}  {phones}  {emails}")
        elif isinstance(results, dict):
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print("Not found")


if __name__ == "__main__":
    main()
