"""CardDAV helpers in tools/contacts.py — vCard build + PROPFIND parse (#110)."""

from __future__ import annotations

import vobject

from tools.contacts import (
    _build_vcard,
    _flatten_vcard,
    _parse_propfind_hrefs,
    _same_origin_url,
)


def test_build_vcard_roundtrips() -> None:
    raw = _build_vcard("uid-1", "Alice Smith", "alice@x.io", "+15551234", "Acme")
    card = vobject.readOne(raw)
    flat = _flatten_vcard(card)
    assert flat["full_name"] == "Alice Smith"
    assert "alice@x.io" in flat["emails"]
    assert any("5551234" in p for p in flat["phones"])
    assert "UID:uid-1" in raw and "ORG:Acme" in raw


def test_build_vcard_minimal_no_optionals() -> None:
    raw = _build_vcard("uid-2", "Bob", "", "", "")
    card = vobject.readOne(raw)
    flat = _flatten_vcard(card)
    assert flat["full_name"] == "Bob"
    assert flat["emails"] == [] and flat["phones"] == []


def test_parse_propfind_hrefs() -> None:
    xml = (
        '<?xml version="1.0"?><d:multistatus xmlns:d="DAV:">'
        "<d:response><d:href>/dav/ab/uid-1.vcf</d:href></d:response>"
        "<d:response><d:href>/dav/ab/</d:href></d:response>"
        "</d:multistatus>"
    )
    assert _parse_propfind_hrefs(xml) == ["/dav/ab/uid-1.vcf", "/dav/ab/"]


def test_parse_propfind_hrefs_bad_xml() -> None:
    assert _parse_propfind_hrefs("not xml") == []
    assert _parse_propfind_hrefs("") == []


def test_same_origin_url_blocks_credential_leaks() -> None:
    base = "https://carddav.example.com/dav/book/"
    # Same-origin relative + absolute-path hrefs are followed.
    assert _same_origin_url(base, "uid-1.vcf") == "https://carddav.example.com/dav/book/uid-1.vcf"
    assert (
        _same_origin_url(base, "/dav/book/uid-2.vcf")
        == "https://carddav.example.com/dav/book/uid-2.vcf"
    )
    # Cross-origin, protocol-relative, and https→http downgrades are rejected —
    # otherwise the session's basic-auth would be sent to the attacker's host.
    assert _same_origin_url(base, "https://evil.com/x.vcf") is None
    assert _same_origin_url(base, "//evil.com/x.vcf") is None
    assert _same_origin_url(base, "http://carddav.example.com/x.vcf") is None
