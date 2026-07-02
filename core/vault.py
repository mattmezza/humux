"""Encryption primitives + key management for the secrets vault (issue #19).

Two vaults, two keys — because the two kinds of secret have different unseal
requirements:

* :class:`InfraVault` — encrypts *infrastructure* secrets (provider keys, bot
  token, …). Its key is a **machine key** read from the environment / a keyfile,
  so the vault unseals at boot, headless, with no human present. Same on-disk
  posture as the existing ``.env`` (a key sitting on the data volume), just
  consolidated and editable in the UI.

* :class:`AgentVault` — encrypts *agent / login* secrets (website logins,
  payment keys, …). It uses **envelope encryption**: a random Data-Encryption
  Key (DEK) encrypts the secrets, and the DEK is stored *wrapped* by a
  Key-Encryption Key ``KEK = PBKDF2(admin_password, salt)``. The KEK is derived
  live at login (never stored), so a stolen disk alone cannot open the vault.
  Changing the admin password re-wraps the DEK only — the secrets are never
  re-encrypted. The vault stays locked until :meth:`AgentVault.unseal`.

This module is pure crypto + key handling: no database, no async. Storage,
ACL, and resolution live in :mod:`core.secret_store`.
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

# PBKDF2 work factor for deriving the agent KEK from the admin password.
# Matches the admin-auth hash cost (core/config_store.py); a *different* salt is
# used so the stored auth hash can never double as the encryption key.
KDF_ITERATIONS = 200_000

# Default on-disk location for the machine key when neither env var is set.
# Lives on the data volume (gitignored), so the infra vault unseals at boot
# without forcing the user to edit their environment.
DEFAULT_KEYFILE = "data/master.key"


def _coerce_fernet_key(value: str | bytes) -> bytes:
    """Turn any string into a valid Fernet key.

    A real 32-byte url-safe base64 Fernet key is used as-is; anything else
    (e.g. a human passphrase) is hashed to 32 bytes and base64-encoded. This
    lets the user set ``HUMUX_MASTER_KEY`` to either a generated key or a memorable
    passphrase.
    """
    raw = value.encode() if isinstance(value, str) else value
    try:
        Fernet(raw)  # validates length + base64 alphabet
        return raw
    except ValueError, TypeError:
        digest = hashlib.sha256(raw).digest()
        return base64.urlsafe_b64encode(digest)


def derive_kek(password: str, salt: bytes) -> bytes:
    """Derive a Fernet-compatible key-encryption key from a password + salt."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, KDF_ITERATIONS)
    return base64.urlsafe_b64encode(dk)


def load_machine_key() -> str | None:
    """Locate the infra-vault machine key.

    Order: ``HUMUX_MASTER_KEY`` env → ``HUMUX_MASTER_KEY_FILE`` path → the default
    keyfile on the data volume. Returns ``None`` when no key is configured (the
    infra vault is then unavailable and ``${vault:...}`` falls back to env).
    """
    env = os.getenv("HUMUX_MASTER_KEY")
    if env:
        return env
    keyfile = os.getenv("HUMUX_MASTER_KEY_FILE") or DEFAULT_KEYFILE
    path = Path(keyfile)
    if path.exists():
        text = path.read_text().strip()
        if text:
            return text
    return None


def generate_and_save_machine_key(keyfile: str = DEFAULT_KEYFILE) -> str:
    """Generate a fresh machine key and persist it to ``keyfile`` (0600).

    Used by the setup wizard so a user can enable the infra vault without
    hand-editing their environment. Never overwrites an existing keyfile.
    """
    path = Path(keyfile)
    if path.exists() and path.read_text().strip():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key().decode()
    # Create with 0600 atomically (no world-readable window between write + chmod).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key.encode())
    finally:
        os.close(fd)
    try:
        path.chmod(0o600)  # tighten if the file pre-existed with looser perms
    except OSError:
        pass  # best-effort on filesystems without POSIX perms
    return key


class InfraVault:
    """Symmetric encryption with a machine key (boot-time, headless)."""

    def __init__(self, key: str | bytes | None) -> None:
        self._fernet = Fernet(_coerce_fernet_key(key)) if key else None

    @property
    def available(self) -> bool:
        return self._fernet is not None

    def encrypt(self, plaintext: str) -> bytes:
        if self._fernet is None:
            raise RuntimeError("InfraVault has no key configured")
        return self._fernet.encrypt(plaintext.encode())

    def decrypt(self, token: bytes) -> str:
        if self._fernet is None:
            raise RuntimeError("InfraVault has no key configured")
        return self._fernet.decrypt(token).decode()


class AgentVault:
    """Envelope encryption: DEK encrypts data, KEK (from password) wraps the DEK.

    Locked until :meth:`unseal`. The DEK lives only in process memory once
    unsealed; it is never written to disk in usable form.
    """

    def __init__(self) -> None:
        self._fernet: Fernet | None = None

    @property
    def unsealed(self) -> bool:
        return self._fernet is not None

    @staticmethod
    def create_wrapped_dek(password: str) -> tuple[str, str]:
        """Mint a new random DEK wrapped by ``password``.

        Returns ``(wrapped_dek_b64, salt_b64)`` for persistence. Call once when
        the admin password is first set.
        """
        dek = Fernet.generate_key()  # this is itself a valid Fernet key
        salt = os.urandom(16)
        wrapped = Fernet(derive_kek(password, salt)).encrypt(dek)
        return base64.b64encode(wrapped).decode(), base64.b64encode(salt).decode()

    def unseal(self, password: str, wrapped_dek_b64: str, salt_b64: str) -> bool:
        """Unwrap the DEK with ``password`` and cache it. Returns success."""
        try:
            salt = base64.b64decode(salt_b64)
            wrapped = base64.b64decode(wrapped_dek_b64)
            dek = Fernet(derive_kek(password, salt)).decrypt(wrapped)
        except InvalidToken, ValueError, TypeError:
            return False
        self._fernet = Fernet(dek)
        return True

    @staticmethod
    def rewrap(
        old_password: str, new_password: str, wrapped_dek_b64: str, salt_b64: str
    ) -> tuple[str, str]:
        """Re-wrap the existing DEK under a new password (password rotation).

        Preserves the DEK so every stored secret stays decryptable; only the
        wrapping changes. Raises :class:`InvalidToken` if ``old_password`` is wrong.
        """
        salt = base64.b64decode(salt_b64)
        wrapped = base64.b64decode(wrapped_dek_b64)
        dek = Fernet(derive_kek(old_password, salt)).decrypt(wrapped)
        new_salt = os.urandom(16)
        new_wrapped = Fernet(derive_kek(new_password, new_salt)).encrypt(dek)
        return base64.b64encode(new_wrapped).decode(), base64.b64encode(new_salt).decode()

    def lock(self) -> None:
        self._fernet = None

    def encrypt(self, plaintext: str) -> bytes:
        if self._fernet is None:
            raise VaultLocked("agent vault is locked")
        return self._fernet.encrypt(plaintext.encode())

    def decrypt(self, token: bytes) -> str:
        if self._fernet is None:
            raise VaultLocked("agent vault is locked")
        return self._fernet.decrypt(token).decode()


class VaultLocked(RuntimeError):
    """Raised when an agent-vault operation is attempted while sealed."""


if __name__ == "__main__":
    # ponytail: one runnable check covering both vaults + the envelope invariants.
    # Infra vault round-trip.
    iv = InfraVault("a-memorable-passphrase")
    assert iv.available
    tok = iv.encrypt("sk-infra-123")
    assert iv.decrypt(tok) == "sk-infra-123"
    # A different key cannot read it.
    try:
        InfraVault("other").decrypt(tok)
        raise AssertionError("cross-key decrypt should fail")
    except InvalidToken:
        pass
    # No key configured -> unavailable.
    assert not InfraVault(None).available

    # Agent vault: locked until unsealed.
    pv = AgentVault()
    assert not pv.unsealed
    try:
        pv.encrypt("nope")
        raise AssertionError("locked vault must refuse encrypt")
    except VaultLocked:
        pass

    wrapped, salt = AgentVault.create_wrapped_dek("hunter2")
    assert pv.unseal("hunter2", wrapped, salt)
    assert pv.unsealed
    ct = pv.encrypt("stripe-key-xyz")
    assert pv.decrypt(ct) == "stripe-key-xyz"

    # Wrong password fails to unseal a fresh vault.
    assert not AgentVault().unseal("wrong", wrapped, salt)

    # Password rotation re-wraps the DEK but preserves stored data.
    new_wrapped, new_salt = AgentVault.rewrap("hunter2", "hunter3", wrapped, salt)
    pv2 = AgentVault()
    assert pv2.unseal("hunter3", new_wrapped, new_salt)
    assert pv2.decrypt(ct) == "stripe-key-xyz"  # same DEK -> old ciphertext still readable
    # Old password no longer unseals the rewrapped DEK.
    assert not AgentVault().unseal("hunter2", new_wrapped, new_salt)

    print("vault.py self-check OK")
