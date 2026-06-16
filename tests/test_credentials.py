"""Tests for ojin.credentials."""

import os

import pytest

from ojin import (
    Credentials,
    MissingCredentialsError,
    load_env,
    resolve_credentials,
)


def test_resolve_returns_credentials(monkeypatch):
    """Both env vars present -> a Credentials with those values."""
    monkeypatch.setenv("OJIN_API_KEY", "key-123")
    monkeypatch.setenv("OJIN_CONFIG_ID", "cfg-456")
    assert resolve_credentials(load_env_file=False) == Credentials("key-123", "cfg-456")


def test_resolve_missing_both_raises_with_guidance(monkeypatch):
    """Missing both vars -> error naming each var and where to get them."""
    monkeypatch.delenv("OJIN_API_KEY", raising=False)
    monkeypatch.delenv("OJIN_CONFIG_ID", raising=False)
    with pytest.raises(MissingCredentialsError) as exc:
        resolve_credentials(load_env_file=False)
    message = str(exc.value)
    assert "OJIN_API_KEY" in message
    assert "OJIN_CONFIG_ID" in message
    assert "ojin.ai" in message


def test_resolve_reports_only_the_missing_one(monkeypatch):
    """Only the absent variable is listed as missing."""
    monkeypatch.setenv("OJIN_API_KEY", "key-123")
    monkeypatch.delenv("OJIN_CONFIG_ID", raising=False)
    with pytest.raises(MissingCredentialsError) as exc:
        resolve_credentials(load_env_file=False)
    first_line = str(exc.value).strip().splitlines()[0]
    assert "OJIN_CONFIG_ID" in first_line
    assert "OJIN_API_KEY" not in first_line


def test_load_env_sets_absent_vars(monkeypatch, tmp_path):
    """load_env populates variables that aren't already set."""
    monkeypatch.delenv("OJIN_API_KEY", raising=False)
    monkeypatch.delenv("OJIN_CONFIG_ID", raising=False)
    (tmp_path / ".env").write_text(
        'OJIN_API_KEY="abc"\n# a comment\nOJIN_CONFIG_ID=def\n'
    )
    load_env(base_dir=tmp_path)
    assert os.environ["OJIN_API_KEY"] == "abc"
    assert os.environ["OJIN_CONFIG_ID"] == "def"


def test_load_env_does_not_override_existing(monkeypatch, tmp_path):
    """An existing environment value wins over the .env file."""
    monkeypatch.setenv("OJIN_API_KEY", "real")
    (tmp_path / ".env").write_text("OJIN_API_KEY=from-file\n")
    load_env(base_dir=tmp_path)
    assert os.environ["OJIN_API_KEY"] == "real"


def test_load_env_missing_file_is_noop(tmp_path):
    """A missing .env file is silently ignored."""
    load_env(base_dir=tmp_path)  # no .env here; must not raise
