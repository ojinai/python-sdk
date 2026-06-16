"""Resolve and validate Ojin credentials from the environment.

Small, dependency-free helpers so applications and examples don't each reinvent
reading ``OJIN_API_KEY`` / ``OJIN_CONFIG_ID`` (and an optional ``.env`` file).

Typical use::

    from ojin import resolve_credentials
    from ojin.stv import OjinSTVClient

    creds = resolve_credentials()  # raises MissingCredentialsError if unset
    client = OjinSTVClient(api_key=creds.api_key, config_id=creds.config_id)
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

API_KEY_ENV = "OJIN_API_KEY"
CONFIG_ID_ENV = "OJIN_CONFIG_ID"

_SETUP_HELP = (
    "  Get them from your Ojin account (https://ojin.ai):\n"
    f"    - {API_KEY_ENV}: authenticates you (create one in your API keys).\n"
    f"    - {CONFIG_ID_ENV}: the Face/avatar model to drive (copy its config id).\n"
    "  Docs: https://docs.ojin.ai  (new accounts get $10 in free credits)\n\n"
    "  Set them in the environment or a .env file:\n"
    f"    export {API_KEY_ENV}=...\n"
    f"    export {CONFIG_ID_ENV}=...\n"
)


class MissingCredentialsError(RuntimeError):
    """Raised when required Ojin credentials are not set.

    The message includes step-by-step instructions for creating them, so callers
    can simply surface ``str(error)`` to the user.
    """


@dataclass(frozen=True)
class Credentials:
    """Resolved Ojin credentials."""

    api_key: str
    config_id: str


def load_env(
    path: str | os.PathLike = ".env", *, base_dir: str | os.PathLike | None = None
) -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Existing environment variables take precedence (values are only set when
    absent), and a missing file is silently ignored. No third-party dependency
    required.

    Args:
        path: Path to the env file, relative to ``base_dir`` when given.
        base_dir: Directory to resolve a relative ``path`` against. Pass
            ``pathlib.Path(__file__).parent`` to load a ``.env`` next to a script
            regardless of the working directory.

    """
    if base_dir is not None:
        env_path = pathlib.Path(base_dir, path)
    else:
        env_path = pathlib.Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_credentials(*, load_env_file: bool = True) -> Credentials:
    """Read and validate Ojin credentials from the environment.

    Args:
        load_env_file: When True (default), first load a ``.env`` file from the
            current working directory via :func:`load_env`. Set False if you have
            already loaded your environment (e.g. with a custom ``base_dir``).

    Returns:
        The resolved :class:`Credentials`.

    Raises:
        MissingCredentialsError: If ``OJIN_API_KEY`` or ``OJIN_CONFIG_ID`` is
            missing. The error message explains how to create them.

    """
    if load_env_file:
        load_env()
    api_key = os.environ.get(API_KEY_ENV)
    config_id = os.environ.get(CONFIG_ID_ENV)
    if api_key and config_id:
        return Credentials(api_key=api_key, config_id=config_id)

    missing = ", ".join(
        name
        for name, value in ((API_KEY_ENV, api_key), (CONFIG_ID_ENV, config_id))
        if not value
    )
    raise MissingCredentialsError(f"\n  Missing {missing}.\n\n{_SETUP_HELP}")
