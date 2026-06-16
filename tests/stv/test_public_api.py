"""Guard the public ojin.stv export surface."""

from ojin import stv


def test_public_exports_present() -> None:
    """Every name in __all__ is importable from the package."""
    for name in stv.__all__:
        assert hasattr(stv, name), f"missing export: {name}"


def test_core_client_exported() -> None:
    """The headline client + config + event enum are exported."""
    assert {"OjinSTVClient", "STVConfig", "STVEvent"} <= set(stv.__all__)
