"""Guard the public ojin.stv export surface."""

from ojin import stv


def test_public_exports_present() -> None:
    """Every name in __all__ is importable from the package."""
    for name in stv.__all__:
        assert hasattr(stv, name), f"missing export: {name}"


def test_core_client_exported() -> None:
    """The headline client + config + event enum are exported."""
    assert {"OjinSTVClient", "STVConfig", "STVEvent"} <= set(stv.__all__)


def test_webrtc_surface_exported() -> None:
    """The direct-WebRTC client, settings and events are public API."""
    assert {"OjinSTVWebRTCClient", "WebRTCSettings"} <= set(stv.__all__)
    assert stv.STVEvent.WEBRTC_CONNECTED.value == "webrtc_connected"
    assert stv.STVEvent.FIRST_FRAME.value == "first_frame"
