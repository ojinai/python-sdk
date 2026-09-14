"""Guard the public ojin.stv export surface."""

import subprocess
import sys

import ojin
from ojin import stv


def test_public_exports_present() -> None:
    """Every name in __all__ is importable from the package."""
    for name in stv.__all__:
        assert hasattr(stv, name), f"missing export: {name}"
    for name in ojin.__all__:
        assert hasattr(ojin, name), f"missing export: {name}"


def test_core_client_exported() -> None:
    """The headline client + config + event enum are exported."""
    assert {"OjinSTVClient", "STVConfig", "STVEvent"} <= set(stv.__all__)


def test_webrtc_surface_exported() -> None:
    """The direct-WebRTC settings, provider enum, helpers and events are public."""
    assert {
        "OjinSTVWebRTCClient",
        "WebRTCProvider",
        "WebRTCSettings",
        "AVATAR_PARTICIPANT_USER_NAME",
        "is_avatar_identity",
        "is_avatar_participant",
    } <= set(stv.__all__)
    assert {
        "AVATAR_PARTICIPANT_USER_NAME",
        "is_avatar_identity",
        "is_avatar_participant",
    } <= set(ojin.__all__)
    assert stv.STVEvent.WEBRTC_CONNECTED.value == "webrtc_connected"
    assert stv.STVEvent.FIRST_FRAME.value == "first_frame"


def test_avatar_participant_helpers_import_without_media_dependencies() -> None:
    """Bots can import the avatar helpers without numpy/opencv being loaded."""
    code = (
        "import sys, ojin.avatar_participant as a; "
        "assert a.AVATAR_PARTICIPANT_USER_NAME == 'ojin-avatar'; "
        "loaded = {'cv2', 'numpy'} & set(sys.modules); "
        "assert not loaded, loaded"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
