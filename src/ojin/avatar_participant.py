"""Identify the Ojin avatar participant in a direct-WebRTC room.

With ``OjinSTVClient(webrtc=...)`` the inference server joins your Daily or
LiveKit room and publishes the avatar's audio and video under a fixed
participant name. If your own bot also sits in that room, it must not listen to
that participant: most room transports (pipecat's ``DailyInputTransport``, for
one) auto-subscribe to *every* participant's microphone, so the bot would
capture the avatar's own voice and transcribe its greeting as user speech — a
self-hearing feedback loop that derails the whole session.

These helpers recognise the avatar so you can unsubscribe from its audio.

Zero-dependency on purpose (no numpy/opencv, unlike ``ojin.stv``) so a bot can
import it anywhere.

CONTRACT: ``AVATAR_PARTICIPANT_USER_NAME`` MUST match ``_AVATAR_USER_NAME`` in
``inference-server/server/webrtc/daily_room_publisher.py``, the identity the C++
LiveKit provider expects (``inference-server/cpp/src/webrtc/webrtc_livekit.cpp``),
and ``LIVEKIT_AVATAR_IDENTITY`` in
``core-api/src/agents/session/livekit-rooms.service.ts``. If one changes they all
must, or bots start capturing the avatar's audio again.

The contract covers WHEN the name is set as well as its value, and the two
providers arrive there very differently:

* **LiveKit**: identity is carried by the access token and there is no pre-join
  setter at all, so it is present in the first participant record *by
  construction* — the race described below cannot happen. The cost moves
  upstream instead: whoever mints the avatar token must set this identity,
  which is why the C++ provider cross-checks what it got back and warns loudly
  on mismatch.

* **Daily**: the server must call ``set_user_name()`` *before* ``join()``, so the
  name ships in the participant record every remote receives. Daily reports
  ``participant-joined`` to remotes before the joining client's own completion
  resolves (~740ms earlier in an observed session) and propagates a post-join
  rename only on its own schedule (~4.3s measured), so a name applied after join
  arrives long after a bot has begun capturing the avatar's mic.
"""

from __future__ import annotations

from typing import Any

# The name the inference server publishes the avatar under: a Daily ``userName``,
# and a LiveKit participant ``identity``. One constant, because it is one
# cross-service contract — see the module docstring.
AVATAR_PARTICIPANT_USER_NAME = "ojin-avatar"


def is_avatar_participant(participant: Any) -> bool:
    """Return True if a Daily participant dict is the inference server's avatar.

    Defensive against partial/malformed participant payloads: any shape that
    isn't a dict carrying ``info.userName == AVATAR_PARTICIPANT_USER_NAME``
    returns False (i.e. "treat as a real user, keep listening").
    """
    if not isinstance(participant, dict):
        return False
    info = participant.get("info")
    if not isinstance(info, dict):
        return False
    return info.get("userName") == AVATAR_PARTICIPANT_USER_NAME


def is_avatar_identity(identity: Any) -> bool:
    """Return True if a LiveKit participant identity is the avatar's.

    LiveKit has no participant dict — ``RemoteParticipant.identity`` is a plain
    string taken from the access token — so the Daily-shaped
    :func:`is_avatar_participant` does not apply. Same defensive posture: any
    shape that is not exactly the expected string returns False, i.e. "treat as
    a real user, keep listening".
    """
    return identity == AVATAR_PARTICIPANT_USER_NAME
