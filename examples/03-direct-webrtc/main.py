"""Speak a WAV file through an Ojin avatar published straight into a WebRTC room.

Usage:
    python main.py [INPUT.wav]

Needs OJIN_API_KEY and OJIN_CONFIG_ID, plus the room the avatar should join:

    OJIN_WEBRTC_PROVIDER   daily | livekit            (default: daily)
    OJIN_WEBRTC_ROOM_URL   Daily room URL, or LiveKit server URL (wss://...)
    OJIN_WEBRTC_TOKEN      credential for the `ojin-avatar` participant

Open the room in a browser first, then run this: the avatar joins as `ojin-avatar`
and speaks the clip. It is the same program you would write for a WebSocket
session — the only difference is the `webrtc=` argument.
"""

import asyncio
import os
import pathlib
import sys
import wave

from ojin import Credentials, MissingCredentialsError, load_env, resolve_credentials
from ojin.stv import OjinSTVClient, STVEvent, WebRTCSettings

JOIN_TIMEOUT_S = 30.0  # covers the room join AND a model cold start


def read_mono_wav(path: pathlib.Path) -> tuple[bytes, int]:
    """Read a mono 16-bit PCM WAV as (pcm_bytes, sample_rate), or exit with a hint."""
    if not path.exists():
        sys.exit(
            f"\n  No audio file at '{path}'.\n"
            "  Pass one:  python main.py myvoice.wav\n"
            "  Need a WAV? Convert anything with ffmpeg:\n"
            f"    ffmpeg -i input.mp3 -ac 1 -ar 24000 {path}\n"
        )
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            sys.exit(
                f"\n  '{path}' must be MONO 16-bit PCM. Convert it:\n"
                f"    ffmpeg -i '{path}' -ac 1 -ar 24000 mono.wav\n"
            )
        return wav.readframes(wav.getnframes()), wav.getframerate()


def webrtc_settings_from_env(sample_rate: int) -> WebRTCSettings:
    """Build the room settings from the environment, or exit with a hint."""
    room_url = os.environ.get("OJIN_WEBRTC_ROOM_URL", "")
    token = os.environ.get("OJIN_WEBRTC_TOKEN", "")
    if not room_url or not token:
        sys.exit(
            "\n  Set OJIN_WEBRTC_ROOM_URL and OJIN_WEBRTC_TOKEN (and\n"
            "  OJIN_WEBRTC_PROVIDER=livekit for LiveKit). See README.md in this\n"
            "  folder for how to create a room and the avatar's token.\n"
        )
    try:
        return WebRTCSettings(
            provider=os.environ.get("OJIN_WEBRTC_PROVIDER", "daily"),
            room_url=room_url,
            token=token,
            audio_sample_rate=sample_rate,  # publish at the clip's own rate
            webrtc_join_timeout_s=JOIN_TIMEOUT_S,
        )
    except ValueError as exc:
        sys.exit(f"\n  Invalid WebRTC settings: {exc}\n")


async def speak(
    creds: Credentials, settings: WebRTCSettings, pcm: bytes, rate: int
) -> None:
    """Have the avatar join the room and speak `pcm`, then leave."""
    seconds = len(pcm) / (rate * 2)  # mono 16-bit, for the finish timeout
    joined, done = asyncio.Event(), asyncio.Event()
    error: dict[str, str] = {}

    def on_error(message: str = "", code: str = "", **_: object) -> None:
        """Capture a fatal error and unblock the waiters."""
        error["message"] = f"{code}: {message}" if code else message
        joined.set()
        done.set()

    client = OjinSTVClient(
        api_key=creds.api_key,
        config_id=creds.config_id,
        webrtc=settings,  # <- the only WebRTC-specific line
    )
    client.add_listener(STVEvent.WEBRTC_CONNECTED, lambda **_: joined.set())
    client.add_listener(STVEvent.BOT_STOPPED_SPEAKING, lambda **_: done.set())
    client.add_listener(STVEvent.ERROR, on_error)

    async with client:
        # Audio sent before the avatar has joined is held and played once it has.
        await client.say(pcm, sample_rate=rate, num_channels=1)
        await joined.wait()  # the join timeout surfaces as an ERROR, so this returns
        if not error:
            print(f"  Avatar joined the {settings.provider} room — speaking...")
            try:
                await asyncio.wait_for(done.wait(), timeout=seconds + 30)
            except asyncio.TimeoutError:
                print("  No end-of-speech from the server in time — leaving.")

    if error:
        sys.exit(f"\n  Stopped: {error['message']}\n")
    print("  Done.\n")


def main() -> None:
    """Resolve credentials and room settings, then speak the clip in the room."""
    load_env(base_dir=pathlib.Path(__file__).parent)  # optional .env beside this file
    try:
        creds = resolve_credentials(load_env_file=False)
    except MissingCredentialsError as exc:
        sys.exit(str(exc))

    in_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "input.wav")
    pcm, rate = read_mono_wav(in_path)
    settings = webrtc_settings_from_env(rate)
    print(f"  Driving Face model '{creds.config_id}' into {settings.room_url} ...")
    asyncio.run(speak(creds, settings, pcm, rate))


if __name__ == "__main__":
    main()
