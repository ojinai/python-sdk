"""Generate a talking-avatar MP4 (with audio) from an audio file via the Ojin STV model.

Usage:
    python main.py [INPUT.wav|INPUT.mp3|...] [OUTPUT.mp4]

Reads speech audio (WAV, MP3, M4A, ... — anything ffmpeg can open), drives an Ojin
avatar with it, and writes a lip-synced MP4 with the original audio muxed in. Needs
OJIN_API_KEY and OJIN_CONFIG_ID — the preflight prints how to get them if they're
missing.
"""

import asyncio
import pathlib
import sys

from audio_input import AudioDecodeError, load_audio
from mp4_writer import Mp4Writer

from ojin import MissingCredentialsError, load_env, resolve_credentials
from ojin.stv import OjinSTVClient, QueueOutput, STVEvent, STVVideoFrame

FPS = 25


def read_audio(path: pathlib.Path) -> tuple[bytes, int]:
    """Read any audio file as mono 16-bit (pcm_bytes, sample_rate), or exit with a hint."""
    if not path.exists():
        sys.exit(
            f"\n  No audio file at '{path}'.\n"
            "  Pass one:  python main.py myvoice.mp3\n"
            "  WAV, MP3, M4A, FLAC, OGG — anything ffmpeg can read works.\n"
        )
    try:
        return load_audio(path)
    except AudioDecodeError as exc:
        sys.exit(
            f"\n  Could not read any audio out of '{path}':\n"
            f"    {exc}\n"
            "  Is it really an audio file? Try converting it first:\n"
            f"    ffmpeg -i '{path}' -ac 1 -ar 16000 speech.wav\n"
        )


async def render(
    creds, pcm: bytes, rate: int, audio_path: pathlib.Path, out: pathlib.Path
) -> None:
    """Drive the avatar with `pcm` and write its frames + `audio_path` audio to `out`."""
    seconds = len(pcm) / (rate * 2)  # mono 16-bit, for the finish timeout
    ready, done, error = asyncio.Event(), asyncio.Event(), {}
    writer: Mp4Writer | None = None

    def on_error(message: str = "", **_: object) -> None:
        """Capture a fatal error and unblock the waiters."""
        error["message"] = message
        ready.set()
        done.set()

    # This is an offline render, so use an effectively unbounded video buffer:
    # we never want frames dropped (the default QueueOutput caps at 60).
    client = OjinSTVClient(
        api_key=creds.api_key,
        config_id=creds.config_id,
        output=QueueOutput(max_video=10**9),
    )
    client.add_listener(STVEvent.SESSION_READY, lambda **_: ready.set())
    client.add_listener(STVEvent.BOT_STOPPED_SPEAKING, lambda **_: done.set())
    client.add_listener(STVEvent.ERROR, on_error)

    await client.start()
    try:
        try:
            await asyncio.wait_for(ready.wait(), timeout=30)
        except asyncio.TimeoutError:
            sys.exit(
                "\n  The session did not become ready within 30s.\n"
                "  Connected, but the server never sent SESSION_READY — check your\n"
                "  network and that OJIN_CONFIG_ID points to a valid Face model.\n"
            )
        if error:
            sys.exit(
                f"\n  Ojin could not start the session: {error['message']}\n"
                "  Double-check OJIN_API_KEY and OJIN_CONFIG_ID.\n"
            )

        await client.say(pcm, sample_rate=rate, num_channels=1)

        async def consume() -> None:
            """Feed each returned RGB frame to the MP4 writer (built on the first one)."""
            nonlocal writer
            async for frame in client.output_stream():
                if (
                    isinstance(frame, STVVideoFrame)
                    and frame.rgb is not None
                    and frame.frame_type != 0  # speech frames only
                ):
                    if writer is None:
                        # Size the MP4 to whatever the server sends (first frame wins).
                        writer = Mp4Writer(
                            out,
                            frame.width,
                            frame.height,
                            fps=FPS,
                            audio_path=audio_path,
                        )
                    writer.write(frame.rgb)
                    print(
                        f"\r  rendering... {writer.frames} frames", end="", flush=True
                    )

        consumer = asyncio.create_task(consume())
        try:
            # The avatar plays in real time, so it finishes ~`seconds` after we send
            # it; allow generous margin, then finalize whatever we captured. (Too
            # short/quiet audio can make the server emit only still frames and never
            # a stopped-speaking event.)
            await asyncio.wait_for(done.wait(), timeout=seconds + 30)
        except asyncio.TimeoutError:
            print(
                "\n  No end-of-speech from the server in time — finalizing what was\n"
                "  rendered (the audio may be too short or too quiet to drive speech)."
            )
        await asyncio.sleep(0.5)  # let the final frames drain
        consumer.cancel()
    finally:
        await client.close()
        if writer is not None:
            writer.close()

    if error:
        sys.exit(f"\n  Stopped: {error['message']}\n")
    if writer is None:
        sys.exit(
            "\n  No video frames were produced — the audio may be too short or too\n"
            "  quiet to drive speech. Try a longer, louder clip.\n"
        )
    secs = writer.frames / FPS
    print(f"\n  Done -> {out}  ({writer.frames} frames, {secs:.1f}s, audio included)\n")


def main() -> None:
    """Resolve credentials, then render the MP4."""
    load_env(base_dir=pathlib.Path(__file__).parent)  # optional .env beside this file
    try:
        creds = resolve_credentials(load_env_file=False)
    except MissingCredentialsError as exc:
        sys.exit(str(exc))

    in_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "input.wav")
    out_path = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "output.mp4")
    pcm, rate = read_audio(in_path)

    seconds = len(pcm) / (rate * 2)
    print(f"  Driving Face model '{creds.config_id}' with {seconds:.1f}s of audio...")
    asyncio.run(render(creds, pcm, rate, in_path, out_path))


if __name__ == "__main__":
    main()
