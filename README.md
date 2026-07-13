# Ojin Python SDK

[![PyPI version](https://img.shields.io/pypi/v/ojin-client.svg)](https://pypi.org/project/ojin-client/)
[![Python versions](https://img.shields.io/pypi/pyversions/ojin-client.svg)](https://pypi.org/project/ojin-client/)
[![License: BSD 2-Clause](https://img.shields.io/badge/license-BSD%202--Clause-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-docs.ojin.ai-111.svg)](https://docs.ojin.ai)

> Real-time talking faces for **human agents that resonate**. Stream audio in, get a lip-synced avatar back — on a real-time clock, straight from Python.

[Ojin](https://ojin.ai) is the real-time GenAI platform. Our focus is **human agents** — lifelike, conversational presences with a real voice, face, and soul, available 24/7. This SDK is the open, framework-agnostic way to drive the models that bring them to life: feed it your TTS audio and it returns a synchronized **speech-to-video** stream (audio + a talking-avatar video), handling buffering, the playback clock, and barge-in for you.

It's the same client that powers Ojin's own [Pipecat](https://github.com/pipecat-ai/pipecat) integration — and it's yours to build agents, demos, and tools on top of.

- 🗣️ **One image → a persona.** Drive any avatar configuration from your Ojin account by its `config_id`.
- ⚡ **Real-time by design.** TCP_NODELAY, off-loop JPEG decode, and an audio-as-clock loop emit a steady 25 fps A/V stream.
- 🔌 **Framework-agnostic.** No Pipecat, no LiveKit, no web framework required. Drop it into anything async.
- 🎚️ **Interruptible.** First-class barge-in with audio fade and server-side cancel re-sync.
- 🧩 **Swappable internals.** Bring your own transport, resampler, JPEG decoder, output sink, or tracer behind small protocols.

---

## Two clients, one package

| | `OjinSTVClient` (high-level) | `OjinClient` (low-level) |
|---|---|---|
| Import | `from ojin.stv import OjinSTVClient` | `from ojin.ojin_client import OjinClient` |
| You give it | TTS audio at **any** sample rate | 16 kHz mono `int16` PCM |
| It gives you | Synced `STVAudioFrame` + `STVVideoFrame` on a 25 fps clock | Raw `OjinInteractionResponseMessage` (JPEG + PCM) |
| Handles for you | Buffering, playback clock, resampling, interruption re-sync, decode | Just the WebSocket + wire protocol |
| Extra deps | `ojin-client[stv]` (numpy / opencv-python-headless / soxr) | none |
| Use it when | You want a working talking avatar fast | You need full control of the loop |

Most applications want `OjinSTVClient`. Reach for `OjinClient` only when you're writing your own playback/sync layer.

---

## Requirements

- **Python 3.10+**
- An **Ojin API key** and a **persona `config_id`** — create both in your [Ojin account](https://ojin.ai/signin) (new accounts start with **$10 in free credits**).

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add "ojin-client[stv]"          # high-level OjinSTVClient (in a uv project)
uv pip install "ojin-client[stv]"  # ...or into the active environment
```

Or with pip:

```bash
pip install "ojin-client[stv]"     # high-level OjinSTVClient
pip install ojin-client            # low-level OjinClient only
```

The `[stv]` extra pulls in `numpy`, `opencv-python-headless`, and `soxr` for the sync math, default JPEG decoder, and resampler. If you inject your own decoder/resampler you can stay on the base install.

> Developing against a local checkout of this repo? Install it editable:
> `uv pip install -e ".[stv]"` (or `pip install -e ".[stv]"`).

## Authentication

The client authenticates with your API key and selects an avatar with a configuration ID. Keep both out of source — read them from the environment:

```bash
export OJIN_API_KEY="…"      # from your Ojin account
export OJIN_CONFIG_ID="…"    # the persona / avatar configuration to drive
```

```python
import os
from ojin.stv import OjinSTVClient

client = OjinSTVClient(
    api_key=os.environ["OJIN_API_KEY"],
    config_id=os.environ["OJIN_CONFIG_ID"],
)
```

Or let the SDK read and validate them for you — `resolve_credentials()` loads an optional `.env` file and raises a `MissingCredentialsError` (with setup instructions) if either value is absent:

```python
from ojin import resolve_credentials
from ojin.stv import OjinSTVClient

creds = resolve_credentials()  # reads OJIN_API_KEY + OJIN_CONFIG_ID (and a .env file)
client = OjinSTVClient(api_key=creds.api_key, config_id=creds.config_id)
```

By default the client connects to `wss://models.ojin.ai/realtime`. Override it with the `ws_url` argument if you're pointed at another environment.

> **Run it server-side.** The realtime transport is a WebSocket built for **server-to-server** delivery over a stable connection — run the client on your backend (not an end-user device), ideally in **US East** near Ojin's inference, and relay the final media to your users over a realtime transport such as WebRTC.

---

## Quickstart

Feed the avatar one utterance of TTS audio and consume the synced stream it speaks back.

```python
import asyncio
import os
import wave

from ojin.stv import OjinSTVClient, STVEvent, STVAudioFrame, STVVideoFrame


async def main() -> None:
    ready = asyncio.Event()
    done = asyncio.Event()

    async with OjinSTVClient(
        api_key=os.environ["OJIN_API_KEY"],
        config_id=os.environ["OJIN_CONFIG_ID"],
    ) as client:
        # Events may be sync or async; register with a decorator or add_listener().
        client.add_listener(STVEvent.SESSION_READY, lambda **_: ready.set())
        client.add_listener(STVEvent.BOT_STOPPED_SPEAKING, lambda **_: done.set())

        @client.on(STVEvent.ERROR)
        def _on_error(message: str, **_):
            print("ojin error:", message)

        # Wait until the avatar session is live (race-free: skip if already up).
        if not client.is_connected:
            await ready.wait()

        # Load mono int16 PCM TTS audio at ANY sample rate — the SDK resamples it.
        # The PCM is treated as a single channel and is NOT downmixed, so feed mono
        # audio (downmix stereo to mono before sending).
        with wave.open("hello.wav", "rb") as wav:  # a mono WAV
            await client.say(
                wav.readframes(wav.getnframes()),
                sample_rate=wav.getframerate(),
                num_channels=1,
            )

        # Consume the synchronized 25 fps audio + video stream.
        async def consume() -> None:
            async for frame in client.output_stream():
                if isinstance(frame, STVVideoFrame) and frame.rgb is not None:
                    handle_video(frame)  # frame.rgb = width*height*3 RGB bytes
                elif isinstance(frame, STVAudioFrame):
                    handle_audio(frame)  # frame.pcm = int16 PCM @ frame.sample_rate

        consumer = asyncio.create_task(consume())
        await done.wait()           # avatar finished the turn
        await asyncio.sleep(0.5)    # let the tail frames drain
        consumer.cancel()


asyncio.run(main())
```

`say()` is a one-shot helper for `start_turn()` + `send_tts_audio()`. For a live agent you'll call `start_turn()` once per utterance and stream chunks with `send_tts_audio()` as your TTS produces them.

---

## Core concepts

**Turns.** A turn is one spoken utterance. Open it with `start_turn()`, then push audio with `send_tts_audio(pcm, sample_rate, num_channels)` as many times as you like. The client buffers and plays the **original** audio you sent while a 16 kHz copy goes to the server for lip-sync — so the voice the user hears is always exactly yours. Because the original is what's played back, you can feed **higher-quality** TTS (e.g. 24 kHz) for better sound and lip-sync still works on the 16 kHz copy — just set your player to the `sample_rate` each `STVAudioFrame` reports.

**Input shaping.** Feed audio at whatever cadence your TTS produces — even tiny 40 ms fragments. The client takes care of the input shape to optimize latency and stability: it primes a ~1 s lead after each `start_turn()`, then combines your audio into **≥400 ms** chunks before forwarding it to the server, so the inference head never starves and lip-sync stays stable. It's automatic — tune it (or restore per-chunk sends) with the `server_feed_*` fields on [`STVConfig`](#configuration).

**Audio is the clock.** The playback loop emits **exactly one audio frame every tick** (real audio or silence so the consumer never starves) plus a video frame whenever one is ready, at `STVConfig.fps` (default 25 → a 40 ms tick). Video falls back to repeating the last frame rather than stalling.

**Interruption (barge-in).** Call `interrupt()` to cut a turn mid-sentence. The client fades the current audio, cancels the in-flight inference server-side, and re-syncs cleanly for the next turn. The `INTERRUPTED` event fires when a barge-in is accepted.

**Output sink.** By default frames land in a `QueueOutput` you drain with `output_stream()`. Inject a custom `STVOutput` (e.g. to push straight into a media transport) and consume it directly — `output_stream()` is only for the default queue.

### Events

Register handlers with `@client.on(STVEvent.X)` or `client.add_listener(STVEvent.X, cb)`. Handlers may be sync or async, and a failing handler never breaks the playback loop.

| Event | Fires when | Handler kwargs |
|---|---|---|
| `SESSION_READY` | The session is live and ready for audio | `session_data: dict \| None` |
| `BOT_STARTED_SPEAKING` | The first speech frame of a turn is played | — |
| `BOT_STOPPED_SPEAKING` | A turn has finished playing | — |
| `INTERRUPTED` | A barge-in was accepted | — |
| `ERROR` | A transport or server error occurred | `message: str`, `code: str` (server only), `fatal: bool` |
| `CLOSED` | The session has been torn down | — |

---

## Recipes

### Render an utterance to MP4

The `[stv]` extra already ships OpenCV, so a few lines turn the RGB stream into a video file.

```python
import asyncio
import os
import wave

import cv2
import numpy as np

from ojin.stv import OjinSTVClient, STVEvent, STVVideoFrame

FPS = 25


async def render(wav_path: str, out_path: str) -> None:
    ready, done = asyncio.Event(), asyncio.Event()
    writer = None  # created on the first frame, sized to whatever the server sends

    async with OjinSTVClient(
        api_key=os.environ["OJIN_API_KEY"],
        config_id=os.environ["OJIN_CONFIG_ID"],
    ) as client:
        client.add_listener(STVEvent.SESSION_READY, lambda **_: ready.set())
        client.add_listener(STVEvent.BOT_STOPPED_SPEAKING, lambda **_: done.set())
        if not client.is_connected:
            await ready.wait()

        with wave.open(wav_path, "rb") as wav:  # mono int16 PCM
            await client.say(
                wav.readframes(wav.getnframes()),
                sample_rate=wav.getframerate(),
                num_channels=1,
            )

        async def consume() -> None:
            nonlocal writer
            async for frame in client.output_stream():
                if isinstance(frame, STVVideoFrame) and frame.rgb is not None:
                    if writer is None:  # size the file to the server's native frame
                        writer = cv2.VideoWriter(
                            out_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS,
                            (frame.width, frame.height),
                        )
                    rgb = np.frombuffer(frame.rgb, np.uint8)
                    rgb = rgb.reshape(frame.height, frame.width, 3)
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        consumer = asyncio.create_task(consume())
        await done.wait()
        await asyncio.sleep(0.5)
        consumer.cancel()

    if writer is not None:
        writer.release()


asyncio.run(render("hello.wav", "avatar.mp4"))
```

> The MP4 above has no audio track — `STVAudioFrame.pcm` carries the synced audio if you want to mux it back in with `ffmpeg`.

### Interrupt a turn (barge-in)

```python
await client.say(long_intro_pcm, sample_rate=24000, num_channels=1)
# …user starts talking…
await client.interrupt()           # fade + server-side cancel
await client.say(reply_pcm, sample_rate=24000, num_channels=1)  # next turn re-syncs
```

### Swap the internals

Everything cross-cutting is injected behind a small protocol, so you can tune or replace it without touching the client.

```python
from ojin.stv import (
    OjinSTVClient,
    STVConfig,
    SoxrResampler,
    PassthroughDecoder,
)

client = OjinSTVClient(
    api_key=os.environ["OJIN_API_KEY"],
    config_id=os.environ["OJIN_CONFIG_ID"],
    config=STVConfig(fps=25, interrupt_audio_fade_s=0.5),
    resampler=SoxrResampler(),       # or NumpyLinearResampler() for zero native deps
    decoder=PassthroughDecoder(),    # skip JPEG decode; read frame.source_bytes yourself
)
```

With a `PassthroughDecoder`, `STVVideoFrame.rgb` is `None` and the raw JPEG arrives in `STVVideoFrame.source_bytes` — ideal when you forward frames to a browser or transport that decodes them itself.

---

## Configuration

`STVConfig` holds the behavioral knobs (connection identity stays on the client constructor):

| Field | Default | What it does |
|---|---|---|
| `client_connect_max_retries` | `3` | Connect attempts before giving up |
| `client_reconnect_delay` | `3.0` | Seconds between connect attempts |
| `fps` | `25` | Playback frame rate (40 ms tick) |
| `initial_buffer_frames` | `6` | Video frames buffered before playback starts (the small jitter buffer) |
| `max_buffered_video_frames` | `700` | Hard cap on queued video frames |
| `align_audio_on_swap` | `True` | Re-align audio/video at buffer swaps |
| `interrupt_audio_fade_s` | `0.75` | Fade length applied on barge-in |
| `lipsync_trace_enabled` | `False` | Emit the per-tick A/V-sync diagnostics |
| `server_feed_batching_enabled` | `True` | Prime + combine the server-bound audio feed (`False` = send each chunk as-is) |
| `server_feed_initial_chunk_ms` | `1000` | Initial lead accumulated after each `start_turn()` |
| `server_feed_min_chunk_ms` | `400` | Steady-state minimum send size |
| `server_feed_flush_idle_ms` | `200` | Quiet time before flushing a sub-threshold tail |

Video frames are emitted at the server's native resolution — read `STVVideoFrame.width`/`height` per frame rather than configuring an output size. Set `OJIN_MODE=dev` in the environment to attach the dev-mode query flag when connecting with the default transport.

---

## Low-level client

`OjinClient` is a thin async WebSocket wrapper over the Ojin wire protocol. You own the loop: connect, wait for the session, send 16 kHz mono `int16` PCM, and read frames.

```python
import asyncio
import os

from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    OjinAudioInputMessage,
    OjinSessionReadyMessage,
    OjinInteractionResponseMessage,
)


async def main(pcm_16k_mono: bytes) -> None:
    client = OjinClient(
        ws_url="wss://models.ojin.ai/realtime",
        api_key=os.environ["OJIN_API_KEY"],
        config_id=os.environ["OJIN_CONFIG_ID"],
    )
    await client.connect()

    # The server announces readiness before it will accept input.
    while True:
        msg = await client.receive_message()
        if isinstance(msg, OjinSessionReadyMessage):
            break
        if isinstance(msg, ErrorResponseMessage):
            raise RuntimeError(msg.payload.message)  # e.g. auth / capacity error

    await client.start_interaction()
    await client.send_message(OjinAudioInputMessage(audio_int16_bytes=pcm_16k_mono))

    # Each response carries one JPEG video frame + the int16 PCM it was synced to.
    while True:
        msg = await client.receive_message()
        if isinstance(msg, OjinInteractionResponseMessage):
            jpeg_bytes = msg.video_frame_bytes
            pcm_bytes = msg.audio_frame_bytes
            frame_type = msg.frame_type  # FrameType: IDLE / SPEECH / FADE_OUT / START_OF_SPEECH
            ...
            if msg.is_final_response:
                break

    await client.close()
```

**Frame types** (`ojin.ojin_client_messages.FrameType`): `IDLE = 0`, `SPEECH = 1`, `FADE_OUT = 2`, `START_OF_SPEECH = 3`.

**Send these:** `OjinAudioInputMessage`, `OjinTextInputMessage`, `OjinCancelInteractionMessage`, `OjinEndInteractionMessage`.
**Receive these:** `OjinSessionReadyMessage`, `OjinInteractionResponseMessage` (from `ojin.ojin_client_messages`), and `ErrorResponseMessage` (from `ojin.entities.interaction_messages`).

> **Send audio in large chunks.** Push as much audio as possible upfront (at least ~1 s), then keep a steady 25 fps input timeline with the biggest chunks you can. Feeding tiny inputs leaves the server unable to sustain speech at 25 fps, so it emits still ("idle") frames in between. The high-level `OjinSTVClient` manages this for you.

---

## Integrations

- **Pipecat** — this SDK is the engine behind Ojin's `OjinVideoService` for [Pipecat](https://github.com/pipecat-ai/pipecat). `OjinSTVClient` exposes the same behavior with no Pipecat dependency, so you can adopt it directly.
- **LiveKit Agents / custom transports** — feed `send_tts_audio()` from your TTS and push `STVVideoFrame` / `STVAudioFrame` into any media pipeline. Inject a custom `STVOutput` to write straight into your transport.
- **WebRTC (roadmap)** — the transport sits behind the `IOjinClient` protocol; a WebRTC transport can be dropped in without touching `OjinSTVClient`.

---

## Troubleshooting

- **`No backend servers available`** — capacity is momentarily exhausted; retry shortly. Surfaced as an `ERROR` event with code `NO_BACKEND_SERVER_AVAILABLE`.
- **Avatar mouth stays still / only idle frames** — the server is starved of audio and can't sustain 25 fps speech, so it emits idle frames in between. `OjinSTVClient` shapes the input to prevent this (see **Input shaping** above); it should only surface if you set `server_feed_batching_enabled=False` or drive the low-level `OjinClient` — then send a ~1 s lead up front, followed by the largest chunks you can.
- **`Inference Server is not ready`** (low-level) — wait for `OjinSessionReadyMessage` before calling `send_message()`.
- **Choppy playback** — keep the event loop free; the SDK already decodes JPEG off-loop, but heavy synchronous work in your frame handlers will stall the 40 ms tick. Enable `STVConfig(lipsync_trace_enabled=True)` to inspect per-tick timing.
- **Latency higher than expected** — the WebSocket transport is built for **server-to-server** use over a stable connection. Run the client on a backend (not an end-user device), ideally in **US East**, close to Ojin's inference; deliver the final media to end users over a realtime transport such as WebRTC. The client keeps a small video buffer (`initial_buffer_frames`) to absorb network jitter, since the server delivers at realtime 25 fps.

---

## Documentation & support

- 📚 Full docs: **[docs.ojin.ai](https://docs.ojin.ai)**
- 🌐 Platform & pricing: **[ojin.ai](https://ojin.ai)** ($10 free credits to start)
- 🐛 Issues & feature requests: [github.com/ojinai/python-sdk/issues](https://github.com/ojinai/python-sdk/issues)
- 🤝 Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md)

## License

BSD 2-Clause License — see [LICENSE](LICENSE).
