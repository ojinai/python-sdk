---
name: ojin-stv-pipecat
description: >-
  Use when adding an Ojin lip-synced talking-avatar (face) to a Pipecat agent —
  the fastest path is to take an existing Pipecat voice-agent example and drop in
  the OjinAvatarService FrameProcessor (built on ojin.stv.OjinSTVClient). Points to
  the exact official pipecat-ai/pipecat examples to start from and the one-spot
  pipeline edit + transport video config needed. Self-contained: the full adapter
  is in this file. For the underlying client model, also read ojin-stv-integration.
---

# Add an Ojin avatar (face) to a Pipecat agent

An Ojin avatar runs as a Pipecat **`FrameProcessor`** placed **after your TTS
service and before `transport.output()`** — the exact same pipeline slot as
Pipecat's built-in avatar services (Simli, Tavus, HeyGen). So the fastest route is:
**take a working Pipecat voice agent and add the face.** This skill gives you a
self-contained `OjinAvatarService` to paste in, and points to the official examples
to start from.

```
transport.input() → STT → LLM → TTS → [OjinAvatarService] → transport.output()
```

## Where to find things — official `pipecat-ai/pipecat` repo

Browse: https://github.com/pipecat-ai/pipecat/tree/main/examples

| Goal | Path in the repo |
|---|---|
| **Start here** — simplest full voice agent (STT→LLM→TTS) | [`examples/getting-started/06-voice-agent.py`](https://github.com/pipecat-ai/pipecat/blob/main/examples/getting-started/06-voice-agent.py) |
| Same, fully local (no web transport) | [`examples/getting-started/06a-voice-agent-local.py`](https://github.com/pipecat-ai/pipecat/blob/main/examples/getting-started/06a-voice-agent-local.py) |
| **The "face added" reference** (a video service in the pipeline — exactly our pattern) | [`examples/video-avatar/video-avatar-simli-video-service.py`](https://github.com/pipecat-ai/pipecat/blob/main/examples/video-avatar/video-avatar-simli-video-service.py) |
| All avatar examples (Simli / Tavus / HeyGen / LemonSlice) | [`examples/video-avatar/`](https://github.com/pipecat-ai/pipecat/tree/main/examples/video-avatar) |
| All starter agents (say-one-thing → function-calling) | [`examples/getting-started/`](https://github.com/pipecat-ai/pipecat/tree/main/examples/getting-started) |
| Setup, API keys, running with `-t daily` / `-t webrtc` | [`examples/README.md`](https://github.com/pipecat-ai/pipecat/blob/main/examples/README.md) |

Read `video-avatar-simli-video-service.py` first: `SimliVideoService(...)` sits
between `tts` and `transport.output()`, and the transport enables `video_out_*`.
`OjinAvatarService` drops into that same slot — you're swapping one face for another.

## Add the face in 3 edits

Start from `getting-started/06-voice-agent.py` (or any agent with STT → LLM → TTS):

**1. Install + paste the adapter** (the full `OjinAvatarService` class is in the section below):

```bash
pip install "ojin-client[stv]"     # or: uv add "ojin-client[stv]"
```

**2. Construct it and insert it into the pipeline — after `tts`, before `transport.output()`:**

```python
avatar = OjinAvatarService(
    api_key=os.environ["OJIN_API_KEY"],
    config_id=os.environ["OJIN_CONFIG_ID"],   # your Face-model id from ojin.ai
)

pipeline = Pipeline([
    transport.input(),
    stt, user_aggregator, llm, tts,
    avatar,                       # <-- the only line you add to the pipeline
    transport.output(),
    assistant_aggregator,
])
```

**3. Turn on video output on the transport** (mirror the Simli example's params).
The avatar emits frames at your Face model's **native resolution**; set
`video_out_width`/`video_out_height` to that resolution so the transport forwards
them without rescaling:

```python
# Daily (use your Face model's output size; 512x512 shown):
DailyParams(audio_in_enabled=True, audio_out_enabled=True,
            video_out_enabled=True, video_out_is_live=True,
            video_out_width=512, video_out_height=512)
# SmallWebRTC (the default `-t webrtc`): same flags on TransportParams(...)
```

That's the whole integration. Run it like any example (e.g. `uv run python bot.py -t daily`).

## The avatar service

Self-contained — paste this into your bot file (or a small local module) and import
`OjinAvatarService`. It needs only `ojin-client[stv]` and `pipecat-ai`.

```python
from __future__ import annotations

from ojin.stv import OjinSTVClient, STVAudioFrame, STVEvent, STVVideoFrame

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _is_trailing_silence(pcm, sample_rate, num_channels):
    """Return True for the ~0.5 s all-zero sentinel some TTS engines emit at end-of-turn."""
    if not pcm:
        return False
    duration = len(pcm) / (sample_rate * num_channels * 2)
    return abs(duration - 0.5) < 0.01 and pcm == b"\x00" * len(pcm)


class _AvatarOutput:
    """STVOutput sink: pushes the client's synced A/V downstream as Pipecat frames."""

    def __init__(self, service):
        self._svc = service

    async def write_audio(self, frame: STVAudioFrame):
        # The client returns your ORIGINAL TTS audio to play (never the server's).
        await self._svc.push_frame(
            OutputAudioRawFrame(frame.pcm, frame.sample_rate, frame.num_channels)
        )

    async def write_video(self, frame: STVVideoFrame):
        # rgb is decoded pixels; it repeats the last frame on held ticks (smooth),
        # and is None only with a passthrough decoder (we use the default).
        if frame.rgb is not None:
            await self._svc.push_frame(
                OutputImageRawFrame(
                    image=frame.rgb, size=(frame.width, frame.height), format=frame.format
                )
            )

    def on_event(self, event, **kwargs):
        pass  # lifecycle events are wired via the client's emitter (see __init__)


class OjinAvatarService(FrameProcessor):
    """Pipecat FrameProcessor that turns the TTS audio stream into a lip-synced avatar.

    Place it in the pipeline after your TTS service and before transport.output().
    All avatar behavior (connect/retry, audio-as-clock playback, A/V sync, re-sync
    after barge-in) lives in ojin.stv.OjinSTVClient; this class is just the mapping.
    """

    def __init__(self, *, api_key, config_id, ws_url="wss://models.ojin.ai/realtime"):
        super().__init__(name="ojin-avatar")
        self._waiting_for_first_tts = False
        self._stv = OjinSTVClient(
            api_key=api_key,
            config_id=config_id,
            ws_url=ws_url,
            output=_AvatarOutput(self),   # push model — do NOT call output_stream()
        )

        @self._stv.on(STVEvent.BOT_STARTED_SPEAKING)
        async def _started(**_):
            await self.stop_ttfb_metrics()        # first avatar frame ⇒ stop TTFB timer

        @self._stv.on(STVEvent.ERROR)
        async def _error(message="", fatal=False, **_):
            await self.push_error(message, fatal=fatal)

    def can_generate_metrics(self):
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._stv.start()                # connect + run the playback loops
        elif isinstance(frame, TTSStartedFrame):
            self._waiting_for_first_tts = True
            await self._stv.start_turn()           # open a buffer for this utterance
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            if _is_trailing_silence(frame.audio, frame.sample_rate, frame.num_channels):
                return                             # drop the end-of-turn silence sentinel
            if self._waiting_for_first_tts:
                self._waiting_for_first_tts = False
                await self.start_ttfb_metrics()    # arm TTFB on the first real audio
            await self._stv.send_tts_audio(
                frame.audio, frame.sample_rate, frame.num_channels
            )
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._stv.interrupt()            # barge-in (no-op if the avatar is idle)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stv.close()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)  # stay transparent for everything else
```

## How it maps (so you can adapt it)

Inbound Pipecat frame → client call:

| Frame | Call | Why |
|---|---|---|
| `StartFrame` | `await stv.start()` | connect + run the playback/receive loops |
| `TTSStartedFrame` | `await stv.start_turn()` | open a buffer for the utterance |
| `TTSAudioRawFrame` | `await stv.send_tts_audio(audio, sample_rate, num_channels)` | the audio to lip-sync to |
| `UserStartedSpeakingFrame` | `await stv.interrupt()` | barge-in (no-op if idle) |
| `EndFrame` / `CancelFrame` | `await stv.close()` | tear down |

Client event → adapter action:

| `STVEvent` | Action |
|---|---|
| `BOT_STARTED_SPEAKING` | `stop_ttfb_metrics()` (first avatar frame) |
| `ERROR` | `push_error(message, fatal=fatal)` |
| `BOT_STOPPED_SPEAKING`, `SESSION_READY`, `INTERRUPTED`, `CLOSED` | not wired here — add `@self._stv.on(STVEvent.X)` handlers if your bot needs them (`SESSION_READY` carries `session_data`) |

## Gotchas

- **The avatar emits the server's native resolution.** Each `OutputImageRawFrame`
  carries `frame.width`/`frame.height` (no client-side resize). Set the transport's
  `video_out_width`/`video_out_height` to your Face model's native size so the
  transport forwards frames without rescaling; a mismatch makes the transport resize
  every frame (works, but wastes CPU and can soften the image).
- **Your transport must carry a video track.** Daily and SmallWebRTC (`-t webrtc`)
  support `video_out_*`; a phone/audio-only transport (e.g. Twilio) has no video out.
- **Feed `TTSAudioRawFrame` as-is.** Pass `frame.sample_rate`/`frame.num_channels`
  through — the client resamples a 16 kHz copy for lip-sync and plays your original
  audio. Don't pre-resample.
- **Drop the trailing-silence sentinel.** Some TTS services append a ~0.5 s all-zero
  frame; the client discards it, and `_is_trailing_silence` drops it in the adapter so
  TTFB anchors on real audio.
- **Render from `rgb` (raw pixels), not `source_bytes`.** `OutputImageRawFrame` wants
  raw pixels; the default decoder always fills `rgb` and repeats the last frame on held
  ticks. (`source_bytes` is the raw JPEG and is empty on held ticks — not what you want
  here.) Don't inject `PassthroughDecoder` — it yields `rgb=None` and emits no frame.
- **Optional, production:** the avatar starts producing frames as soon as it connects.
  If you connect well before the viewer joins, hold the sink's `push_frame` calls behind
  a boolean you flip on join, so the connect→join idle frames don't fill the transport's
  output buffer.

## Verify against your installed versions

Pipecat frame classes and the client API both evolve. Before relying on them:

```python
import inspect, ojin.stv as stv
print(inspect.signature(stv.OjinSTVClient.send_tts_audio))
print([e.name for e in stv.STVEvent])

from pipecat.frames.frames import OutputImageRawFrame, OutputAudioRawFrame, TTSAudioRawFrame
print(inspect.signature(OutputImageRawFrame.__init__))
print(inspect.signature(OutputAudioRawFrame.__init__))
```

If a Pipecat frame's constructor differs, trust your installed Pipecat (and the
official example you started from). If the client API differs, open an issue:
https://github.com/ojinai/python-sdk/issues
