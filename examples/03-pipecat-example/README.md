# 03 · Pipecat agent with an Ojin avatar face

Have a **live video conversation with an AI in your browser — like a Google Meet
call**. You talk; a voice agent (speech-to-text → LLM → text-to-speech) answers;
and an **Ojin avatar lip-syncs to the reply**, streamed back as video over WebRTC.

```
browser mic ─▶ WebRTC ─▶ STT ─▶ LLM ─▶ TTS ─▶ [Ojin avatar] ─▶ WebRTC ─▶ browser video
```

This is a standard [Pipecat](https://github.com/pipecat-ai/pipecat) voice agent
with **one extra line** in the pipeline: the `OjinAvatarService` face, dropped in
after TTS and before the transport output — the same slot Pipecat's Simli / Tavus
/ HeyGen avatars use. The adapter is self-contained in
[`ojin_avatar.py`](ojin_avatar.py); the agent is in [`bot.py`](bot.py).

## What you need

1. **Python 3.10+** and a **Chromium-based browser** (Chrome / Edge) for the call.
2. **An Ojin account** → [ojin.ai](https://ojin.ai) (new accounts get **$10 free credits**):
   - an **API key** → `OJIN_API_KEY`
   - a **Face model** config id → `OJIN_CONFIG_ID`
3. **Keys for the voice pipeline** (each has a free tier):
   - [Deepgram](https://console.deepgram.com) (STT) → `DEEPGRAM_API_KEY`
   - [OpenAI](https://platform.openai.com/api-keys) (LLM) → `OPENAI_API_KEY`
   - [Cartesia](https://play.cartesia.ai) (TTS) → `CARTESIA_API_KEY`

   Missing the Ojin keys? The bot prints exactly where to get them on startup.

## Setup & run

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv && source .venv/bin/activate
uv pip install -e "../..[stv]"      # the Ojin SDK (from this repo, until it's on PyPI)
uv pip install -r requirements.txt  # this example's deps (pipecat-ai + services)
cp .env.example .env                 # paste your keys into .env
python bot.py                        # -> open http://localhost:7860/client
```

> Run `python bot.py` with the venv **activated** (as above), not `uv run python
> bot.py` — this folder has no `pyproject.toml`, so `uv run` walks up to the repo
> root and uses that environment instead of the one you just set up here.

<details><summary>Or with plain pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "../..[stv]"
pip install -r requirements.txt
cp .env.example .env
python bot.py
```

</details>

> Once `ojin-client` is published to PyPI, the editable `-e "../..[stv]"` step is no
> longer needed — `requirements.txt` already lists it.

Then open **<http://localhost:7860/client>**, click **Connect**, allow the mic, and
talk. The avatar greets you and answers, lip-synced, on video.

### Transport options

`bot.py` uses Pipecat's dev runner, which serves a prebuilt browser client:

| Command | What it does |
|---|---|
| `python bot.py` | All transports; browser client at `http://localhost:7860/client` (defaults to WebRTC) |
| `python bot.py -t webrtc` | **Local WebRTC only — no external account needed** (the recommended way to test) |
| `python bot.py -t daily` | Join a [Daily](https://daily.co) room instead (needs a Daily account) |

> **On WSL2:** run `python bot.py` inside WSL and open
> `http://localhost:7860/client` in your normal **Windows** browser. The mic,
> camera, and audio all run in the browser, so no WSLg / X server is needed.

## How it works

- **The face is one stage.** In [`bot.py`](bot.py) the pipeline is
  `transport.input() → stt → user_aggregator → llm → tts → avatar →
  transport.output() → assistant_aggregator`. Remove `avatar` and you have a plain
  voice agent; add it back and the same agent has a lip-synced face.
- **`OjinAvatarService`** ([`ojin_avatar.py`](ojin_avatar.py)) is a Pipecat
  `FrameProcessor` that maps Pipecat frames to `ojin.stv.OjinSTVClient`: it feeds
  each `TTSAudioRawFrame` to the avatar, opens a turn on `TTSStartedFrame`, and
  triggers barge-in on `UserStartedSpeakingFrame`. The client returns your original
  audio plus the synced avatar video, which the adapter pushes downstream as
  `OutputAudioRawFrame` / `OutputImageRawFrame`.
- **The avatar plays your own TTS audio**, lip-synced — only a 16 kHz copy is sent
  to Ojin for inference. A/V sync, the audio-as-clock playback loop, and re-sync
  after interruptions are all handled inside the client.

## Tweak it

- **Swap the face:** change `OJIN_CONFIG_ID` to another Face model. If its output
  size differs from 512×512, update `AVATAR_SIZE` in `bot.py` (the
  `video_out_width/height` follow it automatically).
- **Swap the voice/brain:** change the Cartesia `voice_id`/`model`, the OpenAI
  `model`, or `SYSTEM_PROMPT` in `bot.py`. Any Pipecat STT/LLM/TTS service works —
  this is a vanilla Pipecat pipeline.

## Related

- Framework-agnostic client model: the `ojin-stv-integration` skill.
- This Pipecat integration in depth: the `ojin-stv-pipecat` skill (the
  `OjinAvatarService` here is that skill's adapter).
- Official Pipecat examples this is built on:
  [getting-started](https://github.com/pipecat-ai/pipecat/tree/main/examples/getting-started)
  and [video-avatar](https://github.com/pipecat-ai/pipecat/tree/main/examples/video-avatar).
