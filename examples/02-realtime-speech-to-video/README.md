# 02 · Realtime Speech-To-Video

Talk into your mic and an Ojin avatar talks back — **live, in the browser**. Your
webcam shows on the left, the avatar on the right. When you stop speaking the
avatar speaks your captured audio (lip-synced); start speaking again and it's
interrupted (barge-in). A tiny FastAPI backend + one HTML/JS page.

```
browser mic ──16 kHz PCM──▶ FastAPI ──VAD──▶ OjinSTVClient ──▶ avatar audio + video ──▶ browser
```

## What you need

1. **Python 3.10+** and a **Chromium-based browser** (Chrome / Edge) for the mic + Web Audio APIs.
2. **An Ojin account** → [ojin.ai](https://ojin.ai) (new accounts get **$10 free credits**):
   - an **API key** → `OJIN_API_KEY`
   - a **Face model** config id → `OJIN_CONFIG_ID`

   Missing either? The server prints exactly where to get them on startup.

## Setup & run

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv && source .venv/bin/activate
uv pip install -e "../..[stv]"     # the Ojin SDK (from this repo, until it's on PyPI)
uv pip install -r requirements.txt  # this example's deps (fastapi, webrtcvad, ...)
cp .env.example .env                 # paste your key + config id into .env
uv run python server.py              # -> open http://localhost:8000
```

<details><summary>Or with plain pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "../..[stv]"
pip install -r requirements.txt
cp .env.example .env
python server.py
```

</details>

> Once `ojin-client` is published to PyPI, the editable `-e "../..[stv]"` step is no
> longer needed — `requirements.txt` already lists it. (If you previously installed a
> different `ojin` build, reinstall with the editable step so the credential helpers
> are present.)

Click **Start**, allow the mic + camera, and talk.

> **On WSL2:** run `python server.py` inside WSL and open `http://localhost:8000`
> in your normal **Windows** browser. The mic, camera, and audio all run in the
> browser, so no WSLg / X server is needed.

## How it works

- **`static/app.js`** captures the mic at 16 kHz, streams `int16` PCM over a
  WebSocket, and renders the avatar JPEG frames + PCM audio the backend streams back.
- **`server.py`** runs WebRTC **voice-activity detection** on the mic stream. On
  utterance end it calls `client.say(audio)`; on the next utterance start it calls
  `client.interrupt()`. Avatar frames are forwarded to the browser as they arrive.

Worth tweaking in `server.py`: the VAD `aggressiveness` (0–3) and `end_frames` (how
much trailing silence ends a turn). The avatar video comes back at whatever size the
server sends; the browser scales it into the 512×512 panel via CSS.

## Notes

- The avatar **echoes your own voice** with a lip-synced face — that's the demo.
  Swap `client.say(utterance, ...)` for text→speech (e.g. feed an LLM reply through
  a TTS) to build a real conversational agent.
- `ScriptProcessorNode` is used for mic capture to keep this to a single JS file;
  it's deprecated but universally supported. Production apps should use an
  `AudioWorklet`.
