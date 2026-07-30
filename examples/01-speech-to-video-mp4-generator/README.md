# 01 · Speech-To-Video MP4 Generator

Turn an audio file of speech into a lip-synced talking-avatar **MP4** with the Ojin
Speech-To-Video model. The smallest end-to-end example: one audio file in, one
video file out.

## What you need

1. **Python 3.10+**
2. **An Ojin account** → [ojin.ai](https://ojin.ai) (new accounts get **$10 free credits**):
   - an **API key** → `OJIN_API_KEY`
   - a **Face model** to drive, and its **config id** → `OJIN_CONFIG_ID`

   Don't have these yet? Just run the script — it prints exactly where to get them.
3. **An audio file** of speech — WAV, MP3, M4A, FLAC, OGG, anything ffmpeg reads
   (defaults to `input.wav`).

## Setup

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv && source .venv/bin/activate
uv pip install -e "../..[stv]"     # OPTIONAL the Ojin SDK (from this workspace)
uv pip install -r requirements.txt  # this example's deps (imageio-ffmpeg)
cp .env.example .env                 # paste your key + config id into .env
```

<details><summary>Or with plain pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "../..[stv]"
pip install -r requirements.txt
cp .env.example .env
```

</details>

> Once `ojin-client` is published to PyPI, the editable `-e "../..[stv]"` step is no
> longer needed — `requirements.txt` already lists it.

## Run

```bash
python main.py                       # input.wav  ->  output.mp4
python main.py myvoice.mp3 clip.mp4  # custom paths, any audio format
```

## How it works

- `main.py` reads the audio, opens an `OjinSTVClient`, sends it with `say()`, and
  streams the returned RGB frames to the writer — a few dozen lines of logic.
- `audio_input.py` hands mono 16-bit WAVs straight to the SDK and runs everything
  else (MP3, stereo, other sample rates) through ffmpeg first — the model is fed
  mono 16 kHz PCM either way.
- `mp4_writer.py` muxes those frames **and the original audio** into a single MP4
  with sound, using [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)
  (it ships its own ffmpeg binary, so there's nothing else to install).

The result is a ready-to-play `output.mp4` with both picture and audio — no
separate muxing step needed.
