# 03 · Direct WebRTC into a LiveKit or Daily room

Speak a WAV file through an Ojin avatar that joins **your** LiveKit or Daily room. The inference server joins as a participant named `ojin-avatar` and publishes the avatar's audio and video there, so anyone in the room sees and hears it — no media flows through this script.

The code is the same as a WebSocket session; the only difference is passing `webrtc=WebRTCSettings(...)` to `OjinSTVClient`.

## Run it

```bash
pip install "ojin-client[stv]"

export OJIN_API_KEY="…"                # from your Ojin account
export OJIN_CONFIG_ID="…"              # the persona to drive
export OJIN_WEBRTC_PROVIDER="livekit"  # or "daily"
export OJIN_WEBRTC_ROOM_URL="…"        # see below
export OJIN_WEBRTC_TOKEN="…"           # see below

python main.py hello.wav               # a mono 16-bit WAV
```

You can also put these variables in a `.env` file next to `main.py`.

Open the room in a browser **before** running the script, then watch the avatar join and speak.

## Create a room and the avatar's token

The SDK doesn't create rooms or tokens — use your provider account.

### LiveKit

Use your LiveKit server URL (`wss://<project>.livekit.cloud`) as `OJIN_WEBRTC_ROOM_URL`, and mint an access token with the identity **`ojin-avatar`** (on LiveKit the identity comes from the token, and that claim is how viewers pick the avatar out of the room):

```python
from livekit import api  # pip install livekit-api

token = (
    api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    .with_identity("ojin-avatar")
    .with_grants(
        api.VideoGrants(
            room_join=True,
            room="ojin-demo",
            can_publish=True,
            can_subscribe=False,
            can_publish_data=False,
        )
    )
    .to_jwt()
)
```

Join the same room from the browser (for example with the [LiveKit Meet](https://meet.livekit.io) custom-server tab) using a token with a different identity.

### Daily

Create a room and a meeting token for it with the [Daily REST API](https://docs.daily.co/reference/rest-api):

```bash
curl -s -X POST https://api.daily.co/v1/rooms \
  -H "Authorization: Bearer $DAILY_API_KEY" -H "Content-Type: application/json" \
  -d '{"name": "ojin-demo", "privacy": "private"}'

curl -s -X POST https://api.daily.co/v1/meeting-tokens \
  -H "Authorization: Bearer $DAILY_API_KEY" -H "Content-Type: application/json" \
  -d '{"properties": {"room_name": "ojin-demo"}}'
```

Use the room's `url` as `OJIN_WEBRTC_ROOM_URL` and the returned `token` as `OJIN_WEBRTC_TOKEN`. Mint a second token to join the room yourself from the browser.

## What to expect

- `WEBRTC_CONNECTED` fires once the avatar is in the room; speaking events fire as it talks.
- There is no fallback. If the avatar can't join, the script exits with the error naming the cause — `WEBRTC_AUTH_FAILED`, `WEBRTC_NETWORK_FAILED`, `WEBRTC_INVALID_SETTINGS`, `WEBRTC_JOIN_TIMEOUT` or `WEBRTC_NOT_SUPPORTED`.
- `output_stream()` receives no frames in this mode, because the media goes to the room. To get frames in your process instead, drop the `webrtc=` argument (see example 01).
