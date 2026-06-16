// Realtime Speech-To-Video — browser client.
// Captures the mic (16 kHz mono) + webcam, streams mic PCM to the Python backend,
// and renders the avatar JPEG video + PCM audio the backend streams back.

const RATE = 16000;
const statusEl = document.getElementById("status");
const avatarEl = document.getElementById("avatar");

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.binaryType = "arraybuffer";
ws.onclose = () => (statusEl.textContent = "Disconnected — refresh to reconnect.");

let audioCtx = null; // one 16 kHz context for both capture and playback
let playHead = 0; // scheduled start time of the next avatar audio chunk
let lastFrameUrl = null;

document.getElementById("start").onclick = async () => {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      video: true,
    });
  } catch (err) {
    statusEl.textContent = "Mic + camera permission is required: " + err.message;
    return;
  }

  document.getElementById("me").srcObject = stream; // your webcam, on the left

  // Asking for a 16 kHz context lets the browser downsample the mic for us.
  audioCtx = new AudioContext({ sampleRate: RATE });
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(1024, 1, 1); // ~64 ms blocks
  const mute = audioCtx.createGain(); // route mic to nowhere (no self-monitoring)
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(audioCtx.destination);

  processor.onaudioprocess = (e) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const f32 = e.inputBuffer.getChannelData(0); // Float32 @ 16 kHz
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      i16[i] = s * 0x7fff;
    }
    ws.send(i16.buffer); // mic PCM -> backend
  };

  document.getElementById("start").disabled = true;
  statusEl.textContent = "Listening — talk, pause to hear the avatar, talk again to interrupt.";
};

ws.onmessage = (ev) => {
  if (typeof ev.data === "string") {
    statusEl.textContent = ev.data; // server status / error
    return;
  }
  const msg = new Uint8Array(ev.data);
  const body = msg.subarray(1);
  if (msg[0] === 1) renderVideo(body); // 0x01 = JPEG frame
  else playAudio(body); // 0x00 = int16 PCM @ 16 kHz
};

function renderVideo(jpegBytes) {
  if (lastFrameUrl) URL.revokeObjectURL(lastFrameUrl);
  lastFrameUrl = URL.createObjectURL(new Blob([jpegBytes], { type: "image/jpeg" }));
  avatarEl.src = lastFrameUrl;
}

function playAudio(pcmBytes) {
  if (!audioCtx) return;
  const aligned = pcmBytes.slice(); // copy to a 2-byte-aligned buffer
  const i16 = new Int16Array(aligned.buffer);
  const buffer = audioCtx.createBuffer(1, i16.length, RATE);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < i16.length; i++) channel[i] = i16[i] / 0x8000;
  const node = audioCtx.createBufferSource();
  node.buffer = buffer;
  node.connect(audioCtx.destination);
  playHead = Math.max(playHead, audioCtx.currentTime); // schedule back-to-back
  node.start(playHead);
  playHead += buffer.duration;
}
