"""Replay a recorded client+server trace pair to reproduce the barge-in bug.

This is the diagnosis counterpart to the live ``03-pipecat-example``: instead of
running a session, it loads a *recorded* pair of Perfetto session traces — the
client trace written by ``ojin.stv.OjinSessionTrace`` and the inference server's
``session.json`` — and reconstructs the interruption timeline from them, the same
way you would eyeball the two side by side in https://ui.perfetto.dev.

The bug it reproduces (session ``8b00aa4e27e2``, 2026-06-16): the user barges in,
the client fades its local audio and sends a Cancel immediately, but the avatar
keeps "speaking" for ~9s before the server's fade-out frames arrive. The recorded
traces let us pin *where* the time goes without a live server.

Two signals are computed:

* **client-only (alignment-free):** within the client trace, how long after
  ``cancel_sent`` does the first FADE frame (``recv_frame_type==2``) arrive. This
  needs no cross-clock alignment — it is exactly the "the audio faded but the
  video kept talking" lag the user reported.
* **cross-clock (corroborating):** align the two clocks on the first audio chunk
  (client ``audio_sent`` ↔ server ``audio_input``) and measure ``cancel_sent``
  (client) → ``interrupt_requested`` (server). This localises the lag to the
  server's single FIFO message queue: the Cancel is recorded the instant the
  server *dequeues* it, so a large value means it sat behind buffered audio.

Run::

    python replay_session.py                       # bundled fixtures under traces/
    python replay_session.py CLIENT.json SERVER.json

Exit code is 0 when the bug signature is present (the lag exceeds the threshold),
matching its role as a reproduction harness; see ``test_replay_session.py`` for
the pytest assertion.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

HERE = pathlib.Path(__file__).parent
TRACES = HERE / "traces"

# A fade arriving more than this long after the cancel is the bug, not jitter.
# The healthy path returns a fade within ~1.4s (server output queue is capped so
# a fade reaches the client within ~one chunk of interrupt() running).
BUG_LAG_THRESHOLD_MS = 3000.0

# wire frame_type markers (client recv_frame_type counter / recv:* lanes)
_FADE = 2


def load_trace(path: str | pathlib.Path) -> dict:
    """Load a Chrome/Perfetto trace document from ``path``."""
    with open(path) as fh:
        return json.load(fh)


def _lane_tids(doc: dict, name: str) -> set[int]:
    """Return the thread-ids (lanes) named ``name`` in ``doc``."""
    return {
        e["tid"]
        for e in doc["traceEvents"]
        if e.get("ph") == "M"
        and e.get("name") == "thread_name"
        and e["args"]["name"] == name
    }


def _instants(doc: dict, lane: str, name: Optional[str] = None) -> list[dict]:
    """Return instant/duration events on ``lane`` (optionally filtered by name), by ts."""
    tids = _lane_tids(doc, lane)
    evs = [
        e
        for e in doc["traceEvents"]
        if e.get("tid") in tids
        and e.get("ph") in ("i", "X")
        and (name is None or e.get("name") == name)
    ]
    return sorted(evs, key=lambda e: e.get("ts", 0.0))


def _first_ts(doc: dict, lane: str, name: Optional[str] = None) -> Optional[float]:
    """Return the ts (µs) of the first event on ``lane`` (optionally by name), or None."""
    evs = _instants(doc, lane, name)
    return evs[0]["ts"] if evs else None


def _counter_first_ts(doc: dict, name: str, value: float) -> Optional[float]:
    """Return the ts (µs) of the first counter sample ``name`` equal to ``value``."""
    samples = sorted(
        (e for e in doc["traceEvents"] if e.get("ph") == "C" and e.get("name") == name),
        key=lambda e: e.get("ts", 0.0),
    )
    for e in samples:
        if e["args"].get(name) == value:
            return e["ts"]
    return None


def analyze(client: dict, server: dict) -> dict:
    """Reconstruct the interruption timeline and the cancel→fade lag from both traces."""
    cancel_us = _first_ts(client, "interruption", "cancel_sent")
    if cancel_us is None:
        raise ValueError(
            "client trace has no cancel_sent — not an interruption session"
        )

    fade_us = _counter_first_ts(client, "recv_frame_type", _FADE)
    interrupt_ended_us = _first_ts(client, "interruption", "interrupt_ended")

    # audio_sent enqueues after the cancel (should be a *new* turn, not the old one)
    audio_after_cancel = [
        e["ts"]
        for e in _instants(client, "to_server", "audio_sent")
        if e["ts"] > cancel_us
    ]

    # cross-clock alignment on the first audio chunk
    c_first_audio = _first_ts(client, "to_server", "audio_sent")
    s_first_audio = _first_ts(server, "audio_input")
    offset_us = (
        c_first_audio - s_first_audio
        if c_first_audio is not None and s_first_audio is not None
        else None
    )
    s_interrupt_us = _first_ts(server, "interrupt_requested")
    s_fade_us = _first_ts(server, "fade_out_emitted")

    def ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None else round((a - b) / 1000.0, 1)

    cancel_to_server_interrupt_ms = (
        None
        if s_interrupt_us is None or offset_us is None
        else round(((s_interrupt_us + offset_us) - cancel_us) / 1000.0, 1)
    )

    return {
        "session_id": client["otherData"].get("session_id"),
        # client-only, alignment-free
        "client_cancel_ms": round(cancel_us / 1000.0, 1),
        "client_fade_lag_ms": ms(fade_us, cancel_us),
        "client_interrupt_ended_lag_ms": ms(interrupt_ended_us, cancel_us),
        "audio_sent_after_cancel": len(audio_after_cancel),
        # cross-clock corroboration (localises to the server FIFO)
        "clock_offset_ms": None if offset_us is None else round(offset_us / 1000.0, 1),
        "cancel_to_server_interrupt_ms": cancel_to_server_interrupt_ms,
        "server_fade_latency_ms": ms(s_fade_us, s_interrupt_us),
        # verdict
        "bug_reproduced": (
            fade_us is not None
            and (fade_us - cancel_us) / 1000.0 > BUG_LAG_THRESHOLD_MS
        )
        or (
            cancel_to_server_interrupt_ms is not None
            and cancel_to_server_interrupt_ms > BUG_LAG_THRESHOLD_MS
        ),
    }


def format_report(r: dict) -> str:
    """Render the analysis as a human-readable report."""
    lines = [
        f"Replay session {r['session_id']}",
        "─" * 60,
        "Client-only (no clock alignment needed):",
        f"  cancel_sent at                 t={r['client_cancel_ms']:.0f}ms (client clock)",
        f"  → first FADE frame arrived     +{r['client_fade_lag_ms']:.0f}ms after cancel",
        f"  → interrupt_ended              +{r['client_interrupt_ended_lag_ms']:.0f}ms after cancel",
        f"  audio_sent after cancel        {r['audio_sent_after_cancel']} chunks (next turn)",
        "",
        "Cross-clock (localises the lag to the server FIFO queue):",
        f"  clock offset (client−server)   {r['clock_offset_ms']:.0f}ms",
        f"  cancel → server interrupt      ~{r['cancel_to_server_interrupt_ms']:.0f}ms",
        f"  server fade latency (req→fade) {r['server_fade_latency_ms']:.0f}ms (healthy once it runs)",
        "─" * 60,
        f"BUG REPRODUCED: {r['bug_reproduced']}  "
        f"(threshold {BUG_LAG_THRESHOLD_MS:.0f}ms cancel→fade)",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Load the trace pair (bundled fixtures by default), analyze, print, return status."""
    if len(argv) >= 2:
        client_path, server_path = argv[0], argv[1]
    else:
        client_path = TRACES / "client_session.json"
        server_path = TRACES / "server_session.json"
    r = analyze(load_trace(client_path), load_trace(server_path))
    print(format_report(r))
    return 0 if r["bug_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
