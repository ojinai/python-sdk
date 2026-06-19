#!/usr/bin/env python3
"""Generate a deterministic OjinSessionTrace fixture that includes LIVE lipsync
offsets, for cross-driver lipsync comparison (autoresearch reqs §5.3/§6).

This exercises the real OjinSessionTrace public API (no live session needed, same
approach as the widget's synthesizeWidgetTrace) so the produced trace is genuine
tracer output. Writes a Perfetto/Chrome-Trace JSON to the path given (or stdout).

    python make_lipsync_fixture.py /path/to/sdk_session.json
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ST = os.path.join(HERE, "..", "..", "src", "ojin", "stv", "session_trace.py")
_spec = importlib.util.spec_from_file_location("session_trace", _ST)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
OjinSessionTrace = _m.OjinSessionTrace


def build(correlation_id: str = "oc-1700000000000-w1dget01") -> dict:
    t = {"now": 0.0}
    tr = OjinSessionTrace(session_id="replay-lipsync", config_id="cfg-demo",
                          correlation_id=correlation_id, clock=lambda: t["now"])
    sess_start = tr.now_us()
    t["now"] += 0.05
    conn_start = tr.now_us()
    t["now"] += 0.30
    tr.span("lifecycle", "connect", conn_start)

    # 6 turns: response latency + a live lipsync offset stream within each turn.
    for turn in range(6):
        tts_start = tr.now_us()
        t["now"] += 0.10 + 0.02 * turn
        tr.record_response_latency("recv", tts_start)
        t["now"] += 0.01
        tr.record_response_latency("played", tts_start)
        # ~25 fps played frames + per-frame lipsync offset (deterministic).
        for f in range(20):
            t["now"] += 0.04
            tr.instant("play:speech", "frame_played", args={"frame_type": 1})
            offset = 6.0 * math.sin((turn * 20 + f) / 5.0) + (1.5 if f % 9 == 0 else 0.0)
            tr.record_lipsync_offset(offset)
        if turn == 3:
            tr.instant("interruption", "cancel_sent")
            t["now"] += 0.02
            tr.instant("interruption", "interrupt_ended")

    tr.span("lifecycle", "session", sess_start)
    return tr.build()


def main() -> int:
    doc = build()
    out = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = json.dumps(doc, indent=2)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write(payload)
        print(f"wrote {out} ({doc['otherData']['event_count']} events)")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
