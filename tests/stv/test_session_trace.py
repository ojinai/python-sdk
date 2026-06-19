"""Unit tests for the standalone OjinSessionTrace Perfetto recorder."""

from __future__ import annotations

import json

from ojin.stv.session_trace import LANES, OjinSessionTrace


class FakeClock:
    """A manually-advanced monotonic clock (seconds)."""

    def __init__(self) -> None:
        """Start the clock at zero."""
        self.t = 0.0

    def __call__(self) -> float:
        """Return the current time in seconds."""
        return self.t

    def advance(self, seconds: float) -> None:
        """Advance the clock by ``seconds``."""
        self.t += seconds


def _trace() -> tuple[OjinSessionTrace, FakeClock]:
    clk = FakeClock()
    return OjinSessionTrace(session_id="sess", config_id="cfg", clock=clk), clk


def test_now_us_is_microseconds_since_start() -> None:
    """now_us advances in microseconds relative to the construction time."""
    tr, clk = _trace()
    assert tr.now_us() == 0.0
    clk.advance(0.5)
    assert tr.now_us() == 500_000.0


def test_instant_records_event_on_lane() -> None:
    """Instant emits a Chrome 'i' event on the lane's thread id."""
    tr, clk = _trace()
    clk.advance(0.1)
    tr.instant("recv:speech", "frame_recv", args={"frame_type": 1})
    doc = tr.build()
    evs = [e for e in doc["traceEvents"] if e.get("ph") == "i"]
    assert len(evs) == 1
    assert evs[0]["name"] == "frame_recv"
    assert evs[0]["tid"] == LANES["recv:speech"]
    assert evs[0]["ts"] == 100_000.0
    assert evs[0]["args"] == {"frame_type": 1}


def test_span_records_duration_event() -> None:
    """Span emits a Chrome 'X' event with the elapsed duration."""
    tr, clk = _trace()
    start = tr.mark()
    clk.advance(0.2)
    tr.span("speaking", "bot_speaking", start)
    ev = next(e for e in tr.build()["traceEvents"] if e.get("ph") == "X")
    assert ev["name"] == "bot_speaking"
    assert ev["tid"] == LANES["speaking"]
    assert ev["dur"] == 200_000.0


def test_counter_records_counter_event() -> None:
    """Counter emits a Chrome 'C' event carrying the value series."""
    tr, _clk = _trace()
    tr.counter("output_audio_rms", 1234.5)
    ev = next(e for e in tr.build()["traceEvents"] if e.get("ph") == "C")
    assert ev["name"] == "output_audio_rms"
    assert ev["args"] == {"output_audio_rms": 1234.5}


def test_build_emits_lane_metadata() -> None:
    """Build names the process and every lane via 'M' metadata events."""
    tr, _clk = _trace()
    meta = [e for e in tr.build()["traceEvents"] if e.get("ph") == "M"]
    names = {e["name"] for e in meta}
    assert "process_name" in names and "thread_name" in names
    lane_names = {e["args"]["name"] for e in meta if e["name"] == "thread_name"}
    assert {"recv:speech", "play:idle", "play_audio", "latency"} <= lane_names


def test_correlation_id_emitted_for_stitching() -> None:
    """A correlation id rides otherData + a metadata event so the trace can be
    stitched to the browser/proxy/server traces of the same session."""
    tr = OjinSessionTrace(session_id="sess", correlation_id="oc-1700000000000-abcd1234")
    doc = tr.build()
    assert doc["otherData"]["ojin_correlation_id"] == "oc-1700000000000-abcd1234"
    assert doc["otherData"]["ojin_service"] == "ojin-stv-client"
    corr = [e for e in doc["traceEvents"]
            if e.get("ph") == "M" and e.get("name") == "ojin_correlation"]
    assert corr and corr[0]["args"]["ojin_correlation_id"] == "oc-1700000000000-abcd1234"


def test_record_lipsync_offset_emits_live_offset() -> None:
    """record_lipsync_offset emits a live A/V offset on the lipsync lane (instant
    with offset_ms) plus a counter line-plot (§6: lipsync now emitted live)."""
    tr, _clk = _trace()
    tr.record_lipsync_offset(7.5)
    evs = tr.build()["traceEvents"]
    inst = [e for e in evs if e.get("ph") == "i" and e.get("name") == "av_offset"]
    assert inst and inst[0]["args"]["offset_ms"] == 7.5
    assert any(e.get("ph") == "C" and e.get("name") == "lipsync_offset_ms" for e in evs)


def test_correlation_id_absent_by_default() -> None:
    """Backward compatible: no correlation id unless one is supplied."""
    tr, _clk = _trace()
    doc = tr.build()
    assert doc["otherData"]["ojin_correlation_id"] == ""
    assert not any(e.get("name") == "ojin_correlation" for e in doc["traceEvents"])


def test_record_response_latency_returns_ms_and_spans() -> None:
    """record_response_latency draws a response span and returns the ms value."""
    tr, clk = _trace()
    start = tr.mark()
    clk.advance(0.35)
    ms = tr.record_response_latency("played", start)
    assert ms == 350.0
    assert any(e.get("tid") == LANES["response"] for e in tr.build()["traceEvents"])


def test_dump_writes_loadable_json(tmp_path) -> None:
    """Dump writes a Chrome-format JSON file with otherData metadata."""
    tr, _clk = _trace()
    tr.counter("playback_fps", 25)
    path = tmp_path / "session.json"
    tr.dump(str(path))
    doc = json.loads(path.read_text())
    assert "traceEvents" in doc
    assert doc["otherData"]["session_id"] == "sess"
    assert doc["otherData"]["config_id"] == "cfg"


def test_other_data_extra_is_merged(tmp_path) -> None:
    """Caller-supplied otherData (e.g. the sync verdict) rides along in the doc."""
    tr, _clk = _trace()
    tr.set_other_data("sync_report", {"aligned": True})
    assert tr.build()["otherData"]["sync_report"] == {"aligned": True}
