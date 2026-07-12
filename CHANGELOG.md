# Changelog

All notable changes to `ojin-client` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0 — see CONTRIBUTING.md).

## 0.9.0 - 2026-07-12

### Changed
- TTS audio for a turn that opens **while a barge-in is still settling server-side**
  is now **buffered and replayed** once the interruption clears, instead of being
  fed immediately. The server discards audio sent while it is still cancelling the
  prior turn, so feeding a new turn during that window left the client playing audio
  the server never rendered — desyncing every later turn. `OjinSTVClient` now defers
  the new turn's `start_turn` + audio (in arrival order) from the moment a barge-in
  turn opens until the server's idle/fade frame closes the window, then replays it.
  The interrupted turn's own trailing audio is still dropped (never replayed).

### Added
- Transport-level server-feed **send pacing** in `OjinClient`: a single audio send
  is capped at `max_input_chunk_bytes` (larger payloads are split) and consecutive
  sends are spaced at least `send_chunk_gap_s` apart **by time** — a minimum
  interval between on-wire sends — so any burst (split chunks, a queued backlog,
  or a producer enqueueing one message per await, like the deferred-TTS replay)
  is spread out, while steady-state realtime streaming (batched emits already
  ~400 ms apart) never waits. A cancel landing during the pacing sleep aborts the
  remaining chunks. Wired from new `STVConfig` knobs `server_feed_max_chunk_bytes`
  (default 51200) and `server_feed_send_gap_ms` (default 200). `OjinClient`'s own
  defaults keep the previous behaviour (no gap) for direct users.
- **Server-feed lead cap** (`STVConfig.server_feed_max_lead_ms`, default 8000; 0
  disables): the client never ships audio more than the cap ahead of local
  playback. The server renders at ~1x realtime, so sending a long answer's whole
  audio up front (272 s queued in ~25 s of wall clock, session-7 trace
  2026-07-12) only built a server-side input backlog that a barge-in cancel had
  to flush before acknowledging — ack grew 1.2 s → 2.0 s over a session, and the
  deferred next turn waits on that ack, so post-barge-in freezes grew with
  answer length. Payloads beyond the cap wait client-side and are released as
  playback advances; a barge-in discards them; `close()` flushes best-effort.
  New trace counters: `server_feed_lead_ms`, `server_feed_gated_pending`.

### Fixed
- A `CancelInteractionMessage` now **drops any audio still queued to be sent** in
  `OjinClient` (and the in-flight chunk loop bails between chunks and during the
  pacing sleep), so pre-cancel audio can't outlive the barge-in and desync the
  next turn.

## 0.8.0 - 2026-06-19

### Added
- `OjinSTVClient` now batches the audio it sends to the inference server instead
  of forwarding every TTS chunk immediately. Providers that stream small (~40 ms)
  chunks at realtime previously starved the server's 25 fps timeline (no supply
  lead builds when input rate equals output rate) and churned sub-frame residues
  that disrupted buffer swaps and lip-sync. The client now accumulates a
  lead-establishing initial chunk per `start_turn` (`server_feed_initial_chunk_ms`,
  default 1000 ms), then steady-state chunks of at least `server_feed_min_chunk_ms`
  (default 400 ms), and flushes a sub-threshold tail after
  `server_feed_flush_idle_ms` of quiet (default 200 ms). A barge-in discards the
  un-sent tail; `close` flushes it best-effort. Set
  `server_feed_batching_enabled=False` to restore per-chunk sends. The avatar
  playback path is unchanged — only the server-bound resampled copy is batched.

## 0.7.0 - 2026-06-18

### Added
- `OjinSTVClient(buffer_preinit_tts_audio=True)` (default on): input turns and TTS
  audio sent via `start_turn` / `send_tts_audio` before the session is ready
  (`SESSION_READY`) are now queued in arrival order and replayed once it is,
  instead of being dropped — so a caller can start speaking during the cold-start
  handshake (e.g. an opening line) without losing audio. A barge-in (`interrupt`)
  or `close` before the session is ready discards the queued input. Pass
  `buffer_preinit_tts_audio=False` to restore the previous drop-with-warning
  behaviour.

## 0.6.8 - 2026-06-17

### Added
- Automated release pipeline: `ojin-client` is now built and published to PyPI
  from GitHub Actions whenever the `version` in `pyproject.toml` is bumped on
  `main`, using PyPI Trusted Publishing (OIDC — no stored tokens). See
  [`.github/workflows/release.yml`](.github/workflows/release.yml).

### Changed
- Lint/type-check config: the `ruff` test-file ignores now cover nested test
  directories (`tests/stv`, `tests/integration`, …) and allow guarded lazy
  imports (`PLC0415`); `pyright` resolves imports from the project virtualenv
  and type-checks the shipped package only. The release gate runs `ruff` and
  `pytest` over the shipped package (`src`/`tests`); type-checking is advisory.
