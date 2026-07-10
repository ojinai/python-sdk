# Changelog

All notable changes to `ojin-client` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0 — see CONTRIBUTING.md).

## 0.8.3 - 2026-07-10

### Fixed
- Lip-sync: swap-time alignment now actually fires at real turn entries. Two
  production blockers, measured from staging traces: (a) at a real turn entry
  the video queue is only 1-2 frames deep (just-in-time delivery), so the swap
  tick never had the 4 audible frames the anchor guard demands — alignment was
  silently skipped on every turn (0 `swap_align_trim` events across whole
  sessions), leaving ±1-frame (±40 ms) whole-turn skews uncorrected. Alignment
  is now DEFERRED: the buffer head is snapshotted at the swap and the anchor
  keeps growing from the frames popped on the following ticks, applying the
  shift a few ticks late (a skew is a constant offset, so the remainder of the
  turn still aligns). (b) The 5% absolute error tolerance was too strict for
  cross-sample-rate RMS envelopes (true offsets measured up to 4.5x it; wrong
  offsets 87-374x): a non-head match is now accepted on a best-vs-second-best
  margin (8x) under a loose sanity cap (25%), so confidence comes from how
  uniquely the signature localizes, not from an absolute error bar. Regression
  tests replay the exact RMS envelopes captured from the two misaligned turn
  entries of staging session 4e35f6826fa5.
- Lip-sync: `start_turn` no longer sends the previous turn's resampler tail to
  the server. The tail (last ~30-70 ms held by the soxr filter) belongs to
  audio that played seconds ago; sending it at the next turn boundary landed it
  at the head of the new turn's server feed, where the server rendered it as
  1-2 near-zero speech frames the local buffer does not have — a constant
  40-80 ms video-late offset for the whole turn. Measured on staging: every
  natural-turn entry was offset by exactly the flushed tail's duration, while
  barge-in turns (whose interrupt path already discards the tail) were clean.
  The boundary still flushes the filter so the new turn starts from clean
  state; the stale bytes are discarded, exactly like `interrupt` does.
- Lip-sync: swap-time alignment now also runs on natural turn ends (plain
  SPEECH trigger), not only on the server-marked new-turn boundary. The server
  never marks natural entries, so head anomalies there could never self-heal.
  Safe with the rewritten matcher: it only shifts on a confident envelope
  match and prefers the head, so the steady state stays untouched.
- Lip-sync: mid-speech video stalls are now repaid instead of shifting the video
  timeline permanently. When speech audio drained on a tick with no fresh video
  frame (a delivery stall → repeated frame), the video fell 40 ms behind the
  audio per repeat and never caught up — a single 300 ms network hiccup desynced
  the rest of the turn. The synchronizer now tracks the owed frames (capped at
  1 s) and skips one extra plain-speech frame per tick once frames flow again,
  never across a turn boundary. Disable with
  `STVConfig(video_repeat_catchup_enabled=False)`. Traced as `repeat_catchup`
  instants and the `repeat_catchup_drops_total` counter.
- Lip-sync: swap-time alignment now handles a server turn head with extra
  near-zero speech frames (e.g. padding minted while the server was starved of
  TTS in the turn's first-fragment gap). Previously such a head both defeated
  the aligner (the silent anchor aborted alignment) and needed a correction in
  the direction the aligner could not do — the whole turn then played 40-120 ms
  out of sync. The anchor signature now starts at the first audible frame, and
  the signed onset difference either trims leading buffer audio (server dropped
  it) or prepends silence so the audio waits for the video (server padded).
  `TickResult.align_trim_frames` (and the `swap_align_trim` trace instant) can
  now be negative — silence prepended.

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
