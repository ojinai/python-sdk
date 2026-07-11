# Changelog

All notable changes to `ojin-client` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0 — see CONTRIBUTING.md).

## 0.8.3 - 2026-07-12

### Fixed
- A turn whose `start_turn()` lands while a barge-in is still settling server-side
  (`interrupt()` sent, not yet acknowledged) is now rejected wholesale by the
  client. Previously the client opened a fresh buffer and fed both playback and the
  server, but the server discards that audio (it is still cancelling the prior
  turn) — so the client played audio the server never rendered, desyncing every
  later turn. `start_turn()` now opens no buffer during an in-flight interruption
  and marks the turn discarded; the decision is sticky for the whole turn, so
  `send_tts_audio()` drops every one of its frames even if the interruption clears
  part-way through (the audio still belongs to the turn that opened under it).

### Added
- `OjinSTVClient.is_audio_input_enabled()` — `False` while the current turn is
  being discarded (see above), `True` otherwise. Adapters read it to skip TTS
  bookkeeping (e.g. arming TTFB metrics) for audio that will be dropped.

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
