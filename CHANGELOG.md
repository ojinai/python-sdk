# Changelog

All notable changes to `ojin-client` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0 — see CONTRIBUTING.md).

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
