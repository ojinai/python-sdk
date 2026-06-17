# Changelog

All notable changes to `ojin-client` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0 — see CONTRIBUTING.md).

## 0.7.1 - 2026-06-17

### Added
- Automated release pipeline: `ojin-client` is now built and published to PyPI
  from GitHub Actions whenever the `version` in `pyproject.toml` is bumped on
  `main`, using PyPI Trusted Publishing (OIDC — no stored tokens). See
  [`.github/workflows/release.yml`](.github/workflows/release.yml).

### Changed
- Lint/type-check config: the `ruff` test-file ignores now cover nested test
  directories (`tests/stv`, `tests/integration`, …) and allow guarded lazy
  imports (`PLC0415`); `pyright` resolves imports from the project virtualenv
  and type-checks the shipped package only. Type-checking is advisory in CI.

> The `0.6.8` bump was an unpublished CI shakeout — its run surfaced the
> lint/type-check gaps fixed here, so the first PyPI release through the
> automated pipeline is `0.7.1`.
