# Contributing to the Ojin Python SDK

Thanks for helping build the Ojin Python SDK. Whether you're fixing a typo, sharpening an example, or adding a feature, this guide gets you set up and shipping.

This project is real-time software: the playback loop runs on a 40 ms budget and people watch the latency. We optimize for **correctness, low latency, and small, reviewable changes** — keep those in mind and you'll fit right in.

## Ground rules

- Be respectful and constructive. We assume good intent.
- Open an issue before a large or breaking change so we can agree on the shape first.
- Keep pull requests focused — one logical change per PR is far easier to review and revert.
- Never commit secrets (API keys, `config_id`s, credentials). Read them from the environment, as the README quickstart does.

## Development setup

```bash
git clone https://github.com/ojinai/python-sdk.git
cd python-sdk

# With uv (recommended) — installs the stv + dev extras from the lockfile:
uv sync --all-extras

# …or with pip + a virtualenv:
python -m venv .venv && source .venv/bin/activate
pip install -e ".[stv,dev]"
```

You'll need **Python 3.10+**. The `stv` extra brings in `numpy` / `opencv-python-headless` / `soxr`; the `dev` extra adds the test, lint, and build tooling.

## Project layout

```
src/ojin/
├── ojin_client.py            # low-level WebSocket client (OjinClient)
├── ojin_client_messages.py   # high-level message types + FrameType
├── profiling_utils.py        # lightweight profiling helpers
├── entities/                 # wire-format (de)serialization
│   ├── interaction_messages.py
│   └── session_messages.py
└── stv/                      # high-level OjinSTVClient and its parts
    ├── ojin_stv_client.py    #   the client itself
    ├── config.py             #   STVConfig
    ├── events.py             #   STVEvent + emitter
    ├── frames.py / output.py #   output frame types + sinks
    ├── synchronizer.py       #   audio-as-clock A/V sync
    ├── resampler.py          #   resamplers (soxr / numpy)
    ├── video_decode.py       #   JPEG decoders
    ├── audio_utils.py        #   PCM / RMS helpers
    ├── diagnostics.py        #   playback-loop stall diagnostics
    ├── tracing.py            #   Perfetto trace emitter
    ├── session_trace.py      #   per-session trace assembly
    └── sync_check.py         #   A/V-sync verification
tests/                        # pytest suite
├── stv/                      #   high-level client unit tests
├── integration/             #   live end-to-end tests (start a local server)
└── mock/                     #   inference-proxy mock used by the tests
```

## Checks: run these before you push

Run these locally before every push — there's no PR CI yet, and they're what reviewers will run. The release pipeline **gates publishing on formatting, lint, and tests**. `pyright` runs there too but is **advisory** for now: `src/ojin` carries known type debt, so it's a cleanup target rather than a hard gate — expect it to report errors until that's worked through.

```bash
uv run ruff format --check .   # formatting
uv run ruff check .            # lint (rules configured in pyproject.toml)
uvx pyright                    # type-check (pyright fetched ad hoc by uvx; not a pinned dep)
uv run pytest                  # tests
```

Autofix what you can: `uv run ruff format .` and `uv run ruff check --fix .`.

### Tests

We use `pytest` with `pytest-asyncio` (async mode is on by default — write `async def test_*` directly). The suite lives in `tests/`:

```bash
uv run pytest                                     # everything, with coverage
uv run pytest tests/stv                           # fast high-level unit tests
uv run pytest tests/stv/test_output.py -k stream  # a single test
```

> Heads-up: a plain `uv run pytest` also runs `tests/integration/`, which starts a **local server** and opens a real WebSocket — slower, with external moving parts. To skip it, target a path (e.g. `tests/stv`). The `unit` / `integration` markers are declared in `pyproject.toml` but **not yet applied** to any test, so `-m unit` / `-m integration` currently select nothing — prefer paths until the markers are wired up.

New behavior needs a test. Bug fixes should come with a regression test that fails before your change and passes after. Keep the real-time paths (sync, buffering, interruption) covered.

## Code style

- **Formatting & linting:** [Ruff](https://docs.astral.sh/ruff/) is the single source of truth — see `[tool.ruff]` in `pyproject.toml`. Don't hand-fight it.
- **Types:** type hints are required on function signatures (`reportMissingParameterType` is an error under Pyright). Prefer a precise type — or `object` for genuinely opaque values — over `Any`; use `Optional[...]` / `| None` deliberately.
- **Docstrings:** public modules, classes, and functions carry docstrings (the Ruff `D` rules enforce this).
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- **Async:** this is an async-first codebase — never block the event loop. Push CPU-bound work (e.g. image decode) onto a worker thread, as the existing decode pipeline does.

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/). The format is:

```
<type>(<optional scope>): <description>

[optional body]
[optional footer]
```

| Type | Use for | Version impact (SemVer) |
|---|---|---|
| `feat` | A new feature | minor |
| `fix` | A bug fix | patch |
| `perf` | A performance improvement | patch |
| `refactor` | Code change that neither fixes a bug nor adds a feature | patch |
| `docs` | Documentation only | none |
| `test` | Adding or fixing tests | none |
| `build` / `chore` | Tooling, deps, packaging | none |

**Breaking changes:** add a `!` after the type/scope (`feat!: …`) **and** a `BREAKING CHANGE:` footer describing the migration.

> This SDK is pre-1.0 (currently `0.x`). Under SemVer's 0.x rules a breaking change bumps the **minor** version; everything else bumps the **patch**. Treat the public API as still maturing.

**Examples:**

```
feat(stv): add PassthroughDecoder for raw-JPEG forwarding
fix(client): wait for SessionReady before accepting input
perf(stv): decode JPEG frames off the event loop
docs: document the barge-in recipe in the README
feat!: rename avatar_config_id to config_id

BREAKING CHANGE: OjinClient now takes `config_id` instead of `avatar_config_id`.
```

## Pull request process

1. Branch off `main` (`feat/…`, `fix/…`).
2. Make the change; add or update tests and docs.
3. Run the full check suite above — green across the board.
4. Update `CHANGELOG.md` with a one-line entry for user-facing changes.
5. Open the PR against `main` with a Conventional-Commit-style title and a short "what & why".
6. Address review feedback by pushing follow-up commits (we squash on merge).

## Release process

Releases are **automated**. Publishing `ojin-client` to PyPI is gated on a version bump and runs from GitHub Actions ([`.github/workflows/release.yml`](.github/workflows/release.yml)) on every push to `main`. To cut a release:

1. Bump `version` in `pyproject.toml` (follow the SemVer rules above).
2. Move the `CHANGELOG.md` entries under the new version with the date.
3. Open a PR and merge to `main` (we squash-merge).

That's the whole manual part. On merge, the pipeline:

- detects whether `pyproject.toml`'s version changed vs the previous commit ([`.ci/release_check.py`](.ci/release_check.py)) — if it didn't, the run stops and nothing is published;
- runs the full check suite (format, lint, type-check, unit tests) so a red `main` never ships;
- builds the sdist + wheel (`uv build`);
- publishes to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — **no API tokens are stored**); and
- tags the release `vX.Y.Z` and pushes the tag.

Because publishing keys off the version, a normal feature merge that doesn't touch `version` publishes nothing — it's safe. PyPI also rejects re-uploading an existing version, so each release must bump to a **new** version.

## Reporting bugs & requesting features

Open an [issue](https://github.com/ojinai/python-sdk/issues) with:

- what you expected vs. what happened,
- a minimal repro (SDK version, Python version, OS),
- relevant logs — enable `STVConfig(lipsync_trace_enabled=True)` for A/V-sync issues.

**Security:** please don't file public issues for vulnerabilities. Report them privately via a [GitHub security advisory](https://github.com/ojinai/python-sdk/security/advisories/new), or through [ojin.ai/contact](https://ojin.ai/contact).

## License

By contributing, you agree that your contributions are licensed under the [BSD 2-Clause License](LICENSE), the same license as the project.
