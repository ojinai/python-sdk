# Agent Skills for the Ojin SDK

These are **Agent Skills** — Markdown playbooks that teach an AI coding assistant
(Claude Code, and any tool that reads `SKILL.md` files) how to wire Ojin's
Speech-To-Video client into *your* agent pipeline correctly, the first time.

Ojin is focused on human-agent verticals, and the avatar is usually one stage in a
larger pipeline you already have (STT → LLM → TTS → **avatar** → transport). These
skills encode the integration contract — what to feed in, what comes out, the events,
barge-in, and the non-obvious gotchas — so your assistant doesn't have to guess.

## Skills

| Skill | Use it when you're… |
|---|---|
| [`ojin-stv-integration`](ojin-stv-integration/SKILL.md) | Wiring `ojin.stv.OjinSTVClient` into **any** async pipeline or media transport (the framework-agnostic foundation). |
| [`ojin-stv-pipecat`](ojin-stv-pipecat/SKILL.md) | Adding an Ojin talking-avatar to a **Pipecat** pipeline — the thin `FrameProcessor` adapter. |

Start with `ojin-stv-integration` for the mental model; reach for `ojin-stv-pipecat`
when your transport is Pipecat. The same pattern (inject a custom output sink, feed
`send_tts_audio()`) is designed to adapt to LiveKit Agents or any other framework —
though Pipecat is the only one with a reference adapter shipped so far.

## Using them

**Claude Code / Claude Agent SDK:** copy a skill folder into your project's
`.claude/skills/` directory (or your user-level `~/.claude/skills/`):

```bash
cp -r skills/ojin-stv-pipecat /path/to/your-project/.claude/skills/
```

Claude discovers it automatically and loads it when your request matches the skill's
`description`. No registration step.

**Other assistants:** point your tool at these `SKILL.md` files, or just paste the
relevant one into context. They're plain Markdown — no runtime, no dependencies.

> These skills target the public `ojin-client[stv]` package. They were written against
> a specific version of the API; each skill ends with a short "verify against your
> installed version" step. If a symbol in a skill doesn't match what you have
> installed, trust the installed source and tell us — open an issue at
> https://github.com/ojinai/python-sdk/issues.
