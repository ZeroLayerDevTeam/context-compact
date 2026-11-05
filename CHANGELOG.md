# Changelog

## 0.4.0

- Early summaries: summarising starts at `CONTEXT_COMPACT_EARLY` (75 %) of the trigger and the
  result is applied only once the trigger is really passed, so a summary is usually ready in time
  and no extra tokens are sent.
- `history_search` MCP tool at `/mcp` (streamable HTTP, `CONTEXT_COMPACT_MCP`): the original text of
  everything summarised is archived per conversation and Claude can search it for exact paths,
  commands and errors. The summary header carries the conversation id.
- `ensure_mcp.py`: registers the MCP server in every Claude Code config dir (also ccm accounts);
  idempotent, suitable for a `SessionStart` hook.
- Optional key-fact extraction (`CONTEXT_COMPACT_EXTRACT`, off by default): a second Ollama pass in
  JSON mode adds a "Key facts" list (paths, commands, decisions, pending items) to each summary.
- Dashboard: coverage coloured (red < 30 %, amber < 60 %), recent errors table, active settings.

## 0.3.0

- Trim old tool results (head + tail, `CONTEXT_COMPACT_TRIM_CHARS`) and drop `thinking` blocks of
  finished turns in the tail sent after a summary. Boundaries are fixed per summary, so the prompt
  cache keeps hitting.
- Summary cache: up to `CONTEXT_COMPACT_MAX_STATES` summaries per conversation; after a rewind or
  fork the newest one that still matches is reused instead of summarising from scratch.
- Fidelity check: URLs, paths and error messages missing from the LLM summary are appended
  verbatim ("Verbatim details not covered above"); coverage is logged and shown.
- Dashboard at `/_context-compact/` (tokens and API-equivalent USD saved/spent, cache hit,
  per-conversation summaries). Stats persist across restarts (`stats.json`).
- Ollama: `keep_alive` 30m, `num_predict` cap, `think: false`, per-call speed logged.

## 0.2.0

- Everything in English: files renamed (`context_compact.py`, `configure.py`,
  `test_context_compact.py`, `context-compact.service`), code, logs and summarisation prompt.
- **Breaking:** environment variables renamed `COMPACTADOR_*` -> `CONTEXT_COMPACT_*`, data dir
  `~/.compactador` -> `~/.context-compact`, status endpoint `/_context-compact/status`,
  `configure.py --remove` replaces `--quitar`.
- New `CONTEXT_COMPACT_LANGUAGE` (default `English`): language the summaries are written in.

- Measure the real `usage` Anthropic returns (input, cache read/write, output) per request:
  logged with the size of `system`, `tools` and `messages`, and totals in `/_context-compact/status`.
- Calibrate the token estimator per conversation with that real usage: the trigger and the kept
  tail now follow real tokens (the estimate was about 1.5x low).
- `configure.py` sets `ENABLE_TOOL_SEARCH=true`: behind a proxy (or with `auto` on a 1M context)
  Claude Code sends every MCP tool definition on each request (measured 135 tools, ~60K tokens).

## 0.1.0

- First public release: Anthropic API proxy that compacts long Claude Code conversations with
  Ollama in the background, `configure.py` (then `configurar.py`) to point Claude Code at it, systemd unit and tests.
- Only localhost may connect by default; other machines are allowed with `COMPACTADOR_ALLOW`.
