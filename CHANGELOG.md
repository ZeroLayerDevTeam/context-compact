# Changelog

## Unreleased

- Measure the real `usage` Anthropic returns (input, cache read/write, output) per request:
  logged with the size of `system`, `tools` and `messages`, and totals in `/_compactador/status`.
- Calibrate the token estimator per conversation with that real usage: the trigger and the kept
  tail now follow real tokens (the estimate was about 1.5x low).
- `configurar.py` sets `ENABLE_TOOL_SEARCH=true`: behind a proxy (or with `auto` on a 1M context)
  Claude Code sends every MCP tool definition on each request (measured 135 tools, ~60K tokens).

## 0.1.0

- First public release: Anthropic API proxy that compacts long Claude Code conversations with
  Ollama in the background, `configurar.py` to point Claude Code at it, systemd unit and tests.
- Only localhost may connect by default; other machines are allowed with `COMPACTADOR_ALLOW`.
