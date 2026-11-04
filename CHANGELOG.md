# Changelog

## 0.1.0

- First public release: Anthropic API proxy that compacts long Claude Code conversations with
  Ollama in the background, `configurar.py` to point Claude Code at it, systemd unit and tests.
- Only localhost may connect by default; other machines are allowed with `COMPACTADOR_ALLOW`.
