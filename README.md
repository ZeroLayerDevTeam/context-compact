# context-compact

A local proxy for the Anthropic API that compacts long [Claude Code](https://claude.com/claude-code)
conversations with a local LLM through [Ollama](https://ollama.com), instead of spending your
Claude plan on the native auto-compaction.

**Your conversation stays whole.** Claude Code keeps the full history on screen (scroll,
transcript, rewind); only what is sent to the API changes: `[summary + recent messages]` instead
of the entire history.

- Zero dependencies: Python 3 standard library only, one file.
- Summaries are built in the background, so requests are never delayed while a summary runs.
- Incremental: each new summary starts from the previous one.
- The summary stays fixed until the next compaction, so the prompt cache keeps working.
- Fail-safe: if anything goes wrong (Ollama down, GPU busy, odd JSON, a rewind), the request is
  forwarded untouched. At worst it does not save tokens; it never breaks the session.

## How it works

Claude Code talks to the proxy through `ANTHROPIC_BASE_URL`. Every request is forwarded as-is to
`api.anthropic.com`, except `POST /v1/messages` (and `count_tokens`) of long conversations:

1. When the compactable part of a conversation exceeds `COMPACTADOR_TRIGGER` tokens (35k,
   estimated), a background thread summarises `messages[start:cut]` with Ollama. The cut falls
   on a user message that leaves at least `COMPACTADOR_KEEP` (10k) recent tokens untouched.
2. Once the summary is ready, the following requests carry
   `[system reminders of the first message + summary + messages from the cut]`.
3. Conversation state is keyed by a hash of the first message and validated with a hash of the
   conversation "skeleton" (text and tool ids). If it no longer matches (rewind, edit, native
   compaction), the summary is discarded and the request passes untouched.
4. A failed summary is retried after `COMPACTADOR_RETRY_S` seconds.

## Install

Requirements: Python 3.8+, Ollama with a chat model pulled (default `gemma3:12b`, ~8.4 GB of
VRAM), and Claude Code.

```sh
git clone https://github.com/MattBlackDev1987/context-compact.git
cd context-compact
ollama pull gemma3:12b
python3 compactador.py            # listens on 0.0.0.0:8787
```

To run it as a service on Linux, edit `compactador.service` (user and paths) and:

```sh
sudo cp compactador.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now compactador
```

Point Claude Code at the proxy on every machine that should use it:

```sh
python3 configurar.py http://127.0.0.1:8787     # same machine as the proxy
python3 configurar.py http://<host>:8787        # another machine on your LAN
python3 configurar.py --quitar                  # back to talking to the API directly
```

`configurar.py` sets `ANTHROPIC_BASE_URL` in `~/.claude/settings.json` (and in every
`~/.ccm/accounts/*/settings.json` if you use ccm) and raises `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`
to 85. Restart your Claude Code sessions afterwards: running sessions keep their old settings.

Leave native auto-compaction **on**. With the proxy working, the context Claude Code sees stays
small and the native compaction never triggers; it only acts as a safety net when the proxy
cannot summarise (Ollama down).

Status: `curl http://<host>:8787/_compactador/status`. Log and state: `~/.compactador/`.

## Configuration

All settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `COMPACTADOR_HOST` | `0.0.0.0` | Listen address |
| `COMPACTADOR_PORT` | `8787` | Listen port |
| `COMPACTADOR_ALLOW` | `127.0.0.1,::1` | Client IPs allowed to connect (comma-separated) |
| `COMPACTADOR_UPSTREAM` | `api.anthropic.com` | Upstream API host |
| `COMPACTADOR_OLLAMA` | `http://127.0.0.1:11434` | Ollama URL |
| `COMPACTADOR_MODEL` | `gemma3:12b` | Model used for summaries |
| `COMPACTADOR_TRIGGER` | `35000` | Compactable tokens that trigger a summary |
| `COMPACTADOR_KEEP` | `10000` | Recent tokens always sent unsummarised |
| `COMPACTADOR_CHUNK_CHARS` | `36000` | Characters of history per summarisation step |
| `COMPACTADOR_NUM_CTX` | `16384` | Ollama context window for summaries |
| `COMPACTADOR_RETRY_S` | `120` | Seconds before retrying a failed summary |
| `COMPACTADOR_DATA` | `~/.compactador` | Log and per-conversation state |

The Ollama server can live on another machine (for example the one with the GPU): set
`COMPACTADOR_OLLAMA`, or forward its port over SSH.

## Security

The proxy forwards your Claude credentials (OAuth tokens or API keys). Keep it on your LAN, list
only your own machines in `COMPACTADOR_ALLOW`, and **never expose it to the internet** or put it
behind a public reverse proxy.

## Tests

```sh
python3 -m unittest test_compactador
```

## License

[MIT](LICENSE)
