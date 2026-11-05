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

1. When the compactable part of a conversation exceeds `CONTEXT_COMPACT_TRIGGER` tokens (35k,
   estimated), a background thread summarises `messages[start:cut]` with Ollama. The cut falls
   on a user message that leaves at least `CONTEXT_COMPACT_KEEP` (10k) recent tokens untouched.
2. Once the summary is ready, the following requests carry
   `[system reminders of the first message + summary + messages from the cut]`.
3. Conversation state is keyed by a hash of the first message and validated with a hash of the
   conversation "skeleton" (text and tool ids). If it no longer matches (rewind, edit, native
   compaction), the summary is discarded and the request passes untouched.
4. A failed summary is retried after `CONTEXT_COMPACT_RETRY_S` seconds.
5. Summaries start early, at `CONTEXT_COMPACT_EARLY` (75 %) of the trigger, so one is usually
   ready when the trigger is reached. An early summary is only applied once the history pending
   since its base passes the trigger, so it costs no extra tokens.
6. The original text of everything summarised is archived per conversation and can be searched
   by Claude through the `history_search` MCP tool (see below).

### `history_search` (MCP)

The proxy also serves an MCP server (streamable HTTP) at `/mcp` with one tool, `history_search`:
it returns the original messages behind a summary that match a query, for when Claude needs an
exact path, command or error that the summary lost. The summary header tells Claude which
conversation id to pass. Register it once per Claude Code config dir:

```bash
claude mcp add --transport http -s user context-compact http://<host>:8787/mcp
```

or run `python ensure_mcp.py` (registers it in `~/.claude` and in every ccm account under
`~/.ccm/accounts/`, using `ANTHROPIC_BASE_URL`; it does nothing when already registered, so it can
run from a `SessionStart` hook).

## Install

Requirements: Python 3.8+, Ollama with a chat model pulled (default `gemma3:12b`, ~8.4 GB of
VRAM), and Claude Code.

```sh
git clone https://github.com/ZeroLayerDevTeam/context-compact.git
cd context-compact
ollama pull gemma3:12b
python3 context_compact.py            # listens on 0.0.0.0:8787
```

To run it as a service on Linux, edit `context-compact.service` (user and paths) and:

```sh
sudo cp context-compact.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now context-compact
```

Point Claude Code at the proxy on every machine that should use it:

```sh
python3 configure.py http://127.0.0.1:8787     # same machine as the proxy
python3 configure.py http://<host>:8787        # another machine on your LAN
python3 configure.py --remove                    # back to talking to the API directly
```

`configure.py` sets `ANTHROPIC_BASE_URL` in `~/.claude/settings.json` (and in every
`~/.ccm/accounts/*/settings.json` if you use ccm) and raises `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`
to 85. Restart your Claude Code sessions afterwards: running sessions keep their old settings.

Leave native auto-compaction **on**. With the proxy working, the context Claude Code sees stays
small and the native compaction never triggers; it only acts as a safety net when the proxy
cannot summarise (Ollama down).

Status: `curl http://<host>:8787/_context-compact/status`. Log and state: `~/.context-compact/`.

## Configuration

All settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CONTEXT_COMPACT_HOST` | `0.0.0.0` | Listen address |
| `CONTEXT_COMPACT_PORT` | `8787` | Listen port |
| `CONTEXT_COMPACT_ALLOW` | `127.0.0.1,::1` | Client IPs allowed to connect (comma-separated) |
| `CONTEXT_COMPACT_UPSTREAM` | `api.anthropic.com` | Upstream API host |
| `CONTEXT_COMPACT_OLLAMA` | `http://127.0.0.1:11434` | Ollama URL |
| `CONTEXT_COMPACT_MODEL` | `gemma3:12b` | Model used for summaries |
| `CONTEXT_COMPACT_LANGUAGE` | `English` | Language the summaries are written in |
| `CONTEXT_COMPACT_TRIGGER` | `35000` | Compactable tokens that trigger a summary |
| `CONTEXT_COMPACT_KEEP` | `10000` | Recent tokens always sent unsummarised |
| `CONTEXT_COMPACT_CHUNK_CHARS` | `36000` | Characters of history per summarisation step |
| `CONTEXT_COMPACT_NUM_CTX` | `16384` | Ollama context window for summaries |
| `CONTEXT_COMPACT_RETRY_S` | `120` | Seconds before retrying a failed summary |
| `CONTEXT_COMPACT_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded |
| `CONTEXT_COMPACT_NUM_PREDICT` | `2500` | Max tokens of each summary |
| `CONTEXT_COMPACT_TRIM_CHARS` | `3000` | Old tool results longer than this are cut (0 = off) |
| `CONTEXT_COMPACT_TRIM_KEEP_MSGS` | `8` | Newest messages never trimmed |
| `CONTEXT_COMPACT_STRIP_THINKING` | `1` | Drop thinking blocks of finished turns (0 = keep) |
| `CONTEXT_COMPACT_MAX_STATES` | `6` | Cached summaries per conversation (rewind/fork reuse) |
| `CONTEXT_COMPACT_EARLY` | `0.75` | Fraction of the trigger at which summarising starts |
| `CONTEXT_COMPACT_MCP` | `1` | Serve `/mcp` with `history_search` and archive summarised text (0 = off) |
| `CONTEXT_COMPACT_EXTRACT` | `0` | Extra Ollama pass adding a "Key facts" list to each summary (+1-2k tokens per request) |
| `CONTEXT_COMPACT_PRICE_IN` / `_OUT` | `5` / `25` | USD per MTok, dashboard estimate only |

Dashboard: `http://<host>:8787/_context-compact/` (same allow-list as the proxy).
| `CONTEXT_COMPACT_DATA` | `~/.context-compact` | Log and per-conversation state |

The Ollama server can live on another machine (for example the one with the GPU): set
`CONTEXT_COMPACT_OLLAMA`, or forward its port over SSH.

## Security

The proxy forwards your Claude credentials (OAuth tokens or API keys). Keep it on your LAN, list
only your own machines in `CONTEXT_COMPACT_ALLOW`, and **never expose it to the internet** or put it
behind a public reverse proxy.

## Tests

```sh
python3 -m unittest test_context_compact
```

## License

[MIT](LICENSE)
