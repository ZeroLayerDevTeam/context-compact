#!/usr/bin/env python3
"""context-compact — Anthropic API proxy that compacts the context with a local LLM.

Claude Code points here through ANTHROPIC_BASE_URL. Every request is forwarded as-is to
api.anthropic.com except POST /v1/messages (and count_tokens) of long conversations: when the
compactable part exceeds CONTEXT_COMPACT_TRIGGER tokens, a background thread summarises the old
messages with Ollama and, once the summary is ready, the following requests carry
[summary + recent messages] instead of the whole history. Claude Code's UI keeps the full
conversation (scroll, transcript); only what reaches the API changes.

If anything fails (Ollama down, GPU busy, odd JSON) the request passes untouched: it never breaks
the session, at worst it does not save tokens. Standard library only.
"""
import copy
import hashlib
import http.client
import json
import logging
import logging.handlers
import os
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

E = os.environ.get
HOST = E("CONTEXT_COMPACT_HOST", "0.0.0.0")
PORT = int(E("CONTEXT_COMPACT_PORT", "8787"))
UPSTREAM = E("CONTEXT_COMPACT_UPSTREAM", "api.anthropic.com")
ALLOW = {a.strip() for a in E("CONTEXT_COMPACT_ALLOW", "127.0.0.1,::1").split(",") if a.strip()}
OLLAMA = E("CONTEXT_COMPACT_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
MODEL = E("CONTEXT_COMPACT_MODEL", "gemma3:12b")
LANGUAGE = E("CONTEXT_COMPACT_LANGUAGE", "English")   # language of the summaries
TRIGGER = int(E("CONTEXT_COMPACT_TRIGGER", "35000"))   # compactable tokens that trigger a summary
KEEP = int(E("CONTEXT_COMPACT_KEEP", "10000"))         # recent tokens always sent unsummarised
CHUNK_CHARS = int(E("CONTEXT_COMPACT_CHUNK_CHARS", "36000"))
NUM_CTX = int(E("CONTEXT_COMPACT_NUM_CTX", "16384"))
RETRY_S = int(E("CONTEXT_COMPACT_RETRY_S", "120"))
KEEP_ALIVE = E("CONTEXT_COMPACT_KEEP_ALIVE", "30m")          # keep the model loaded between summaries
NUM_PREDICT = int(E("CONTEXT_COMPACT_NUM_PREDICT", "2500"))  # cap on summary length (tokens)
TRIM_CHARS = int(E("CONTEXT_COMPACT_TRIM_CHARS", "3000"))    # old tool results longer than this are cut
TRIM_KEEP_MSGS = int(E("CONTEXT_COMPACT_TRIM_KEEP_MSGS", "8"))  # newest messages never trimmed
STRIP_THINKING = E("CONTEXT_COMPACT_STRIP_THINKING", "1") != "0"
MAX_STATES = int(E("CONTEXT_COMPACT_MAX_STATES", "6"))       # cached summaries per conversation
PRICE_IN = float(E("CONTEXT_COMPACT_PRICE_IN", "5"))          # USD per MTok, dashboard estimate only
PRICE_OUT = float(E("CONTEXT_COMPACT_PRICE_OUT", "25"))
EARLY = float(E("CONTEXT_COMPACT_EARLY", "0.75"))   # start summarising at this fraction of TRIGGER
EXTRACT = E("CONTEXT_COMPACT_EXTRACT", "0") != "0"  # extra Ollama pass extracting key facts as JSON
MCP = E("CONTEXT_COMPACT_MCP", "1") != "0"           # /mcp endpoint with the history_search tool
DATA = os.path.expanduser(E("CONTEXT_COMPACT_DATA", "~/.context-compact"))
STATE_DIR = os.path.join(DATA, "state")
ARCHIVE_DIR = os.path.join(DATA, "archive")
CHARS_PER_TOKEN = 3.5
IMAGE_TOKENS = 1600

HEADER = ("[context-compact] The earlier messages of this conversation have been replaced by the "
          "summary below to save context. The conversation continues normally afterwards; do not "
          "mention this summary unless the user asks.\n\n")
HEADER_MCP = ("[context-compact] The earlier messages of this conversation have been replaced by the "
              "summary below to save context. The conversation continues normally afterwards; do not "
              "mention this summary unless the user asks. If you need an exact detail from before the "
              "summary (a path, command, error, value), the `history_search` tool of the "
              "`context-compact` MCP server returns the original text: conversation=\"{key}\".\n\n")


def header(key=None):
    return HEADER_MCP.format(key=key) if MCP and key else HEADER

log = logging.getLogger("context-compact")

# ---------------------------------------------------------------- pure logic (tested)

def blocks(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return c if isinstance(c, list) else []


def text_of(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") in ("image", "document"):
                out.append(f"[{b['type']}]")
    return "\n".join(out)


def est_block(b):
    t = b.get("type")
    if t in ("image", "document"):
        return IMAGE_TOKENS
    if t == "tool_result":
        c = b.get("content")
        if isinstance(c, list):
            return sum(est_block(x) for x in c if isinstance(x, dict)) + 10
        return len(str(c or "")) / CHARS_PER_TOKEN + 10
    if t == "text":
        return len(b.get("text", "")) / CHARS_PER_TOKEN
    if t == "thinking":
        return len(b.get("thinking", "")) / CHARS_PER_TOKEN
    return len(json.dumps(b, ensure_ascii=False)) / CHARS_PER_TOKEN


def est_msg(msg):
    return sum(est_block(b) for b in blocks(msg) if isinstance(b, dict)) + 4


def is_reminder(b):
    return b.get("type") == "text" and b.get("text", "").lstrip().startswith("<system-reminder>")


def has_tool_result(msg):
    return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks(msg))


def skeleton_hash(msgs):
    """Stable fingerprint of the history: ignores cache_control and tool_result content
    (Claude Code moves/clears them between requests), detects rewinds/edits."""
    h = hashlib.sha256()
    for m in msgs:
        h.update(m.get("role", "").encode())
        for b in blocks(m):
            if not isinstance(b, dict):
                continue
            t = b.get("type", "")
            h.update(t.encode())
            if t == "text":
                h.update(b.get("text", "").encode())
            elif t == "tool_use":
                h.update(str(b.get("id")).encode())
            elif t == "tool_result":
                h.update(str(b.get("tool_use_id")).encode())
        h.update(b"\x00")
    return h.hexdigest()


def conv_key(msgs):
    return hashlib.sha256(json.dumps(blocks(msgs[0]) if msgs else [], sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:24]


def compactable_tokens(msgs, start=0):
    """Tokens of msgs[start:], not counting the system reminders of the first message
    (CLAUDE.md etc., which are always kept)."""
    total = 0.0
    for i, m in enumerate(msgs[start:], start):
        if i == 0:
            total += sum(est_block(b) for b in blocks(m) if isinstance(b, dict) and not is_reminder(b))
        else:
            total += est_msg(m)
    return total


def choose_cut(msgs, keep, min_cut=1):
    """Index of the user message that becomes the first one after compaction. Prefers a real
    user prompt; if the turn is a long run of tool calls, cuts at a tool_result."""
    tail = 0.0
    best_any = None
    for i in range(len(msgs) - 1, min_cut - 1, -1):
        tail += est_msg(msgs[i])
        if tail < keep or msgs[i].get("role") != "user":
            continue
        if not has_tool_result(msgs[i]):
            return i
        if best_any is None:
            best_any = i
        if tail > 2 * keep:          # no real prompt nearby: cut inside the turn
            return best_any
    return best_any


def tool_result_as_text(b):
    flag = " (error)" if b.get("is_error") else ""
    body = text_of(b.get("content"))
    if len(body) > 4000:
        body = body[:4000] + " …[truncated]"
    return {"type": "text", "text": f"[Tool result{flag}]\n{body}"}


def trim_tool_result(b, limit):
    """Copy of a tool_result whose text is cut to head + tail of `limit` chars (images kept)."""
    def cut(t):
        if len(t) <= limit:
            return t
        head, tail = int(limit * 0.75), int(limit * 0.25)
        return f"{t[:head]}\n…[context-compact: {len(t) - head - tail} chars trimmed]…\n{t[-tail:]}"
    c = b.get("content")
    nb = dict(b)
    if isinstance(c, str):
        nb["content"] = cut(c)
    elif isinstance(c, list):
        nb["content"] = [dict(x, text=cut(x.get("text", ""))) if isinstance(x, dict) and x.get("type") == "text"
                         else x for x in c]
    return nb


def matches(msgs, state):
    cut = state["cut"]
    return len(msgs) > cut and skeleton_hash(msgs[:cut + 1]) == state["hash"]


def trim_old(msgs, start, trim_before, think_before):
    """msgs[start:] with long tool results trimmed (index < trim_before) and thinking blocks
    removed from finished turns (index < think_before). The boundaries are fixed per summary, so
    the output stays byte-identical between compactions and the prompt cache keeps working."""
    out = []
    for i, m in enumerate(msgs[start:], start):
        bl = m.get("content")
        if i >= trim_before or not isinstance(bl, list):
            out.append(m)
            continue
        nb = []
        for b in bl:
            if not isinstance(b, dict):
                nb.append(b)
            elif b.get("type") == "tool_result" and TRIM_CHARS:
                nb.append(trim_tool_result(b, TRIM_CHARS))
            elif (STRIP_THINKING and i < think_before and m.get("role") == "assistant"
                  and b.get("type") in ("thinking", "redacted_thinking")):
                continue
            else:
                nb.append(b)
        if not any(isinstance(b, dict) and b.get("type") not in ("thinking", "redacted_thinking") for b in nb):
            nb = bl                       # never leave an assistant message without content
        out.append(dict(m, content=nb))
    return out


def trim_bounds(msgs, cut):
    """(trim_before, think_before) for a summary created now with this history."""
    trim_before = max(cut + 1, len(msgs) - TRIM_KEEP_MSGS)
    think_before = cut + 1
    for i in range(trim_before - 1, cut, -1):
        m = msgs[i]
        if m.get("role") == "user" and not has_tool_result(m):
            think_before = i              # last real prompt: the turns before it are finished
            break
    return trim_before, think_before


def armed(msgs, state, ratio=1.0):
    """A summary started early (CONTEXT_COMPACT_EARLY) is only used once the history since its
    base (the previous cut) really exceeds TRIGGER: until then the previous summary stays."""
    base = state.get("base")
    return base is None or compactable_tokens(msgs, base) * ratio > TRIGGER


def apply_state(msgs, state, key=None):
    """Return the compacted message list, or None if the state does not match this history."""
    cut = state["cut"]
    if not matches(msgs, state):
        return None
    reminders = [copy.deepcopy(b) for b in blocks(msgs[0]) if isinstance(b, dict) and is_reminder(b)]
    for b in reminders:
        b.pop("cache_control", None)
    first_blocks = []
    for b in blocks(msgs[cut]):
        if isinstance(b, dict) and b.get("type") == "tool_result":
            first_blocks.append(tool_result_as_text(b))
        else:
            first_blocks.append(b)
    summary = {"type": "text", "text": header(key) + state["summary"]}
    first = {"role": "user", "content": reminders + [summary] + first_blocks}
    rest = trim_old(msgs, cut + 1, state.get("trim_before", 0), state.get("think_before", 0))
    return [first] + rest


REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def clip(s, n):
    s = s.strip()
    return s if len(s) <= n else s[:n] + " …"


def render(msgs):
    out = []
    for m in msgs:
        who = "USER" if m.get("role") == "user" else "CLAUDE"
        for b in blocks(m):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                txt = REMINDER_RE.sub("", b.get("text", "")).strip()
                if txt:
                    out.append(f"[{who}] {clip(txt, 4000)}")
            elif t == "tool_use":
                out.append(f"[CLAUDE uses {b.get('name')}] {clip(json.dumps(b.get('input'), ensure_ascii=False), 800)}")
            elif t == "tool_result":
                flag = " ERROR" if b.get("is_error") else ""
                out.append(f"[result{flag}] {clip(REMINDER_RE.sub('', text_of(b.get('content'))), 1200)}")
            elif t in ("image", "document"):
                out.append(f"[{who} attaches {t}]")
    return "\n".join(out)


FILE_TOOLS = {"Edit": "mod", "Write": "mod", "MultiEdit": "mod", "NotebookEdit": "mod", "Read": "read"}


def touched_files(msgs):
    mod, read = [], []
    for m in msgs:
        for b in blocks(m):
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in FILE_TOOLS:
                inp = b.get("input") or {}
                p = inp.get("file_path") or inp.get("notebook_path") or inp.get("path")
                if p:
                    (mod if FILE_TOOLS[b["name"]] == "mod" else read).append(p)
    return mod, read


def archive_entries(msgs, start=0):
    """[index, role, text] of each message, for history_search (much less clipped than render)."""
    out = []
    for i, m in enumerate(msgs[start:], start):
        parts = []
        for b in blocks(m):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                txt = REMINDER_RE.sub("", b.get("text", "")).strip()
                if txt:
                    parts.append(clip(txt, 20000))
            elif t == "tool_use":
                parts.append(f"[uses {b.get('name')}] {clip(json.dumps(b.get('input'), ensure_ascii=False), 6000)}")
            elif t == "tool_result":
                flag = " ERROR" if b.get("is_error") else ""
                parts.append(f"[result{flag}] {clip(REMINDER_RE.sub('', text_of(b.get('content'))), 20000)}")
        if parts:
            out.append([i, m.get("role", "?"), "\n".join(parts)])
    return out


WORD_RE = re.compile(r"[\w./\\:-]{2,}", re.U)


def search_entries(entries, query, limit=5, max_chars=8000):
    """Entries ranked by how many query terms they contain (exact phrase first), with a snippet
    around the first hit. Plain text, newest first on ties."""
    q = query.strip().lower()
    terms = [t for t in dict.fromkeys(WORD_RE.findall(q))] or ([q] if q else [])
    if not terms:
        return "Empty query."
    scored = []
    for idx, role, text in entries:
        low = text.lower()
        hits = sum(1 for t in terms if t in low)
        if not hits:
            continue
        score = hits + (len(terms) if len(terms) > 1 and q in low else 0)
        scored.append((score, idx, role, text, low))
    if not scored:
        return f"No match for {query!r} in the summarised history."
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    out, budget = [], max_chars
    per = max(600, max_chars // max(1, min(limit, len(scored))))
    for score, idx, role, text, low in scored[:limit]:
        pos = next((low.find(t) for t in ([q] + terms) if t and low.find(t) >= 0), 0)
        a = max(0, pos - per // 3)
        snip = text[a:a + per]
        snip = ("…" if a else "") + snip + ("…" if a + per < len(text) else "")
        piece = f"--- message {idx} ({role}) ---\n{snip}"
        if len(piece) > budget:
            break
        out.append(piece)
        budget -= len(piece)
    return f"{len(scored)} matching message(s), showing {len(out)}:\n\n" + "\n\n".join(out)


URL_RE = re.compile(r"https?://[^\s\"'`<>()\[\]{}]+")
PATH_RE = re.compile(r"(?<![\w/])(?:[A-Za-z]:\\{1,2}|~/|/(?:home|etc|tmp|var|opt|usr|mnt|srv)/)[^\s\"'`<>|,;()\[\]{}]+")
ERR_RE = re.compile(r"^\[result ERROR\] (.{0,200})", re.M)


def facts(fragment):
    """Verbatim identifiers worth keeping: URLs, paths and error messages, in order, no duplicates."""
    found = [u.rstrip(".:") for u in URL_RE.findall(fragment)]
    found += [p.replace("\\\\", "\\").rstrip(".:") for p in PATH_RE.findall(fragment)]
    found += [e.strip() for e in ERR_RE.findall(fragment)]
    return [f for f in dict.fromkeys(found) if len(f) > 6]


def missing_facts(fragment, summary):
    """(facts not present in the summary, coverage 0..1)."""
    fs = facts(fragment)
    miss = [f for f in fs if f not in summary]
    return miss, (1 - len(miss) / len(fs)) if fs else 1.0


def chunks(text, size):
    lines, cur, n = [], [], 0
    for line in text.split("\n"):
        if n + len(line) > size and cur:
            lines.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    if cur:
        lines.append("\n".join(cur))
    return lines

# ---------------------------------------------------------------- summaries with Ollama

SYSTEM = ("You summarise working conversations between a user and Claude Code (a coding "
          "assistant) so that Claude can continue without the history. Write in {lang}. Keep "
          "paths, function names, commands, error messages, IDs, URLs, numbers and decisions "
          "verbatim. Do not invent anything that does not appear.")


def ollama_chat(system, prompt, json_mode=False):
    req_body = {"model": MODEL, "stream": False, "keep_alive": KEEP_ALIVE, "think": False,
                "options": {"num_ctx": NUM_CTX, "temperature": 0.2, "num_predict": NUM_PREDICT},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    if json_mode:
        req_body["format"] = "json"
    req = urllib.request.Request(OLLAMA + "/api/chat", json.dumps(req_body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.loads(r.read())
    out = res["message"]["content"].strip()
    if not out:
        raise RuntimeError("Ollama returned an empty answer")
    n, ns = res.get("eval_count", 0), res.get("eval_duration", 0) / 1e9
    log.info("ollama%s: prompt %d tok %.1fs, output %d tok %.1fs (%.0f tok/s), load %.1fs",
             " [json]" if json_mode else "", res.get("prompt_eval_count", 0),
             res.get("prompt_eval_duration", 0) / 1e9, n, ns, n / ns if ns else 0,
             res.get("load_duration", 0) / 1e9)
    return out


def ask_ollama(prev, fragment):
    prompt = (f"PREVIOUS SUMMARY:\n{prev or '(none)'}\n\nNEW CONVERSATION FRAGMENT:\n{fragment}\n\n"
              "Write the updated summary (about 1000 words at most) merging the previous summary "
              f"and the new fragment, in {LANGUAGE}, with these sections:\n## User requests\n"
              "## Work status\n## User decisions and preferences\n## Relevant files and commands\n"
              "## Errors found and how they were solved\n## Pending / next step")
    return ollama_chat(SYSTEM.format(lang=LANGUAGE), prompt)


EXTRACT_KINDS = ("decisions", "paths", "commands", "identifiers", "errors", "pending")
EXTRACT_SYSTEM = ("You extract facts from a conversation between a user and Claude Code. Answer only "
                  "with JSON. Copy every value verbatim from the text; never invent or paraphrase "
                  "identifiers.")


def parse_extract(raw):
    """Facts JSON from Ollama -> list of "kind: value" strings (tolerant to odd shapes)."""
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    out = []
    if not isinstance(data, dict):
        return out
    for kind in EXTRACT_KINDS:
        vals = data.get(kind)
        if isinstance(vals, str):
            vals = [vals]
        for v in vals if isinstance(vals, list) else []:
            if isinstance(v, (str, int, float)) and str(v).strip():
                out.append(f"{kind}: {clip(str(v), 300)}")
    return out


def extract_facts(fragment):
    prompt = (f"CONVERSATION FRAGMENT:\n{fragment}\n\nReturn a JSON object with these keys, each a "
              "list of short strings (at most 15 per key): " + ", ".join(EXTRACT_KINDS) + ". "
              "decisions = what the user decided or asked for; paths = files and directories; "
              "commands = shell commands that were run or must be run; identifiers = IDs, URLs, "
              "ports, versions, names of services; errors = error messages and their fix; "
              "pending = what is still to do.")
    return parse_extract(ollama_chat(EXTRACT_SYSTEM, prompt, json_mode=True))


def summarize(prev_summary, msgs):
    """(summary, verbatim facts missing from it, coverage, extracted facts)."""
    summary = prev_summary
    text = render(msgs)
    extracted = []
    for frag in chunks(text, CHUNK_CHARS):
        summary = ask_ollama(summary, frag)
        if EXTRACT:
            try:
                extracted += extract_facts(frag)
            except Exception as e:  # noqa: BLE001 — extraction is optional
                log.warning("fact extraction failed: %s", e)
    miss, cov = missing_facts(text, summary)
    return summary, miss, cov, extracted

# ---------------------------------------------------------------- per-conversation state

LOCK = threading.Lock()
STATES = {}      # key -> cached states [{cut, hash, summary, files_mod, files_read, facts, t, ...}]
JOBS = set()     # keys with a summary in progress
FAILED = {}      # key -> time of the last failure
RECENT_ERRORS = []   # [(time, key, message)] last summarisation / proxy errors, for the dashboard
STATS = {"requests": 0, "compacted": 0, "summaries": 0, "errors": 0, "saved_tokens": 0,
         # real usage returned by Anthropic (all /v1/messages requests)
         "input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
         "output_tokens": 0, "trimmed_chars": 0}
STATS_PERSIST = tuple(STATS)
_last_stats_save = [0.0]
USAGE_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
USAGE_RE = re.compile(rb'"(input_tokens|cache_read_input_tokens|cache_creation_input_tokens|output_tokens)"\s*:\s*(\d+)')


def parse_usage(data):
    """Maximum of each usage field in a response (JSON or SSE: message_start + message_delta)."""
    u = dict.fromkeys(USAGE_FIELDS, 0)
    for k, v in USAGE_RE.findall(data):
        k = k.decode()
        u[k] = max(u[k], int(v))
    return u


RATIO = {}       # key -> real / estimated tokens (calibrates the estimator with Anthropic usage)


def est_body(body):
    """Estimate of the whole input with the same estimator as compactable_tokens."""
    fixed = sum(len(json.dumps(body.get(k) or "", ensure_ascii=False)) for k in ("system", "tools"))
    return fixed / CHARS_PER_TOKEN + sum(est_msg(m) for m in body.get("messages") or [])


def record_usage(u, compacted, key, conv=None, est=0.0):
    total = u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]
    with LOCK:
        for k in USAGE_FIELDS:
            STATS[k] += u[k]
        if conv and est > 1000 and total:
            RATIO[conv] = min(max(total / est, 0.5), 5.0)
    save_stats()
    if total:
        log.info("usage %s%s: in=%d cache_read=%d cache_write=%d out=%d (hit %.0f%%)",
                 key or "-", " [compacted]" if compacted else "", u["input_tokens"],
                 u["cache_read_input_tokens"], u["cache_creation_input_tokens"], u["output_tokens"],
                 100 * u["cache_read_input_tokens"] / total)


def state_path(key):
    return os.path.join(STATE_DIR, key + ".json")


def stats_path():
    return os.path.join(DATA, "stats.json")


def load_stats():
    try:
        with open(stats_path(), encoding="utf-8") as f:
            saved = json.load(f)
        for k in STATS_PERSIST:
            STATS[k] = int(saved.get(k, 0))
    except (OSError, ValueError):
        pass


def save_stats(force=False):
    if not force and time.time() - _last_stats_save[0] < 30:
        return
    _last_stats_save[0] = time.time()
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(stats_path() + ".tmp", "w", encoding="utf-8") as f:
            json.dump({k: STATS[k] for k in STATS_PERSIST}, f)
        os.replace(stats_path() + ".tmp", stats_path())
    except OSError:
        pass


def load_states(key):
    """Cached summaries of a conversation (also reads the old single-state file)."""
    sts = STATES.get(key)
    if sts is None:
        try:
            with open(state_path(key), encoding="utf-8") as f:
                data = json.load(f)
            sts = data["states"] if isinstance(data, dict) and "states" in data else [data]
            STATES[key] = sts
        except (OSError, ValueError, KeyError, TypeError):
            return []
    return sts


def best_state(msgs, sts):
    """The cached summary with the largest cut that still matches this history: after a rewind
    or a fork an older summary is reused instead of summarising again from scratch."""
    for st in sorted(sts, key=lambda s: s["cut"], reverse=True):
        if matches(msgs, st):
            return st
    return None


def save_state(key, st):
    sts = [s for s in load_states(key) if s["cut"] != st["cut"]] + [st]
    sts = sorted(sts, key=lambda s: s["t"])[-MAX_STATES:]
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = state_path(key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"states": sts}, f, ensure_ascii=False)
    os.replace(tmp, state_path(key))
    STATES[key] = sts


def archive_path(key):
    return os.path.join(ARCHIVE_DIR, key + ".json")


def save_archive(key, msgs, cut):
    """Original text of msgs[:cut] (what the summaries replace), for history_search."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    tmp = archive_path(key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cut": cut, "t": time.time(), "entries": archive_entries(msgs[:cut])}, f, ensure_ascii=False)
    os.replace(tmp, archive_path(key))


def find_archive(conversation):
    """Archive of a conversation by key or key prefix (the dashboard shows 8 chars)."""
    conversation = re.sub(r"[^0-9a-f]", "", (conversation or "").lower())
    if len(conversation) < 6:
        return None
    try:
        names = [n for n in os.listdir(ARCHIVE_DIR) if n.startswith(conversation) and n.endswith(".json")]
    except OSError:
        return None
    if not names:
        return None
    names.sort(key=lambda n: os.path.getmtime(os.path.join(ARCHIVE_DIR, n)), reverse=True)
    with open(os.path.join(ARCHIVE_DIR, names[0]), encoding="utf-8") as f:
        return json.load(f)


def note_error(key, msg):
    with LOCK:
        STATS["errors"] += 1
        RECENT_ERRORS.append((time.time(), (key or "-")[:8], clip(str(msg), 300)))
        del RECENT_ERRORS[:-10]


def cleanup_states(max_age_days=14):
    limit = time.time() - max_age_days * 86400
    for d in (STATE_DIR, ARCHIVE_DIR):
        try:
            for n in os.listdir(d):
                p = os.path.join(d, n)
                if os.path.getmtime(p) < limit:
                    os.remove(p)
        except OSError:
            pass


def build_summary(key, msgs, prev):
    """Background thread: summarise msgs[prev.cut:new_cut] on top of the previous summary and save the state."""
    try:
        start = prev["cut"] if prev else 0
        cut = choose_cut(msgs, KEEP / RATIO.get(key, 1.0), min_cut=start + 1)
        if cut is None:
            log.info("%s: no valid cut point", key)
            return
        span = msgs[start:cut]
        t0 = time.time()
        summary, miss, cov, extracted = summarize(prev["summary"] if prev else "", span)
        mod, read = touched_files(span)
        files_mod = list(dict.fromkeys((prev or {}).get("files_mod", []) + mod))[-60:]
        files_read = list(dict.fromkeys((prev or {}).get("files_read", []) + read))[-60:]
        listed = set(files_mod) | set(files_read)
        kept = [f for f in (prev or {}).get("facts", []) if f not in summary]
        fact_list = list(dict.fromkeys(kept + [f for f in miss if f not in listed]))[-40:]
        key_facts = list(dict.fromkeys((prev or {}).get("key_facts", []) + extracted))[-80:]
        extra = ""
        if key_facts:
            extra += "\n\n## Key facts (extracted)\n" + "\n".join(f"- {f}" for f in key_facts)
        if files_mod:
            extra += "\n\n## Modified files (exact list)\n" + "\n".join(f"- {p}" for p in files_mod)
        if files_read:
            extra += "\n\n## Read files (exact list)\n" + "\n".join(f"- {p}" for p in files_read)
        if fact_list:
            extra += "\n\n## Verbatim details not covered above\n" + "\n".join(f"- {f}" for f in fact_list)
        trim_before, think_before = trim_bounds(msgs, cut)
        # summary_for_next: the next summary starts from the LLM text, not from the lists
        # (they are regenerated every time)
        st = {"cut": cut, "hash": skeleton_hash(msgs[:cut + 1]), "summary": summary + extra,
              "summary_for_next": summary, "files_mod": files_mod, "files_read": files_read,
              "facts": fact_list, "key_facts": key_facts, "coverage": round(cov, 3),
              "trim_before": trim_before, "think_before": think_before,
              "seconds": round(time.time() - t0), "msgs": len(msgs), "t": time.time(),
              # early summary: only used once the history since `base` exceeds TRIGGER
              "base": start + 1 if prev else 0}
        if MCP:
            try:
                save_archive(key, msgs, cut)
            except OSError as e:
                log.warning("%s: archive not saved: %s", key, e)
        with LOCK:
            save_state(key, st)
            STATS["summaries"] += 1
        FAILED.pop(key, None)
        log.info("%s: summary ready (%d msgs -> cut at %d, %.0fs, %d chars, coverage %.0f%%, %d facts kept)",
                 key, len(span), cut, time.time() - t0, len(st["summary"]), cov * 100, len(fact_list))
    except Exception as e:  # noqa: BLE001 — any failure leaves the session uncompacted
        FAILED[key] = time.time()
        note_error(key, f"summarisation failed: {e}")
        log.warning("%s: summarisation failed: %s", key, e)
    finally:
        with LOCK:
            JOBS.discard(key)


def transform(body, count_only=False):
    """Return the body (dict) to send. May start a background summary."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return body
    key = conv_key(msgs)
    ratio = RATIO.get(key, 1.0)
    with LOCK:
        sts = load_states(key)
        # newest: latest matching summary (may be an early one not in use yet);
        # st: the one actually applied (newest summary that is armed)
        newest = best_state(msgs, sts) if sts else None
        st = best_state(msgs, [s for s in sts if armed(msgs, s, ratio)]) if newest else None
    out = apply_state(msgs, st, key) if st else None
    if sts and newest is None and key not in JOBS:
        log.info("%s: no cached summary matches this history (rewind/fork/native compaction)", key)
    pending = compactable_tokens(msgs, newest["cut"] + 1 if newest else 0) * ratio
    if not count_only and pending > TRIGGER * EARLY:
        with LOCK:
            busy = key in JOBS or time.time() - FAILED.get(key, 0) < RETRY_S
            if not busy:
                JOBS.add(key)
        if not busy:
            prev = None
            if newest:
                prev = dict(newest, summary=newest.get("summary_for_next", newest["summary"]))
            threading.Thread(target=build_summary, args=(key, copy.deepcopy(msgs), prev), daemon=True).start()
            log.info("%s: %d compactable tokens > %d (%.0f%% of trigger), summarising in the background",
                     key, pending, TRIGGER * EARLY, EARLY * 100)
    if out is None:
        return body
    new = dict(body)
    new["messages"] = out
    if not count_only:
        saved = int(compactable_tokens(msgs) - compactable_tokens(out))
        STATS["trimmed_chars"] += max(len(json.dumps(msgs[st["cut"] + 1:], ensure_ascii=False))
                                      - len(json.dumps(out[1:], ensure_ascii=False)), 0)
        STATS["compacted"] += 1
        STATS["saved_tokens"] += max(saved, 0)
    return new

# ---------------------------------------------------------------- status + dashboard

def status():
    sess = []
    with LOCK:
        for key, sts in STATES.items():
            if not sts:
                continue
            st = max(sts, key=lambda s: s["t"])
            sess.append({"key": key[:8], "cut": st["cut"], "msgs": st.get("msgs"), "chars": len(st["summary"]),
                         "coverage": st.get("coverage"), "seconds": st.get("seconds"),
                         "cached": len(sts), "age_min": round((time.time() - st["t"]) / 60)})
        errs = [{"age_min": round((time.time() - t) / 60), "key": k, "message": m}
                for t, k, m in reversed(RECENT_ERRORS)]
    sess.sort(key=lambda s: s["age_min"])
    m = 1e6
    spent = (STATS["input_tokens"] * PRICE_IN + STATS["cache_read_input_tokens"] * PRICE_IN * 0.1
             + STATS["cache_creation_input_tokens"] * PRICE_IN * 1.25 + STATS["output_tokens"] * PRICE_OUT) / m
    return dict(STATS, jobs=len(JOBS), sessions=len(STATES), trigger=TRIGGER, keep=KEEP, model=MODEL,
                language=LANGUAGE, early=EARLY, num_ctx=NUM_CTX, extract=EXTRACT, mcp=MCP,
                recent_errors=errs, usd_spent=round(spent, 2),
                # saved tokens would mostly have been cache reads (0.1x) and at worst fresh input (1x)
                usd_saved_min=round(STATS["saved_tokens"] * PRICE_IN * 0.1 / m, 2),
                usd_saved_max=round(STATS["saved_tokens"] * PRICE_IN / m, 2), session_list=sess[:30])


DASHBOARD = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>context-compact</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c1f24;--mut:#6b7280;--line:#e5e7eb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8eb;--mut:#9aa1ac;--line:#262b35}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin-bottom:18px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.c b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.c span{color:var(--mut);font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:500;font-size:12px}
.wrap{overflow-x:auto}.bad{color:#dc2626;font-weight:600}.warn{color:#d97706}
h2{font-size:14px;margin:18px 0 6px}#e td{text-align:left}#e td:first-child{white-space:nowrap}
</style><h1>context-compact</h1><div class="sub" id="sub"></div>
<div class="grid" id="g"></div><div class="wrap"><table id="t"></table></div>
<div id="ew"></div>
<script>
const f=n=>n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':String(n);
async function load(){try{const s=await (await fetch('status')).json();
const tot=Math.max(1,s.input_tokens+s.cache_read_input_tokens+s.cache_creation_input_tokens);
document.getElementById('sub').textContent=s.model+' · '+s.language+' · trigger '+f(s.trigger)+' (starts at '+Math.round(s.early*100)+'%) · keep '+f(s.keep)+' · num_ctx '+f(s.num_ctx)+(s.extract?' · extract':'')+(s.mcp?' · mcp':'')+' · '+s.jobs+' summary job(s) running';
const esc=t=>String(t).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const cv=x=>x==null?'':'<span class="'+(x<0.3?'bad':x<0.6?'warn':'')+'">'+Math.round(x*100)+'%</span>';
document.getElementById('ew').innerHTML=s.recent_errors.length?'<h2 class="bad">Recent errors</h2><div class="wrap"><table id="e">'+
s.recent_errors.map(x=>'<tr><td>'+x.age_min+' min ago</td><td>'+esc(x.key)+'</td><td>'+esc(x.message)+'</td></tr>').join('')+'</table></div>':'';
const cards=[['Tokens saved',f(s.saved_tokens)],['Saved (API-equivalent USD)',s.usd_saved_min+' – '+s.usd_saved_max],
['Spent (API-equivalent USD)',s.usd_spent],['Cache hit',(100*s.cache_read_input_tokens/tot).toFixed(0)+'%'],
['Compacted requests',f(s.compacted)+' / '+f(s.requests)],['Summaries',f(s.summaries)+(s.errors?' · '+s.errors+' errors':'')],
['Old tool output trimmed',f(s.trimmed_chars)+' chars'],
['Input / cache read / write',f(s.input_tokens)+' / '+f(s.cache_read_input_tokens)+' / '+f(s.cache_creation_input_tokens)],
['Output tokens',f(s.output_tokens)]];
document.getElementById('g').innerHTML=cards.map(c=>'<div class="c"><b>'+c[1]+'</b><span>'+c[0]+'</span></div>').join('');
document.getElementById('t').innerHTML='<tr><th>Conversation</th><th>Messages</th><th>Cut</th><th>Summary chars</th><th>Coverage</th><th>Took</th><th>Cached</th><th>Age</th></tr>'+
s.session_list.map(x=>'<tr><td>'+x.key+'</td><td>'+(x.msgs??'')+'</td><td>'+x.cut+'</td><td>'+f(x.chars)+'</td><td>'+
cv(x.coverage)+'</td><td>'+(x.seconds==null?'':x.seconds+'s')+'</td><td>'+x.cached+'</td><td>'+x.age_min+' min</td></tr>').join('');
}catch(e){document.getElementById('sub').textContent='status unavailable: '+e}}
load();setInterval(load,15000);
</script></html>"""

# ---------------------------------------------------------------- MCP (history_search)

MCP_TOOLS = [{
    "name": "history_search",
    "description": ("Search the original text of the part of this conversation that context-compact "
                    "replaced with a summary. Use it when you need an exact path, command, error, ID "
                    "or value from before the summary. `conversation` is the id given in the "
                    "[context-compact] header of the summary."),
    "inputSchema": {"type": "object", "properties": {
        "conversation": {"type": "string", "description": "Conversation id from the summary header"},
        "query": {"type": "string", "description": "Words or exact phrase to look for"},
        "limit": {"type": "integer", "description": "Max messages returned (default 5)"}},
        "required": ["conversation", "query"]}}]


def mcp_call(name, args):
    if name != "history_search":
        raise ValueError(f"unknown tool {name}")
    arch = find_archive(str(args.get("conversation", "")))
    if not arch:
        return "No summarised history for that conversation id (nothing has been compacted yet, or it expired)."
    try:
        limit = min(max(int(args.get("limit") or 5), 1), 20)
    except (TypeError, ValueError):
        limit = 5
    return search_entries(arch.get("entries", []), str(args.get("query", "")), limit)


def mcp_handle(req):
    """One JSON-RPC message of MCP's streamable HTTP transport -> response dict, or None for
    notifications."""
    if not isinstance(req, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
    rid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
    if rid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion") or "2025-03-26",
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "context-compact", "version": "0.4.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": MCP_TOOLS}
    elif method == "tools/call":
        try:
            text, err = mcp_call(params.get("name"), params.get("arguments") or {}), False
        except Exception as e:  # noqa: BLE001
            text, err = f"history_search failed: {e}", True
        result = {"content": [{"type": "text", "text": text}], "isError": err}
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}

# ---------------------------------------------------------------- proxy HTTP

HOP = {"host", "content-length", "connection", "keep-alive", "proxy-connection", "transfer-encoding",
       "te", "trailer", "upgrade", "accept-encoding"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silent: our own log is enough
        pass

    def _deny(self):
        ip = self.client_address[0].removeprefix("::ffff:")
        if ip in ALLOW:
            return False
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def _reply(self, code, data=b"", ctype="application/json"):
        self.send_response(code)
        if data:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _mcp(self):
        if self._deny():
            return
        if not MCP or self.command != "POST":
            self._reply(405)
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) if n else b"null")
        except ValueError:
            self._reply(400, json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700, "message": "parse error"}}).encode())
            return
        if isinstance(req, list):
            res = [r for r in (mcp_handle(x) for x in req) if r is not None]
        else:
            res = mcp_handle(req)
        if not res:
            self._reply(202)
            return
        self._reply(200, json.dumps(res, ensure_ascii=False).encode())

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/mcp":
            self._mcp()
            return
        if p.startswith("/_context-compact"):
            if self._deny():
                return
            if p.startswith("/_context-compact/status"):
                data, ctype = json.dumps(status(), ensure_ascii=False).encode(), "application/json"
            elif p == "/_context-compact":
                self.send_response(301)
                self.send_header("Location", "/_context-compact/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            else:
                data, ctype = DASHBOARD.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._forward()

    def do_POST(self):
        if self.path.split("?")[0] == "/mcp":
            self._mcp()
            return
        self._forward()

    do_PUT = do_DELETE = do_PATCH = do_POST

    def do_HEAD(self):
        self._forward()

    def _forward(self):
        if self._deny():
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        path = self.path.split("?")[0]
        measure = self.command == "POST" and path == "/v1/messages"
        compacted, key, conv, est = False, None, None, 0.0
        if self.command == "POST" and path in ("/v1/messages", "/v1/messages/count_tokens") and raw:
            STATS["requests"] += 1
            try:
                body = json.loads(raw)
                msgs = body.get("messages")
                if isinstance(msgs, list) and msgs:
                    conv = conv_key(msgs)
                    key = conv[:8]
                new = transform(body, count_only=path.endswith("count_tokens"))
                if new is not body:
                    compacted = True
                    raw = json.dumps(new, ensure_ascii=False).encode()
                if measure:
                    est = est_body(new)
                    sz = {k: len(json.dumps(new.get(k, ""), ensure_ascii=False)) // 1000
                          for k in ("system", "tools", "messages")}
                    by = {}
                    for m in new.get("messages") or []:
                        for b in blocks(m):
                            if isinstance(b, dict):
                                t = b.get("type", "?")
                                by[t] = by.get(t, 0) + len(json.dumps(b, ensure_ascii=False))
                    key = f"{key} sys={sz['system']}k tools={sz['tools']}k({len(new.get('tools') or [])}) " \
                          f"msgs={sz['messages']}k({len(new.get('messages') or [])}) est={int(compactable_tokens(new['messages']))} " \
                          + " ".join(f"{t}={v // 1000}k" for t, v in sorted(by.items()))
            except Exception as e:  # noqa: BLE001 — never break the request
                note_error(key, f"transform failed: {e}")
                log.warning("transform failed, forwarding untouched: %s", e)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Host"] = UPSTREAM
        headers["Accept-Encoding"] = "identity"
        headers["Content-Length"] = str(len(raw))
        try:
            conn = http.client.HTTPSConnection(UPSTREAM, timeout=900)
            conn.request(self.command, self.path, body=raw if raw or self.command != "GET" else None,
                         headers=headers)
            resp = conn.getresponse()
        except Exception as e:  # noqa: BLE001
            log.warning("upstream unreachable: %s", e)
            msg = json.dumps({"type": "error", "error": {"type": "api_error",
                                                         "message": f"context-compact: upstream unreachable ({e})"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        try:
            self.send_response(resp.status, resp.reason)
            length = resp.getheader("Content-Length")
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            chunked = length is None and self.command != "HEAD"
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", length or "0")
            self.end_headers()
            if self.command == "HEAD":
                return
            seen = []          # chunks with "usage" (message_start first, message_delta last)
            while True:
                data = resp.read1(65536)
                if not data:
                    break
                if measure and resp.status == 200 and b"_tokens" in data:
                    seen.append(data[-65536:])
                if chunked:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                else:
                    self.wfile.write(data)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            if seen:
                try:
                    record_usage(parse_usage(b"".join(seen[:1] + seen[-2:])), compacted,
                                 f"{key} ratio={(RATIO.get(conv) or 0):.2f}", conv, est)
                except Exception as e:  # noqa: BLE001 — measuring never breaks the response
                    log.warning("usage not measurable: %s", e)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            conn.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # client closing keep-alive: normal
        import sys
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, TimeoutError)):
            super().handle_error(request, client_address)


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(os.path.join(DATA, "context-compact.log"),
                                                   maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)
    cleanup_states()
    load_stats()
    srv = Server((HOST, PORT), Handler)
    log.info("context-compact on %s:%d -> %s (trigger %d, early %.0f%%, keep %d, num_ctx %d, model %s, "
             "language %s, extract %s, mcp %s)", HOST, PORT, UPSTREAM, TRIGGER, EARLY * 100, KEEP, NUM_CTX,
             MODEL, LANGUAGE, EXTRACT, MCP)
    srv.serve_forever()


if __name__ == "__main__":
    main()
