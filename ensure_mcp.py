#!/usr/bin/env python3
"""Register the context-compact MCP server (history_search) in every Claude Code config dir of this
machine: ~/.claude (default) plus, if you use ccm, every account in ~/.ccm/config.json or
~/.ccm/accounts/*. Idempotent and fast when nothing is missing (only reads JSON), so it can run
from a SessionStart hook: accounts added later get it on their next session.

  python ensure_mcp.py                       # <ANTHROPIC_BASE_URL>/mcp, else http://127.0.0.1:8787/mcp
  python ensure_mcp.py http://host:8787/mcp
"""
import glob
import json
import os
import shutil
import subprocess
import sys

NAME = "context-compact"
base = (os.environ.get("ANTHROPIC_BASE_URL") or "http://127.0.0.1:8787").rstrip("/")
url = sys.argv[1] if len(sys.argv) > 1 else base + "/mcp"
home = os.path.expanduser("~")


def account_dirs():
    dirs = {os.path.join(home, ".claude")}
    try:
        with open(os.path.join(home, ".ccm", "config.json"), encoding="utf-8-sig") as f:
            dirs |= {a["dir"] for a in json.load(f).get("accounts", []) if a.get("dir")}
    except (OSError, ValueError, KeyError):
        pass
    dirs |= set(glob.glob(os.path.join(home, ".ccm", "accounts", "*")))
    return sorted(os.path.normcase(os.path.normpath(d)) for d in dirs if os.path.isdir(d))


def state_file(d):
    # the default config dir keeps its state next to it (~/.claude.json), the rest inside
    if d == os.path.normcase(os.path.normpath(os.path.join(home, ".claude"))):
        return os.path.join(home, ".claude.json")
    return os.path.join(d, ".claude.json")


def registered(d):
    try:
        with open(state_file(d), encoding="utf-8-sig") as f:
            return (json.load(f).get("mcpServers") or {}).get(NAME, {}).get("url") == url
    except (OSError, ValueError, AttributeError):
        return False


claude = shutil.which("claude")
for d in account_dirs():
    if registered(d):
        continue
    if not claude:
        print(f"{NAME}: claude CLI not found, cannot register in {d}")
        break
    env = dict(os.environ)
    if state_file(d) == os.path.join(home, ".claude.json"):
        env.pop("CLAUDE_CONFIG_DIR", None)
    else:
        env["CLAUDE_CONFIG_DIR"] = d
    subprocess.run([claude, "mcp", "remove", "-s", "user", NAME], env=env, capture_output=True, timeout=60)
    r = subprocess.run([claude, "mcp", "add", "--transport", "http", "-s", "user", NAME, url],
                       env=env, capture_output=True, text=True, timeout=60)
    print(f"{NAME}: {'registered in' if r.returncode == 0 else 'FAILED in'} {d} {r.stderr.strip()[:200]}")
