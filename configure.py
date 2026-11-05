#!/usr/bin/env python3
"""Enable (or disable with --remove) context-compact in this machine's Claude Code settings.json:
~/.claude and every ccm account (~/.ccm/accounts/*).

  python configure.py http://<host>:8787            # another machine on the LAN
  python configure.py http://127.0.0.1:8787         # the machine running the proxy
  python configure.py --remove                       # talk to the API directly again

Sets ENABLE_TOOL_SEARCH=true (deferred MCP tools: behind a gateway Claude Code would send all of
them on every request). Raises CLAUDE_AUTOCOMPACT_PCT_OVERRIDE to 85: native compaction stays only
as a safety net for when the proxy cannot summarise (Ollama down). --remove sets it back to 30.
"""
import glob
import json
import os
import sys

remove = "--remove" in sys.argv
if not remove and len(sys.argv) < 2:
    sys.exit(__doc__)
url = None if remove else sys.argv[1]
home = os.path.expanduser("~")
files = [os.path.join(home, ".claude", "settings.json")] + glob.glob(
    os.path.join(home, ".ccm", "accounts", "*", "settings.json"))
for path in files:
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    env = data.setdefault("env", {})
    if remove:
        env.pop("ANTHROPIC_BASE_URL", None)
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "30"
    else:
        env["ANTHROPIC_BASE_URL"] = url
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "85"
        # With ANTHROPIC_BASE_URL (gateway) or "auto" (10 % threshold = 100K on a 1M context)
        # MCP tools are sent in full on every request (~60K tokens with 135 tools).
        env["ENABLE_TOOL_SEARCH"] = "true"
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(path + ".tmp", path)
    print("ok", path)
