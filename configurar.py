#!/usr/bin/env python3
"""Activa (o desactiva con --quitar) el compactador en los settings.json de Claude Code de esta
máquina: ~/.claude y todas las cuentas de ccm (~/.ccm/accounts/*).

  python configurar.py http://<host>:8787            # otra máquina de la red
  python configurar.py http://127.0.0.1:8787         # la máquina donde corre el proxy
  python configurar.py --quitar                       # vuelve a hablar directo con la API

Sube CLAUDE_AUTOCOMPACT_PCT_OVERRIDE a 85: la compactación nativa queda solo como red de
seguridad para cuando el compactador no pueda resumir (Ollama caído). Con --quitar vuelve a 30.
"""
import glob
import json
import os
import sys

quitar = "--quitar" in sys.argv
url = None if quitar else sys.argv[1]
home = os.path.expanduser("~")
files = [os.path.join(home, ".claude", "settings.json")] + glob.glob(
    os.path.join(home, ".ccm", "accounts", "*", "settings.json"))
for path in files:
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    env = data.setdefault("env", {})
    if quitar:
        env.pop("ANTHROPIC_BASE_URL", None)
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "30"
    else:
        env["ANTHROPIC_BASE_URL"] = url
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "85"
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(path + ".tmp", path)
    print("ok", path)
