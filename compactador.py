#!/usr/bin/env python3
"""compactador — proxy de la API de Anthropic que compacta el contexto con un LLM local.

Claude Code apunta aquí con ANTHROPIC_BASE_URL. Cada petición se reenvía tal cual a
api.anthropic.com salvo POST /v1/messages (y count_tokens) de conversaciones largas: cuando la
parte compactable pasa de COMPACTADOR_TRIGGER tokens, un hilo de fondo resume los mensajes
antiguos con Ollama y, en cuanto el resumen está listo, las peticiones siguientes llevan
[resumen + últimos mensajes] en vez del historial entero. La interfaz de Claude Code conserva la
conversación completa (scroll, transcript); solo cambia lo que llega a la API.

Si algo falla (Ollama caído, GPU ocupada, JSON raro) la petición pasa sin tocar: nunca rompe
la sesión, como mucho no ahorra. Solo stdlib.
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
HOST = E("COMPACTADOR_HOST", "0.0.0.0")
PORT = int(E("COMPACTADOR_PORT", "8787"))
UPSTREAM = E("COMPACTADOR_UPSTREAM", "api.anthropic.com")
ALLOW = {a.strip() for a in E("COMPACTADOR_ALLOW", "127.0.0.1,::1").split(",") if a.strip()}
OLLAMA = E("COMPACTADOR_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
MODEL = E("COMPACTADOR_MODEL", "gemma3:12b")
TRIGGER = int(E("COMPACTADOR_TRIGGER", "35000"))   # tokens compactables que disparan el resumen
KEEP = int(E("COMPACTADOR_KEEP", "10000"))         # tokens recientes que se mandan sin resumir
CHUNK_CHARS = int(E("COMPACTADOR_CHUNK_CHARS", "36000"))
NUM_CTX = int(E("COMPACTADOR_NUM_CTX", "16384"))
RETRY_S = int(E("COMPACTADOR_RETRY_S", "120"))
DATA = os.path.expanduser(E("COMPACTADOR_DATA", "~/.compactador"))
STATE_DIR = os.path.join(DATA, "state")
CHARS_PER_TOKEN = 3.5
IMAGE_TOKENS = 1600

HEADER = ("[compactador] Los mensajes anteriores de esta conversación se han sustituido por el "
          "resumen siguiente para ahorrar contexto. La conversación continúa después con "
          "normalidad; no menciones este resumen salvo que el usuario pregunte.\n\n")

log = logging.getLogger("compactador")

# ---------------------------------------------------------------- lógica pura (testeada)

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
    """Huella estable del historial: ignora cache_control y el contenido de tool_result
    (Claude Code los mueve/vacía entre peticiones), detecta rewind/edición."""
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
    """Tokens de msgs[start:] sin contar los system-reminder del primer mensaje (CLAUDE.md etc.,
    que se conservan siempre)."""
    total = 0.0
    for i, m in enumerate(msgs[start:], start):
        if i == 0:
            total += sum(est_block(b) for b in blocks(m) if isinstance(b, dict) and not is_reminder(b))
        else:
            total += est_msg(m)
    return total


def choose_cut(msgs, keep, min_cut=1):
    """Índice del mensaje user que pasa a ser el primero tras compactar. Prefiere un prompt real
    del usuario; si el turno es una tirada larga de herramientas, corta en un tool_result."""
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
        if tail > 2 * keep:          # no hay prompt real cerca: cortamos dentro del turno
            return best_any
    return best_any


def tool_result_as_text(b):
    flag = " (error)" if b.get("is_error") else ""
    body = text_of(b.get("content"))
    if len(body) > 4000:
        body = body[:4000] + " …[recortado]"
    return {"type": "text", "text": f"[Resultado de herramienta{flag}]\n{body}"}


def apply_state(msgs, state):
    """Devuelve la lista de mensajes compactada, o None si el estado no casa con este historial."""
    cut = state["cut"]
    if len(msgs) <= cut or skeleton_hash(msgs[:cut + 1]) != state["hash"]:
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
    summary = {"type": "text", "text": HEADER + state["summary"]}
    first = {"role": "user", "content": reminders + [summary] + first_blocks}
    return [first] + msgs[cut + 1:]


REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def clip(s, n):
    s = s.strip()
    return s if len(s) <= n else s[:n] + " …"


def render(msgs):
    out = []
    for m in msgs:
        who = "USUARIO" if m.get("role") == "user" else "CLAUDE"
        for b in blocks(m):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                txt = REMINDER_RE.sub("", b.get("text", "")).strip()
                if txt:
                    out.append(f"[{who}] {clip(txt, 4000)}")
            elif t == "tool_use":
                out.append(f"[CLAUDE usa {b.get('name')}] {clip(json.dumps(b.get('input'), ensure_ascii=False), 800)}")
            elif t == "tool_result":
                flag = " ERROR" if b.get("is_error") else ""
                out.append(f"[resultado{flag}] {clip(REMINDER_RE.sub('', text_of(b.get('content'))), 1200)}")
            elif t in ("image", "document"):
                out.append(f"[{who} adjunta {t}]")
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

# ---------------------------------------------------------------- resumen con Ollama

SYSTEM = ("Resumes conversaciones de trabajo entre un usuario y Claude Code (un asistente de "
          "programación) para que Claude pueda continuar sin el historial. Escribe en español. "
          "Conserva literalmente rutas, nombres de funciones, comandos, mensajes de error, IDs, "
          "URLs, números y decisiones. No inventes nada que no aparezca.")


def ask_ollama(prev, fragment):
    prompt = (f"RESUMEN PREVIO:\n{prev or '(ninguno)'}\n\nNUEVO FRAGMENTO DE CONVERSACIÓN:\n{fragment}\n\n"
              "Escribe el resumen actualizado (máximo unas 1000 palabras) integrando el resumen previo "
              "y el fragmento nuevo, con estas secciones:\n## Peticiones del usuario\n## Estado del trabajo\n"
              "## Decisiones y preferencias del usuario\n## Ficheros y comandos relevantes\n"
              "## Errores encontrados y cómo se resolvieron\n## Pendiente / siguiente paso")
    body = json.dumps({"model": MODEL, "stream": False, "keep_alive": "5m",
                       "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())["message"]["content"].strip()
    if not out:
        raise RuntimeError("Ollama devolvió un resumen vacío")
    return out


def summarize(prev_summary, msgs):
    summary = prev_summary
    for frag in chunks(render(msgs), CHUNK_CHARS):
        summary = ask_ollama(summary, frag)
    return summary

# ---------------------------------------------------------------- estado por conversación

LOCK = threading.Lock()
STATES = {}      # key -> estado aplicado {cut, hash, summary, files_mod, files_read, saved, t}
JOBS = set()     # keys con resumen en curso
FAILED = {}      # key -> instante del último fallo
STATS = {"requests": 0, "compacted": 0, "summaries": 0, "errors": 0, "saved_tokens": 0}


def state_path(key):
    return os.path.join(STATE_DIR, key + ".json")


def load_state(key):
    st = STATES.get(key)
    if st is None:
        try:
            with open(state_path(key), encoding="utf-8") as f:
                st = json.load(f)
            STATES[key] = st
        except (OSError, ValueError):
            return None
    return st


def save_state(key, st):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = state_path(key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, state_path(key))
    STATES[key] = st


def cleanup_states(max_age_days=14):
    try:
        limit = time.time() - max_age_days * 86400
        for n in os.listdir(STATE_DIR):
            p = os.path.join(STATE_DIR, n)
            if os.path.getmtime(p) < limit:
                os.remove(p)
    except OSError:
        pass


def build_summary(key, msgs, prev):
    """Hilo de fondo: resume msgs[prev.cut:new_cut] sobre el resumen previo y guarda el estado."""
    try:
        start = prev["cut"] if prev else 0
        cut = choose_cut(msgs, KEEP, min_cut=start + 1)
        if cut is None:
            log.info("%s: sin punto de corte válido", key)
            return
        span = msgs[start:cut]
        t0 = time.time()
        summary = summarize(prev["summary"] if prev else "", span)
        mod, read = touched_files(span)
        files_mod = list(dict.fromkeys((prev or {}).get("files_mod", []) + mod))[-60:]
        files_read = list(dict.fromkeys((prev or {}).get("files_read", []) + read))[-60:]
        extra = ""
        if files_mod:
            extra += "\n\n## Ficheros modificados (lista exacta)\n" + "\n".join(f"- {p}" for p in files_mod)
        if files_read:
            extra += "\n\n## Ficheros leídos (lista exacta)\n" + "\n".join(f"- {p}" for p in files_read)
        # summary_for_next: el siguiente resumen parte del texto del LLM, no de las listas
        # (se regeneran cada vez)
        st = {"cut": cut, "hash": skeleton_hash(msgs[:cut + 1]), "summary": summary + extra,
              "summary_for_next": summary, "files_mod": files_mod, "files_read": files_read,
              "t": time.time()}
        with LOCK:
            save_state(key, st)
            STATS["summaries"] += 1
        FAILED.pop(key, None)
        log.info("%s: resumen listo (%d msgs -> corte en %d, %.0fs, %d chars)",
                 key, len(span), cut, time.time() - t0, len(st["summary"]))
    except Exception as e:  # noqa: BLE001 — cualquier fallo deja la sesión sin compactar
        FAILED[key] = time.time()
        STATS["errors"] += 1
        log.warning("%s: fallo al resumir: %s", key, e)
    finally:
        with LOCK:
            JOBS.discard(key)


def transform(body, count_only=False):
    """Devuelve el cuerpo (dict) a enviar. Puede lanzar un resumen de fondo."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return body
    key = conv_key(msgs)
    with LOCK:
        st = load_state(key)
    out = None
    if st:
        out = apply_state(msgs, st)
        if out is None:
            log.info("%s: el historial ya no casa con el resumen (rewind/compactación nativa), se descarta", key)
            with LOCK:
                STATES.pop(key, None)
                try:
                    os.remove(state_path(key))
                except OSError:
                    pass
            st = None
    pending = compactable_tokens(msgs, st["cut"] + 1 if st else 0)
    if not count_only and pending > TRIGGER:
        with LOCK:
            busy = key in JOBS or time.time() - FAILED.get(key, 0) < RETRY_S
            if not busy:
                JOBS.add(key)
        if not busy:
            prev = None
            if st:
                prev = dict(st, summary=st.get("summary_for_next", st["summary"]))
            threading.Thread(target=build_summary, args=(key, copy.deepcopy(msgs), prev), daemon=True).start()
            log.info("%s: %d tokens compactables > %d, resumiendo en segundo plano", key, pending, TRIGGER)
    if out is None:
        return body
    new = dict(body)
    new["messages"] = out
    if not count_only:
        saved = int(compactable_tokens(msgs) - compactable_tokens(out))
        STATS["compacted"] += 1
        STATS["saved_tokens"] += max(saved, 0)
    return new

# ---------------------------------------------------------------- proxy HTTP

HOP = {"host", "content-length", "connection", "keep-alive", "proxy-connection", "transfer-encoding",
       "te", "trailer", "upgrade", "accept-encoding"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silencio: el log propio basta
        pass

    def _deny(self):
        ip = self.client_address[0].removeprefix("::ffff:")
        if ip in ALLOW:
            return False
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def do_GET(self):
        if self.path.startswith("/_compactador/status"):
            if self._deny():
                return
            data = json.dumps(dict(STATS, jobs=len(JOBS), sessions=len(STATES), trigger=TRIGGER,
                                   keep=KEEP, model=MODEL), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._forward()

    def do_POST(self):
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
        if self.command == "POST" and path in ("/v1/messages", "/v1/messages/count_tokens") and raw:
            STATS["requests"] += 1
            try:
                body = json.loads(raw)
                new = transform(body, count_only=path.endswith("count_tokens"))
                if new is not body:
                    raw = json.dumps(new, ensure_ascii=False).encode()
            except Exception as e:  # noqa: BLE001 — nunca romper la petición
                log.warning("transform falló, se reenvía sin tocar: %s", e)
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
            log.warning("upstream inaccesible: %s", e)
            msg = json.dumps({"type": "error", "error": {"type": "api_error",
                                                         "message": f"compactador: upstream inaccesible ({e})"}}).encode()
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
            while True:
                data = resp.read1(65536)
                if not data:
                    break
                if chunked:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                else:
                    self.wfile.write(data)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            conn.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # cliente que cierra keep-alive: normal
        import sys
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, TimeoutError)):
            super().handle_error(request, client_address)


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(os.path.join(DATA, "compactador.log"),
                                                   maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)
    cleanup_states()
    srv = Server((HOST, PORT), Handler)
    log.info("compactador en %s:%d -> %s (trigger %d, keep %d, modelo %s)", HOST, PORT, UPSTREAM, TRIGGER, KEEP, MODEL)
    srv.serve_forever()


if __name__ == "__main__":
    main()
