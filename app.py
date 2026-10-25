"""
DrowseGuard — Flask Backend (v2: per-session detectors)

Routes:
  GET  /                                  → index.html  (landing page)
  GET  /detect.html (/detector, /detector.html) → detect.html (live detector)
  POST /detect                            → process one JPEG frame, return JSON
  POST /reset                             → reset the CALLER's detector state
  GET  /status                            → health check (+ caller's session stats)

What changed vs v1:
  * One detector PER SESSION (header "X-Session-Id") instead of one global
    detector shared by every visitor. Sessions expire after SESSION_TTL_S.
  * A lock around inference: the MediaPipe landmarkers in detector.py are
    module-level and shared, and Flask serves requests on multiple threads.
  * /reset only resets the caller's session. /status no longer leaks other
    users' state.
  * Input validation: empty body, oversize body (413), bad session id,
    undecodable frames now return 4xx instead of HTTP 200.
  * Open CORS removed (the pages are served from this same app). Set the
    CORS_ORIGINS env var (comma separated) only if you host the UI elsewhere.
  * Pages are served from this file's directory, not the process CWD.

Run locally:   python app.py
Production:    gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT app:app
               (keep ONE worker — sessions live in this process's memory)
"""

import logging
import os
import re
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

from detector import DrowsinessDetector

BASE_DIR = Path(__file__).resolve().parent

SESSION_TTL_S = 15 * 60          # drop sessions idle for 15 minutes
MAX_SESSIONS = 200               # hard cap; oldest-idle session is evicted
MAX_FRAME_BYTES = 2 * 1024 * 1024
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
ANON_SESSION = "anonymous"       # used when an old frontend sends no header

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drowseguard")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    from flask_cors import CORS
    CORS(app, origins=_cors_origins)

# ─────────────────────────────────────────────
#  SESSION STORE
# ─────────────────────────────────────────────

_sessions = {}                    # sid -> {"detector": ..., "last_seen": float}
_store_lock = threading.Lock()    # protects _sessions
_infer_lock = threading.Lock()    # serialises use of the shared MediaPipe models


class BadSession(ValueError):
    pass


def _session_id():
    sid = request.headers.get("X-Session-Id")
    if not sid:
        return ANON_SESSION
    if not SESSION_ID_RE.match(sid):
        raise BadSession("X-Session-Id must be 8-64 chars of [A-Za-z0-9_-]")
    return sid


def _evict_locked(now):
    """Caller must hold _store_lock."""
    for sid in [s for s, v in _sessions.items() if now - v["last_seen"] > SESSION_TTL_S]:
        del _sessions[sid]
    while len(_sessions) > MAX_SESSIONS:
        oldest = min(_sessions, key=lambda s: _sessions[s]["last_seen"])
        del _sessions[oldest]


def _get_detector(sid, create=True):
    now = time.monotonic()
    with _store_lock:
        _evict_locked(now)
        entry = _sessions.get(sid)
        if entry is None:
            if not create:
                return None
            entry = {"detector": DrowsinessDetector(), "last_seen": now}
            _sessions[sid] = entry
            _evict_locked(now)      # enforce the cap AFTER inserting
        entry["last_seen"] = now
        return entry["detector"]


def _reset_session(sid):
    now = time.monotonic()
    with _store_lock:
        _sessions[sid] = {"detector": DrowsinessDetector(), "last_seen": now}
        _evict_locked(now)


# ─────────────────────────────────────────────
#  SERVE PAGES
# ─────────────────────────────────────────────

def _page(name):
    return send_from_directory(BASE_DIR, name)


@app.route("/")
def index():
    return _page("index.html")


def _detect_page():
    return _page("detect.html")


# /detector and /detector.html are kept as aliases for old links.
for _path, _endpoint in (("/detect.html", "page_detect_html"),
                         ("/detector", "page_detector"),
                         ("/detector.html", "page_detector_html")):
    app.add_url_rule(_path, endpoint=_endpoint, view_func=_detect_page)


# ─────────────────────────────────────────────
#  CORE DETECTION
# ─────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    """Body = raw JPEG bytes. Header X-Session-Id identifies the browser."""
    try:
        sid = _session_id()
    except BadSession as e:
        return jsonify({"error": str(e)}), 400

    frame_bytes = request.get_data()
    if not frame_bytes:
        return jsonify({"error": "No frame received"}), 400

    detector = _get_detector(sid)
    try:
        with _infer_lock:
            result = detector.process_frame(frame_bytes)
    except Exception:
        log.exception("process_frame failed (session=%s)", sid[:8])
        return jsonify({"error": "processing_failed"}), 500

    if isinstance(result, dict) and result.get("error"):
        return jsonify(result), 400

    resp = jsonify(result)
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ─────────────────────────────────────────────
#  RESET
# ─────────────────────────────────────────────

@app.route("/reset", methods=["POST"])
def reset():
    try:
        sid = _session_id()
    except BadSession as e:
        return jsonify({"error": str(e)}), 400
    _reset_session(sid)
    return jsonify({"message": "Detector reset successfully"})


# ─────────────────────────────────────────────
#  STATUS / HEALTH CHECK
# ─────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    with _store_lock:
        _evict_locked(time.monotonic())
        active = len(_sessions)

    session = None
    if request.headers.get("X-Session-Id"):
        try:
            d = _get_detector(_session_id(), create=False)
        except BadSession as e:
            return jsonify({"error": str(e)}), 400
        if d is not None:
            session = {
                "frames_processed":   d.frames,
                "current_status":     d.status,
                "total_eye_alerts":   d.eye_alerts,
                "total_yawn_alerts":  d.yawn_alerts,
                "total_nod_alerts":   d.nod_alerts,
            }

    return jsonify({"server": "running", "active_sessions": active, "session": session})


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n==========================================")
    print("  DrowseGuard Server running!")
    print(f"  Landing page : http://localhost:{port}")
    print(f"  Detector     : http://localhost:{port}/detect.html")
    print(f"  API status   : http://localhost:{port}/status")
    print("==========================================\n")
    app.run(host="0.0.0.0", port=port)