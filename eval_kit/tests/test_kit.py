import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze as A  # noqa: E402
import protocol as P  # noqa: E402
import synth  # noqa: E402

DELAYS = {"eye_closure": 2.0, "yawn": 1.6, "head_lowering": 1.4, "side_turn": 1.0,
          "raised_hands": 1.7, "face_not_detected": 1.3, "looking_down": 2.2}


def load(d, pid):
    return A.load_session(os.path.join(d, pid))


# ── protocol ──────────────────────────────────────────────────
def test_schedule_is_deterministic_and_complete():
    a, b = P.build_schedule(7, reps=4), P.build_schedule(7, reps=4)
    assert [(i.type, i.t_on) for i in a] == [(i.type, i.t_on) for i in b]
    assert [(i.type) for i in P.build_schedule(8, reps=4)] != [i.type for i in a]
    ev = [i for i in a if i.kind == "event"]
    assert len(ev) == 4 * len(P.CUES)
    for c in P.CUES:
        assert sum(i.type == c for i in ev) == 4
    assert all(a[k].type != a[k + 1].type for k in range(len(a) - 1))
    # timeline is gap-free and ordered
    for x, y in zip(a, a[1:]):
        assert y.t_prep == pytest.approx(x.t_end)
    assert all(i.t_prep < i.t_on < i.t_off < i.t_end for i in a)
    assert 8 * 60 < a[-1].t_end < 13 * 60          # a full session takes about 10 minutes


def test_rest_period_is_longer_than_scoring_tolerance():
    assert P.REST_S > P.TOL_S + 1.0                 # an event window never reaches the next item's countdown


# ── perfect detector ──────────────────────────────────────────
def test_perfect_detector_metrics(tmp_path):
    d = str(tmp_path)
    items = synth.make_session(d, "P1", seed=1, delay_s=DELAYS)
    s = load(d, "P1")
    sc = A.score_session(s, P.CUES)
    for c in P.CUES:
        r = sc["cues"][c]
        assert (r["n_events"], r["tp"], r["fn"], r["fp_all"], r["fp_nonevent"]) == (4, 4, 0, 0, 0)
        # alert onset is quantised to the 10 fps frame grid: delay within one frame of the planted value
        assert all(-5 <= x - DELAYS[c] * 1000 <= 105 for x in r["delays"])   # first frame at/after the planted time
    assert sc["nonevent_s"] == pytest.approx(synth.expected_nonevent_seconds(items))
    m = A.pooled({"P1": sc}, ["P1"], "eye_closure")
    assert m["recall"] == 100 and m["precision"] == 100 and m["fp_per_min"] == 0
    assert -5 <= m["median_delay_ms"] - 2000 <= 105
    # the cross-cue matrix is diagonal
    for et, row in sc["cross"].items():
        assert row["n"] == 4 and row[et] == 4
        assert all(row[c] == 0 for c in P.CUES if c != et)


# ── misses ────────────────────────────────────────────────────
def test_missed_events_lower_recall(tmp_path):
    d = str(tmp_path)
    synth.make_session(d, "P1", delay_s=DELAYS, miss={"yawn": {0, 2}, "raised_hands": {0, 1, 2, 3}})
    sc = A.score_session(load(d, "P1"), P.CUES)
    per = {"P1": sc}
    assert A.pooled(per, ["P1"], "yawn")["recall"] == 50
    r = A.pooled(per, ["P1"], "raised_hands")
    assert r["recall"] == 0 and np.isnan(r["median_delay_ms"]) and np.isnan(r["precision"])


# ── false warnings ────────────────────────────────────────────
def test_false_warning_rate_uses_only_alert_free_time(tmp_path):
    d = str(tmp_path)
    items = synth.make_session(d, "P1", delay_s=DELAYS)
    talk = next(i for i in items if i.type == "talking")
    mirror = next(i for i in items if i.type == "mirror_glance")
    ev_yawn = next(i for i in items if i.type == "yawn")
    ev_eye = next(i for i in items if i.type == "eye_closure")
    S = synth.SCHED_START
    extra = [
        ("yawn", S + talk.t_on + 4.0),             # false yawn while talking          -> counts
        ("side_turn", S + mirror.t_on + 2.0),      # false side-turn at the mirror     -> counts
        ("side_turn", S + ev_yawn.t_on + 2.5),     # side-turn DURING a yawn           -> cross trigger, not in FP/min
        ("yawn", S + ev_eye.t_prep + 1.0),         # yawn alert in an eye event's countdown -> excluded time
    ]
    synth.make_session(d, "P1", delay_s=DELAYS, extra_onsets=extra)
    sc = A.score_session(load(d, "P1"), P.CUES)
    per = {"P1": sc}
    ne_min = sc["nonevent_s"] / 60
    assert A.pooled(per, ["P1"], "yawn")["fp_nonevent"] == 1
    assert A.pooled(per, ["P1"], "side_turn")["fp_nonevent"] == 1
    assert A.pooled(per, ["P1"], "side_turn")["fp_all"] == 2           # includes the cross trigger
    assert A.pooled(per, ["P1"], "yawn")["fp_all"] == 2
    assert A.pooled(per, ["P1"], "side_turn")["fp_per_min"] == pytest.approx(1 / ne_min)
    # precision counts every false alert: 4 detected / (4 + 2)
    assert A.pooled(per, ["P1"], "yawn")["precision"] == pytest.approx(100 * 4 / 6)
    # which behaviour each alert landed in
    assert sc["cross"]["yawn"]["side_turn"] == 1
    assert sc["neg"]["talking"]["yawn"] == 1 and sc["neg"]["mirror_glance"]["side_turn"] == 1


def test_calibration_period_and_http_failures_do_not_create_alerts(tmp_path):
    d = str(tmp_path)
    items = synth.make_session(d, "P1", delay_s=DELAYS, early_onsets=[("eye_closure", 5.0)])
    s = load(d, "P1")
    assert all(o > synth.CAL_END for o in s.onsets["eye_closure"])        # pre-calibration alert ignored
    # a dropped frame in the middle of a latched alert must not look like a new alert
    ev = next(i for i in items if i.type == "eye_closure")
    mid = synth.SCHED_START + ev.t_on + 3.0
    synth.make_session(d, "P2", delay_s=DELAYS, http_fail_times=[mid, mid + 0.1])
    s2 = load(d, "P2")
    assert len(s2.onsets["eye_closure"]) == 4


# ── pooling / bootstrap ───────────────────────────────────────
def test_bootstrap_ci_contains_point_estimate(tmp_path):
    d = str(tmp_path)
    for k in range(6):
        synth.make_session(d, f"P{k}", seed=k + 1, delay_s=DELAYS, miss={"yawn": {0}} if k % 2 == 0 else {})
    per = {f"P{k}": A.score_session(load(d, f"P{k}"), P.CUES) for k in range(6)}
    ci = A.bootstrap(per, P.CUES)
    pt = A.pooled(per, list(per), "yawn")
    lo, hi = ci["yawn"]["recall"]
    assert 0 <= lo <= pt["recall"] <= hi <= 100
    assert ci["eye_closure"]["recall"] == (100.0, 100.0)               # nothing to resample away


def test_too_few_participants_gives_no_ci(tmp_path):
    d = str(tmp_path)
    synth.make_session(d, "P1", delay_s=DELAYS)
    synth.make_session(d, "P2", seed=2, delay_s=DELAYS)
    per = {p: A.score_session(load(d, p), P.CUES) for p in ("P1", "P2")}
    assert A.bootstrap(per, P.CUES) == {}


# ── end-to-end analyze.py ─────────────────────────────────────
def test_analyze_writes_all_outputs_and_filters_sessions(tmp_path):
    d, out = str(tmp_path / "data"), str(tmp_path / "res")
    for k in range(4):
        synth.make_session(d, f"P{k}", seed=k + 1, delay_s=DELAYS, latency_ms=30 + 10 * k)
    synth.make_session(d, "DEV", seed=9, delay_s=DELAYS, role="dev")
    synth.make_session(d, "FAKE", seed=10, delay_s=DELAYS, synthetic=True)
    synth.make_session(d, "PARTIAL", seed=11, delay_s=DELAYS, complete=False)
    assert A.main(["--data", d, "--out", out, "--allow-synthetic"]) == 0
    for f in ("table4.csv", "table4.md", "table4.tex", "figE_precision_recall.png", "figE_precision_recall.pdf",
              "figF_latency.png", "figF_latency.pdf", "cross_cue_matrix.csv", "fp_by_negative.csv",
              "per_participant.csv", "dataset_manifest.csv", "latency_summary.json", "report.txt"):
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    rep = open(os.path.join(out, "report.txt")).read()
    assert "DEV" in rep and "PARTIAL" in rep and "role=dev" in rep and "incomplete" in rep
    manifest = open(os.path.join(out, "dataset_manifest.csv")).read()
    assert manifest.count("\n") == 1 + 5                                # 4 test + FAKE (allowed), no DEV / PARTIAL
    lat = json.load(open(os.path.join(out, "latency_summary.json")))
    assert 30 <= lat["capture_to_response_ms"]["p50"] <= 80
    rows = open(os.path.join(out, "table4.csv")).read().strip().splitlines()
    assert len(rows) == 1 + len(P.CUES) and rows[1].split(",")[1] == str(5 * 4)   # 5 sessions x 4 events


def test_synthetic_sessions_are_refused_by_default(tmp_path):
    d, out = str(tmp_path / "data"), str(tmp_path / "res")
    synth.make_session(d, "FAKE", delay_s=DELAYS, synthetic=True)
    assert A.main(["--data", d, "--out", out]) == 1


def test_looking_down_row_dropped_for_older_detector(tmp_path):
    d, out = str(tmp_path / "data"), str(tmp_path / "res")
    for k in range(3):
        synth.make_session(d, f"P{k}", seed=k + 1, delay_s=DELAYS, has_lookdown=False)
    assert A.main(["--data", d, "--out", out]) == 0
    assert "Looking down" not in open(os.path.join(out, "table4.csv")).read()


# ── the recorder, against a mock server with a fake camera ────────────
class _Mock(BaseHTTPRequestHandler):
    n = 0
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"server": "running"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path == "/reset":
            return self._send({"ok": True})
        _Mock.n += 1
        _Mock.seen.append((self.headers.get("X-Session-Id"), body[:2]))
        r = {"status": "awake", "calibrated": _Mock.n > 15, "calibration_failed": False, "ear": 0.27,
             "mar": 0.01, "nod_drop": 0.0, "yaw": 0.0, "perclos": 0.0, "look_down": 0.1, "message": "Awake",
             "ear_alerting": False, "mar_alerting": False, "nod_alerting": False, "yaw_alerting": False,
             "hand_alerting": 150 < _Mock.n < 180, "down_alerting": False, "distract_alert": False}
        self._send(r)


def test_recorder_end_to_end_with_mock_server(tmp_path):
    import run_session
    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    out = str(tmp_path / "data")
    argv = ["run_session.py", "--server", url, "--participant", "T-01", "--out", out, "--no-gui",
            "--fake-camera", "--fps", "40", "--reps", "1", "--time-scale", "0.04"]
    old = sys.argv
    sys.argv = argv
    try:
        assert run_session.main() == 0
    finally:
        sys.argv = old
        srv.shutdown()
    assert _Mock.seen[0][0].startswith("evalT01") and _Mock.seen[0][1] == b"\xff\xd8"     # session id + JPEG magic
    meta = json.load(open(os.path.join(out, "T01_meta.json")))
    assert meta["complete"] and meta["synthetic"] and meta["has_looking_down"]
    assert meta["scoring"]["time_scale"] == 0.04
    labels = open(os.path.join(out, "T01_labels.csv")).read().strip().splitlines()
    assert len(labels) == 1 + len(P.CUES) + 4
    s = A.load_session(os.path.join(out, "T01"))
    assert len(s.t) > 100 and s.ok.all() and s.calibrated.sum() > 0
    assert len(s.onsets["raised_hands"]) == 1                            # latched flag -> exactly one onset


def test_recorder_refuses_time_scale_with_real_camera(tmp_path):
    import run_session
    old = sys.argv
    sys.argv = ["run_session.py", "--time-scale", "0.5", "--out", str(tmp_path)]
    try:
        with pytest.raises(SystemExit):
            run_session.main()
    finally:
        sys.argv = old
