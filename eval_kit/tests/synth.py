"""Write fake sessions in the exact on-disk format of run_session.py, with KNOWN ground truth,
so the analysis can be verified against numbers worked out independently.
These files are SYNTHETIC and must never be used as results."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import protocol as P  # noqa: E402

CAL_END = 11.0
SCHED_START = 14.0
FPS = 10.0


def make_session(out_dir, pid, seed=1, reps=4, delay_s=None, miss=None, extra_onsets=None,
                 http_fail_times=(), role="test", synthetic=False, has_lookdown=True,
                 latency_ms=40.0, latched_s=1.0, complete=True, early_onsets=None):
    """delay_s: {cue: seconds from t_on to alert onset}  (default 1.5)
    miss: {cue: set of occurrence indices (0-based, in schedule order) that are NOT detected}
    extra_onsets: list of (cue, absolute_time) extra alert onsets (false warnings / cross triggers)
    early_onsets: onsets placed BEFORE calibration end (must be ignored)
    http_fail_times: frame times whose HTTP call 'failed'."""
    delay_s = delay_s or {}
    miss = miss or {}
    extra_onsets = extra_onsets or []
    early_onsets = early_onsets or []
    items = P.build_schedule(seed, reps=reps)
    end = SCHED_START + items[-1].t_end + 3.0

    intervals = {c: [] for c in P.CUES}            # (start, stop) during which the alert flag is 1
    occ = {c: 0 for c in P.CUES}
    for it in items:
        if it.kind != "event":
            continue
        k = occ[it.type]
        occ[it.type] += 1
        if k in miss.get(it.type, set()):
            continue
        on = SCHED_START + it.t_on + delay_s.get(it.type, 1.5)
        intervals[it.type].append((on, SCHED_START + it.t_off + latched_s))
    for cue, t in extra_onsets + early_onsets:
        intervals[cue].append((t, t + 0.5))

    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, pid)
    fail = {round(t, 1) for t in http_fail_times}
    with open(prefix + "_frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_capture", "t_encoded", "t_response", "http_ok", "status", "calibrated"]
                   + ["a_" + c for c in P.CUES] + ["ear", "mar", "nod_drop", "yaw", "perclos", "look_down", "message"])
        t = 0.0
        while t < end:
            ok = 0 if round(t, 1) in fail else 1
            flags = [1 if (ok and any(a <= t < b for a, b in intervals[c])) else 0 for c in P.CUES]
            w.writerow([f"{t:.4f}", f"{t + 0.01:.4f}", f"{t + latency_ms / 1000:.4f}", ok, "awake",
                        1 if t >= CAL_END else 0] + flags + ["", "", "", "", "", "", ""])
            t += 1.0 / FPS
    with open(prefix + "_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "kind", "type", "t_prep", "t_on", "t_off", "t_end"])
        for it in items:
            w.writerow([it.item_id, it.kind, it.type, SCHED_START + it.t_prep, SCHED_START + it.t_on,
                        SCHED_START + it.t_off, SCHED_START + it.t_end])
    meta = {"participant": pid, "role": role, "complete": complete, "calibration_failed": False,
            "glasses": "no", "lighting": "indoor", "camera_distance_cm": "60", "camera_height": "eye-level",
            "seed": seed, "reps": reps, "synthetic": synthetic, "scoring": P.scoring_constants(1.0),
            "schedule_start_s": SCHED_START, "frames_sent": int(end * FPS), "frames_failed": len(fail),
            "has_looking_down": has_lookdown, "achieved_fps_overall": FPS,
            "environment": {"platform": "test", "server": "mock"}}
    json.dump(meta, open(prefix + "_meta.json", "w"))
    return items


def expected_nonevent_seconds(items, tol=P.TOL_S):
    total = items[-1].t_end - items[0].t_prep
    excl = sum((it.t_off + tol) - it.t_prep for it in items if it.kind == "event")
    return total - excl
