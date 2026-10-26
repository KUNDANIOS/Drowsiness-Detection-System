"""Record ONE evaluation session against a running DrowseGuard server.

    python run_session.py --smoke-test                      # check the server contract first
    python run_session.py --participant P01 --role test --glasses no --lighting daylight

What it does
  1. Sends webcam frames (JPEG, ~10 fps) to POST /detect exactly like the browser does.
  2. Waits until the server reports calibration done (look straight at the camera).
  3. Walks through the scripted cue schedule (see protocol.py) with on-screen
     text + beeps, logging every frame's capture / encode / response times and
     the detector's alert flags.
  4. Writes data/<participant>_frames.csv, _labels.csv, _meta.json.
No video is saved unless --save-video is given (privacy by default).

IMPORTANT: assumptions about app.py that I could not verify (see --smoke-test):
  * POST /detect takes the raw JPEG bytes as the request body
  * the session is identified by the X-Session-Id header
  * POST /reset (same header) resets that session
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import secrets
import subprocess
import sys
import time
import zlib
import urllib.error
import urllib.request

import protocol as P

# detector JSON -> per-cue "alert currently latched" flag
def alert_flags(r: dict) -> dict:
    g = lambda k: 1 if r.get(k) else 0  # noqa: E731
    return {
        "a_eye_closure": g("ear_alerting"),
        "a_yawn": g("mar_alerting"),
        "a_head_lowering": g("nod_alerting"),
        "a_side_turn": g("yaw_alerting"),
        "a_raised_hands": g("hand_alerting"),
        "a_face_not_detected": 1 if (r.get("status") == "no_face" and r.get("distract_alert")) else 0,
        "a_looking_down": g("down_alerting"),
    }


FRAME_COLS = (["t_capture", "t_encoded", "t_response", "http_ok", "status", "calibrated"]
              + ["a_" + c for c in P.CUES]
              + ["ear", "mar", "nod_drop", "yaw", "perclos", "look_down", "message"])


def http(method: str, url: str, body: bytes | None, headers: dict, timeout: float = 5.0):
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:                              # connection refused, timeout, ...
        return 0, str(e).encode()


class Camera:
    def __init__(self, index: int, width: int, height: int, fake: bool):
        import cv2
        self.cv2, self.fake = cv2, fake
        if fake:
            import numpy as np
            self.np = np
            self.w, self.h = width, height
            return
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit("Could not open the camera (close the browser detector tab / other apps).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        if self.fake:
            return self.np.full((self.h, self.w, 3), 127, self.np.uint8)
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if not self.fake:
            self.cap.release()


class Cue:
    """Big-text cue window (tkinter) with a console fallback."""

    def __init__(self, gui: bool):
        self.root = self.label = None
        self.gui = False
        if gui:
            try:
                import tkinter as tk
                self.root = tk.Tk()
                self.root.title("DrowseGuard evaluation - keep looking at the camera")
                self.root.geometry("760x300+80+20")
                self.root.attributes("-topmost", True)
                self.label = tk.Label(self.root, text="", font=("Arial", 30, "bold"), justify="center",
                                      wraplength=720)
                self.label.pack(expand=True, fill="both")
                self.gui = True
            except Exception as e:                     # no display / no tkinter
                print(f"(no cue window: {e}; using console cues)")
        self._last = None

    def show(self, text: str):
        if text == self._last:
            return
        self._last = text
        if self.gui:
            self.label.config(text=text)
        else:
            print("\n>>> " + text.replace("\n", " | "))

    def beep(self, kind="on"):
        try:
            import winsound
            winsound.Beep(1000 if kind == "on" else 600, 180)
        except Exception:
            if self.gui:
                self.root.bell()
            else:
                print("\a", end="", flush=True)

    def pump(self):
        if self.gui:
            self.root.update()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.getcwd()).decode().strip()
    except Exception:
        return "unknown"


def env_info(args, cam, server_status) -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "processor": platform.processor(), "machine": platform.machine(),
            "client_git_commit": git_commit(), "server": args.server,
            "server_status": server_status, "camera_resolution": [cam.w, cam.h],
            "jpeg_quality": args.jpeg_quality, "target_fps": args.fps}
    from importlib import metadata
    for dist in ("opencv-python", "opencv-python-headless", "numpy", "mediapipe", "flask"):
        try:
            info[dist] = metadata.version(dist)
        except Exception:
            pass
    return info


def smoke_test(args) -> int:
    import cv2
    import numpy as np
    img = np.full((args.height, args.width, 3), 127, np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    sid = "smoke" + secrets.token_hex(6)
    h = {"X-Session-Id": sid, "Content-Type": args.content_type}
    code, body = http("GET", args.server + "/status", None, {})
    print("GET /status      ->", code, body[:200])
    code, body = http("POST", args.server + "/detect", buf.tobytes(), h)
    print("POST /detect     ->", code, body[:300])
    if code != 200:
        print("\nThe server did not accept the frame the way this script sends it.\n"
              "Open app.py, find the /detect route, and check: how the body is read, which header carries\n"
              "the session id, and the expected Content-Type (try --content-type application/octet-stream).")
        return 1
    try:
        r = json.loads(body)
        print("keys present     ->", sorted(k for k in ("status", "calibrated", "ear_alerting", "mar_alerting",
                                                       "nod_alerting", "yaw_alerting", "hand_alerting",
                                                       "down_alerting", "distract_alert") if k in r))
        if "down_alerting" not in r:
            print("NOTE: no 'down_alerting' in the reply -> this is not detector v4.2; "
                  "the looking_down row will be skipped.")
    except Exception:
        print("Reply is not JSON.")
        return 1
    code, body = http("POST", args.server + "/reset", b"", h)
    print("POST /reset      ->", code, body[:120])
    print("\nSmoke test OK." if code == 200 else "\n/reset failed (non-fatal for evaluation).")
    return 0


def run(args) -> int:
    import cv2
    os.makedirs(args.out, exist_ok=True)
    pid = "".join(ch for ch in args.participant if ch.isalnum()) or "P"
    prefix = os.path.join(args.out, pid)
    for suffix in ("_frames.csv", "_labels.csv", "_meta.json"):
        if os.path.exists(prefix + suffix) and not args.overwrite:
            raise SystemExit(f"{prefix + suffix} exists. Use a new --participant id or --overwrite.")

    sid = ("eval" + pid + secrets.token_hex(6))[:60]
    hdr = {"X-Session-Id": sid, "Content-Type": args.content_type}
    code, status_body = http("GET", args.server + "/status", None, {})
    if code != 200:
        raise SystemExit(f"Server not reachable at {args.server} (GET /status -> {code}). Start python app.py.")
    http("POST", args.server + "/reset", b"", hdr)

    seed = args.seed if args.seed is not None else (zlib.crc32(pid.encode()) % 100000)
    items = P.build_schedule(seed, reps=args.reps, time_scale=args.time_scale)
    cam = Camera(args.camera, args.width, args.height, args.fake_camera)
    cue = Cue(gui=not args.no_gui)
    try:
        server_status = json.loads(status_body)
    except Exception:
        server_status = {}

    writer_file = open(prefix + "_frames.csv", "w", newline="", encoding="utf-8")
    fw = csv.DictWriter(writer_file, fieldnames=FRAME_COLS)
    fw.writeheader()
    video = None
    if args.save_video:
        video = cv2.VideoWriter(prefix + "_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                (cam.w, cam.h))

    t0 = time.monotonic()
    period = 1.0 / args.fps
    next_tick = t0
    sched_start = None            # session time when the schedule begins
    cal_done_at = None
    n_frames = n_http_fail = 0
    has_lookdown = None
    cal_failed = False
    idx = 0                       # current schedule item
    beeped_on = beeped_off = -1
    cue.show("LOOK STRAIGHT AT THE CAMERA\ncalibrating (about 10 seconds)")

    try:
        while True:
            cue.pump()
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.005, next_tick - now))
                continue
            next_tick += period
            if next_tick < now - 0.5:                  # fell far behind: do not burst
                next_tick = now + period

            frame = cam.read()
            if frame is None:
                continue
            t_c = time.monotonic() - t0
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            t_e = time.monotonic() - t0
            code, body = http("POST", args.server + "/detect", buf.tobytes(), hdr)
            t_r = time.monotonic() - t0
            if video is not None:
                video.write(frame)
            n_frames += 1

            row = {"t_capture": f"{t_c:.4f}", "t_encoded": f"{t_e:.4f}", "t_response": f"{t_r:.4f}",
                   "http_ok": 1 if code == 200 else 0, "status": "", "calibrated": 0}
            for c in P.CUES:
                row["a_" + c] = 0
            r = {}
            if code == 200:
                try:
                    r = json.loads(body)
                except Exception:
                    r = {}
            if not r or "error" in r:
                n_http_fail += 1
            else:
                if has_lookdown is None:
                    has_lookdown = "down_alerting" in r
                row.update(alert_flags(r))
                row["status"] = r.get("status", "")
                row["calibrated"] = 1 if r.get("calibrated") else 0
                for k_src, k_dst in (("ear", "ear"), ("mar", "mar"), ("nod_drop", "nod_drop"), ("yaw", "yaw"),
                                     ("perclos", "perclos"), ("look_down", "look_down"), ("message", "message")):
                    row[k_dst] = r.get(k_src, "")
            fw.writerow(row)

            # ── phase control ──
            if sched_start is None:
                if r.get("calibration_failed"):
                    cal_failed = True
                    break
                if r.get("calibrated") and cal_done_at is None:
                    cal_done_at = t_c
                    cue.show("CALIBRATED\nget ready...")
                if cal_done_at is not None and t_c >= cal_done_at + P.SETTLE_AFTER_CAL_S:
                    sched_start = t_c
                if sched_start is None and t_c > P.CAL_TIMEOUT_S:
                    cal_failed = True
                    break
                continue

            ts = t_c - sched_start
            if idx >= len(items):
                break
            it = items[idx]
            if ts >= it.t_end:
                idx += 1
                continue
            if ts < it.t_on:
                cue.show(f"GET READY ({it.t_on - ts:0.0f})\n" + it.text)
            elif ts < it.t_off:
                cue.show("NOW\n" + it.text)
                if beeped_on != it.item_id:
                    beeped_on = it.item_id
                    cue.beep("on")
            else:
                cue.show("REST - sit normally,\nlook at the camera")
                if beeped_off != it.item_id:
                    beeped_off = it.item_id
                    cue.beep("off")
    except KeyboardInterrupt:
        print("\nInterrupted - saving what was recorded (session will be marked incomplete).")
    finally:
        writer_file.close()
        if video is not None:
            video.release()
        cam.release()

    complete = (idx >= len(items) and sched_start is not None and not cal_failed)
    with open(prefix + "_labels.csv", "w", newline="", encoding="utf-8") as fh:
        lw = csv.writer(fh)
        lw.writerow(["item_id", "kind", "type", "t_prep", "t_on", "t_off", "t_end"])
        for it in items:
            base = sched_start or 0.0
            lw.writerow([it.item_id, it.kind, it.type, f"{base + it.t_prep:.4f}", f"{base + it.t_on:.4f}",
                         f"{base + it.t_off:.4f}", f"{base + it.t_end:.4f}"])

    meta = {
        "participant": pid, "role": args.role, "complete": bool(complete), "calibration_failed": cal_failed,
        "glasses": args.glasses, "lighting": args.lighting, "camera_distance_cm": args.camera_distance_cm,
        "camera_height": args.camera_height, "notes": args.notes,
        "seed": seed, "reps": args.reps, "synthetic": bool(args.fake_camera),
        "scoring": P.scoring_constants(args.time_scale),
        "schedule_start_s": sched_start, "calibration_done_s": cal_done_at,
        "frames_sent": n_frames, "frames_failed": n_http_fail,
        "has_looking_down": bool(has_lookdown),
        "achieved_fps_overall": round(n_frames / max(1e-9, (time.monotonic() - t0)), 2),
        "environment": env_info(args, cam, server_status),
    }
    with open(prefix + "_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    if cal_failed:
        print("Calibration did not complete (face not found or not frontal). Re-run in better light.")
        return 2
    print(f"\nSaved {prefix}_frames.csv / _labels.csv / _meta.json   (complete={complete})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:5000")
    ap.add_argument("--participant", default="P01")
    ap.add_argument("--role", choices=["test", "dev"], default="test",
                    help="'dev' = people whose data was used to choose thresholds (the authors); excluded from results")
    ap.add_argument("--out", default="data")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=10.0, help="send rate; the detector assumes about 10 fps")
    ap.add_argument("--jpeg-quality", type=int, default=70)
    ap.add_argument("--content-type", default="image/jpeg")
    ap.add_argument("--reps", type=int, default=4, help="repetitions per cue (4 = about 10 min session)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--glasses", default="unknown", help="no | glasses | sunglasses")
    ap.add_argument("--lighting", default="unknown", help="daylight | indoor | dim")
    ap.add_argument("--camera-distance-cm", default="unknown")
    ap.add_argument("--camera-height", default="unknown", help="eye-level | below | above")
    ap.add_argument("--notes", default="")
    ap.add_argument("--save-video", action="store_true", help="also save a video (off by default: privacy)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-gui", action="store_true", help="console cues instead of a cue window")
    ap.add_argument("--smoke-test", action="store_true", help="check the /detect contract and exit")
    ap.add_argument("--fake-camera", action="store_true", help="SOFTWARE TESTING ONLY")
    ap.add_argument("--time-scale", type=float, default=1.0, help="SOFTWARE TESTING ONLY (must be 1.0 for real data)")
    args = ap.parse_args()
    if args.smoke_test:
        return smoke_test(args)
    if not args.fake_camera and args.time_scale != 1.0:
        raise SystemExit("--time-scale is for software tests only; real sessions must use 1.0.")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
