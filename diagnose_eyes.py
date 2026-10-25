"""
Eye-signal diagnostic v2 — measures what the detector "sees" in several poses
so thresholds can be chosen from real numbers.

HOW TO RUN (from the drowsiness-main folder, venv active):
    1. Close the browser detector tab (only one program can use the webcam).
    2. python diagnose_eyes.py
    3. For each pose: press Enter, then GET INTO the pose during the 2-second
       settle time. Measuring starts after "HOLD" appears. Hold it steadily.
    4. Paste the final tables back into the chat.

Nothing is saved or uploaded; it only prints numbers.
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector as D  # noqa: E402  (reuses the exact EAR / nose code)

SETTLE_SECONDS = 2      # frames in this window are thrown away
SECONDS = 6             # frames in this window are measured
CONDITIONS = [
    ("A", "EYES OPEN, looking straight at the camera"),
    ("B", "EYES OPEN, looking at your phone held low in front of you, EXACTLY like\n"
          "        when the false alarm fired (eyes down, head fairly still)"),
    ("F", "EYES OPEN, looking even LOWER (phone in your lap / low dashboard height)"),
    ("E", "HALF-CLOSED eyelids, like fighting sleep (heavy, droopy, not fully shut)"),
    ("C", "EYES FULLY CLOSED, as if asleep"),
    ("D", "NORMAL BLINKING, looking straight"),
]
EAR_LEVELS = (0.19, 0.15, 0.12)       # 0.19 = what the detector uses now
BLINK_LEVELS = (0.30, 0.40, 0.50)


def blend_dict(result):
    bs = result.face_blendshapes
    if not bs or not bs[0]:
        return {}
    return {b.category_name: float(b.score) for b in bs[0]}


def sample_from_result(result, w, h):
    """One measurement dict from one landmarker result, or None if no face."""
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    ear = (D.calc_ear(lm, D.LEFT_EYE, w, h) + D.calc_ear(lm, D.RIGHT_EYE, w, h)) / 2
    b = blend_dict(result)
    pair = lambda a, c: (b[a] + b[c]) / 2 if a in b and c in b else None  # noqa: E731
    return {
        "ear": ear,
        "blink": pair("eyeBlinkLeft", "eyeBlinkRight"),
        "look_down": pair("eyeLookDownLeft", "eyeLookDownRight"),
    }


def measure(cap, landmarker, settle, seconds, mp, on_phase=None):
    """Discard `settle` seconds, then record `seconds`. Returns (samples, frames)."""
    samples, frames = [], 0
    t0 = time.time()
    announced = False
    while True:
        elapsed = time.time() - t0
        if elapsed >= settle + seconds:
            break
        ok, frame = cap.read()
        if not ok:
            continue
        if elapsed < settle:
            continue
        if not announced:
            announced = True
            if on_phase:
                on_phase()
        frames += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        s = sample_from_result(landmarker.detect(img), w, h)
        if s is not None:
            samples.append(s)
    return samples, frames


def _p(vals):
    return tuple(float(x) for x in np.percentile(np.asarray(vals, float), [10, 50, 90]))


def summarize(samples):
    """Percentiles (p10/p50/p90) and threshold hit-rates (% of frames)."""
    out = {"n": len(samples), "pct": {}, "ear_below": {}, "blink_above": {}}
    for m in ("ear", "blink", "look_down"):
        vals = [s[m] for s in samples if s[m] is not None]
        out["pct"][m] = _p(vals) if vals else None
    ears = [s["ear"] for s in samples]
    blinks = [s["blink"] for s in samples if s["blink"] is not None]
    for t in EAR_LEVELS:
        out["ear_below"][t] = 100.0 * sum(e < t for e in ears) / len(ears) if ears else None
    for t in BLINK_LEVELS:
        out["blink_above"][t] = 100.0 * sum(b > t for b in blinks) / len(blinks) if blinks else None
    return out


def _fmt3(t):
    return "  n/a" + " " * 14 if t is None else f"{t[0]:.2f}/{t[1]:.2f}/{t[2]:.2f}  "


def _fmt_rate(v):
    return "   n/a" if v is None else f"{v:5.0f}%"


def main():
    import mediapipe as mp

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam. Close the browser detector tab "
              "(or any other app using the camera) and try again.")
        return
    landmarker = D.get_face_landmarker()

    results = {}
    for key, text in CONDITIONS:
        print(f"\n[{key}] {text}")
        input(f"    Press Enter, then get into the pose (settle {SETTLE_SECONDS}s, "
              f"then I measure {SECONDS}s)... ")
        for _ in range(5):
            cap.read()
        samples, frames = measure(cap, landmarker, SETTLE_SECONDS, SECONDS, mp,
                                  on_phase=lambda: print("    >>> HOLD the pose <<<"))
        results[key] = samples
        print(f"    captured {len(samples)} face frames out of {frames}")
    cap.release()

    print("\n" + "=" * 86)
    print("PASTE EVERYTHING BELOW THIS LINE INTO THE CHAT")
    print("=" * 86)
    print("Table 1 - p10 / p50(median) / p90 per signal")
    print(f"{'cond':<5}{'n':>5}  {'EAR':<20}{'eyeBlink':<20}{'eyeLookDown':<20}")
    for key, _ in CONDITIONS:
        if not results[key]:
            print(f"{key:<5}{0:>5}  (no face found)")
            continue
        s = summarize(results[key])
        print(f"{key:<5}{s['n']:>5}  {_fmt3(s['pct']['ear']):<20}"
              f"{_fmt3(s['pct']['blink']):<20}{_fmt3(s['pct']['look_down']):<20}")

    print("\nTable 2 - % of frames below / above a level")
    head = "".join(f"EAR<{t:<5}" for t in EAR_LEVELS) + "".join(f"blink>{t:<4}" for t in BLINK_LEVELS)
    print(f"{'cond':<5}{head}")
    for key, _ in CONDITIONS:
        if not results[key]:
            continue
        s = summarize(results[key])
        row = "".join(f"{_fmt_rate(s['ear_below'][t]):<10}" for t in EAR_LEVELS)
        row += "".join(f"{_fmt_rate(s['blink_above'][t]):<10}" for t in BLINK_LEVELS)
        print(f"{key:<5}{row}")
    print("\nLegend: A open/straight  B phone (false-alarm pose)  F looking lower  "
          "E half-closed  C closed  D blinking")


if __name__ == "__main__":
    main()