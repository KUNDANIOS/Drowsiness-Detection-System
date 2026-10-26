"""Scripted evaluation protocol for DrowseGuard.

Ground truth comes from the schedule itself: the participant is cued to perform
one behaviour at a known time, so every labelled event has a known onset/offset
without manual frame-by-frame annotation. These are deliberately PERFORMED
behaviours ("controlled facial-action simulations"), not real drowsiness; the
paper must say so.

Everything that affects scoring is a constant here and is written into each
session's meta.json, so the analysis uses exactly what the session used.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

# Alert types the detector can raise; one Table-4 row each.
CUES = ["eye_closure", "yawn", "head_lowering", "side_turn",
        "raised_hands", "face_not_detected", "looking_down"]

CUE_TITLES = {
    "eye_closure": "Sustained eye closure",
    "yawn": "Yawning",
    "head_lowering": "Head lowering",
    "side_turn": "Sideward head rotation",
    "raised_hands": "Visible raised hands",
    "face_not_detected": "Face not detected",
    "looking_down": "Looking down (eyes open)",
}

# Seconds. PREP is the countdown before the behaviour (excluded from scoring);
# TOL is the pre-specified window after the behaviour ends in which an alert
# still counts as detecting it; REST separates items.
PREP_S = 3.0
TOL_S = 2.0
REST_S = 6.0
SETTLE_AFTER_CAL_S = 2.0
CAL_TIMEOUT_S = 45.0

# What the participant does. hold = how long (s) the behaviour is held.
CUE_SHORT = {
    "eye_closure": "Eye\nclosure", "yawn": "Yawning", "head_lowering": "Head\nlowering",
    "side_turn": "Sideward\nrotation", "raised_hands": "Raised\nhands",
    "face_not_detected": "Face not\ndetected", "looking_down": "Looking\ndown",
}

POSITIVES: Dict[str, dict] = {
    "eye_closure": dict(hold=6.0, text="CLOSE YOUR EYES\nas if falling asleep\n(open them at the second beep)"),
    "yawn": dict(hold=4.0, text="YAWN WIDELY\nmouth open, hold it"),
    "head_lowering": dict(hold=4.0, text="LOWER YOUR HEAD\nchin toward your chest, hold"),
    "side_turn": dict(hold=3.0, text="TURN YOUR HEAD SIDEWAYS\nlook out of the side window, hold"),
    "raised_hands": dict(hold=3.0, text="RAISE BOTH HANDS\nopen palms beside your face, hold"),
    "face_not_detected": dict(hold=3.0, text="LEAN OUT OF THE CAMERA VIEW\nface must not be visible, hold"),
    "looking_down": dict(hold=4.0, text="LOOK DOWN AT YOUR LAP / PHONE\neyes OPEN, head fairly still, hold"),
}

# Normal behaviour that looks a bit like a cue. Nothing should alert.
NEGATIVES: Dict[str, dict] = {
    "normal": dict(hold=15.0, text="DRIVE NORMALLY\nlook ahead, blink naturally"),
    "talking": dict(hold=12.0, text="TALK OR READ ALOUD\nnormal conversation"),
    "mirror_glance": dict(hold=8.0, text="MIRROR GLANCES\nglance quickly at a side mirror and back\n(about 1 second each, 2-3 times)"),
    "fast_blinking": dict(hold=10.0, text="BLINK DELIBERATELY\nabout once every 2 seconds"),
}
NEGATIVE_COUNTS = {"normal": 4, "talking": 2, "mirror_glance": 3, "fast_blinking": 2}


@dataclass
class Item:
    item_id: int
    kind: str          # "event" | "negative"
    type: str          # cue name or negative name
    text: str
    t_prep: float      # schedule time (s) when the countdown starts
    t_on: float        # behaviour start
    t_off: float       # behaviour end
    t_end: float       # end of the rest period


def build_schedule(seed: int, reps: int = 4, time_scale: float = 1.0,
                   cues: List[str] | None = None) -> List[Item]:
    """Deterministic, shuffled schedule (same seed -> same schedule).

    time_scale < 1 is for software tests only; real sessions must use 1.0.
    """
    rng = random.Random(seed)
    cues = cues or CUES
    pool = [("event", c) for c in cues for _ in range(reps)]
    pool += [("negative", n) for n, k in NEGATIVE_COUNTS.items()
             for _ in range(max(1, int(round(k * reps / 4))))]

    # shuffle; avoid the same item type twice in a row
    for _ in range(200):
        rng.shuffle(pool)
        if all(pool[i][1] != pool[i + 1][1] for i in range(len(pool) - 1)):
            break

    items: List[Item] = []
    t = 0.0
    for i, (kind, typ) in enumerate(pool):
        spec = (POSITIVES if kind == "event" else NEGATIVES)[typ]
        hold = spec["hold"] * time_scale
        prep, rest = PREP_S * time_scale, REST_S * time_scale
        items.append(Item(i, kind, typ, spec["text"], t, t + prep, t + prep + hold,
                          t + prep + hold + rest))
        t += prep + hold + rest
    return items


def schedule_duration(items: List[Item]) -> float:
    return items[-1].t_end if items else 0.0


def scoring_constants(time_scale: float = 1.0) -> dict:
    return dict(prep_s=PREP_S * time_scale, tol_s=TOL_S * time_scale,
                rest_s=REST_S * time_scale, time_scale=time_scale)
