"""
DrowseGuard — Drowsiness Detector  v4.1

Changes vs v4.0 (output JSON keys are unchanged; two keys were added:
"calibration_failed" and "calibrating"):

  1. PERCLOS is only evaluated once the rolling window is mostly full
     (EAR_PERCLOS_MIN_FILL). Before, a nearly empty window (after
     calibration, after a lost face, after an alert cleared) made 5 closed
     frames read as 100% PERCLOS and fire an eye alert.
  2. Calibration rewritten:
       * also records the driver's "straight ahead" yaw, so an off-axis
         camera no longer makes calibration reject every frame;
       * the nod baseline is filtered the same way as the EAR baseline
         (before, it used every frame, including turned ones);
       * it keeps going until enough valid samples exist, with a timeout and
         an explicit calibration_failed flag instead of silently falling back.
  3. While the head is turned (|yaw| > YAW_HOLD_THRESH) eye / yawn / nod
     metrics are held, not updated: EAR and nose-Y are unreliable on a turned
     face and used to cause extra eye alerts on top of the distraction alert.
  4. _build() no longer reports calibrated=True / 100% when the face is lost
     during calibration.
  5. Hand logic is unchanged in behaviour ("every visible hand is raised"),
     now written that way. NOTE: "no hands visible" counts as normal, so this
     cannot detect hands resting in the lap. See README note.
  6. Face loss also resets the partial yawn / nod / eye-lock / hand counters.
  7. Models load lazily on first use (no network download at import time) and
     cv2 / mediapipe are imported lazily; scipy is no longer needed.
  8. Hand inference is skipped while there is no face or calibration is still
     running (the result is not used then) — roughly halves work in those
     phases.

Still frame-count based (assumes ~10 fps) and `yaw` is still a unitless ratio,
NOT degrees. Both are listed as follow-ups.
"""

import logging
import math
import os
import threading
import urllib.request
from collections import deque

import numpy as np

log = logging.getLogger("drowseguard.detector")

# ─────────────────────────────────────────────────────────────
#  THRESHOLDS
# ─────────────────────────────────────────────────────────────

# EAR — PERCLOS method
EAR_THRESHOLD        = 0.19     # lowest value the personal EAR threshold may take
EAR_DROP_FROM_BASE   = 0.12     # personal drop from baseline
EAR_PERCLOS_WINDOW   = 40       # 4s rolling window at 10fps
EAR_PERCLOS_MIN_FILL = 0.75     # window must be this full before PERCLOS counts
EAR_PERCLOS_MIN_SAMPLES = int(EAR_PERCLOS_WINDOW * EAR_PERCLOS_MIN_FILL)
EAR_PERCLOS_THRESH   = 0.35     # >35% frames closed
EAR_LOCK_FRAMES      = 5        # must stay over PERCLOS threshold for N frames
EAR_SMOOTH_ALPHA     = 0.25

# Calibration
CAL_FRAMES           = 90       # valid samples to collect (~9s at 10fps)
CAL_MAX_FRAMES       = 300      # give up after this many face frames (~30s)
CAL_MIN_VALID        = 20       # minimum usable samples after yaw filtering
CAL_EAR_MIN          = 0.20     # sample EAR must be inside this band
CAL_EAR_MAX          = 0.48
CAL_MAX_YAW          = 0.18     # keep samples within this yaw of the median

# MAR — blendshape jawOpen
JAW_OPEN_THRESHOLD   = 0.42
MAR_GEO_THRESHOLD    = 0.12
MAR_FRAMES           = 14
MAR_RESET_FRAMES     = 6
MAR_SMOOTH_ALPHA     = 0.40

# NOD
NOD_THRESHOLD        = 0.07
NOD_FRAMES           = 12
NOD_RESET_FRAMES     = 8
NOD_SMOOTH_ALPHA     = 0.35

# Distraction — side face / no face
NO_FACE_FRAMES       = 12       # 1.2s
YAW_DISTRACT_THRESH  = 0.35     # |yaw - baseline| ratio (NOT degrees)
YAW_DISTRACT_FRAMES  = 8        # 0.8s sustained side-face before alert
YAW_HOLD_THRESH      = 0.25     # above this, eye/yawn/nod metrics are held (tune on data)

# Hands
HAND_RAISED_FRAMES   = 15       # 1.5s with hands raised before alert
HAND_RESET_FRAMES    = 10
HAND_MODEL_PATH      = "hand_landmarker.task"
HAND_MODEL_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ─────────────────────────────────────────────────────────────
#  LANDMARK INDICES
# ─────────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_TOP = 0
MOUTH_BOT = 17
FOREHEAD  = 10
CHIN      = 152
NOSE_TIP  = 1
LEFT_EYE_INNER  = 362
RIGHT_EYE_INNER = 133
LEFT_EAR_POINT  = 234   # left face edge
RIGHT_EAR_POINT = 454   # right face edge

# ─────────────────────────────────────────────────────────────
#  MODELS (lazy — nothing is downloaded or loaded at import time)
# ─────────────────────────────────────────────────────────────
FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

_models_lock = threading.Lock()
_face_landmarker = None
_hand_landmarker = None
_hand_load_attempted = False


def ensure_model(path, url):
    if not os.path.exists(path):
        log.info("Downloading %s (~30 MB)…", path)
        urllib.request.urlretrieve(url, path)
        log.info("Download finished: %s", path)


def get_face_landmarker():
    global _face_landmarker
    with _models_lock:
        if _face_landmarker is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.50,
                min_face_presence_confidence=0.50,
                min_tracking_confidence=0.50,
            )
            _face_landmarker = vision.FaceLandmarker.create_from_options(options)
        return _face_landmarker


def get_hand_landmarker():
    """Optional: returns None (and remembers it) if the hand model can't load."""
    global _hand_landmarker, _hand_load_attempted
    with _models_lock:
        if not _hand_load_attempted:
            _hand_load_attempted = True
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
                ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)
                options = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                    num_hands=2,
                    min_hand_detection_confidence=0.50,
                    min_hand_presence_confidence=0.50,
                    min_tracking_confidence=0.50,
                )
                _hand_landmarker = vision.HandLandmarker.create_from_options(options)
                log.info("Hand landmarker loaded.")
            except Exception as e:
                log.warning("Hand landmarker not available: %s — hand detection disabled.", e)
        return _hand_landmarker


# ─────────────────────────────────────────────────────────────
#  METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def calc_ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.30


def calc_mar_geometry(lm, w, h):
    top  = np.array([lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h])
    bot  = np.array([lm[MOUTH_BOT].x * w, lm[MOUTH_BOT].y * h])
    fore = np.array([lm[FOREHEAD].x  * w, lm[FOREHEAD].y  * h])
    chin = np.array([lm[CHIN].x      * w, lm[CHIN].y      * h])
    vert = float(np.linalg.norm(top - bot))
    fh   = float(np.linalg.norm(fore - chin))
    return vert / fh if fh > 0 else 0.0


def get_jaw_blendshape(blendshapes):
    if not blendshapes or not blendshapes[0]:
        return None
    for bs in blendshapes[0]:
        if bs.category_name == 'jawOpen':
            return float(bs.score)
    return None


def calc_nose_y_norm(lm, w, h):
    ny  = lm[NOSE_TIP].y * h
    fy  = lm[FOREHEAD].y * h
    cy  = lm[CHIN].y * h
    fh  = abs(cy - fy)
    return (ny - fy) / fh if fh > 0 else 0.5


def calc_face_yaw(lm, w, h):
    """
    Horizontal head turn estimated from landmark geometry: horizontal offset
    of the nose tip from the eye midpoint, divided by face width, scaled.

    Returns a SIGNED unitless ratio clipped to [-1, 1] (NOT degrees):
        0 = nose centred between the eyes; +/-1 = strongly turned.
    Use the difference from the driver's calibrated baseline, not the raw value.
    """
    le = np.array([lm[LEFT_EYE_INNER].x * w,  lm[LEFT_EYE_INNER].y * h])
    re = np.array([lm[RIGHT_EYE_INNER].x * w, lm[RIGHT_EYE_INNER].y * h])
    eye_mid = (le + re) / 2.0
    nose = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])

    left_edge  = np.array([lm[LEFT_EAR_POINT].x  * w, lm[LEFT_EAR_POINT].y  * h])
    right_edge = np.array([lm[RIGHT_EAR_POINT].x * w, lm[RIGHT_EAR_POINT].y * h])
    face_width = float(np.linalg.norm(left_edge - right_edge))

    if face_width < 1.0:
        return 0.0

    offset = float(nose[0] - eye_mid[0]) / face_width
    return float(np.clip(offset * 2.5, -1.0, 1.0))


def calc_hand_raised(hand_result, lm, h):
    """
    'raised'  — at least one hand is visible and EVERY visible hand is above
                the reference line (chin minus 25% of face height)
    'normal'  — no hands visible, or some visible hand is below the line
    None      — hand detection unavailable

    Limitation: a camera can't see hands resting on the wheel or in the lap, so
    "no hands visible" is treated as normal. This detects raised hands (phone,
    face touching, waving), not hands that are merely off the wheel.
    """
    if hand_result is None:
        return None

    chin_y = lm[CHIN].y * h
    fore_y = lm[FOREHEAD].y * h
    face_h = abs(chin_y - fore_y)
    ref_y  = chin_y - face_h * 0.25

    if not hand_result.hand_landmarks:
        return 'normal'

    raised_flags = []
    for hand_lm in hand_result.hand_landmarks:
        key_pts = [hand_lm[i] for i in (0, 5, 9, 13, 17)]   # wrist + finger bases
        cy = float(np.mean([p.y * h for p in key_pts]))
        raised_flags.append(cy < ref_y)

    return 'raised' if all(raised_flags) else 'normal'


def exp_smooth(prev, val, alpha):
    return val if prev is None else alpha * val + (1.0 - alpha) * prev


# ─────────────────────────────────────────────────────────────
#  DETECTOR
# ─────────────────────────────────────────────────────────────

class DrowsinessDetector:

    def __init__(self):
        # EAR / PERCLOS
        self.ear_sm           = None
        self.ear_baseline     = None
        self.ear_window       = deque(maxlen=EAR_PERCLOS_WINDOW)
        self.ear_alerting     = False
        self.ear_fired        = False
        self.ear_clear        = 0
        self.ear_lock_count   = 0

        # MAR / yawn
        self.mar_sm           = None
        self.mar_frames       = 0
        self.mar_alerting     = False
        self.mar_fired        = False
        self.mar_clear        = 0
        self.using_blendshape = False

        # NOD
        self.nod_sm           = None
        self.nod_baseline     = None
        self.nod_frames       = 0
        self.nod_alerting     = False
        self.nod_fired        = False
        self.nod_clear        = 0

        # Side-face / yaw distraction
        self.yaw_baseline     = 0.0
        self.yaw_sm           = None
        self.yaw_frames       = 0
        self.yaw_alerting     = False
        self.yaw_fired        = False
        self.yaw_clear        = 0

        # No-face / distraction
        self.no_face_frames   = 0
        self.no_face_fired    = False

        # Hands
        self.hand_frames      = 0
        self.hand_alerting    = False
        self.hand_fired       = False
        self.hand_clear       = 0
        self.hand_alerts      = 0
        self._hand_err_logged = False

        # Calibration
        self.cal_frame        = 0          # face frames seen while calibrating
        self._cal_samples     = []         # (ear, yaw, nose_y)
        self.cal_done         = False
        self.cal_failed       = False

        # Stats
        self.frames           = 0
        self.eye_alerts       = 0
        self.yawn_alerts      = 0
        self.nod_alerts       = 0
        self.dist_alerts      = 0
        self.status           = "no_face"

    # ── helpers ───────────────────────────────────────────
    @property
    def ear_thr(self):
        if self.ear_baseline:
            return max(self.ear_baseline - EAR_DROP_FROM_BASE, EAR_THRESHOLD)
        return EAR_THRESHOLD

    def _perclos(self):
        n = len(self.ear_window)
        if n < EAR_PERCLOS_MIN_SAMPLES:
            return 0.0
        return sum(self.ear_window) / n

    def _cal_pct(self):
        if self.cal_done:
            return 100
        by_samples = len(self._cal_samples) / CAL_FRAMES
        by_time    = self.cal_frame / CAL_MAX_FRAMES
        return min(99, int(max(by_samples, by_time) * 100))

    def _clear_ear(self):
        self.ear_alerting   = False
        self.ear_fired      = False
        self.ear_clear      = 0
        self.ear_lock_count = 0
        self.ear_window.clear()

    def _clear_mar(self):
        self.mar_alerting = False
        self.mar_fired    = False
        self.mar_frames   = 0
        self.mar_clear    = 0

    def _clear_nod(self):
        self.nod_alerting = False
        self.nod_fired    = False
        self.nod_frames   = 0
        self.nod_clear    = 0

    def _clear_yaw(self):
        self.yaw_alerting = False
        self.yaw_fired    = False
        self.yaw_frames   = 0
        self.yaw_clear    = 0

    def _clear_hand(self):
        self.hand_alerting = False
        self.hand_fired    = False
        self.hand_frames   = 0
        self.hand_clear    = 0

    # ─────────────────────────────────────────────────────
    def process_frame(self, frame_bytes: bytes) -> dict:
        import cv2
        import mediapipe as mp

        arr   = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "decode_failed"}

        self.frames += 1
        h, w = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = get_face_landmarker().detect(mp_img)

        # Hand result is only used after calibration, with a face present.
        hand_result = None
        if self.cal_done and face_result.face_landmarks:
            hand_model = get_hand_landmarker()
            if hand_model is not None:
                try:
                    hand_result = hand_model.detect(mp_img)
                except Exception as e:
                    if not self._hand_err_logged:
                        log.warning("hand detection failed: %s", e)
                        self._hand_err_logged = True

        return self._analyze(face_result, hand_result, w, h)

    # ─────────────────────────────────────────────────────
    def _analyze(self, face_result, hand_result, w, h):
        # ── NO FACE ───────────────────────────────────────
        if not face_result.face_landmarks:
            self.ear_sm = None
            self.mar_sm = None
            self.nod_sm = None
            self.yaw_sm = None
            self.ear_window.clear()
            self.ear_lock_count = 0
            self.mar_frames     = 0
            self.nod_frames     = 0
            self.hand_frames    = 0
            self._clear_yaw()
            self.no_face_frames += 1
            self.status = "no_face"

            dist = self.no_face_frames >= NO_FACE_FRAMES
            if dist and not self.no_face_fired:
                self.dist_alerts += 1
                self.no_face_fired = True

            return self._build(None, None, None, None,
                               False, False, False, dist, False,
                               perclos=0.0, jaw_raw=None)

        self.no_face_frames = 0
        self.no_face_fired  = False
        lm = face_result.face_landmarks[0]
        raw_yaw = calc_face_yaw(lm, w, h)

        if not self.cal_done:
            return self._calibrate(lm, raw_yaw, w, h)

        # ── FACE YAW — side face = distraction ────────────
        yaw_dev = raw_yaw - self.yaw_baseline
        self.yaw_sm = exp_smooth(self.yaw_sm, abs(yaw_dev), 0.30)

        yaw_alert = False
        if self.yaw_sm > YAW_DISTRACT_THRESH:
            self.yaw_frames += 1
            self.yaw_clear   = 0
            if self.yaw_frames >= YAW_DISTRACT_FRAMES:
                self.yaw_alerting = True
                yaw_alert         = True
                if not self.yaw_fired:
                    self.dist_alerts += 1
                    self.yaw_fired    = True
        else:
            self.yaw_frames = 0
            if self.yaw_alerting:
                self.yaw_clear += 1
                if self.yaw_clear >= 6:
                    self._clear_yaw()

        # ── EYES / YAWN / NOD (held while head is turned) ──
        if self.yaw_sm > YAW_HOLD_THRESH:
            eye_alert = yawn_alert = nod_alert = False
            perclos   = self._perclos()
            mar_display = self.mar_sm
            jaw_score = None
            nod_drop  = None
        else:
            eye_alert, perclos = self._update_eyes(lm, w, h)
            yawn_alert, mar_display, jaw_score = self._update_yawn(face_result, lm, w, h)
            nod_alert, nod_drop = self._update_nod(lm, w, h)

        # ── HANDS ─────────────────────────────────────────
        hand_alert = self._update_hands(hand_result, lm, h)

        # ── OVERALL STATUS ────────────────────────────────
        dist_alert = yaw_alert

        if self.ear_alerting or self.nod_alerting or self.mar_alerting:
            self.status = "drowsy"
        elif dist_alert or hand_alert:
            self.status = "distracted"
        else:
            self.status = "awake"

        return self._build(
            self.ear_sm, mar_display, nod_drop, abs(yaw_dev),
            eye_alert, yawn_alert, nod_alert, dist_alert, hand_alert,
            perclos=perclos, jaw_raw=jaw_score
        )

    # ── calibration ───────────────────────────────────────
    def _calibrate(self, lm, raw_yaw, w, h):
        self.cal_frame += 1
        raw_e = (calc_ear(lm, LEFT_EYE, w, h) + calc_ear(lm, RIGHT_EYE, w, h)) / 2

        if CAL_EAR_MIN < raw_e < CAL_EAR_MAX:
            self._cal_samples.append((raw_e, raw_yaw, calc_nose_y_norm(lm, w, h)))

        if len(self._cal_samples) >= CAL_FRAMES or self.cal_frame >= CAL_MAX_FRAMES:
            self._finish_calibration()

        self.status = "awake"
        return self._build(raw_e, 0.0, None, None,
                           False, False, False, False, False,
                           perclos=0.0, jaw_raw=None)

    def _finish_calibration(self):
        samples = self._cal_samples
        kept = []
        yaw_med = 0.0
        if samples:
            yaw_med = float(np.median([s[1] for s in samples]))
            kept = [s for s in samples if abs(s[1] - yaw_med) < CAL_MAX_YAW]

        if len(kept) >= CAL_MIN_VALID:
            self.yaw_baseline = yaw_med
            self.ear_baseline = float(np.median([s[0] for s in kept]))
            self.nod_baseline = float(np.median([s[2] for s in kept]))
        else:
            self.cal_failed = True
            log.warning("Calibration failed: only %d usable samples; using fixed thresholds.",
                        len(kept))
        self.cal_done = True

    # ── per-signal updates ────────────────────────────────
    def _update_eyes(self, lm, w, h):
        raw_ear = (calc_ear(lm, LEFT_EYE, w, h) + calc_ear(lm, RIGHT_EYE, w, h)) / 2.0
        self.ear_sm = exp_smooth(self.ear_sm, raw_ear, EAR_SMOOTH_ALPHA)

        self.ear_window.append(1 if self.ear_sm < self.ear_thr else 0)
        perclos = self._perclos()

        eye_alert = False
        if perclos >= EAR_PERCLOS_THRESH:
            self.ear_lock_count += 1
            if self.ear_lock_count >= EAR_LOCK_FRAMES:
                self.ear_alerting = True
                eye_alert         = True
                if not self.ear_fired:
                    self.eye_alerts += 1
                    self.ear_fired   = True
        else:
            self.ear_lock_count = max(0, self.ear_lock_count - 1)
            if self.ear_alerting:
                self.ear_clear += 1
                if self.ear_clear >= 10:
                    self._clear_ear()
            else:
                self.ear_clear = 0
        return eye_alert, perclos

    def _update_yawn(self, face_result, lm, w, h):
        jaw_score = get_jaw_blendshape(face_result.face_blendshapes)

        if jaw_score is not None:
            self.using_blendshape = True
            self.mar_sm   = exp_smooth(self.mar_sm, jaw_score, MAR_SMOOTH_ALPHA)
            mar_triggered = self.mar_sm > JAW_OPEN_THRESHOLD
        else:
            self.using_blendshape = False
            raw_mar       = calc_mar_geometry(lm, w, h)
            self.mar_sm   = exp_smooth(self.mar_sm, raw_mar, MAR_SMOOTH_ALPHA)
            mar_triggered = self.mar_sm > MAR_GEO_THRESHOLD

        yawn_alert = False
        if mar_triggered:
            self.mar_frames += 1
            self.mar_clear   = 0
            if self.mar_frames >= MAR_FRAMES:
                self.mar_alerting = True
                yawn_alert        = True
                if not self.mar_fired:
                    self.yawn_alerts += 1
                    self.mar_fired    = True
        else:
            self.mar_frames = 0
            if self.mar_alerting:
                self.mar_clear += 1
                if self.mar_clear >= MAR_RESET_FRAMES:
                    self._clear_mar()
        return yawn_alert, self.mar_sm, jaw_score

    def _update_nod(self, lm, w, h):
        ny = calc_nose_y_norm(lm, w, h)
        self.nod_sm = exp_smooth(self.nod_sm, ny, NOD_SMOOTH_ALPHA)

        nod_alert = False
        if self.nod_baseline is None:
            return nod_alert, None

        nod_drop = self.nod_sm - self.nod_baseline
        if nod_drop > NOD_THRESHOLD:
            self.nod_frames += 1
            self.nod_clear   = 0
            if self.nod_frames >= NOD_FRAMES:
                self.nod_alerting = True
                nod_alert         = True
                if not self.nod_fired:
                    self.nod_alerts += 1
                    self.nod_fired   = True
        else:
            self.nod_frames = 0
            if self.nod_alerting:
                self.nod_clear += 1
                if self.nod_clear >= NOD_RESET_FRAMES:
                    self._clear_nod()
        return nod_alert, nod_drop

    def _update_hands(self, hand_result, lm, h):
        hand_alert  = False
        hand_status = calc_hand_raised(hand_result, lm, h)

        if hand_status == 'raised':
            self.hand_frames += 1
            self.hand_clear   = 0
            if self.hand_frames >= HAND_RAISED_FRAMES:
                self.hand_alerting = True
                hand_alert         = True
                if not self.hand_fired:
                    self.hand_alerts += 1
                    self.hand_fired   = True
        else:
            self.hand_frames = max(0, self.hand_frames - 1)
            if self.hand_alerting:
                self.hand_clear += 1
                if self.hand_clear >= HAND_RESET_FRAMES:
                    self._clear_hand()
        return hand_alert

    # ─────────────────────────────────────────────────────
    def _build(self, ear_v, mar_v, nod_v, yaw_v,
               eye_a, yawn_a, nod_a, dist_a, hand_a,
               perclos=0.0, jaw_raw=None):

        any_alert = eye_a or yawn_a or nod_a or dist_a or hand_a

        ec = min(100, int(perclos / EAR_PERCLOS_THRESH * 100)) if perclos else 0
        mc = 0
        if mar_v is not None:
            thr = JAW_OPEN_THRESHOLD if self.using_blendshape else MAR_GEO_THRESHOLD
            mc  = min(100, int(mar_v / thr * 100))
        nc = min(100, int(max(0.0, nod_v) / NOD_THRESHOLD * 100)) if nod_v is not None else 0
        yc = min(100, int((yaw_v or 0) / YAW_DISTRACT_THRESH * 100)) if yaw_v is not None else 0
        hc = min(100, int(self.hand_frames / HAND_RAISED_FRAMES * 100))

        return {
            "status":                self.status,
            "ear":                   round(ear_v,  4) if ear_v  is not None else None,
            "mar":                   round(mar_v,  4) if mar_v  is not None else None,
            "nod_drop":              round(nod_v,  4) if nod_v  is not None else None,
            "yaw":                   round(yaw_v,  4) if yaw_v  is not None else None,
            "jaw_raw":               round(jaw_raw, 4) if jaw_raw is not None else None,
            "perclos":               round(perclos, 3),
            "using_blendshape":      self.using_blendshape,
            "eye_alert":             eye_a,
            "yawn_alert":            yawn_a,
            "nod_alert":             nod_a,
            "distract_alert":        dist_a,
            "hand_alert":            hand_a,
            "alert":                 any_alert,
            "ear_alerting":          self.ear_alerting,
            "mar_alerting":          self.mar_alerting,
            "nod_alerting":          self.nod_alerting,
            "yaw_alerting":          self.yaw_alerting,
            "hand_alerting":         self.hand_alerting,
            "ear_frames":            int(sum(self.ear_window)) if self.ear_window else 0,
            "mar_frames":            self.mar_frames,
            "nod_frames":            self.nod_frames,
            "yaw_frames":            self.yaw_frames,
            "hand_frames":           self.hand_frames,
            "ear_confidence":        ec,
            "mar_confidence":        mc,
            "nod_confidence":        nc,
            "yaw_confidence":        yc,
            "hand_confidence":       hc,
            "calibrated":            self.cal_done,
            "calibrating":           not self.cal_done,
            "calibration_failed":    self.cal_failed,
            "calibration_progress":  self._cal_pct(),
            "ear_baseline":          round(self.ear_baseline, 4) if self.ear_baseline else None,
            "ear_threshold":         round(self.ear_thr, 4),
            "total_eye_alerts":      self.eye_alerts,
            "total_yawn_alerts":     self.yawn_alerts,
            "total_nod_alerts":      self.nod_alerts,
            "total_distract_alerts": self.dist_alerts,
            "total_hand_alerts":     self.hand_alerts,
            "frames_processed":      self.frames,
            "no_face_frames":        self.no_face_frames,
            "message":               self._msg(eye_a, yawn_a, nod_a, dist_a, hand_a),
        }

    def _msg(self, e, y, n, d, h):
        if d: return "Eyes off road!"
        if h: return "Hands off wheel!"
        parts = []
        if e: parts.append("Eyes closing!")
        if y: parts.append("Yawning!")
        if n: parts.append("Head drooping!")
        return " + ".join(parts) if parts else "Awake"