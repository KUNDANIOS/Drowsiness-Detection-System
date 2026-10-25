"""
Unit tests for detector.py using synthetic face/hand landmarks — no camera,
no model files and no network needed.

Run:  python -m unittest test_detector -v
"""
import os
import sys
import unittest
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector as D  # noqa: E402

W = H = 1000


# ─────────────────────────────────────────────────────────────
#  Synthetic landmark helpers (shared with the old-vs-new comparison)
# ─────────────────────────────────────────────────────────────

def make_landmarks(ear=0.30, yaw=0.0, nose_dy=0.0):
    """
    478 landmarks on a 1000x1000 frame.
      ear     -> EAR of both eyes (A=B=V, EAR = V / eye_width)
      yaw     -> value calc_face_yaw() will return (clipped to +-1)
      nose_dy -> extra nose-tip drop (frame fraction) for nod tests
    """
    lm = [NS(x=0.5, y=0.5) for _ in range(478)]

    def put(i, x, y):
        lm[i] = NS(x=x, y=y)

    eye_w, ey = 0.06, 0.40
    v = ear * eye_w
    # left eye (image right): 362 inner, 263 outer
    put(362, 0.54, ey); put(263, 0.60, ey)
    put(385, 0.55, ey - v / 2); put(380, 0.55, ey + v / 2)
    put(387, 0.58, ey - v / 2); put(373, 0.58, ey + v / 2)
    # right eye (image left): 133 inner, 33 outer
    put(133, 0.46, ey); put(33, 0.40, ey)
    put(160, 0.42, ey - v / 2); put(144, 0.42, ey + v / 2)
    put(158, 0.44, ey - v / 2); put(153, 0.44, ey + v / 2)
    # face edges / forehead / chin / mouth
    put(234, 0.30, 0.45); put(454, 0.70, 0.45)
    put(10, 0.50, 0.20); put(152, 0.50, 0.80)
    put(0, 0.50, 0.66); put(17, 0.50, 0.69)
    # nose: yaw = dx * 1000 / 400 * 2.5  ->  dx = yaw / 6.25
    put(1, 0.50 + yaw / 6.25, 0.55 + nose_dy)
    return lm


def face_result(ear=0.30, yaw=0.0, nose_dy=0.0, jaw=None):
    blend = [[NS(category_name="jawOpen", score=jaw)]] if jaw is not None else None
    return NS(face_landmarks=[make_landmarks(ear, yaw, nose_dy)], face_blendshapes=blend)


def no_face():
    return NS(face_landmarks=[], face_blendshapes=None)


def hands_result(*ys):
    """One 21-point hand per y value (frame fraction). Raised = y < 0.65."""
    return NS(hand_landmarks=[[NS(x=0.5, y=y) for _ in range(21)] for y in ys])


def step(det, hands=None, **kw):
    return det._analyze(face_result(**kw), hands, W, H)


def feed(det, n, hands=None, **kw):
    out = []
    for _ in range(n):
        out.append(step(det, hands=hands, **kw))
    return out


def calibrated(**kw):
    det = D.DrowsinessDetector()
    feed(det, D.CAL_FRAMES, **kw)
    assert det.cal_done
    return det


ORIGINAL_KEYS = [
    "status", "ear", "mar", "nod_drop", "yaw", "jaw_raw", "perclos", "using_blendshape",
    "eye_alert", "yawn_alert", "nod_alert", "distract_alert", "hand_alert", "alert",
    "ear_alerting", "mar_alerting", "nod_alerting", "yaw_alerting", "hand_alerting",
    "ear_frames", "mar_frames", "nod_frames", "yaw_frames", "hand_frames",
    "ear_confidence", "mar_confidence", "nod_confidence", "yaw_confidence",
    "hand_confidence", "calibrated", "calibration_progress", "ear_baseline",
    "ear_threshold", "total_eye_alerts", "total_yawn_alerts", "total_nod_alerts",
    "total_distract_alerts", "total_hand_alerts", "frames_processed", "no_face_frames",
    "message",
]


class SyntheticSanity(unittest.TestCase):
    def test_landmark_generator_produces_expected_metrics(self):
        lm = make_landmarks(ear=0.30, yaw=0.5)
        self.assertAlmostEqual(D.calc_ear(lm, D.LEFT_EYE, W, H), 0.30, places=3)
        self.assertAlmostEqual(D.calc_ear(lm, D.RIGHT_EYE, W, H), 0.30, places=3)
        self.assertAlmostEqual(D.calc_face_yaw(lm, W, H), 0.5, places=3)
        self.assertAlmostEqual(D.calc_nose_y_norm(make_landmarks(), W, H), 0.5833, places=3)


class Calibration(unittest.TestCase):
    def test_completes_with_sane_baselines(self):
        det = D.DrowsinessDetector()
        outs = feed(det, D.CAL_FRAMES - 1)
        self.assertTrue(all(o["calibrated"] is False and o["calibrating"] for o in outs))
        self.assertTrue(all(o["status"] == "awake" for o in outs))
        last = step(det)
        self.assertTrue(last["calibrated"])
        self.assertEqual(last["calibration_progress"], 100)
        self.assertFalse(last["calibration_failed"])
        self.assertAlmostEqual(det.ear_baseline, 0.30, places=2)
        self.assertAlmostEqual(det.nod_baseline, 0.5833, places=2)
        self.assertAlmostEqual(det.yaw_baseline, 0.0, places=2)

    def test_face_lost_mid_calibration_does_not_report_calibrated(self):
        det = D.DrowsinessDetector()
        feed(det, 30)
        out = det._analyze(no_face(), None, W, H)
        self.assertFalse(out["calibrated"])
        self.assertTrue(out["calibrating"])
        self.assertLess(out["calibration_progress"], 100)

    def test_off_axis_camera_calibrates_and_does_not_alert_at_same_pose(self):
        det = D.DrowsinessDetector()
        feed(det, D.CAL_FRAMES, yaw=0.5)            # camera sees the driver 'turned'
        self.assertTrue(det.cal_done)
        self.assertFalse(det.cal_failed)
        self.assertAlmostEqual(det.yaw_baseline, 0.5, places=2)
        self.assertAlmostEqual(det.ear_baseline, 0.30, places=2)
        outs = feed(det, 60, yaw=0.5)
        self.assertFalse(any(o["distract_alert"] for o in outs))
        # now really turn away from the road
        outs = feed(det, 30, yaw=1.0)
        self.assertTrue(any(o["distract_alert"] for o in outs))

    def test_failure_is_explicit_and_falls_back_to_fixed_threshold(self):
        det = D.DrowsinessDetector()
        out = None
        for _ in range(D.CAL_MAX_FRAMES):
            out = step(det, ear=0.10)               # eyes never open -> no valid sample
        self.assertTrue(det.cal_done)
        self.assertTrue(out["calibration_failed"])
        self.assertIsNone(det.ear_baseline)
        self.assertIsNone(det.nod_baseline)
        self.assertEqual(out["ear_threshold"], D.EAR_THRESHOLD)

    def test_turned_frames_are_excluded_from_baselines(self):
        det = D.DrowsinessDetector()
        feed(det, 70)                                # straight, nose_y_norm 0.583
        feed(det, 20, yaw=0.9, nose_dy=0.10)         # a minority of turned/nodded frames
        self.assertTrue(det.cal_done)
        self.assertAlmostEqual(det.nod_baseline, 0.5833, places=2)
        self.assertAlmostEqual(det.yaw_baseline, 0.0, places=2)


class EyeAlerts(unittest.TestCase):
    def test_short_closure_right_after_calibration_does_not_alert(self):
        det = calibrated()
        outs = feed(det, 6, ear=0.08) + feed(det, 40)
        self.assertFalse(any(o["eye_alert"] for o in outs))
        self.assertEqual(det.eye_alerts, 0)

    def test_sustained_closure_alerts(self):
        det = calibrated()
        feed(det, 60)                                # fill the window with open frames
        outs = feed(det, 60, ear=0.08)
        self.assertTrue(any(o["eye_alert"] for o in outs))
        self.assertEqual(outs[-1]["status"], "drowsy")
        self.assertEqual(det.eye_alerts, 1)

    def test_open_eyes_never_alert(self):
        det = calibrated()
        outs = feed(det, 200)
        self.assertFalse(any(o["alert"] for o in outs))
        self.assertTrue(all(o["status"] == "awake" for o in outs))

    def test_perclos_is_zero_until_window_is_mostly_full(self):
        det = calibrated()
        outs = feed(det, D.EAR_PERCLOS_MIN_SAMPLES - 1, ear=0.08)
        self.assertTrue(all(o["perclos"] == 0.0 for o in outs))


class TurnedHead(unittest.TestCase):
    def test_eye_nod_and_yawn_metrics_are_held_while_turned(self):
        det = calibrated()
        feed(det, 60)
        window_before = list(det.ear_window)
        # head turned away, eyes look "closed" from the camera, nose low, mouth open
        outs = feed(det, 60, yaw=0.75, ear=0.08, nose_dy=0.10, jaw=0.9)
        self.assertTrue(any(o["distract_alert"] for o in outs))
        self.assertFalse(any(o["eye_alert"] or o["nod_alert"] or o["yawn_alert"] for o in outs))
        self.assertEqual(list(det.ear_window), window_before)
        self.assertFalse(det.ear_alerting)
        self.assertEqual(outs[-1]["status"], "distracted")

    def test_metrics_resume_when_head_returns(self):
        det = calibrated()
        feed(det, 60)
        feed(det, 20, yaw=0.75)
        outs = feed(det, 80, ear=0.08)               # facing forward again, eyes closed
        self.assertTrue(any(o["eye_alert"] for o in outs))


class Hands(unittest.TestCase):
    def test_all_visible_hands_raised_alerts(self):
        det = calibrated()
        outs = feed(det, 20, hands=hands_result(0.5, 0.5))
        self.assertTrue(outs[-1]["hand_alert"])
        self.assertEqual(det.hand_alerts, 1)
        self.assertEqual(outs[-1]["status"], "distracted")

    def test_single_visible_hand_raised_still_alerts(self):
        det = calibrated()
        outs = feed(det, 20, hands=hands_result(0.5))
        self.assertTrue(outs[-1]["hand_alert"])

    def test_one_hand_low_means_no_alert(self):
        det = calibrated()
        outs = feed(det, 30, hands=hands_result(0.5, 0.9))
        self.assertFalse(any(o["hand_alert"] for o in outs))

    def test_no_hands_visible_is_normal(self):
        det = calibrated()
        outs = feed(det, 30, hands=hands_result())
        self.assertFalse(any(o["hand_alert"] for o in outs))

    def test_hand_model_unavailable(self):
        det = calibrated()
        outs = feed(det, 30, hands=None)
        self.assertFalse(any(o["hand_alert"] for o in outs))

    def test_hand_alert_clears(self):
        det = calibrated()
        feed(det, 20, hands=hands_result(0.5, 0.5))
        outs = feed(det, 15, hands=hands_result(0.9, 0.9))
        self.assertFalse(outs[-1]["hand_alerting"])


class YawnAndNod(unittest.TestCase):
    def test_yawn_alerts_and_clears(self):
        det = calibrated()
        outs = feed(det, 30, jaw=0.8)
        self.assertTrue(any(o["yawn_alert"] for o in outs))
        self.assertEqual(outs[-1]["status"], "drowsy")
        self.assertEqual(det.yawn_alerts, 1)
        outs = feed(det, 15, jaw=0.05)
        self.assertFalse(outs[-1]["mar_alerting"])

    def test_normal_mouth_never_alerts(self):
        det = calibrated()
        outs = feed(det, 60, jaw=0.05)
        self.assertFalse(any(o["yawn_alert"] for o in outs))

    def test_geometric_fallback_without_blendshapes(self):
        det = calibrated()
        outs = feed(det, 20)                         # jaw=None -> geometric MAR (0.05 < 0.12)
        self.assertFalse(outs[-1]["using_blendshape"])
        self.assertFalse(any(o["yawn_alert"] for o in outs))

    def test_head_nod_alerts(self):
        det = calibrated()
        outs = feed(det, 30, nose_dy=0.06)
        self.assertTrue(any(o["nod_alert"] for o in outs))
        self.assertEqual(det.nod_alerts, 1)

    def test_no_nod_when_baseline_missing(self):
        det = D.DrowsinessDetector()
        for _ in range(D.CAL_MAX_FRAMES):
            step(det, ear=0.10)
        outs = feed(det, 30, nose_dy=0.10)
        self.assertFalse(any(o["nod_alert"] for o in outs))
        self.assertTrue(all(o["nod_drop"] is None for o in outs))


class NoFace(unittest.TestCase):
    def test_no_face_distraction_after_threshold(self):
        det = calibrated()
        outs = [det._analyze(no_face(), None, W, H) for _ in range(D.NO_FACE_FRAMES)]
        self.assertFalse(any(o["distract_alert"] for o in outs[:-1]))
        self.assertTrue(outs[-1]["distract_alert"])
        self.assertEqual(det.dist_alerts, 1)
        det._analyze(no_face(), None, W, H)
        self.assertEqual(det.dist_alerts, 1)         # fires once per episode

    def test_face_loss_resets_partial_counters(self):
        det = calibrated()
        feed(det, 5, jaw=0.8, nose_dy=0.06)
        self.assertGreater(det.mar_frames, 0)
        self.assertGreater(det.nod_frames, 0)
        det._analyze(no_face(), None, W, H)
        self.assertEqual((det.mar_frames, det.nod_frames, det.ear_lock_count), (0, 0, 0))


class Contract(unittest.TestCase):
    def test_all_original_json_keys_still_present(self):
        det = calibrated()
        out = step(det)
        missing = [k for k in ORIGINAL_KEYS if k not in out]
        self.assertEqual(missing, [])

    def test_no_face_response_has_same_keys_as_normal_response(self):
        det = calibrated()
        a = set(step(det).keys())
        b = set(det._analyze(no_face(), None, W, H).keys())
        self.assertEqual(a, b)


class ProcessFrameWiring(unittest.TestCase):
    """process_frame() with real cv2 but fake landmarkers (no model files)."""

    def setUp(self):
        import cv2
        self.cv2 = cv2
        ok, buf = cv2.imencode(".jpg", np.zeros((500, 500, 3), np.uint8))
        self.jpeg = buf.tobytes()
        self.hand_calls = 0
        outer = self

        class FakeFace:
            def detect(self, _img):
                return face_result()

        class FakeHand:
            def detect(self, _img):
                outer.hand_calls += 1
                return hands_result(0.9)

        self._orig = (D.get_face_landmarker, D.get_hand_landmarker)
        D.get_face_landmarker = lambda: FakeFace()
        D.get_hand_landmarker = lambda: FakeHand()

    def tearDown(self):
        D.get_face_landmarker, D.get_hand_landmarker = self._orig

    def test_decode_failure(self):
        det = D.DrowsinessDetector()
        self.assertEqual(det.process_frame(b"not a jpeg"), {"error": "decode_failed"})
        self.assertEqual(det.frames, 0)

    def test_hand_model_only_runs_after_calibration(self):
        det = D.DrowsinessDetector()
        for _ in range(D.CAL_FRAMES):
            out = det.process_frame(self.jpeg)
        self.assertTrue(det.cal_done)
        self.assertEqual(self.hand_calls, 0)
        det.process_frame(self.jpeg)
        self.assertEqual(self.hand_calls, 1)
        self.assertIn("status", out)

    def test_import_has_no_side_effects(self):
        self.assertIsNone(D._face_landmarker)


if __name__ == "__main__":
    unittest.main(verbosity=2)