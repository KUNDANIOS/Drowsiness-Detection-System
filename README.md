# 🛡️ DrowseGuard — Real-Time Drowsiness & Distraction Detection

A browser-based driver drowsiness and distraction detector built on **MediaPipe** face and hand landmark models and served by a **Flask** backend.

🚀 **Live demo:** <https://drowsiness-production.up.railway.app/>

> **Status: prototype.** DrowseGuard is a research / learning project. It is **not** a certified safety system and must not be relied on in a real vehicle. See [Limitations](#-limitations) before using it for anything beyond experiments.

---

## 🧠 How it works

The browser captures webcam frames (`getUserMedia`) and POSTs each JPEG frame to the Flask backend. The backend runs MediaPipe's Face Landmarker (with blendshapes) and, once calibrated, the Hand Landmarker, derives several signals, and returns them as JSON together with any active alerts.

### Detection signals

| Signal | Method | Alert condition |
| --- | --- | --- |
| **Eye closure** | Eye Aspect Ratio (EAR) + PERCLOS, confirmed by `eyeLookDown` blendshape | ≥35 % of frames closed over a 4 s window, sustained for 5 frames |
| **Looking down** | Low EAR **and** low `eyeLookDown` (eyes open, gaze low) | ~2 s → *distraction* alert (not drowsy) |
| **Yawning** | `jawOpen` blendshape (geometric MAR fallback) | Jaw open for ~1.4 s |
| **Head nod** | Nose-Y drop from the calibrated baseline | Drop > 0.07 for 1.2 s |
| **Looking away** | Face-yaw ratio relative to your calibrated "straight ahead" | Yaw deviation > 0.35 for 0.8 s |
| **No face** | No landmarks returned | 1.2 s without a face |
| **Hand raised** | Every visible hand above chin level | 1.5 s |

Frame-count thresholds assume roughly **10 fps** (the demo page sends about 10 frames per second).

### Calibration

For the first ~9 seconds (90 valid frames) the detector learns *your* neutral state: median EAR, nose-Y (nod baseline) and yaw ("straight ahead" for your camera position). Look straight at the road with your eyes open during this phase.

- Samples are filtered, so turned-head frames don't skew the baselines.
- If too few valid samples are collected (camera off-axis, eyes never detected open, …) calibration ends after ~30 s with `calibration_failed: true` and the detector falls back to fixed thresholds.

### Looking down vs. eyes closed

Low EAR alone cannot tell "eyes shut" from "eyes open but looking at a phone". When the blendshape scores are available, a low-EAR frame counts as **eye closure** only if the smoothed `eyeLookDown` score is ≥ `LOOKDOWN_CLOSED_MIN` (0.62). Otherwise it counts as **looking down** and raises the *"Eyes off road"* distraction alert instead of a drowsy one.

The 0.62 value was chosen from measurements on one person's webcam (looking low: 90th percentile 0.49; eyes closed: 10th percentile 0.76). **Re-measure with `diagnose_eyes.py` on your own setup** — see [Tuning](#-tuning-with-diagnose_eyespy).

### Other behaviour worth knowing

- While the head is turned away (`YAW_HOLD_THRESH`), eye, yawn and nod metrics are **held**, not updated — a turned face makes the eyes look narrower and used to cause extra false eye alerts.
- PERCLOS only counts once the 4 s window is at least 75 % full, so a short blink right after a reset can't fire an alert.
- Half-closed ("droopy") eyelids look like looking down to this model, so only **fully shut** eyes count as closure. This is a known trade-off.

---

## 🗂️ Project structure

```
drowsiness/
├── app.py                 # Flask backend: routes, per-session detectors
├── detector.py            # DrowsinessDetector class (v4.2)
├── detect.html            # Live detector UI
├── index.html             # Landing page
├── test_detector.py       # Unit tests (synthetic landmarks, no camera/models needed)
├── diagnose_eyes.py       # Webcam tool for measuring eye signals / tuning thresholds
├── face_landmarker.task   # MediaPipe Face Landmarker model
├── hand_landmarker.task   # MediaPipe Hand Landmarker model
├── requirements.txt
├── Dockerfile
└── nixpacks.toml          # Railway build config
```

The two `.task` model files are downloaded automatically on first use if they are missing (~30 MB each). Models load **lazily on the first frame**, so the first request after starting the server is slower.

---

## 🔌 API

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Landing page |
| `GET` | `/detect.html` (aliases: `/detector`, `/detector.html`) | Live detector page |
| `POST` | `/detect` | Body = raw JPEG bytes → JSON with all metrics and alerts |
| `POST` | `/reset` | Reset **your** session's detector state |
| `GET` | `/status` | Health check; includes your session's stats if `X-Session-Id` is sent |

### Sessions

Every browser should send an `X-Session-Id` header (8–64 characters of `A–Z a–z 0–9 _ -`) on `/detect`, `/reset` and `/status`. Each id gets its **own detector** (own calibration, windows and counters). Sessions expire after 15 minutes of inactivity, and at most 200 are kept (oldest idle evicted).

```js
const SID = sessionStorage.getItem('dg_sid') ||
            (sessionStorage.setItem('dg_sid', crypto.randomUUID()), sessionStorage.getItem('dg_sid'));

fetch('/detect', { method: 'POST', body: jpegBlob, headers: { 'X-Session-Id': SID } });
```

Requests **without** the header share a single `"anonymous"` session — fine for solo testing, but multiple visitors would then share state.

### Responses

| Status | Meaning |
| --- | --- |
| `200` | Frame processed |
| `400` | Empty body, bad `X-Session-Id`, or frame could not be decoded (`{"error": "decode_failed"}`) |
| `413` | Frame larger than 2 MB |
| `500` | Unexpected processing error (`{"error": "processing_failed"}`) |

### Sample `/detect` response (abridged)

```json
{
  "status": "drowsy",
  "ear": 0.0512,
  "perclos": 0.62,
  "look_down": 0.79,
  "eye_alert": true,
  "yawn_alert": false,
  "nod_alert": false,
  "distract_alert": false,
  "hand_alert": false,
  "alert": true,
  "calibrated": true,
  "calibrating": false,
  "calibration_failed": false,
  "calibration_progress": 100,
  "ear_baseline": 0.301,
  "ear_threshold": 0.19,
  "message": "Eyes closing!"
}
```

`status` is one of `awake`, `drowsy`, `distracted` or `no_face`. The full response also contains per-signal confidences, frame counters, and running totals for each alert type.

---

## 🛠️ Local setup

**Prerequisites:** Python 3.9+, a webcam.

```bash
git clone https://github.com/KUNDANIOS/drowsiness.git
cd drowsiness

python -m venv venv
# Windows:  .\venv\Scripts\Activate.ps1
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open **<http://localhost:5000/detect.html>** (use `localhost` — browsers only allow camera access on `localhost` or HTTPS), allow the camera, and look straight ahead for ~10 seconds while it calibrates.

> `scipy` is no longer required by `detector.py` and can be removed from `requirements.txt`.

### Docker

```bash
docker build -t drowseguard .
docker run -p 5000:5000 drowseguard
```

---

## ✅ Tests

```bash
python -m unittest test_detector -v
```

The tests use synthetic face/hand landmarks, so they need no camera, model files or network. They cover calibration (including off-axis cameras and failure), PERCLOS, eye closure vs. looking down, turned-head behaviour, yawns, nods, hands, face loss, and the JSON contract.

---

## 🎚️ Tuning with `diagnose_eyes.py`

Thresholds depend on the person, camera angle, lighting and distance. `diagnose_eyes.py` measures what the detector sees in six poses and prints percentile tables:

```bash
# close the browser detector tab first — only one app can use the webcam
python diagnose_eyes.py
```

For each pose (open/straight, looking at a low phone, looking even lower, half-closed, fully closed, normal blinking) press Enter, get into the pose during the 2 s settle time, and hold it for 6 s. Use the tables to adjust `LOOKDOWN_CLOSED_MIN`, `EAR_THRESHOLD`, etc. at the top of `detector.py`.

### Key thresholds (`detector.py`)

| Constant | Default | Meaning |
| --- | --- | --- |
| `EAR_THRESHOLD` | 0.19 | Lowest value the personal EAR threshold may take |
| `EAR_DROP_FROM_BASE` | 0.12 | Personal threshold = baseline − this (floored at the line above) |
| `EAR_PERCLOS_THRESH` | 0.35 | Fraction of closed frames in the 4 s window that triggers an alert |
| `LOOKDOWN_CLOSED_MIN` | 0.62 | `eyeLookDown` score above which low EAR counts as *closed* |
| `JAW_OPEN_THRESHOLD` | 0.42 | Yawn threshold (blendshape) |
| `NOD_THRESHOLD` | 0.07 | Head-nod drop from baseline |
| `YAW_DISTRACT_THRESH` | 0.35 | Looking-away threshold (unitless ratio, **not degrees**) |

---

## 🚢 Deployment

Deployed on **Railway** (Docker / Nixpacks). The server reads the `PORT` environment variable.

For production use a WSGI server with **one worker** — sessions live in that process's memory, so multiple worker processes would each see only some of a user's frames:

```bash
gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT app:app
```

(`gunicorn` does not run on Windows; use `python app.py` locally.)

Open CORS is **off** by default because the pages are served by the same app. If you host the UI on another origin, set `CORS_ORIGINS` (comma-separated).

---

## ⚠️ Limitations

- **Not certified.** Indian rules for driver drowsiness and attention warning systems in larger vehicles are tied to AIS-184 and require type-approval testing. DrowseGuard has had no such validation and has no published accuracy figures.
- **Thresholds are tuned on one person and one webcam.** Expect different results for other faces, glasses/sunglasses, headwear, lighting and camera positions. Night driving needs an infrared camera.
- **Cloud round-trip.** Every frame goes to the server, which adds latency and a privacy cost, and needs connectivity. A vehicle system would need on-device inference.
- **"Hands off wheel" is really "hands raised."** A camera can't see hands resting on the wheel or in the lap; "no hands visible" counts as normal. Real hands-off detection needs steering torque or a grip sensor.
- **Yaw is a unitless ratio, not degrees**, and head pitch is approximated from nose position.
- **Frame-count timing** assumes ~10 fps; variable network latency shifts the effective durations.
- **Sessions are in memory** — a server restart clears all calibration.
- No vehicle integration (CAN bus, speed gating) and no persistent logging.

### Ideas for future work

Time-based (not frame-count) windows · true yaw/pitch from the facial transformation matrix · on-device inference (Jetson / Android) · IR camera support · baseline-scaled eye threshold · speed gating and bump filtering · yawn-frequency tracking · validation on real in-cab data.

---

## 🧰 Tech stack

- **Backend:** Python, Flask (optional Flask-CORS)
- **CV / AI:** MediaPipe Face Landmarker & Hand Landmarker, OpenCV, NumPy
- **Frontend:** Vanilla HTML/JS (`getUserMedia`)
- **Deployment:** Railway (Docker / Nixpacks)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.