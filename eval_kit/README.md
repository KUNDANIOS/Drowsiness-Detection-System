# DrowseGuard evaluation kit

Produces the numbers for **Table 4**, **Figure E** and **Figure F** of the paper from real recordings.
Nothing here invents results: until you record sessions, those cells stay empty.

## What it measures

Participants follow a scripted schedule (about 10 min): 7 behaviours x 4 repetitions
(eye closure, yawn, head lowering, sideward turn, both hands raised, face out of view, looking down with eyes
open) mixed with normal behaviour that looks a bit like a cue (talking, blinking, quick mirror glances, driving
normally). The schedule is the ground truth, so no frame-by-frame labelling is needed. The recorder sends frames
to your running server exactly like the browser does and logs the detector's alert flags with timestamps.

These are **performed** behaviours, not real drowsiness. Say so in the paper ("controlled facial-action simulations").

## Setup (once)

```powershell
# in your venv, inside the eval_kit folder
python -m pip install -r requirements-eval.txt
```

## Before the first session

1. Freeze the code. `git commit` the final `detector.py`, `app.py`, `detect.html`, then `git tag eval-v1` and note
   `git rev-parse HEAD` for the paper. Do not change thresholds after test sessions begin.
2. `python app.py` in one terminal (close the browser detector tab: one program per camera).
3. In a second terminal: `python run_session.py --smoke-test`. It checks how /detect is called.
   I could not see `app.py`, so this checks my assumptions (raw JPEG body, `X-Session-Id` header).
   If it fails, the message says what to look at; try `--content-type application/octet-stream`.

## Record sessions

```powershell
# the authors / anyone whose data was used to choose thresholds -> role dev (never scored as held-out)
python run_session.py --participant A01 --role dev

# volunteers who never touched the thresholds -> role test
python run_session.py --participant P01 --role test --glasses no --lighting daylight --camera-distance-cm 60 --camera-height eye-level
python run_session.py --participant P02 --role test --glasses glasses --lighting indoor ...
```

* Aim for **10 or more test participants**, ideally including glasses, different lighting and different camera distances.
  The analysis warns below 8 and gives no confidence intervals below 3.
* Participant sits where they would in the demo, looks at the camera for calibration (about 10 s), then follows the
  cue window (big text + countdown + beeps: first beep = start the behaviour, second = relax).
* Put the cue window near the camera so reading it does not look like "looking down".
* Do not give hints about the thresholds. Use `--reps 2` only for a quick rehearsal.
* No video is stored unless you pass `--save-video`.

## Analyse

```powershell
python analyze.py --data data --out results
```

| Output | Goes to |
|---|---|
| `results/table4.tex` / `.md` / `.csv` | Table 4 (values with 95% participant-bootstrap CIs) |
| `results/figE_precision_recall.pdf` | Figure E |
| `results/figF_latency.pdf` | Figure F |
| `results/report.txt`, `latency_summary.json` | numbers to quote in the text (fps, p50/p95 latency) |
| `results/dataset_manifest.csv` | the dataset manifest the paper promises in Sec. 6.5 |
| `results/cross_cue_matrix.csv` | which alert fired during which behaviour (confusion matrix) |
| `results/fp_by_negative.csv` | false warnings during talking / mirror glances / blinking |

`analyze.py` skips `role=dev`, incomplete and fake-camera sessions and lists them in `report.txt`.

## Definitions (fixed before looking at results)

* Event detected = an alert of the matching type starts within [behaviour start, behaviour end + 2 s].
* Alert delay = first such alert start minus behaviour start. It includes the person's reaction to the cue and the
  detector's persistence window; it is not pure software latency.
* False warning = an alert start outside every event window of that type. **False warnings / min** counts only
  alert-free footage (no countdown, no event window).
* Precision counts every false alert of that type, including ones that fired during a different behaviour.
* Latency (Fig. F a) = frame captured -> server response received, measured by the Python client on the same
  machine. It includes JPEG encoding, HTTP and inference, not browser rendering or speech.

## Tests

`python -m pytest tests -q` (13 tests: schedule, scoring against hand-computed values, bootstrap, file outputs,
and the recorder against a mock server with a fake camera). Synthetic data is stamped
"SYNTHETIC TEST DATA - NOT RESULTS" and refused by `analyze.py` unless `--allow-synthetic` is given.
