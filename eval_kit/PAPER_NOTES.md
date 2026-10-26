# Paper notes: what to fix and fill in

## 0. Rules for the results

* Every number in Table 4 / Figures E-F must come from `analyze.py` run on real sessions. Do not estimate, round up
  or copy numbers from the screenshots (Fig. 5 shows that an alert path fires, not how often it works).
* Report the number of participants, events and minutes next to every figure. With few participants say so.
* The thresholds were chosen from the authors' own recordings, so the authors are the **development** set
  (`--role dev`). Only other people can be the held-out **test** set. Freeze the code before test sessions;
  if you change a threshold afterwards, say so and re-collect or relabel that data as development data.

## 1. The paper no longer matches the code (fix BEFORE recording)

The paper describes detector v4.1. The code you are running now is v4.2 plus hand-rule changes. Update:

| Where in the paper | Says | Code now |
|---|---|---|
| Sec. 4.12, Eq. 11 text | alert needs every visible hand raised; "mean of wrist and finger-base keypoints" | needs **two** hands visible and both raised (`HAND_MIN_HANDS = 2`); a hand counts when its **highest landmark** (a fingertip) is above the line `y_ref` (`HAND_REF_FACTOR = 0.25`); a hidden face with two hands in view also counts as raised |
| Sec. 4.12 | "the code treats no visible hands as ordinary" | still true; one visible hand is now also ordinary |
| Sec. 5.6 (first discrepancy) | looking-down rule is in the README but not in the code (v4.1) | v4.2 contains it: low EAR counts as eye closure only if smoothed `eyeLookDown` >= 0.62 (`LOOKDOWN_CLOSED_MIN`); otherwise it is "looking down" and raises a distraction alert after 20 frames (`LOOKDOWN_FRAMES`), reset 8 |
| Table 2 | no looking-down or hand-count rows | add: look-down cut-off 0.62, look-down smoothing 0.35, look-down persistence / reset 20 / 8 frames, minimum hands 2, hand reference factor 0.25 |
| Sec. 4.14 (fusion) | distracted if yaw or raised-hand alert | add: or looking-down alert |
| Sec. 5.6 title and last paragraph, "inspected detector.py (labelled v4.1)" | v4.1 | v4.2 + commit hash |
| Fig. 3 / Fig. 5 | screenshots from the earlier version | re-capture with the frozen version or add "captured with v4.1" to the caption |
| Table 2, Fig. 4 | yaw 8 frames, no-face 12 frames | **only if you changed them** (for example to 20 frames to tolerate mirror checks) |
| Abstract, Sec. 7.1, Sec. 10, Highlights ("no accuracy claims made yet") | no measurements exist | rewrite after the results are in; keep "performed behaviours, one camera setup" |

Also run `test_detector.py` now: the two-hand rule changes `calc_hand_raised`, so any test that expects a single
raised hand to count as "raised" has to be updated.

## 2. Method text to replace (Sec. 6.3-6.5 and 6.7), matching what the kit does

> **Protocol.** Each participant completed one session of about 10 minutes in front of a webcam after the
> automatic calibration phase. A schedule generated from a participant-specific seed cued, with a 3 s countdown
> and audible markers, 4 repetitions of each of seven behaviours (sustained eye closure for 6 s, yawning, head
> lowering, sideward head turn, both hands raised, leaning out of view, looking down with eyes open), interleaved
> with normal behaviour (normal driving posture, talking, deliberate blinking, brief mirror glances). Ground truth
> is therefore defined by the cue schedule rather than by post-hoc annotation. The behaviours are performed
> simulations and not episodes of real drowsiness.
>
> **Matching.** An event was counted as detected if an alert of the corresponding type started between the
> behaviour onset and 2 s after its offset. Alert delay is measured from the cued onset and therefore includes
> the participant's reaction time and the detector's persistence window. Precision counts every alert of a type
> outside that type's event windows. False warnings per minute count alerts of a type that started in alert-free
> footage (neither a countdown nor an event window), divided by the duration of that footage.
>
> **Statistics.** Confidence intervals are 95% percentile intervals from a bootstrap that resamples participants
> (2000 resamples).
>
> **Latency.** A Python client on the same machine sent JPEG frames (640 x 480, quality 70, nominal 10 frames/s)
> to the unchanged `/detect` endpoint and recorded the time from frame capture to response. This includes JPEG
> encoding, HTTP transfer, model inference and rule evaluation, but not browser rendering or speech output.
> Network conditions: [localhost / LAN / hosted - fill in].

Sec. 6.4 currently mentions two independent raters and adjudication. Remove that, or add a second step where two
authors check a sample of recorded events (use `--save-video` with consent) and report the agreement.

## 3. How the outputs fill the placeholders

* **Table 4**: copy `results/table4.tex`. Add a seventh row for "Looking down" (the code now has that cue). New caption
  idea: "Event-level results on N held-out participants (K labelled events, M min of alert-free footage). Values are
  point estimates with 95% participant-level bootstrap CIs in brackets. Alert delay includes persistence and
  reaction time."
* **Figure E**: `results/figE_precision_recall.pdf`. Caption: state N participants, event-level scoring, CI method.
* **Figure F**: `results/figF_latency.pdf`. Caption: environment (CPU, OS, versions are in each `*_meta.json`), achieved
  frame rate, network condition. Panel (b) can be compared with the nominal persistence times in Fig. 4.
* **Fig. 6 caption** ("Reserved layouts ... no values measured") and **Table 4 caption** ("unfilled until validation"): replace.
* **Sec. 6.5 manifest**: use `dataset_manifest.csv`; the sentence "these quantities are not yet available" can then go.
* **Discussion material already computed**: `cross_cue_matrix.csv` (e.g. does a sideward turn also trigger face loss?),
  `fp_by_negative.csv` (do talking or mirror glances cause false alerts?). Report these honestly even if unflattering.

## 4. Placeholders only you can fill

| Placeholder | Note |
|---|---|
| Corresponding author name / e-mail | author decision |
| CRediT roles for each author, supervision statement | agree among authors and supervisor |
| Competing interests | all authors |
| **Generative AI declaration** | must be truthful. If an AI assistant helped write code, tests or text, name the tool and say how, and that the authors reviewed and take responsibility. Check the target journal's wording |
| Funding | statement or "no specific grant" |
| Target journal in the footer | after choosing |
| Commit hash, Zenodo DOI (Data and code availability) | tag and archive the exact version used for the experiments |
| Ethics statement | currently says no formal human-subject experiment. That stops being true once volunteers are recorded. Get your institution's ethics/IRB determination or exemption and write it in |

## 5. Volunteer checklist (put in the ethics statement)

* Written informed consent (purpose, 10 min, face video frames are sent to a server, what is stored, right to withdraw).
* Stored: numeric logs only (timestamps, alert flags, a few metrics). No video unless the person agrees separately.
* Pseudonymous IDs (P01, ...), no names in files; keep the consent forms apart from the data.
* Retention and deletion rule stated, plus who can access the data.
* Participants are not driving: label the scenes as controlled facial-action simulations (as the paper already does).
