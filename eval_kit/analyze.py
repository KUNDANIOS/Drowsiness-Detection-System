"""Turn recorded sessions into the paper's Table 4, Figure E and Figure F.

    python analyze.py --data data --out results

Reads every  data/<participant>_{frames.csv,labels.csv,meta.json}  and writes to results/:
    table4.csv / table4.md / table4.tex     Table 4 (events, recall, false warnings/min, median delay) + CIs
    figE_precision_recall.png/.pdf          Figure E  (per-cue precision / recall, participant-bootstrap CIs)
    figF_latency.png/.pdf                   Figure F  (capture->response latency + onset->alert delay)
    cross_cue_matrix.csv                    which alerts fired during which labelled behaviour
    fp_by_negative.csv                      false warnings during normal behaviour (talking, mirror glance...)
    per_participant.csv, dataset_manifest.csv, latency_summary.json, report.txt

Definitions (fixed BEFORE looking at results; quote them in the paper)
  * A labelled event = one scripted behaviour with known onset t_on and offset t_off.
  * An alert ONSET of cue X = the frame where X's latched alert flag turns on (after calibration).
  * Event detected = an onset of the matching cue in [t_on, t_off + TOL]  (TOL = 2 s).
    Alert delay = first such onset - t_on. It INCLUDES the performer's reaction time and the
    detector's persistence window; it is not pure software latency.
  * False warning of cue X = an onset of X outside every X-event window [t_on, t_off + TOL].
  * False warnings / min (Table 4) counts only onsets in NON-EVENT time (neither a countdown nor an event
    window, TOL included) - i.e. alert-free footage - divided by that time in minutes.
  * Precision (Fig E) = detected events / (detected events + ALL false warnings of that cue), so alerts that
    fired during a different behaviour count against precision (conservative).
  * Recall = detected events / labelled events.
  * CIs: 95 % percentile bootstrap, resampling PARTICIPANTS (frames and events within a person are correlated).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import protocol as P

N_BOOT = 2000
MIN_PARTICIPANTS_FOR_CI = 3
RECOMMENDED_PARTICIPANTS = 8


# ───────────────────────── loading ─────────────────────────
@dataclass
class Session:
    pid: str
    meta: dict
    t: np.ndarray
    t_enc: np.ndarray
    t_resp: np.ndarray
    ok: np.ndarray
    calibrated: np.ndarray
    flags: Dict[str, np.ndarray]
    labels: List[dict]
    onsets: Dict[str, List[float]] = field(default_factory=dict)


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_session(prefix: str) -> Session:
    with open(prefix + "_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = list(csv.DictReader(open(prefix + "_frames.csv", encoding="utf-8")))
    labels = [dict(item_id=int(r["item_id"]), kind=r["kind"], type=r["type"],
                   t_prep=float(r["t_prep"]), t_on=float(r["t_on"]),
                   t_off=float(r["t_off"]), t_end=float(r["t_end"]))
              for r in csv.DictReader(open(prefix + "_labels.csv", encoding="utf-8"))]
    col = lambda k: np.array([_f(r[k]) for r in rows])  # noqa: E731
    flags = {c: np.array([int(_f(r["a_" + c], 0)) for r in rows]) for c in P.CUES}
    s = Session(pid=meta["participant"], meta=meta, t=col("t_capture"), t_enc=col("t_encoded"),
                t_resp=col("t_response"), ok=col("http_ok").astype(int),
                calibrated=col("calibrated").astype(int), flags=flags, labels=labels)
    s.onsets = {c: alert_onsets(s, c) for c in P.CUES}
    return s


def alert_onsets(s: Session, cue: str) -> List[float]:
    """Rising edges of the latched alert flag; failed HTTP frames keep the previous state."""
    out, prev = [], 0
    for i in range(len(s.t)):
        if not s.ok[i]:
            continue
        cur = int(s.flags[cue][i]) if s.calibrated[i] else 0
        if cur and not prev:
            out.append(float(s.t[i]))
        prev = cur
    return out


# ───────────────────────── scoring ─────────────────────────
def _merge(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _inside(x: float, ivs) -> bool:
    return any(a <= x <= b for a, b in ivs)


def score_session(s: Session, cues: List[str]) -> dict:
    tol = s.meta["scoring"]["tol_s"]
    events = [l for l in s.labels if l["kind"] == "event"]
    negatives = [l for l in s.labels if l["kind"] == "negative"]
    sched_start = min(l["t_prep"] for l in s.labels)
    sched_end = max(l["t_end"] for l in s.labels)
    excl = _merge([(e["t_prep"], e["t_off"] + tol) for e in events])
    excl_clipped = [(max(a, sched_start), min(b, sched_end)) for a, b in excl]
    nonevent_s = (sched_end - sched_start) - sum(max(0.0, b - a) for a, b in excl_clipped)

    out = {"nonevent_s": nonevent_s, "cues": {}, "cross": {}, "neg": {}}
    for cue in cues:
        ev_c = [e for e in events if e["type"] == cue]
        wins_c = [(e["t_on"], e["t_off"] + tol) for e in ev_c]
        delays, tp = [], 0
        for e in ev_c:
            cand = [o for o in s.onsets[cue] if e["t_on"] <= o <= e["t_off"] + tol]
            if cand:
                tp += 1
                delays.append((cand[0] - e["t_on"]) * 1000.0)
        fp_all = [o for o in s.onsets[cue] if sched_start <= o <= sched_end and not _inside(o, wins_c)]
        fp_nonevent = [o for o in fp_all if not _inside(o, excl)]
        out["cues"][cue] = dict(n_events=len(ev_c), tp=tp, fn=len(ev_c) - tp, fp_all=len(fp_all),
                                fp_nonevent=len(fp_nonevent), delays=delays)

    # which alert fired during which labelled behaviour
    for e in events:
        row = out["cross"].setdefault(e["type"], {"n": 0, **{c: 0 for c in cues}})
        row["n"] += 1
        for c in cues:
            if any(e["t_on"] <= o <= e["t_off"] + tol for o in s.onsets[c]):
                row[c] += 1

    # false warnings during each kind of normal behaviour (hold windows only)
    for n in negatives:
        row = out["neg"].setdefault(n["type"], {"seconds": 0.0, **{c: 0 for c in cues}})
        row["seconds"] += n["t_off"] - n["t_on"]
        for c in cues:
            row[c] += sum(1 for o in s.onsets[c] if n["t_on"] <= o <= n["t_off"])
    return out


def pooled(per: Dict[str, dict], pids: List[str], cue: str) -> dict:
    n = tp = fp_all = fp_ne = 0
    delays: List[float] = []
    ne_s = 0.0
    for p in pids:
        c = per[p]["cues"][cue]
        n += c["n_events"]
        tp += c["tp"]
        fp_all += c["fp_all"]
        fp_ne += c["fp_nonevent"]
        delays += c["delays"]
        ne_s += per[p]["nonevent_s"]
    return dict(
        n_events=n, tp=tp,
        recall=100.0 * tp / n if n else float("nan"),
        precision=100.0 * tp / (tp + fp_all) if (tp + fp_all) else float("nan"),
        fp_per_min=fp_ne / (ne_s / 60.0) if ne_s > 0 else float("nan"),
        fp_nonevent=fp_ne, fp_all=fp_all, nonevent_min=ne_s / 60.0,
        median_delay_ms=float(np.median(delays)) if delays else float("nan"),
        iqr_delay_ms=(float(np.percentile(delays, 25)), float(np.percentile(delays, 75))) if delays else (float("nan"),) * 2,
        n_detected=len(delays),
    )


def bootstrap(per: Dict[str, dict], cues: List[str], seed: int = 0) -> Dict[str, Dict[str, Tuple[float, float]]]:
    pids = list(per)
    if len(pids) < MIN_PARTICIPANTS_FOR_CI:
        return {}
    rng = np.random.default_rng(seed)
    draws: Dict[str, Dict[str, List[float]]] = {c: defaultdict(list) for c in cues}
    for _ in range(N_BOOT):
        sample = [pids[i] for i in rng.integers(0, len(pids), len(pids))]
        for c in cues:
            m = pooled(per, sample, c)
            for k in ("recall", "precision", "fp_per_min", "median_delay_ms"):
                draws[c][k].append(m[k])
    ci = {}
    for c in cues:
        ci[c] = {}
        for k, v in draws[c].items():
            a = np.array(v, float)
            ci[c][k] = (float("nan"), float("nan")) if np.all(np.isnan(a)) else \
                (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))
    return ci


# ───────────────────────── latency ─────────────────────────
def latency_summary(sessions: List[Session]) -> dict:
    tot, enc, rnd, fps = [], [], [], []
    for s in sessions:
        m = s.ok == 1
        tot += list((s.t_resp - s.t)[m])
        enc += list((s.t_enc - s.t)[m])
        rnd += list((s.t_resp - s.t_enc)[m])
        if m.sum() > 1:
            fps.append(float(m.sum() / (s.t[m][-1] - s.t[m][0])))
    pct = lambda a: {f"p{q}": float(np.percentile(np.array(a) * 1000, q)) for q in (50, 95, 99)} | \
        {"mean": float(np.mean(a) * 1000)}  # noqa: E731
    return {"frames": len(tot), "capture_to_response_ms": pct(tot), "jpeg_encode_ms": pct(enc),
            "post_to_response_ms": pct(rnd), "achieved_fps_median": float(np.median(fps)) if fps else None,
            "achieved_fps_min": float(min(fps)) if fps else None,
            "note": "post_to_response = network + server inference + decision (cannot be split without server timestamps)"}


# ───────────────────────── output ─────────────────────────
def _fmt(v, nd=1):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _ci(ci, cue, k, nd=1):
    if not ci or cue not in ci or k not in ci[cue]:
        return ""
    lo, hi = ci[cue][k]
    return "" if math.isnan(lo) else f" [{lo:.{nd}f}-{hi:.{nd}f}]"


def write_table4(out, cues, point, ci):
    rows = []
    for c in cues:
        m = point[c]
        rows.append([P.CUE_TITLES[c], str(m["n_events"]),
                     _fmt(m["recall"]) + _ci(ci, c, "recall"),
                     _fmt(m["fp_per_min"], 2) + _ci(ci, c, "fp_per_min", 2),
                     (_fmt(m["median_delay_ms"], 0) + _ci(ci, c, "median_delay_ms", 0)) if m["n_detected"] else "n/a (no detections)"])
    head = ["Scenario", "Number of labelled events", "Recall (%)", "False warnings / min", "Median alert delay (ms)"]
    with open(os.path.join(out, "table4.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    with open(os.path.join(out, "table4.md"), "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(r) + " |\n")
    esc = lambda s: s.replace("-", "--") if "[" in s else s  # CI ranges: 88.0-100.0 -> 88.0--100.0  # noqa: E731
    with open(os.path.join(out, "table4.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{lcccc}\n\\hline\n" + " & ".join(head).replace("%", "\\%") + " \\\\\n\\hline\n")
        for r in rows:
            fh.write(" & ".join(esc(x) for x in r) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
    return rows


def figure_e(out, cues, point, ci, n_part, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(cues))
    wid = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for j, (k, lab, col) in enumerate((("precision", "Precision", "#4C6A8D"), ("recall", "Recall", "#C9803A"))):
        vals = np.array([point[c][k] for c in cues], float)
        yerr = None
        if ci:
            lo = np.array([ci[c][k][0] for c in cues], float)
            hi = np.array([ci[c][k][1] for c in cues], float)
            yerr = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
            yerr = np.nan_to_num(yerr)
        ax.bar(x + (j - 0.5) * wid, np.nan_to_num(vals), wid, yerr=yerr, capsize=3, label=lab, color=col)
        his = (np.array([ci[c][k][1] for c in cues], float) if ci else vals)
        for xi, v, hi in zip(x + (j - 0.5) * wid, vals, his):
            if not np.isnan(v):
                top = hi if not np.isnan(hi) else v
                ax.text(xi, min(top + 2.5, 106), f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([P.CUE_SHORT[c] for c in cues], fontsize=9)
    ax.set_ylim(0, 114)
    ax.set_ylabel("%")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    note = f"{n_part} participants; event-level; error bars = 95% participant-level bootstrap CI" if ci else \
        f"{n_part} participants; event-level; CIs not computed (<{MIN_PARTICIPANTS_FOR_CI} participants)"
    fig.text(0.01, 0.01, note, fontsize=7, color="#555")
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC TEST DATA - NOT RESULTS", fontsize=26, color="red", alpha=0.25,
                 ha="center", va="center", rotation=20)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figE_precision_recall.{ext}"), dpi=200)
    plt.close(fig)


def figure_f(out, cues, sessions, point, lat, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    tot = np.sort(np.concatenate([((s.t_resp - s.t)[s.ok == 1]) * 1000 for s in sessions]))
    a.plot(tot, np.arange(1, len(tot) + 1) / len(tot), color="#4C6A8D")
    for q, ls in ((50, "--"), (95, ":")):
        v = np.percentile(tot, q)
        a.axvline(v, color="#C9803A", ls=ls)
        a.text(v, 0.05 + (0.1 if q == 95 else 0), f" p{q}={v:.0f} ms", fontsize=8, color="#C9803A")
    a.set_xlabel("capture \u2192 response per frame (ms)")
    a.set_ylabel("cumulative fraction of frames")
    a.set_title("(a) pipeline latency per frame", fontsize=9)
    a.spines[["top", "right"]].set_visible(False)

    labs = [P.CUE_SHORT[c] for c in cues]
    data = [point[c]["delays_all"] for c in cues]
    pos = np.arange(1, len(cues) + 1)
    nonempty = [i for i, d in enumerate(data) if len(d)]
    if nonempty:
        b.boxplot([data[i] for i in nonempty], positions=[pos[i] for i in nonempty], widths=0.5, showfliers=True,
                  boxprops=dict(color="#4C6A8D"), medianprops=dict(color="#C9803A"))
    b.set_xticks(pos)
    b.set_xticklabels([f"{l}\n(n={len(d)})" for l, d in zip(labs, data)], fontsize=8)
    b.set_ylabel("event onset \u2192 alert (ms)")
    b.set_title("(b) alert delay (includes persistence window + performer reaction)", fontsize=9)
    b.spines[["top", "right"]].set_visible(False)
    env = sessions[0].meta.get("environment", {})
    fig.text(0.01, 0.005, f"{env.get('platform', '')} | server {env.get('server', '')} | "
             f"{lat['frames']} frames | median {lat['achieved_fps_median']:.1f} fps"
             if lat.get("achieved_fps_median") else "", fontsize=6.5, color="#555")
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC TEST DATA - NOT RESULTS", fontsize=26, color="red", alpha=0.25,
                 ha="center", va="center", rotation=20)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figF_latency.{ext}"), dpi=200)
    plt.close(fig)


# ───────────────────────── main ─────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--include-dev", action="store_true",
                    help="also score role=dev sessions (people used to pick thresholds) - NOT for the paper's held-out result")
    ap.add_argument("--allow-synthetic", action="store_true", help="SOFTWARE TESTING ONLY")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    sessions, skipped = [], []
    for mp in sorted(glob.glob(os.path.join(args.data, "*_meta.json"))):
        prefix = mp[:-len("_meta.json")]
        meta = json.load(open(mp, encoding="utf-8"))
        why = None
        if meta.get("synthetic") and not args.allow_synthetic:
            why = "synthetic (fake camera)"
        elif not meta.get("complete"):
            why = "incomplete session"
        elif meta.get("role") != "test" and not args.include_dev:
            why = f"role={meta.get('role')} (not held-out)"
        if why:
            skipped.append((meta.get("participant", os.path.basename(prefix)), why))
            continue
        sessions.append(load_session(prefix))
    if not sessions:
        print("No usable sessions.", skipped)
        return 1

    synthetic = any(s.meta.get("synthetic") for s in sessions)
    cues = [c for c in P.CUES if c != "looking_down" or all(s.meta.get("has_looking_down") for s in sessions)]
    per = {s.pid: score_session(s, cues) for s in sessions}
    pids = list(per)
    point = {c: pooled(per, pids, c) for c in cues}
    for c in cues:
        point[c]["delays_all"] = [d for p in pids for d in per[p]["cues"][c]["delays"]]
    ci = bootstrap(per, cues)
    lat = latency_summary(sessions)

    write_table4(args.out, cues, point, ci)
    figure_e(args.out, cues, point, ci, len(pids), synthetic)
    figure_f(args.out, cues, sessions, point, lat, synthetic)

    # cross-cue matrix
    with open(os.path.join(args.out, "cross_cue_matrix.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["labelled_behaviour", "n_events"] + [f"alerts_of_{c}" for c in cues])
        for et in cues:
            n = sum(per[p]["cross"].get(et, {}).get("n", 0) for p in pids)
            w.writerow([et, n] + [sum(per[p]["cross"].get(et, {}).get(c, 0) for p in pids) for c in cues])
    # false warnings during normal behaviour
    with open(os.path.join(args.out, "fp_by_negative.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["normal_behaviour", "total_seconds"] + [f"false_{c}" for c in cues])
        for nt in P.NEGATIVES:
            secs = sum(per[p]["neg"].get(nt, {}).get("seconds", 0.0) for p in pids)
            w.writerow([nt, f"{secs:.1f}"] + [sum(per[p]["neg"].get(nt, {}).get(c, 0) for p in pids) for c in cues])
    # per participant + manifest
    with open(os.path.join(args.out, "per_participant.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant"] + [f"{c}_{k}" for c in cues for k in ("events", "detected", "false_all")])
        for p in pids:
            w.writerow([p] + [per[p]["cues"][c][k] for c in cues for k in ("n_events", "tp", "fp_all")])
    with open(os.path.join(args.out, "dataset_manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant", "role", "glasses", "lighting", "camera_distance_cm", "camera_height",
                    "usable_minutes", "frames_sent", "frames_failed", "achieved_fps", "labelled_events", "normal_blocks"])
        for s in sessions:
            ev = sum(1 for l in s.labels if l["kind"] == "event")
            ng = sum(1 for l in s.labels if l["kind"] == "negative")
            dur = (max(l["t_end"] for l in s.labels) - min(l["t_prep"] for l in s.labels)) / 60
            w.writerow([s.pid, s.meta.get("role"), s.meta.get("glasses"), s.meta.get("lighting"),
                        s.meta.get("camera_distance_cm"), s.meta.get("camera_height"), f"{dur:.1f}",
                        s.meta.get("frames_sent"), s.meta.get("frames_failed"),
                        s.meta.get("achieved_fps_overall"), ev, ng])
    json.dump(lat, open(os.path.join(args.out, "latency_summary.json"), "w"), indent=2)

    # human-readable report
    lines = [f"Participants analysed: {len(pids)}  (skipped: {skipped or 'none'})",
             f"Recommended minimum for meaningful CIs: {RECOMMENDED_PARTICIPANTS}"]
    if len(pids) < RECOMMENDED_PARTICIPANTS:
        lines.append("WARNING: few participants - CIs will be wide or missing; say so in the paper.")
    lines += ["", "cue                       events  detected  recall%  precision%  FW/min   median delay ms"]
    for c in cues:
        m = point[c]
        lines.append(f"{c:<25} {m['n_events']:>6} {m['n_detected']:>9} {_fmt(m['recall']):>8} "
                     f"{_fmt(m['precision']):>10} {_fmt(m['fp_per_min'], 2):>7} {_fmt(m['median_delay_ms'], 0):>10}")
    lines += ["", f"Non-event (alert-free) footage: {sum(per[p]['nonevent_s'] for p in pids) / 60:.1f} min",
              "Latency: " + json.dumps(lat["capture_to_response_ms"]),
              f"Achieved fps (median per session): {lat['achieved_fps_median']}"]
    open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote tables/figures to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
