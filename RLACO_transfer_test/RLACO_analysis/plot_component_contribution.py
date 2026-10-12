"""
Visualise the contribution of each component (RL / ACO / TabuCol) to the
final conflict reduction, aggregated over all runs of each instance.

Design (per user request):
  Panel A: stacked bar per instance = RL gain + ACO gain + final residual
  Panel B: RL vs ACO share (%) horizontal bar per instance
  Panel C: RLTabuCol vs pure TabuCol final score (from RLTabucol-Experienment
           old data — comparison bar per instance)

Data sources:
  New RLACO experiments  : RLACO_project/results/*_rlaco_soft
                            (has episode_traces with rl_start/rl_end,
                             aco_seed/aco_end fields)
  Old RLTabuCol experiments : src/RLTabucol-Experienment/output_*.zip
                            (only per-episode "score" print, no trace fields)
"""
import glob
import io
import json
import os
import re
import statistics
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RLACO_DIR = REPO / "RLACO_project" / "results"
OLD_DIR = REPO / "src" / "RLTabucol-Experienment"
FIG_OUT = Path(__file__).resolve().parent / "figures"
FIG_OUT.mkdir(exist_ok=True)


def parse_result_json(text):
    if "[DONE]" not in text:
        return None
    idx = text.rfind("[DONE]")
    tail = text[idx:].split("\n", 1)
    if len(tail) < 2:
        return None
    for line in tail[1].splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue
    return None


# =====================================================================
# NEW: RLACO_soft component breakdown (has episode_traces)
# =====================================================================

def rlaco_contributions(inst_method_dir):
    """Return dict {rl_median, aco_median, final_median, rl_share_pct, n}."""
    rl_gains, aco_gains, finals = [], [], []
    for fp in sorted(inst_method_dir.glob("*.txt")):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        obj = parse_result_json(text)
        if obj is None:
            continue
        traces = obj.get("episode_traces") or []
        # sum per-episode contributions across the run
        run_rl = 0.0
        run_aco = 0.0
        for t in traces:
            rl_s, rl_e = t.get("rl_start_score"), t.get("rl_end_score")
            aco_s, aco_e = t.get("aco_seed_score"), t.get("aco_end_score")
            try:
                if rl_s is not None and rl_e is not None:
                    run_rl += float(rl_s) - float(rl_e)
                if aco_s is not None and aco_e is not None:
                    run_aco += float(aco_s) - float(aco_e)
            except (TypeError, ValueError):
                continue
        rl_gains.append(run_rl)
        aco_gains.append(run_aco)
        finals.append(int(obj.get("score", 0)))
    if not rl_gains:
        return None
    tot_rl = sum(rl_gains)
    tot_aco = sum(aco_gains)
    denom = tot_rl + tot_aco
    share = 100 * tot_rl / denom if denom > 0 else 0
    return {
        "n": len(rl_gains),
        "rl_median": statistics.median(rl_gains),
        "aco_median": statistics.median(aco_gains),
        "final_median": statistics.median(finals),
        "final_min": min(finals),
        "final_max": max(finals),
        "rl_share_pct": share,
    }


# =====================================================================
# OLD: RLTabucol vs pure TabuCol final scores (from zip'd logs)
# =====================================================================

_ZIP_CACHE = {}


def _iter_txt_in_zip(zip_path):
    """Yield (name, text) for each .txt inside a zip. Cached."""
    if zip_path in _ZIP_CACHE:
        return _ZIP_CACHE[zip_path]
    out = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".txt"):
                    continue
                with zf.open(name) as fh:
                    raw = fh.read()
                try:
                    text = raw.decode("utf-8", errors="replace")
                except Exception:
                    text = raw.decode("cp1252", errors="replace")
                out.append((name, text))
    except (zipfile.BadZipFile, FileNotFoundError):
        pass
    _ZIP_CACHE[zip_path] = out
    return out


def old_final_scores(zip_paths):
    """Return list of final `score` across all runs in the given zip files."""
    scores = []
    for zp in zip_paths:
        for name, text in _iter_txt_in_zip(zp):
            obj = parse_result_json(text)
            if obj is None:
                continue
            sc = obj.get("score")
            if sc is not None:
                try:
                    scores.append(int(sc))
                except (TypeError, ValueError):
                    pass
    return scores


# =====================================================================
# Main
# =====================================================================

# The 10 instances covered by RLACO_project. keyed name = (instance, k).
RLACO_CELLS = [
    ("DSJC250.5",    28),
    ("flat300_26_0", 26),
    ("flat300_28_0", 28),
    ("DSJC500.1",    12),
    ("DSJC500.5",    48),
    ("DSJC500.9",   126),
    ("le450_15c",    15),
    ("le450_15d",    15),
    ("le450_25c",    25),
    ("le450_25d",    25),
]

# Manual mapping of instance name -> which OLD zips are RLTabuCol vs pure TabuCol.
# The zip filenames are inconsistent so we hand-map them.
OLD_ZIP_MAP = {
    "DSJC250.5":    {"rl": ["output_RL_250.5.zip"],       "tc": ["output_Tabucol_250.5.zip"]},
    "DSJC500.1":    {"rl": ["output_RL_500.1.zip"],       "tc": ["output_DSJC500.1_tabucol.zip", "output_500.1_tabucol.zip"]},
    "DSJC500.5":    {"rl": ["output_RL_500.5.zip"],       "tc": ["output_500.5_tabucol.zip"]},
    "DSJC500.9":    {"rl": ["output_RL_500.9.zip"],       "tc": ["output_500.9_tabucol.zip", "output_DSJC500.9_tabucol.zip"]},
    "flat300_26_0": {"rl": ["output_flat300_26_0_RL.zip"],"tc": ["output_flat300_26_0_tabucol.zip"]},
    "flat300_28_0": {"rl": ["output_RL_flat300_28.zip", "output_flat300_28_0_RL.zip"], "tc": ["output_flat300_28_0_tabucol.zip"]},
    "le450_15c":    {"rl": ["output_RL_le450_15c.zip",  "output_le450_15c_RL.zip"],    "tc": ["output_le450_15c_tabucol.zip"]},
    "le450_15d":    {"rl": [],  "tc": ["output_le450_15d_tabucol.zip"]},
    "le450_25c":    {"rl": ["output_RL_le450_25c.zip",  "output_le450_25c_RL.zip"],   "tc": ["output_le450_25c_tabucol.zip"]},
    "le450_25d":    {"rl": ["output_RL_le450_25d.zip",  "output_le450_25d_RL.zip"],   "tc": ["output_le450_25d_tabucol.zip"]},
}


def main():
    # ---------- Collect data ----------
    rows = []
    for inst, k in RLACO_CELLS:
        # NEW: RLACO_soft component breakdown
        rlaco_dir = RLACO_DIR / f"{inst}_k{k}_rlaco_soft"
        contrib = rlaco_contributions(rlaco_dir) if rlaco_dir.exists() else None

        # OLD: RLTabucol (with RL) vs pure Tabucol final scores
        m = OLD_ZIP_MAP.get(inst, )
        rl_zips = [OLD_DIR / z for z in m.get("rl", [])]
        tc_zips = [OLD_DIR / z for z in m.get("tc", [])]
        rl_scores = old_final_scores(rl_zips)
        tc_scores = old_final_scores(tc_zips)

        rows.append({
            "instance": inst,
            "k": k,
            "contrib": contrib,
            "old_rl_scores": rl_scores,
            "old_tc_scores": tc_scores,
        })

    # Print a quick console summary
    print(f"{'Instance':<14} {'k':<4} {'RLACO_soft_finals':<24} {'RL_share%':<10} {'oldRL':<20} {'oldTC':<20}")
    for r in rows:
        c = r["contrib"]
        if c:
            fin = f"n={c['n']} med={c['final_median']:.0f} [{c['final_min']}-{c['final_max']}]"
            share = f"{c['rl_share_pct']:.1f}"
        else:
            fin = "(no data)"
            share = "-"
        rls = f"n={len(r['old_rl_scores'])} med={statistics.median(r['old_rl_scores']):.0f}" if r["old_rl_scores"] else "-"
        tcs = f"n={len(r['old_tc_scores'])} med={statistics.median(r['old_tc_scores']):.0f}" if r["old_tc_scores"] else "-"
        print(f"{r['instance']:<14} {r['k']:<4} {fin:<24} {share:<10} {rls:<20} {tcs:<20}")

    # ---------- Build the 3-panel figure ----------
    valid = [r for r in rows if r["contrib"] is not None]
    labels = [f"{r['instance']}\n(k={r['k']})" for r in valid]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    # -- Panel A: Stacked contribution bar --
    ax = axes[0]
    rl_med = np.array([r["contrib"]["rl_median"] for r in valid])
    aco_med = np.array([r["contrib"]["aco_median"] for r in valid])
    final_med = np.array([r["contrib"]["final_median"] for r in valid])

    # Per-episode median gains + final residual. For readability we stack
    # cumulative "typical episode" contributions.
    b1 = ax.bar(x, rl_med, color="#1f77b4", label="RL contribution (median gain / ep)", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, aco_med, bottom=rl_med, color="#ff7f0e", label="ACO contribution (median gain / ep)", edgecolor="black", linewidth=0.5)
    # Print the final median as a small label on top of the bar
    for i, r in enumerate(valid):
        top = rl_med[i] + aco_med[i]
        ax.text(i, top + 0.5, f"best@5h\nmed={int(final_med[i])}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Median conflict reduction per episode")
    ax.set_title("(A) Per-episode conflict reduction: RL vs ACO contribution (RLACO_soft, median across all runs)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # -- Panel B: RL share % horizontal bar --
    ax = axes[1]
    rl_share = np.array([r["contrib"]["rl_share_pct"] for r in valid])
    aco_share = 100 - rl_share
    y = np.arange(len(labels))
    ax.barh(y, rl_share, color="#1f77b4", edgecolor="black", linewidth=0.5, label="RL share of total gain")
    ax.barh(y, aco_share, left=rl_share, color="#ff7f0e", edgecolor="black", linewidth=0.5, label="ACO share of total gain")
    for i, sh in enumerate(rl_share):
        ax.text(sh / 2, i, f"RL {sh:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(sh + (100 - sh) / 2, i, f"ACO {100 - sh:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total conflict reduction (%)")
    ax.set_title("(B) Cumulative RL-vs-ACO share across all episodes (RLACO_soft)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # -- Panel C: RLTabucol vs pure TabuCol final score comparison --
    ax = axes[2]
    # Only include instances with both RL and TC data
    inst_labels, rl_bar, tc_bar, rl_err, tc_err = [], [], [], [], []
    for r in rows:
        if r["old_rl_scores"] and r["old_tc_scores"]:
            inst_labels.append(f"{r['instance']}\n(k={r['k']})")
            rl_bar.append(statistics.median(r["old_rl_scores"]))
            tc_bar.append(statistics.median(r["old_tc_scores"]))
            rl_err.append((max(r["old_rl_scores"]) - min(r["old_rl_scores"])) / 2)
            tc_err.append((max(r["old_tc_scores"]) - min(r["old_tc_scores"])) / 2)
    if inst_labels:
        xc = np.arange(len(inst_labels))
        w = 0.36
        b1 = ax.bar(xc - w/2, rl_bar, w, color="#1f77b4", edgecolor="black", linewidth=0.5, label="RLTabuCol (RL + TabuCol)", yerr=rl_err, capsize=3, error_kw={"ecolor":"gray","alpha":0.6})
        b2 = ax.bar(xc + w/2, tc_bar, w, color="#7f7f7f", edgecolor="black", linewidth=0.5, label="Pure TabuCol", yerr=tc_err, capsize=3, error_kw={"ecolor":"gray","alpha":0.6})
        for rect, val in zip(b1, rl_bar):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.3, f"{int(val)}", ha="center", va="bottom", fontsize=8, color="#1f77b4", fontweight="bold")
        for rect, val in zip(b2, tc_bar):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.3, f"{int(val)}", ha="center", va="bottom", fontsize=8, color="#7f7f7f", fontweight="bold")
        ax.set_xticks(xc)
        ax.set_xticklabels(inst_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Median final conflict count")
        ax.set_title("(C) RLTabuCol vs Pure TabuCol: median final score across runs (5h budget)", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "(no old RL vs TabuCol data available)", ha="center", va="center", fontsize=11, transform=ax.transAxes)

    fig.suptitle("Figure X. Component-level Analysis of RL, ACO, TabuCol", fontsize=13, y=1.00)
    fig.tight_layout()
    out = FIG_OUT / "component_contribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
