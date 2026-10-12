"""
Standalone figure: RL vs ACO cumulative share of conflict reduction
across every RLACO_soft instance.

One horizontal bar per instance. Length = 100%. Blue segment = RL's share,
orange segment = ACO's share. In-bar white label shows the exact percentage.
"""
import glob
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RLACO_DIR = REPO / "RLACO_project" / "results"
FIG_OUT = Path(__file__).resolve().parent / "figures"
FIG_OUT.mkdir(exist_ok=True)

INSTANCES = [
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


def parse_result_json(text):
    if "[DONE]" not in text:
        return None
    tail = text[text.rfind("[DONE]"):].split("\n", 1)
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


def shares(cell_dir):
    """Return (rl_share_pct, aco_share_pct, n_runs)."""
    tot_rl = 0.0
    tot_aco = 0.0
    n_runs = 0
    for fp in sorted(cell_dir.glob("*.txt")):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        obj = parse_result_json(text)
        if obj is None:
            continue
        n_runs += 1
        for t in obj.get("episode_traces") or []:
            rl_s, rl_e = t.get("rl_start_score"), t.get("rl_end_score")
            aco_s, aco_e = t.get("aco_seed_score"), t.get("aco_end_score")
            try:
                if rl_s is not None and rl_e is not None:
                    tot_rl += max(0.0, float(rl_s) - float(rl_e))
                if aco_s is not None and aco_e is not None:
                    tot_aco += max(0.0, float(aco_s) - float(aco_e))
            except (TypeError, ValueError):
                continue
    denom = tot_rl + tot_aco
    if denom == 0:
        return None
    return (100 * tot_rl / denom, 100 * tot_aco / denom, n_runs)


def main():
    rows = []
    for inst, k in INSTANCES:
        d = RLACO_DIR / f"{inst}_k{k}_rlaco_soft"
        if not d.exists():
            continue
        s = shares(d)
        if s is None:
            continue
        rows.append((inst, k, s[0], s[1], s[2]))

    if not rows:
        print("No data.")
        return

    labels = [f"{r[0]} (k={r[1]})" for r in rows]
    rl_share = np.array([r[2] for r in rows])
    aco_share = np.array([r[3] for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.barh(y, rl_share, color="#1f77b4",
            edgecolor="black", linewidth=0.5, label="RL contribution")
    ax.barh(y, aco_share, left=rl_share, color="#ff7f0e",
            edgecolor="black", linewidth=0.5, label="ACO contribution")

    for i in range(len(rows)):
        # RL label — only draw if bar wide enough
        if rl_share[i] >= 8:
            ax.text(rl_share[i] / 2, i, f"{rl_share[i]:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        # ACO label — likewise
        if aco_share[i] >= 8:
            ax.text(rl_share[i] + aco_share[i] / 2, i, f"{aco_share[i]:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        # For very narrow slices put the number outside the bar
        if rl_share[i] < 8:
            ax.text(rl_share[i] + 1, i, f"RL {rl_share[i]:.1f}%",
                    ha="left", va="center", fontsize=8, color="#1f77b4", fontweight="bold")
        if aco_share[i] < 8:
            ax.text(rl_share[i] - 1, i, f"ACO {aco_share[i]:.1f}%",
                    ha="right", va="center", fontsize=8, color="#ff7f0e", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total conflict reduction (%)", fontsize=11)
    ax.set_title(
        "Cumulative RL vs ACO share of conflict reduction across RLACO_soft experiments",
        fontsize=12, pad=10,
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = FIG_OUT / "rl_aco_share.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    # Console table
    print(f"\n{'Instance':<14} {'k':<4} {'n_runs':<7} {'RL%':<7} {'ACO%'}")
    for r in rows:
        print(f"{r[0]:<14} {r[1]:<4} {r[4]:<7} {r[2]:<7.2f} {r[3]:.2f}")


if __name__ == "__main__":
    main()
