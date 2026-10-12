"""
Generate convergence figures for the RLACO experiment.

For every results/<cell> directory we read the JSON record at the end of each
run log, extract the per-episode `running_best` trace, and plot the best-so-far
conflict count against episode index. 10 runs per cell are overlaid plus their
median, so you can see whether the search has flattened (converged) or is still
improving at the time budget.

Outputs (into analysis/figures/):
    conv_<cell>.png            one panel per cell (10 runs + median)
    compare_<instance>.png     soft vs baseline (rlaco0) median curves
    convergence_grid.png       all cells in one grid
    convergence_summary.csv    per-cell converged? + final-best stats
"""
import csv
import json
import os
import re
from pathlib import Path
from statistics import median, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENCODINGS = ["utf-8", "utf-16", "utf-16-le", "cp1252"]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"


def read_text(path):
    for enc in ENCODINGS:
        try:
            return open(path, "r", encoding=enc, errors="strict").read()
        except (UnicodeDecodeError, Exception):
            continue
    return open(path, "r", encoding="utf-8", errors="replace").read()


def parse_log(path):
    """Return the runner.py JSON result (last JSON object line) or None."""
    text = read_text(path)
    if "[DONE]" not in text:
        return None
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue
    return None


def running_best_trace(result):
    traces = result.get("episode_traces", [])
    out = []
    for t in traces:
        v = t.get("running_best")
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def converged(trace, tail_frac=0.3, rel_tol=0.02):
    """Practically converged if the best-so-far has plateaued over the last
    `tail_frac` of episodes. We compare the median of the first half of the
    tail to the median of the second half: if the search improved by less than
    `rel_tol` (default 2%, min 1 conflict) across the tail, it has converged.
    Using medians (not the raw tail endpoints) makes this robust to the tiny
    ±1-2 conflict wiggle in running_best that a strict 'bit-identical' test
    would wrongly flag as non-convergence."""
    if len(trace) < 4:
        return False
    tail_start = max(1, int(len(trace) * (1 - tail_frac)))
    tail = trace[tail_start:]
    if len(tail) < 4:
        return False
    half = len(tail) // 2
    early = median(tail[:half])
    late = median(tail[half:])
    final = trace[-1]
    tol = max(1.0, abs(final) * rel_tol)
    return (early - late) <= tol


def cell_label(name):
    # le450_25d_k25_rlaco_soft -> ("le450_25d", "25", "soft")
    m = re.match(r"(.+?)_k(\d+)_rlaco(0|_soft)$", name)
    if not m:
        return name, None, name
    inst, k, suf = m.group(1), m.group(2), m.group(3)
    method = "baseline" if suf == "0" else "soft"
    return inst, k, method


def load_cell(cell_dir):
    """Return list of running_best traces for all valid DONE runs in a cell."""
    traces = []
    for fp in sorted(cell_dir.glob("*.txt")):
        if "checkpoint" in fp.name:
            continue
        res = parse_log(fp)
        if res is None:
            continue
        tr = running_best_trace(res)
        if tr:
            traces.append(tr)
    return traces


def median_curve(traces):
    """Episode-wise median across runs of differing length."""
    if not traces:
        return []
    maxlen = max(len(t) for t in traces)
    med = []
    for i in range(maxlen):
        vals = [t[i] for t in traces if i < len(t)]
        med.append(median(vals))
    return med


def main():
    FIG_DIR.mkdir(exist_ok=True)
    cells = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        traces = load_cell(d)
        if traces:
            cells[d.name] = traces

    summary_rows = []

    # --- per-cell panels ---
    for name, traces in cells.items():
        inst, k, method = cell_label(name)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for tr in traces:
            ax.plot(range(1, len(tr) + 1), tr, color="0.75", lw=0.8, alpha=0.7)
        med = median_curve(traces)
        ax.plot(range(1, len(med) + 1), med, color="C0", lw=2.2, label="median")
        finals = [t[-1] for t in traces]
        conv_n = sum(converged(t) for t in traces)
        ax.set_title(
            f"{name}\n"
            f"n={len(traces)}  final best: median={int(median(finals))} "
            f"[{int(min(finals))}–{int(max(finals))}]  "
            f"converged {conv_n}/{len(traces)}"
        )
        ax.set_xlabel("episode")
        ax.set_ylabel("best conflicts so far")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"conv_{name}.png", dpi=120)
        plt.close(fig)

        summary_rows.append({
            "cell": name, "instance": inst, "k": k, "method": method,
            "n_runs": len(traces),
            "final_min": int(min(finals)),
            "final_median": median(finals),
            "final_max": int(max(finals)),
            "final_sd": round(pstdev(finals), 2) if len(finals) > 1 else 0.0,
            "converged_runs": conv_n,
        })

    # --- soft vs baseline comparison per instance ---
    by_inst = {}
    for name, traces in cells.items():
        inst, k, method = cell_label(name)
        by_inst.setdefault((inst, k), {})[method] = traces
    for (inst, k), d in sorted(by_inst.items()):
        if "soft" not in d or "baseline" not in d:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, color in [("baseline", "C3"), ("soft", "C0")]:
            med = median_curve(d[method])
            ax.plot(range(1, len(med) + 1), med, color=color, lw=2.2,
                    label=f"{method} (median of {len(d[method])})")
        ax.set_title(f"{inst}  k={k}  —  RLACO soft vs baseline")
        ax.set_xlabel("episode")
        ax.set_ylabel("best conflicts so far")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"compare_{inst}_k{k}.png", dpi=120)
        plt.close(fig)

    # --- one big grid ---
    n = len(cells)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()
    for ax, (name, traces) in zip(axes, cells.items()):
        for tr in traces:
            ax.plot(range(1, len(tr) + 1), tr, color="0.8", lw=0.6)
        med = median_curve(traces)
        ax.plot(range(1, len(med) + 1), med, color="C0", lw=1.8)
        conv_n = sum(converged(t) for t in traces)
        ax.set_title(f"{name}\nconv {conv_n}/{len(traces)}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[len(cells):]:
        ax.axis("off")
    fig.suptitle("RLACO convergence — best conflicts vs episode (all cells)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG_DIR / "convergence_grid.png", dpi=110)
    plt.close(fig)

    # --- CSV summary ---
    with open(FIG_DIR / "convergence_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote {len(cells)} cell panels + comparisons + grid to {FIG_DIR}")
    print(f"{'cell':38s} {'n':>2} {'conv':>5} {'final med':>9}")
    for r in summary_rows:
        print(f"{r['cell']:38s} {r['n_runs']:2d} "
              f"{r['converged_runs']:2d}/{r['n_runs']:<2d} {r['final_median']:9.1f}")


if __name__ == "__main__":
    main()
