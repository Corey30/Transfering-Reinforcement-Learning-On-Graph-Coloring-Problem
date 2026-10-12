"""
Analyze convergence test results.

Convergence criterion: running_best of the LAST 30% of episodes is constant.
"""
import json
import os
import re
import sys
from pathlib import Path
from statistics import median

ENCODINGS = ["utf-16", "utf-16-le", "utf-8", "cp1252"]


def read_log(path):
    """Read a log file trying multiple encodings."""
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                text = f.read()
            # Quick sanity check — JSON line should be readable
            if "[DONE]" in text or "Episode " in text or "{" in text:
                return text
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    # Last resort
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_log(path):
    """Return the runner.py JSON result (last JSON line) or None."""
    text = read_log(path)
    # Find the last line that looks like a JSON object
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue
    return None


def analyze_run(result):
    """Compute convergence + RL degradation stats from a parsed JSON result."""
    traces = result.get("episode_traces", [])
    if not traces:
        return None

    n_ep = len(traces)

    def f(t, key, default=None):
        v = t.get(key, default)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    bests = [f(t, "running_best") for t in traces]
    bests = [b for b in bests if b is not None]
    if not bests:
        return None
    final_best = bests[-1]
    times = [f(t, "t_end", 0.0) for t in traces]

    ttb_idx = next((i for i, b in enumerate(bests) if b == final_best), n_ep - 1)
    ttb_time = times[ttb_idx]
    ttb_ep = ttb_idx + 1

    # Convergence: last 30% of bests are constant
    tail_start = max(1, int(n_ep * 0.7))
    tail_bests = bests[tail_start:]
    converged = len(set(tail_bests)) == 1 if len(tail_bests) >= 2 else False

    total_time = result.get("time", times[-1] if times else 0.0)

    # RL degradation analysis
    rl_starts = [f(t, "rl_start_score") for t in traces]
    rl_ends = [f(t, "rl_end_score") for t in traces]            # after keep-best restore
    rl_actual = [f(t, "rl_actual_end_score") for t in traces]   # before restore
    rl_best_step = [f(t, "rl_best_step") for t in traces]
    aco_seeds = [f(t, "aco_seed_score") for t in traces]
    aco_ends = [f(t, "aco_end_score") for t in traces]

    def med(xs):
        xs = [x for x in xs if x is not None]
        return median(xs) if xs else None

    def degradation_count(starts, ends):
        cnt = 0
        for s, e in zip(starts, ends):
            if s is not None and e is not None and e > s:
                cnt += 1
        return cnt

    # Prefer "actual" RL_end for degradation analysis (raw policy behavior).
    # Fallback to rl_end if actual is missing.
    rl_post_field = rl_actual if any(x is not None for x in rl_actual) else rl_ends
    deg_n = degradation_count(rl_starts, rl_post_field)

    return {
        "episodes": n_ep,
        "first_best": bests[0],
        "final_best": final_best,
        "ttb_episode": ttb_ep,
        "ttb_time": ttb_time,
        "total_time": total_time,
        "converged": converged,
        "tail_unique": sorted(set(tail_bests)),
        # RL stats
        "median_rl_start": med(rl_starts),
        "median_rl_end": med(rl_ends),
        "median_rl_actual_end": med(rl_actual),
        "median_rl_best_step": med(rl_best_step),
        "median_aco_seed": med(aco_seeds),
        "median_aco_end": med(aco_ends),
        "rl_deg_count": deg_n,
    }


def collect_logs(out_dir, suffix_regex):
    """Group logs by instance, preferring RERUN files when present.

    Returns dict: { (instance, k) -> Path }
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return {}

    # All matching files
    files = sorted(out_dir.glob("*.txt"))
    matches = []
    for fp in files:
        m = re.match(suffix_regex, fp.name)
        if m:
            instance, k = m.group(1), m.group(2)
            is_rerun = "RERUN" in fp.name
            matches.append((instance, k, is_rerun, fp))

    # Group by (instance, k); prefer RERUN
    grouped = {}
    for instance, k, is_rerun, fp in matches:
        key = (instance, k)
        if key not in grouped:
            grouped[key] = fp
        else:
            # Keep RERUN if both exist
            existing_is_rerun = "RERUN" in grouped[key].name
            if is_rerun and not existing_is_rerun:
                grouped[key] = fp
    return grouped


def fmt(v, prec=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def fmt_signed(v, prec=0):
    """Format with explicit sign, '-' if None."""
    if v is None:
        return "-"
    return f"{v:+.{prec}f}"


def main():
    # Force stdout to UTF-8 so we can emit unicode markers on Windows GBK consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Accept an optional directory argument; default to convergence_results
    main_dir = sys.argv[1] if len(sys.argv) > 1 else "convergence_results"

    print(f"# Convergence Test Results — RLACO soft, 1 h, chi* (best-known)\n")
    print(f"_Source: `{main_dir}/`_\n")

    # --- Main test ---
    main_pattern = r"(.+?)_k(\d+)_rlaco(?:_RERUN)?\.txt"
    main_logs = collect_logs(main_dir, main_pattern)

    if not main_logs:
        print(f"ERROR: no logs found in {main_dir}/", file=sys.stderr)
        sys.exit(1)

    print("## Part 1 — Convergence Summary (without keep-rl-best)\n")
    print(
        "| Instance | k | Eps | First→Final | TTB ep | TTB (s) | Total (s) | Converged? |"
    )
    print("|---|---|---:|---|---:|---:|---:|:---:|")

    main_results = {}
    summary = {"converged": [], "not_converged": [], "broken": []}

    for (instance, k), fp in sorted(main_logs.items()):
        result = parse_log(fp)
        if result is None:
            print(f"| {instance} | {k} | - | - | - | - | - | BROKEN |")
            summary["broken"].append(instance)
            continue
        info = analyze_run(result)
        if info is None:
            print(f"| {instance} | {k} | 0 | - | - | - | - | NO TRACES |")
            summary["broken"].append(instance)
            continue

        status = "OK " if info["converged"] else "WARN"
        (summary["converged" if info["converged"] else "not_converged"]).append(
            instance
        )
        main_results[(instance, k)] = info

        rerun_tag = " (RERUN)" if "RERUN" in fp.name else ""
        print(
            f"| {instance}{rerun_tag} | {k} | {info['episodes']} | "
            f"{int(info['first_best'])}→**{int(info['final_best'])}** | "
            f"{info['ttb_episode']} | {info['ttb_time']:.0f} | "
            f"{info['total_time']:.0f} | {status} |"
        )

    print(
        f"\n**Summary:** OK {len(summary['converged'])}  "
        f"WARN {len(summary['not_converged'])}  "
        f"BROKEN {len(summary['broken'])} (out of {len(main_logs)})\n"
    )
    if summary["not_converged"]:
        print("**Not converged (best still improving in last 30% of episodes):**")
        for inst in summary["not_converged"]:
            print(f"- {inst}")
        print()

    # --- RL degradation analysis ---
    print("## Part 2 — RL Phase Diagnostics\n")
    print(
        "Median values per instance. `RL_end (raw)` = score after RL's 300 steps "
        "(before any keep-rl-best restoration). `RL_best_step` = step at which RL "
        "found its episode-internal best (0 means RL never improved on the start).\n"
    )
    print(
        "| Instance | k | RL_start | RL_end (raw) | Δ (degraded) | RL_best_step | "
        "ACO_seed | ACO_end | Deg eps |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for (instance, k), info in sorted(main_results.items()):
        rls = info["median_rl_start"]
        rle = info["median_rl_actual_end"] or info["median_rl_end"]
        delta = (rle - rls) if (rls is not None and rle is not None) else None
        bs = info["median_rl_best_step"]
        acs = info["median_aco_seed"]
        ace = info["median_aco_end"]
        deg_str = f"{info['rl_deg_count']}/{info['episodes']}"
        print(
            f"| {instance} | {k} | {fmt(rls,0)} | {fmt(rle,0)} | "
            f"{fmt_signed(delta,0)} | {fmt(bs,0)} | {fmt(acs,0)} | "
            f"{fmt(ace,0)} | {deg_str} |"
        )

    # --- Keep-best comparison (if available) ---
    keepbest_pattern = r"(.+?)_k(\d+)_rlaco_keepbest\.txt"
    kb_logs = collect_logs("convergence_results_keepbest", keepbest_pattern)

    if kb_logs:
        print("\n## Part 3 — keep-rl-best vs no keep-rl-best\n")
        print(
            "Compared on the 4 instances where keep-rl-best was tested. "
            "Final best is what would be reported. RL_best_step → 0 means "
            "RL learned to 'do nothing' (the start frame was best).\n"
        )
        print(
            "| Instance | k | Mode | Eps | Final Best | TTB (s) | Med RL_best_step |"
        )
        print("|---|---|---|---:|---:|---:|---:|")

        for (instance, k), fp_kb in sorted(kb_logs.items()):
            # No keep-best version
            no_info = main_results.get((instance, k))
            if no_info:
                bs_str = fmt(no_info["median_rl_best_step"], 0)
                print(
                    f"| {instance} | {k} | _no keep-best_ | "
                    f"{no_info['episodes']} | {int(no_info['final_best'])} | "
                    f"{no_info['ttb_time']:.0f} | {bs_str} |"
                )
            # Keep-best version
            kb_result = parse_log(fp_kb)
            if kb_result:
                kb_info = analyze_run(kb_result)
                if kb_info:
                    bs_str = fmt(kb_info["median_rl_best_step"], 0)
                    print(
                        f"| {instance} | {k} | **keep-best** | "
                        f"{kb_info['episodes']} | **{int(kb_info['final_best'])}** | "
                        f"{kb_info['ttb_time']:.0f} | {bs_str} |"
                    )

    print()


if __name__ == "__main__":
    main()
