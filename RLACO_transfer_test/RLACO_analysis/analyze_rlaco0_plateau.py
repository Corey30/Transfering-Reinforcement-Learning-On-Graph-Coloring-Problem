"""
Analyze RLACO0 convergence runs to verify whether the main-experiment
time limit is long enough for RLACO0 to plateau.

"""
import json
import sys
from pathlib import Path

ENCODINGS = ["utf-16", "utf-16-le", "utf-8", "cp1252"]


def read_log(path):
    for enc in ENCODINGS:
        try:
            text = Path(path).read_text(encoding=enc, errors="strict")
            if "Episode " in text or "{" in text:
                return text
        except Exception:
            continue
    return Path(path).read_text(encoding="utf-8", errors="replace")


def parse_log(path):
    text = read_log(path)
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue
    return None


def best_at_time(traces, t_cutoff):
    """Return running_best at first episode whose t_end <= t_cutoff."""
    best_so_far = None
    for tr in traces:
        t = tr.get("t_end")
        rb = tr.get("running_best")
        if t is None or rb is None:
            continue
        if t <= t_cutoff:
            best_so_far = rb
        else:
            break
    return best_so_far


def plateau_time(traces):
    """First t_end at which running_best equals the final running_best."""
    rbs = [(tr.get("t_end"), tr.get("running_best")) for tr in traces]
    rbs = [(t, b) for t, b in rbs if t is not None and b is not None]
    if not rbs:
        return None, None
    final = rbs[-1][1]
    for t, b in rbs:
        if b == final:
            return t, final
    return rbs[-1][0], final


# Per-instance main-experiment cutoff (seconds)
MAIN_CUTOFF = {
    "flat1000_76_0": 10800,  # 3h
}
DEFAULT_CUTOFF = 3600  # 1h


def cutoff_for(tag):
    for key, v in MAIN_CUTOFF.items():
        if tag.startswith(key):
            return v
    return DEFAULT_CUTOFF


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "rlaco0_convergence")
    if not out_dir.exists():
        print(f"ERROR: {out_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    files = sorted(out_dir.glob("*_rlaco0.txt"))
    if not files:
        print(f"ERROR: no rlaco0 logs in {out_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"# RLACO0 Plateau Verification\n")
    print(f"_Source: `{out_dir}/`_\n")
    print(
        "Goal: verify that the main-experiment time limit (1h for most "
        "instances, 3h for flat1000_76_0) is long enough for RLACO0 to "
        "plateau. RLACO0 is run here with 2x the main-experiment budget.\n"
    )
    print(
        "| Instance | k | Cutoff | Best@cutoff | Best@end | Δ (forfeit) | "
        "Plateau (s) | Eps | Total (s) | Verdict |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|")

    for fp in files:
        # Filename: <tag>_k<K>_rlaco0.txt
        name = fp.stem  # strip .txt
        # Strip trailing _rlaco0
        if name.endswith("_rlaco0"):
            name = name[: -len("_rlaco0")]
        # Split _k<K>
        if "_k" in name:
            tag, k = name.rsplit("_k", 1)
        else:
            tag, k = name, "?"

        result = parse_log(fp)
        if result is None:
            print(f"| {tag} | {k} | - | - | - | - | - | - | - | BROKEN |")
            continue

        traces = result.get("episode_traces", [])
        if not traces:
            print(f"| {tag} | {k} | - | - | - | - | - | 0 | - | BROKEN |")
            continue

        cutoff = cutoff_for(tag)
        b_cut = best_at_time(traces, cutoff)
        plat_t, b_end = plateau_time(traces)
        total = result.get("time", 0)
        eps = len(traces)

        if b_cut is None:
            verdict = "NO DATA <cutoff"
            forfeit_str = "-"
            b_cut_str = "-"
        elif b_end is not None and b_cut == b_end:
            verdict = "OK plateau"
            forfeit_str = "+0"
            b_cut_str = f"{int(b_cut)}"
        else:
            forfeit = (b_cut - b_end) if (b_cut is not None and b_end is not None) else None
            verdict = "WARN improving"
            forfeit_str = f"-{int(forfeit)}" if forfeit else "?"
            b_cut_str = f"{int(b_cut)}"

        b_end_str = f"{int(b_end)}" if b_end is not None else "-"
        plat_str = f"{plat_t:.0f}" if plat_t is not None else "-"

        print(
            f"| {tag} | {k} | {cutoff} | {b_cut_str} | {b_end_str} | "
            f"{forfeit_str} | {plat_str} | {eps} | {total:.0f} | {verdict} |"
        )

    print()
    print(
        "**Δ (forfeit)** = Best@cutoff − Best@end. "
        "0 means the cutoff sacrifices nothing; a negative number means "
        "RLACO0 would have improved past the cutoff. "
        "Plateau (s) = first time at which the final best was reached."
    )


if __name__ == "__main__":
    main()
