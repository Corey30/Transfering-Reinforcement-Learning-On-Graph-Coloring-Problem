"""
Batch wrapper for runner.py.

Spawns `--num-jobs` independent runs of runner.py with `--concurrency`
processes at a time. Each run gets a fresh random seed (the env's `reset`
uses a fresh PRNG for each episode), so the jobs differ in trajectory.

Output: one .txt per job under --output_dir, named `{k}_{timestamp}.txt`.
A successful run ends with "[DONE]" followed by a JSON result.
"""

import argparse
import os
import subprocess
import time
from datetime import datetime
from multiprocessing.dummy import Pool


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description="Run a batch of independent runner.py jobs in parallel."
    )
    parser.add_argument("policy", type=str, help="Policy state_dict (.pt) path.")
    parser.add_argument("graph", type=str, help="DIMACS .col path.")
    parser.add_argument("k", type=int, help="Number of colors.")
    parser.add_argument("output_dir", type=str, help="Output directory.")
    parser.add_argument("-I", "--max-steps", dest="max_steps", type=int, default=300)
    parser.add_argument("-B", "--beta", dest="beta", type=float, default=0.2)
    parser.add_argument("--time-limit", dest="time_limit", type=int, default=18000)
    parser.add_argument("--RL", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--num-jobs", dest="num_jobs", type=int, default=10)
    parser.add_argument("--concurrency", dest="concurrency", type=int, default=2)
    parser.add_argument("--aco-ants", dest="aco_ants", type=int, default=40)
    parser.add_argument("--aco-iter", dest="aco_iter", type=int, default=50)
    parser.add_argument(
        "--inject-mode", dest="inject_mode",
        choices=["hard", "soft"], default="hard",
    )
    parser.add_argument(
        "--use-aco", dest="use_aco", type=str2bool, nargs="?",
        const=True, default=True,
        help="Compatibility flag (always True in this build).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pool = Pool(args.concurrency)

    def run_job(i):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = f"{args.output_dir}/{args.k}_{timestamp}.txt"
        print(f"Starting job {i + 1}", flush=True)
        with open(log_path, "w") as f:
            cmd = (
                f"python runner.py {args.policy} {args.graph} {args.k} "
                f"-I {args.max_steps} -B {args.beta} "
                f"--time-limit {args.time_limit} --RL {args.RL} "
                f"--use-aco --aco-ants {args.aco_ants} "
                f"--aco-iter {args.aco_iter} --inject-mode {args.inject_mode}"
            )
            print(f"running: {cmd}", flush=True)
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            proc.wait()
        print(f"Finished job {i + 1}", flush=True)

    pool.map_async(run_job, range(args.num_jobs)).wait()
    print("All jobs finished", flush=True)


if __name__ == "__main__":
    main()
