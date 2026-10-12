from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "longrun_results" / "_new_code_from_zip" / "RLTCol-main" / "src"
POLICY_DIR = ROOT / "probe_policies"
OUT_ROOT = ROOT / "longrun_results" / "20260710_flat30028_aco_hybrids_beta02" / "results"


POLICIES = {
    "train300_15": POLICY_DIR / "train300_15.pth",
    "train350_15": POLICY_DIR / "train350_15.pth",
    "train300_28": POLICY_DIR / "train300_28.pth",
    "train350_28": POLICY_DIR / "train350_28.pth",
    "train400_28": POLICY_DIR / "train400_28.pth",
    "train400_15": POLICY_DIR / "train400_15.pth",
    "train450_15": POLICY_DIR / "train450_15.pth",
    "train450_28": POLICY_DIR / "train450_28.pth",
    "train500_15": POLICY_DIR / "train500_15.pth",
    "train500_28": POLICY_DIR / "train500_28.pth",
}


ARTICLE_DEFAULTS = {
    "instance": "le450_15d",
    "k": 15,
    "policy": "train300_15",
    "seeds": 10,
    "time_limit": 18000,
    "rl_steps": 300,
    "aco_ants": 40,
    "aco_iter": 50,
    "tabu_iters": 100000,
    "tabu_a": 10,
    "tabu_alpha": 1.2,
    "beta": 0.2,
    "inject_mode": "soft",
    "aco_alpha": 1.0,
    "aco_beta": 3.0,
    "aco_rho": 0.20,
    "aco_q": 1.0,
    "aco_tau0": 1.0,
    "aco_tau_min": 0.05,
    "aco_tau_max": 8.0,
    "aco_q0": 0.85,
    "aco_elite_count": 3,
    "aco_global_best_weight": 2.0,
    "injection_strength": 8.0,
}


def parse_json_from_stdout(text: str) -> dict | None:
    text = text.replace("\x00", "")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    start = text.rfind('{"mode"')
    if start >= 0:
        candidate = text[start:].strip()
        end = candidate.rfind("}")
        if end >= 0:
            return json.loads(candidate[: end + 1])
    return None


def median(vals: list[float]) -> float | None:
    vals = sorted(vals)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def resolve_policy(policy: str) -> Path:
    if policy in POLICIES:
        return POLICIES[policy]
    path = Path(policy)
    if path.exists():
        return path.resolve()
    if not policy.endswith(".pth"):
        candidate = POLICY_DIR / f"{policy}.pth"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve policy: {policy}")


def resolve_graph(instance: str) -> Path:
    path = Path(instance)
    if path.exists():
        return path.resolve()
    name = instance if instance.endswith(".col") else f"{instance}.col"
    candidate = SRC / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve graph instance: {instance}")


def ensure_inputs(args: argparse.Namespace) -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Cannot find source directory: {SRC}")
    for rel in [
        "network.py",
        "gcp_env/gcp_env.py",
        "gcp_env/ACO.py",
        "tabucol/python/tabucol/__init__.py",
    ]:
        if not (SRC / rel).exists():
            raise FileNotFoundError(f"Missing required source file: {SRC / rel}")
    resolve_graph(args.instance)
    resolve_policy(args.policy)


def write_summaries(rows: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "all_summaries.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = [
        "run_id",
        "model",
        "instance",
        "k",
        "policy",
        "seed",
        "score",
        "recalc_score",
        "cycles",
        "time_limit",
        "reported_time",
        "elapsed_wall",
        "rl_steps",
        "aco_ants",
        "aco_iter",
        "tabu_iters",
        "tabu_a",
        "tabu_alpha",
        "beta",
        "inject_mode",
        "keep_rl_best",
        "returncode",
        "parse_ok",
        "last_start",
        "last_rl1_end",
        "last_aco_end",
        "last_rl2_end",
        "last_tabucol_end",
        "stdout",
        "stderr",
    ]
    with (OUT_ROOT / "summary_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["model"], r["seed"])):
            writer.writerow({key: row.get(key, "") for key in fields})

    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        if row.get("recalc_score") is None:
            continue
        key = (
            row["model"],
            row["instance"],
            row["k"],
            row["policy"],
            row["rl_steps"],
            row["aco_ants"],
            row["aco_iter"],
            row["tabu_iters"],
            row["beta"],
            row["inject_mode"],
            row["keep_rl_best"],
        )
        grouped.setdefault(key, []).append(float(row["recalc_score"]))

    summary_rows = []
    for key, vals in sorted(grouped.items()):
        mean = sum(vals) / len(vals)
        var = sum((value - mean) ** 2 for value in vals) / len(vals)
        summary_rows.append(
            {
                "model": key[0],
                "instance": key[1],
                "k": key[2],
                "policy": key[3],
                "rl_steps": key[4],
                "aco_ants": key[5],
                "aco_iter": key[6],
                "tabu_iters": key[7],
                "beta": key[8],
                "inject_mode": key[9],
                "keep_rl_best": key[10],
                "runs": len(vals),
                "best": min(vals),
                "mean": mean,
                "median": median(vals),
                "std": var**0.5,
                "zeros": sum(1 for value in vals if value == 0),
                "values": sorted(vals),
            }
        )
    (OUT_ROOT / "policy_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_child(job: dict, args: argparse.Namespace) -> dict:
    model = job["model"]
    seed = int(job["seed"])
    policy_path = resolve_policy(args.policy)
    graph_path = resolve_graph(args.instance)
    run_id = (
        f"{args.instance}_k{args.k}_{model}_{policy_path.stem}_"
        f"{args.inject_mode}_rl{args.rl_steps}_aco{args.aco_ants}x{args.aco_iter}_"
        f"T{args.tabu_iters}_beta{args.beta:g}_keep{int(args.keep_rl_best)}_seed{seed}"
    )
    out_dir = (
        OUT_ROOT
        / args.instance
        / f"k{args.k}"
        / policy_path.stem
        / model
        / args.inject_mode
        / f"beta_{args.beta:g}"
        / ("rl_keep_best" if args.keep_rl_best else "rl_no_keep_best")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{run_id}.stdout.txt"
    stderr_path = out_dir / f"{run_id}.stderr.txt"
    result_path = out_dir / f"{run_id}.result.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run",
        "--model",
        model,
        "--policy",
        str(policy_path),
        "--instance",
        str(graph_path),
        "--k",
        str(args.k),
        "--seed",
        str(seed),
        "--time-limit",
        str(args.time_limit),
        "--cycles",
        str(args.cycles),
        "--rl-steps",
        str(args.rl_steps),
        "--aco-ants",
        str(args.aco_ants),
        "--aco-iter",
        str(args.aco_iter),
        "--tabu-iters",
        str(args.tabu_iters),
        "--tabu-a",
        str(args.tabu_a),
        "--tabu-alpha",
        str(args.tabu_alpha),
        "--beta",
        str(args.beta),
        "--inject-mode",
        args.inject_mode,
        "--aco-alpha",
        str(args.aco_alpha),
        "--aco-beta",
        str(args.aco_beta),
        "--aco-rho",
        str(args.aco_rho),
        "--aco-q",
        str(args.aco_q),
        "--aco-tau0",
        str(args.aco_tau0),
        "--aco-tau-min",
        str(args.aco_tau_min),
        "--aco-tau-max",
        str(args.aco_tau_max),
        "--aco-q0",
        str(args.aco_q0),
        "--aco-elite-count",
        str(args.aco_elite_count),
        "--aco-global-best-weight",
        str(args.aco_global_best_weight),
        "--injection-strength",
        str(args.injection_strength),
    ]
    if args.keep_rl_best:
        cmd.append("--keep-rl-best")
    if args.deterministic_rl:
        cmd.append("--deterministic-rl")

    env = dict(os.environ)
    pythonpath_parts = [str(SRC), str(SRC / "tabucol" / "python")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=args.time_limit + args.timeout_grace,
        env=env,
    )
    elapsed = time.time() - started
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    data = parse_json_from_stdout(proc.stdout)
    cycles = data.get("cycle_traces", []) if data else []
    last = cycles[-1] if cycles else {}
    if data is not None:
        result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "run_id": run_id,
        "model": model,
        "instance": args.instance,
        "k": args.k,
        "policy": policy_path.stem,
        "seed": seed,
        "score": data.get("score") if data else None,
        "recalc_score": data.get("recalc_score") if data else None,
        "cycles": data.get("cycles") if data else None,
        "time_limit": args.time_limit,
        "reported_time": data.get("time") if data else None,
        "elapsed_wall": elapsed,
        "rl_steps": args.rl_steps,
        "aco_ants": args.aco_ants,
        "aco_iter": args.aco_iter,
        "tabu_iters": args.tabu_iters,
        "tabu_a": args.tabu_a,
        "tabu_alpha": args.tabu_alpha,
        "beta": args.beta,
        "inject_mode": args.inject_mode,
        "keep_rl_best": bool(args.keep_rl_best),
        "returncode": proc.returncode,
        "parse_ok": data is not None,
        "last_start": last.get("cycle_start_score"),
        "last_rl1_end": (last.get("rl1") or {}).get("end"),
        "last_aco_end": (last.get("aco") or {}).get("end"),
        "last_rl2_end": (last.get("rl2") or {}).get("end"),
        "last_tabucol_end": (last.get("tabucol") or {}).get("end"),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (out_dir / f"{run_id}.summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return row


def read_graph_from_file(path: str | Path):
    import networkx as nx

    graph = nx.Graph()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                n, _m = line.split()[2:4]
                graph.add_nodes_from(range(int(n)))
                continue
            if line.startswith("e"):
                u, v = line.split()[1:3]
                graph.add_edge(int(u) - 1, int(v) - 1)
    return graph


def calculate_score(graph, solution) -> int:
    return int(sum(1 for u, v in graph.edges() if int(solution[u]) == int(solution[v])))


def set_all_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy(policy_path: Path, graph, k: int, env, device: str):
    import torch
    from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
    from tianshou.utils.net.common import ActorCritic

    actor = ActorNetwork(3, 3, device=device).to(device)
    critic = CriticNetwork(3, 3, device=device).to(device)
    actor_critic = ActorCritic(actor, critic).to(device)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)
    policy = GCPPPOPolicy(
        actor,
        critic,
        optim,
        torch.distributions.Categorical,
        k=k,
        nodes=len(graph),
        action_space=env.action_space,
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    return policy


def actor_col_distribution(policy, obs, device: str):
    import numpy as np
    import torch

    with torch.no_grad():
        col_feats = torch.as_tensor(
            np.expand_dims(obs["col_features"], 0),
            device=device,
            dtype=torch.float32,
        )
        col_probs = torch.squeeze(policy.actor.col_model(col_feats), -1)
        return col_probs[0].cpu().numpy()


def select_action(policy, obs, deterministic: bool, device: str):
    import numpy as np
    import torch

    with torch.no_grad():
        batch_obs = {
            "node_features": np.expand_dims(obs["node_features"], 0),
            "col_features": np.expand_dims(obs["col_features"], 0),
            "k": np.array([obs["k"]]),
        }
        logits, _ = policy.actor(batch_obs)
        probs = logits[0]
        if deterministic:
            flat = int(torch.argmax(probs).item())
        else:
            dist = torch.distributions.Categorical(probs=probs)
            flat = int(dist.sample().item())
        return policy.map_action(np.array([flat]))[0]


def run_rl_phase(env, policy, graph, steps: int, deterministic: bool, device: str, keep_best: bool):
    start_score = int(env.score)
    best_score = start_score
    best_solution = env._solution.copy()
    actions = 0
    good = 0
    bad = 0
    same = 0
    for _ in range(steps):
        if int(env.score) == 0:
            break
        obs = env._get_obs()
        before = int(env.score)
        action = select_action(policy, obs, deterministic, device)
        if int(env._solution[int(action[0])]) == int(action[1]):
            same += 1
        _obs, _rew, terminated, _truncated, _info = env.step(action)
        after = int(env.score)
        actions += 1
        if after < before:
            good += 1
        elif after > before:
            bad += 1
        if after < best_score:
            best_score = after
            best_solution = env._solution.copy()
        if terminated:
            break
    if keep_best:
        env._solution = best_solution.copy()
        env._prev_solution = env._solution.copy()
    env.score = calculate_score(graph, env._solution)
    env._initialize_obs()
    return {
        "start": start_score,
        "end": int(env.score),
        "best": int(best_score),
        "keep_best": bool(keep_best),
        "actions": actions,
        "good": good,
        "bad": bad,
        "same": same,
    }


def run_aco_phase(env, policy, graph, args: argparse.Namespace, device: str):
    import numpy as np
    from gcp_env.ACO import AntColony

    start_score = int(env.score)
    aco = AntColony(
        graph,
        args.k,
        alpha=args.aco_alpha,
        beta=args.aco_beta,
        rho=args.aco_rho,
        q=args.aco_q,
        tau0=args.aco_tau0,
        tau_min=args.aco_tau_min,
        tau_max=args.aco_tau_max,
        q0=args.aco_q0,
        elite_count=args.aco_elite_count,
        global_best_weight=args.aco_global_best_weight,
    )
    if args.inject_mode == "soft":
        dist = actor_col_distribution(policy, env._get_obs(), device)
        aco.bias_from_distribution(
            dist,
            solution=env._solution,
            strength=args.injection_strength,
            reset_tau=True,
        )
    else:
        aco.bias_from_solution(
            env._solution,
            strength=args.injection_strength,
            reset_tau=True,
        )
    sol = aco.run(m=args.aco_ants, max_iter=args.aco_iter, verbose=False)
    env._solution = np.asarray(sol, dtype=np.int32)
    env._prev_solution = env._solution.copy()
    env.score = calculate_score(graph, env._solution)
    env._initialize_obs()
    return {
        "start": start_score,
        "seed": int(aco.seed_conflicts),
        "end": int(env.score),
        "curve": list(aco.iter_y),
        "params": {
            "alpha": args.aco_alpha,
            "beta": args.aco_beta,
            "rho": args.aco_rho,
            "q": args.aco_q,
            "tau0": args.aco_tau0,
            "tau_min": args.aco_tau_min,
            "tau_max": args.aco_tau_max,
            "q0": args.aco_q0,
            "elite_count": args.aco_elite_count,
            "global_best_weight": args.aco_global_best_weight,
            "injection_strength": args.injection_strength,
        },
    }


def run_tabucol_phase(env, graph, args: argparse.Namespace):
    import numpy as np
    import tabucol

    start_score = int(env.score)
    adj_list = [[] for _ in range(len(graph))]
    for node in graph:
        for neighbor in graph.neighbors(node):
            adj_list[node].append(neighbor)
    solver = tabucol.TabuColSolver(
        adj_list,
        env.k,
        max_iterations=args.tabu_iters,
        tabu_a=args.tabu_a,
        tabu_alpha=args.tabu_alpha,
        beta=args.beta,
    )
    solver.set_solution(env._solution)
    sol, _ = solver.solve()
    env._solution = np.asarray(sol, dtype=np.int32)
    env._prev_solution = env._solution.copy()
    env.score = calculate_score(graph, env._solution)
    env._initialize_obs()
    return {"start": start_score, "end": int(env.score)}


def single_run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import numpy as np
    import torch
    from gymnasium.envs.registration import register

    local_tabucol = SRC / "tabucol" / "python"
    if local_tabucol.exists():
        sys.path.insert(0, str(local_tabucol))
    sys.path.insert(1, str(SRC))

    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_path = resolve_policy(args.policy)
    graph_path = resolve_graph(args.instance)
    graph = read_graph_from_file(graph_path)

    env_id = f"ArticleAcoHybridGcpEnv-{args.model}-{args.seed}-v0"
    register(
        id=env_id,
        entry_point="gcp_env:GcpEnv",
        max_episode_steps=max(10_000_000, args.rl_steps * 4),
    )
    gym_env = gym.make(env_id, graph=graph, k=args.k, disable_tabucol=True)
    _obs, _info = gym_env.reset(seed=args.seed, options={"random_init": True})
    env = gym_env.unwrapped
    policy = load_policy(policy_path, graph, args.k, env, device)

    start_time = time.time()
    initial_score = int(env.score)
    best_score = initial_score
    best_solution = env._solution.copy()
    cycle_traces = []
    cycles_done = 0

    print("-------------------------", flush=True)
    print(f"Mode: {args.model}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Graph: {graph_path}", flush=True)
    print(f"k: {args.k}", flush=True)
    print(f"Policy: {policy_path}", flush=True)
    print(f"Initial score: {initial_score}", flush=True)
    print(f"Time limit: {args.time_limit}", flush=True)
    print(f"RL keep-best: {args.keep_rl_best}", flush=True)
    print("-------------------------", flush=True)

    while True:
        if args.cycles and cycles_done >= args.cycles:
            break
        if time.time() - start_time >= args.time_limit:
            break
        if int(env.score) == 0 or int(best_score) == 0:
            break

        cycles_done += 1
        trace = {
            "cycle": cycles_done,
            "t_start": time.time() - start_time,
            "cycle_start_score": int(env.score),
            "running_best_start": int(best_score),
        }

        trace["rl1"] = run_rl_phase(
            env,
            policy,
            graph,
            args.rl_steps,
            args.deterministic_rl,
            device,
            args.keep_rl_best,
        )
        if int(env.score) < best_score:
            best_score = int(env.score)
            best_solution = env._solution.copy()
        if best_score == 0:
            trace["t_end"] = time.time() - start_time
            cycle_traces.append(trace)
            break

        trace["aco"] = run_aco_phase(env, policy, graph, args, device)
        if int(env.score) < best_score:
            best_score = int(env.score)
            best_solution = env._solution.copy()
        if best_score == 0:
            trace["t_end"] = time.time() - start_time
            cycle_traces.append(trace)
            break

        if args.model == "alternating":
            trace["rl2"] = run_rl_phase(
                env,
                policy,
                graph,
                args.rl_steps,
                args.deterministic_rl,
                device,
                args.keep_rl_best,
            )
            if int(env.score) < best_score:
                best_score = int(env.score)
                best_solution = env._solution.copy()
            if best_score == 0:
                trace["t_end"] = time.time() - start_time
                cycle_traces.append(trace)
                break
        else:
            trace["rl2"] = None

        trace["tabucol"] = run_tabucol_phase(env, graph, args)
        if int(env.score) < best_score:
            best_score = int(env.score)
            best_solution = env._solution.copy()

        trace["cycle_end_score"] = int(env.score)
        trace["running_best"] = int(best_score)
        trace["t_end"] = time.time() - start_time
        cycle_traces.append(trace)

        print(
            f"Cycle {cycles_done} done: "
            f"start={trace['cycle_start_score']} "
            f"rl1={trace['rl1']['end']} "
            f"aco={trace['aco']['end']} "
            f"rl2={(trace.get('rl2') or {}).get('end')} "
            f"tabu={trace['tabucol']['end']} "
            f"best={best_score} "
            f"time={trace['t_end']:.1f}s",
            flush=True,
        )

    elapsed = time.time() - start_time
    final_solution = np.asarray(best_solution, dtype=np.int32)
    final_score = int(best_score)
    recalc_score = calculate_score(graph, final_solution)
    result = {
        "mode": args.model,
        "score": final_score,
        "recalc_score": recalc_score,
        "score_verified": final_score == recalc_score,
        "cycles": cycles_done,
        "time": elapsed,
        "graph": str(graph_path),
        "k": args.k,
        "policy": str(policy_path),
        "seed": args.seed,
        "initial_score": initial_score,
        "inject_mode": args.inject_mode,
        "rl_steps": args.rl_steps,
        "aco_ants": args.aco_ants,
        "aco_iter": args.aco_iter,
        "tabu_iters": args.tabu_iters,
        "tabu_a": args.tabu_a,
        "tabu_alpha": args.tabu_alpha,
        "beta": args.beta,
        "keep_rl_best": bool(args.keep_rl_best),
        "deterministic_rl": bool(args.deterministic_rl),
        "article_parameter_defaults": ARTICLE_DEFAULTS,
        "solution": final_solution.tolist(),
        "cycle_traces": cycle_traces,
    }
    print("[DONE]", flush=True)
    print(json.dumps(result, default=str), flush=True)
    os._exit(0)


def batch_run(args: argparse.Namespace) -> None:
    ensure_inputs(args)
    jobs = []
    for model in args.models:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            jobs.append({"model": model, "seed": seed})

    print(f"Using src: {SRC}", flush=True)
    print(f"Output: {OUT_ROOT}", flush=True)
    print(f"Models: {args.models}", flush=True)
    print(f"Instance: {args.instance}, k={args.k}", flush=True)
    print(f"Policy: {args.policy}", flush=True)
    print(f"Seeds: {args.seed_start}..{args.seed_start + args.seeds - 1}", flush=True)
    print(f"Time limit per seed: {args.time_limit}s", flush=True)
    print(f"Concurrency: {args.concurrency}", flush=True)
    print(f"RL keep-best: {args.keep_rl_best}", flush=True)
    print(f"Beta: {args.beta}", flush=True)
    print(f"Total jobs: {len(jobs)}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_child, job, args) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_summaries(rows)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{row['model']} seed={row['seed']} "
                f"score={row['score']} recalc={row['recalc_score']} "
                f"cycles={row['cycles']} "
                f"last={row['last_start']}->{row['last_rl1_end']}->"
                f"{row['last_aco_end']}->{row['last_rl2_end']}->"
                f"{row['last_tabucol_end']} "
                f"wall={row['elapsed_wall']:.1f}s",
                flush=True,
            )
    write_summaries(rows)
    print(f"Done. Summary: {OUT_ROOT / 'policy_summary.json'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="le450_15d beta=0.2 runner for RL-ACO-TabuCol and RL-ACO-RL-TabuCol."
    )
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=["rlacot", "alternating"], default="rlacot")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rlacot", "alternating"],
        default=["rlacot", "alternating"],
    )
    parser.add_argument("--policy", default=ARTICLE_DEFAULTS["policy"])
    parser.add_argument("--instance", default=ARTICLE_DEFAULTS["instance"])
    parser.add_argument("--k", type=int, default=ARTICLE_DEFAULTS["k"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=ARTICLE_DEFAULTS["seeds"])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=ARTICLE_DEFAULTS["time_limit"])
    parser.add_argument("--cycles", type=int, default=0, help="0 means stop by time-limit or score 0.")
    parser.add_argument("--rl-steps", type=int, default=ARTICLE_DEFAULTS["rl_steps"])
    parser.add_argument("--aco-ants", type=int, default=ARTICLE_DEFAULTS["aco_ants"])
    parser.add_argument("--aco-iter", type=int, default=ARTICLE_DEFAULTS["aco_iter"])
    parser.add_argument("--tabu-iters", type=int, default=ARTICLE_DEFAULTS["tabu_iters"])
    parser.add_argument("--tabu-a", type=int, default=ARTICLE_DEFAULTS["tabu_a"])
    parser.add_argument("--tabu-alpha", type=float, default=ARTICLE_DEFAULTS["tabu_alpha"])
    parser.add_argument("--beta", type=float, default=ARTICLE_DEFAULTS["beta"])
    parser.add_argument("--inject-mode", choices=["soft", "hard"], default=ARTICLE_DEFAULTS["inject_mode"])
    parser.add_argument("--aco-alpha", type=float, default=ARTICLE_DEFAULTS["aco_alpha"])
    parser.add_argument("--aco-beta", type=float, default=ARTICLE_DEFAULTS["aco_beta"])
    parser.add_argument("--aco-rho", type=float, default=ARTICLE_DEFAULTS["aco_rho"])
    parser.add_argument("--aco-q", type=float, default=ARTICLE_DEFAULTS["aco_q"])
    parser.add_argument("--aco-tau0", type=float, default=ARTICLE_DEFAULTS["aco_tau0"])
    parser.add_argument("--aco-tau-min", type=float, default=ARTICLE_DEFAULTS["aco_tau_min"])
    parser.add_argument("--aco-tau-max", type=float, default=ARTICLE_DEFAULTS["aco_tau_max"])
    parser.add_argument("--aco-q0", type=float, default=ARTICLE_DEFAULTS["aco_q0"])
    parser.add_argument("--aco-elite-count", type=int, default=ARTICLE_DEFAULTS["aco_elite_count"])
    parser.add_argument(
        "--aco-global-best-weight",
        type=float,
        default=ARTICLE_DEFAULTS["aco_global_best_weight"],
    )
    parser.add_argument("--injection-strength", type=float, default=ARTICLE_DEFAULTS["injection_strength"])
    parser.add_argument("--keep-rl-best", action="store_true", help="Default is off.")
    parser.add_argument("--deterministic-rl", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-grace", type=int, default=1800)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.single_run:
        single_run(args)
    else:
        batch_run(args)


if __name__ == "__main__":
    main()
