"""
Inference runner for ACO+RL graph coloring.

Modes:
    RLACO  : --RL True  --inject-mode soft
        Uses the trained PPO policy at every RL step, then biases ACO pheromone
        with the policy's per-vertex color distribution (soft) or argmax (hard).

    RLACO0 : --RL False
        Uses a uniform-random policy at every RL step, then runs ACO with hard
        injection of the random solution. This is the no-policy baseline.

Each episode = (RL phase, max_steps actions) + (ACO phase, ants * iter constructions).
The run terminates when --time-limit (wall-clock) is reached or --max-episodes is hit.
"""

import argparse
import json
import os
import random
import sys
import time

import networkx as nx
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register
from tianshou.data import Batch, Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import ActorCritic

from network import ActorNetwork, CriticNetwork, GCPPPOPolicy


def read_graph_from_file(path):
    graph = nx.Graph()
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                n, _ = line.split()[2:4]
                graph.add_nodes_from(range(int(n)))
                continue
            if line.startswith("e"):
                u, v = line.split()[1:3]
                graph.add_edge(int(u) - 1, int(v) - 1)
    return graph


def calculate_score(graph, solution):
    score = 0
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            if solution[node] == solution[neighbor]:
                score += 1
    return score // 2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class RandomGCPPolicy(BasePolicy):
    """Uniform-random policy used for the RLACO0 baseline."""

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, batch, state=None, **kwargs):
        return Batch(act=np.array([self.action_space.sample()]))

    def learn(self, batch, **kwargs):
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="ACO + PPO inference runner for fixed-k graph coloring."
    )
    parser.add_argument("policy", type=str, help="Path to PPO policy state_dict (.pt).")
    parser.add_argument("graph", type=str, help="Path to DIMACS .col graph file.")
    parser.add_argument("k", type=int, help="Number of colors.")
    parser.add_argument(
        "-I", "--max-steps", dest="max_steps", type=int, default=300,
        help="Max RL steps per episode (default 300).",
    )
    parser.add_argument(
        "-E", "--max-episodes", dest="episodes", type=int, default=None,
        help="Max episodes per run; None = unlimited (use --time-limit).",
    )
    parser.add_argument(
        "-B", "--beta", dest="beta", type=float, default=0.2,
        help="Compatibility flag, retained for log parsers.",
    )
    parser.add_argument(
        "--time-limit", dest="time_limit", type=int, default=3600,
        help="Wall-clock cap per run, in seconds.",
    )
    parser.add_argument(
        "--RL", type=str2bool, nargs="?", const=True, default=True,
        help="True = use trained policy. False = uniform random (RLACO0).",
    )
    parser.add_argument(
        "--aco-ants", dest="aco_ants", type=int, default=40,
        help="Number of ants per ACO call.",
    )
    parser.add_argument(
        "--aco-iter", dest="aco_iter", type=int, default=50,
        help="ACO iterations per call.",
    )
    parser.add_argument(
        "--inject-mode", dest="inject_mode", type=str,
        choices=["hard", "soft"], default="hard",
        help="hard = boost tau at argmax(policy). "
             "soft = interpolate tau by per-vertex color distribution.",
    )
    parser.add_argument(
        "--keep-rl-best", dest="keep_rl_best", action="store_true", default=False,
        help="Restore best score frame seen during the RL phase before ACO.",
    )
    parser.add_argument(
        "--use-aco", dest="use_aco", type=str2bool, nargs="?", const=True, default=True,
        help="Compatibility flag, retained for log parsers (always True in this build).",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = read_graph_from_file(args.graph)

    register(
        id="GcpEnvMaxIters-v0",
        entry_point="gcp_env:GcpEnv",
        max_episode_steps=args.max_steps,
    )
    spec = gym.spec("GcpEnvMaxIters-v0")

    env = gym.make(
        spec,
        graph=graph,
        k=args.k,
        beta=args.beta,
        aco_num_ants=args.aco_ants,
        aco_max_iter=args.aco_iter,
        inject_mode=args.inject_mode,
        keep_rl_best=args.keep_rl_best,
    )
    vector_env = DummyVectorEnv([lambda: env])

    node_features = 3
    col_features = 3

    if args.RL:
        actor = ActorNetwork(node_features, col_features, device=device).to(device)
        critic = CriticNetwork(node_features, col_features, device=device).to(device)
        actor_critic = ActorCritic(actor, critic).to(device)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

        dist_fn = torch.distributions.Categorical
        policy = GCPPPOPolicy(
            actor, critic, optim, dist_fn,
            k=args.k, nodes=len(graph),
            action_space=env.action_space,
        ).to(device)
        policy.load_state_dict(torch.load(args.policy, map_location=device))
    else:
        policy = RandomGCPPolicy(action_space=env.action_space)

    policy.eval()
    eval_collector = Collector(policy, vector_env)
    eval_collector.reset()

    if args.RL and args.inject_mode == "soft":
        actor_for_prior = policy.actor

        def _soft_prior_fn(obs):
            with torch.no_grad():
                col_feats = torch.as_tensor(
                    np.expand_dims(obs["col_features"], 0),
                    device=device, dtype=torch.float32,
                )
                col_probs = torch.squeeze(actor_for_prior.col_model(col_feats), -1)
                return col_probs[0].cpu().numpy()

        env.env.set_soft_prior_fn(_soft_prior_fn)

    if args.RL:
        mode_label = f"RLACO ({args.inject_mode})"
    else:
        mode_label = f"RLACO0 (random + ACO {args.inject_mode})"

    print("-------------------------")
    print("Starting evaluation, info:")
    print(f"Mode: {mode_label}")
    print(f"Device: {device}")
    print(f"Policy: {args.policy if args.RL else '(random policy)'}")
    print(f"Graph: {args.graph}")
    print(f"k: {args.k}")
    print(f"Beta: {args.beta}")
    print(f"Max RL steps: {args.max_steps}")
    print(f"Max episodes: {args.episodes}")
    print(f"Time limit: {args.time_limit}")
    print(f"ACO ants: {args.aco_ants}")
    print(f"ACO iterations: {args.aco_iter}")
    print("-------------------------")
    sys.stdout.flush()

    episodes = 0
    tot_start = time.time()
    best_score = float("inf")
    episode_traces = []

    while True:
        if args.episodes is not None and episodes >= args.episodes:
            break
        if time.time() - tot_start > args.time_limit:
            break

        episodes += 1
        start_time = time.time()
        result = eval_collector.collect(n_episode=1)
        end_time = time.time()

        graph = env.env.get_graph()
        solution = env.env.get_solution()
        current_score = calculate_score(graph, solution)

        rl_start_score = getattr(env.env, "last_episode_rl_start_score", None)
        rl_end_score = getattr(env.env, "last_episode_rl_end_score", None)
        rl_actual_end_score = getattr(env.env, "last_episode_rl_actual_end_score", None)
        rl_best_step = getattr(env.env, "last_episode_rl_best_step", None)
        aco_seed_score = getattr(env.env, "last_episode_aco_seed_score", None)
        aco_end_score = getattr(env.env, "last_episode_aco_end_score", None)
        aco_iter_x, aco_iter_y = env.env.get_last_aco_curve()

        if current_score < best_score:
            best_score = current_score

        episode_traces.append({
            "episode": episodes,
            "t_start": start_time - tot_start,
            "t_end": end_time - tot_start,
            "rl_start_score": rl_start_score,
            "rl_end_score": rl_end_score,
            "rl_actual_end_score": rl_actual_end_score,
            "rl_best_step": rl_best_step,
            "aco_seed_score": aco_seed_score,
            "aco_end_score": aco_end_score,
            "final_score": current_score,
            "running_best": best_score,
            "aco_iter_x": aco_iter_x,
            "aco_iter_y": aco_iter_y,
        })

        print(
            f"Episode {episodes} done, "
            f"score: {current_score}, "
            f"RL_start: {rl_start_score}, "
            f"RL_end: {rl_end_score}, "
            f"RL_actual_end: {rl_actual_end_score}, "
            f"RL_best_step: {rl_best_step}, "
            f"ACO_seed: {aco_seed_score}, "
            f"ACO_end: {aco_end_score}, "
            f"time: {end_time - start_time:.2f}s, "
            f"result: {result}",
            flush=True,
        )

        if current_score == 0:
            print(f"[Perfect solution found in episode {episodes}!]", flush=True)
            break

    tot_end = time.time()
    iterations = max(episodes * (args.max_steps + args.aco_ants * args.aco_iter), 1)

    print("[DONE]")
    json_result = {
        "mode": mode_label,
        "score": best_score,
        "episodes": episodes,
        "iterations": iterations,
        "time": tot_end - tot_start,
        "time_per_iteration": (tot_end - tot_start) / iterations,
        "solution": solution,
        "graph": args.graph,
        "k": args.k,
        "max_steps": args.max_steps,
        "policy": args.policy,
        "RL": args.RL,
        "use_aco": True,
        "inject_mode": args.inject_mode if args.RL else None,
        "aco_ants": args.aco_ants,
        "aco_iter": args.aco_iter,
        "last_rl_start_score": getattr(env.env, "last_episode_rl_start_score", None),
        "last_rl_end_score": getattr(env.env, "last_episode_rl_end_score", None),
        "last_aco_seed_score": getattr(env.env, "last_episode_aco_seed_score", None),
        "last_aco_end_score": getattr(env.env, "last_episode_aco_end_score", None),
        "episode_traces": episode_traces,
    }
    print(json.dumps(json_result, default=str), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
