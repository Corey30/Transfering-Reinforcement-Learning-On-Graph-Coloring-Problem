"""
Gymnasium environment for fixed-k graph coloring with PPO + ACO.

Each episode:
    1. RL phase: the PPO actor proposes vertex/color updates for `max_episode_steps` steps.
    2. ACO phase (at episode end): an MMAS-style ant colony reconstructs a coloring
       from scratch, with pheromone biased by the RL policy.
       - hard injection: tau is boosted at the RL solution's argmax color per vertex.
       - soft injection: tau is interpolated using the policy's per-vertex color
         distribution (Sun et al. 2022, ML-ACO style).
"""

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from gcp_env.ACO import AntColony


class GcpEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(
        self,
        graph,
        k,
        beta=0.2,
        aco_num_ants=40,
        aco_max_iter=50,
        inject_mode="hard",          # "hard" or "soft"
        soft_prior_fn=None,          # callable: (obs_dict) -> np.ndarray (n, k)
        keep_rl_best=False,          # restore best RL frame before ACO
        # Compatibility kwargs (ignored in ACO-only build):
        tabucol_iters=None,
        render_mode=None,
        base_filename=None,
        tabucol_init=False,
        disable_tabucol=True,
        use_aco=True,
    ):
        self._graph = graph
        self._k = k
        self.k = k
        self._inject_mode = inject_mode
        self._soft_prior_fn = soft_prior_fn
        self._keep_rl_best = keep_rl_best

        n = len(self._graph)
        self._n = n

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0, high=n, shape=(n, 3), dtype=np.float32
                ),
                "col_features": spaces.Box(
                    low=0, high=n, shape=(n, k, 3), dtype=np.float32
                ),
                "k": spaces.Discrete(k, start=1),
            }
        )
        self.solution_space = spaces.Box(
            low=0, high=k - 1, shape=(n,), dtype=np.int32
        )
        self.action_space = spaces.MultiDiscrete([n, k])

        self._step_counter = 0

        self._aco_local = AntColony(graph, k)
        self._aco_num_ants = aco_num_ants
        self._aco_max_iter = aco_max_iter

        self._initial_score = 0
        self._rl_improvement = 0
        self._aco_improvement = 0

        self.graph = graph
        self.final_score = 0
        self.final_solution = None

        # Per-episode score tracking
        self.episode_rl_start_score = 0
        self.episode_rl_end_score = 0
        self.episode_rl_actual_end_score = 0
        self.episode_rl_best_step = 0
        self.episode_aco_seed_score = 0
        self.episode_aco_end_score = 0

        # Last-episode mirror (read by runner after collector.collect)
        self.last_episode_rl_start_score = 0
        self.last_episode_rl_end_score = 0
        self.last_episode_rl_actual_end_score = 0
        self.last_episode_rl_best_step = 0
        self.last_episode_aco_seed_score = 0
        self.last_episode_aco_end_score = 0
        self.last_episode_solution = None

        self._prev_solution = None
        self._solution = None
        self.render_mode = None

    # ------------------------------------------------------------------
    # External hooks
    # ------------------------------------------------------------------

    def set_soft_prior_fn(self, fn):
        """Inject a callable (obs_dict) -> np.ndarray of shape (n, k).
        Used only when inject_mode='soft'."""
        self._soft_prior_fn = fn

    def get_last_aco_curve(self):
        """Per-iteration best-conflict trace of the most recent ACO call."""
        return list(self._aco_local.iter_x), list(self._aco_local.iter_y)

    def get_graph(self):
        return self._graph

    def get_solution(self):
        return self._prev_solution

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.solution_space = spaces.Box(
            low=0, high=self._k - 1, shape=(self._n,), dtype=np.int32, seed=seed
        )
        self.action_space = spaces.MultiDiscrete([self._n, self._k], seed=seed)

        if options is not None and "initial_solution" in options:
            self._solution = options["initial_solution"]
            self._prev_solution = self._solution
        elif options is not None and options.get("random_init", False):
            self._solution = self.solution_space.sample()
            self._prev_solution = self._solution
        elif self._prev_solution is not None:
            self._solution = self._prev_solution
        else:
            self._solution = self.solution_space.sample()
            self._prev_solution = self._solution

        self._step_counter = 0
        self._initialize_obs()
        self.score = self._calculate_score()

        self._initial_score = self.score
        self._rl_improvement = 0
        self._aco_improvement = 0

        self.episode_rl_start_score = self.score
        self.episode_rl_end_score = self.score
        self.episode_rl_actual_end_score = self.score
        self.episode_rl_best_step = 0
        self.episode_aco_seed_score = self.score
        self.episode_aco_end_score = self.score

        # Track the best score frame seen during the RL phase.
        self._rl_best_score = self.score
        self._rl_best_solution = self._solution.copy()
        self._rl_best_step = 0

        return self._get_obs(), {"solution": self._solution}

    def step(self, action):
        action_node, new_group = action[0], action[1]
        prev_group = self._solution[action_node]
        reward = 0

        if prev_group != new_group:
            conflicts = self._node_feats[action_node, 1]
            self._solution[action_node] = new_group

            new_conflicts = 0
            neighbors_groups = set()
            for neighbor in self._graph.neighbors(action_node):
                if self._solution[action_node] == self._solution[neighbor]:
                    new_conflicts += 1
                neighbors_groups.add(self._solution[neighbor])

                self._col_feats[neighbor, prev_group, 0] -= 1
                self._col_feats[neighbor, new_group, 0] += 1

                if self._solution[neighbor] == prev_group:
                    self._node_feats[neighbor, 1] -= 1
                if self._solution[neighbor] == new_group:
                    self._node_feats[neighbor, 1] += 1
                if self._col_feats[neighbor, prev_group, 0] == 0:
                    self._node_feats[neighbor, 2] -= 1
                if self._col_feats[neighbor, new_group, 0] == 1:
                    self._node_feats[neighbor, 2] += 1

            self._node_feats[action_node, 1] = new_conflicts
            self._node_feats[action_node, 2] = len(neighbors_groups)
            self._col_feats[:, prev_group, 1] -= 1
            self._col_feats[:, new_group, 1] += 1

            reward = conflicts - new_conflicts
            self._rl_improvement += reward

        self.score -= reward

        if self.score < self._rl_best_score:
            self._rl_best_score = self.score
            self._rl_best_solution = self._solution.copy()
            self._rl_best_step = self._step_counter

        terminated = self.score == 0
        self._step_counter += 1
        observation = self._get_obs()

        if self._step_counter == self.spec.max_episode_steps:
            self.episode_rl_actual_end_score = self.score
            self.episode_rl_best_step = self._rl_best_step

            if self._keep_rl_best and self._rl_best_score < self.score:
                degraded_score = self.score
                self._solution = self._rl_best_solution.copy()
                self.score = self._rl_best_score
                self._initialize_obs()
                print(
                    f"[keep_rl_best] restored frame at step {self._rl_best_step}: "
                    f"score {degraded_score} -> {self._rl_best_score}",
                    flush=True,
                )

            score_before_aco = self.score
            print(
                f"[ACO local] start: seed_score={score_before_aco}, "
                f"ants={self._aco_num_ants}, iter={self._aco_max_iter}, "
                f"inject_mode={self._inject_mode}",
                flush=True,
            )
            self.episode_rl_end_score = score_before_aco

            if self._inject_mode == "soft" and self._soft_prior_fn is not None:
                dist = self._soft_prior_fn(self._get_obs())
                self._aco_local.bias_from_distribution(dist, solution=self._solution)
            else:
                self._aco_local.bias_from_solution(self._solution)

            self.episode_aco_seed_score = int(self._aco_local.seed_conflicts)
            aco_solution = self._aco_local.run(
                m=self._aco_num_ants,
                max_iter=self._aco_max_iter,
                verbose=False,
            )
            self._solution = aco_solution
            self._prev_solution = self._solution

            score_after_aco = self._calculate_score()
            self.score = score_after_aco
            self.episode_aco_end_score = score_after_aco
            self._aco_improvement = score_before_aco - score_after_aco
            print(
                f"[ACO local] done: before={score_before_aco}, "
                f"seed={self.episode_aco_seed_score}, "
                f"after={score_after_aco}, "
                f"improvement={self._aco_improvement}",
                flush=True,
            )

        if terminated or self._step_counter == self.spec.max_episode_steps:
            if terminated and self._step_counter < self.spec.max_episode_steps:
                self.episode_rl_end_score = self.score
                self.episode_aco_seed_score = self.score
                self.episode_aco_end_score = self.score

            self.final_score = self.score
            self.final_solution = self._solution.copy()

            self.last_episode_rl_start_score = self.episode_rl_start_score
            self.last_episode_rl_end_score = self.episode_rl_end_score
            self.last_episode_rl_actual_end_score = self.episode_rl_actual_end_score
            self.last_episode_rl_best_step = self.episode_rl_best_step
            self.last_episode_aco_seed_score = self.episode_aco_seed_score
            self.last_episode_aco_end_score = self.episode_aco_end_score
            self.last_episode_solution = self._solution.copy()

        info = {
            "solution": self._solution,
            "graph": self._graph,
            "initial_score": self._initial_score,
            "final_score": self.score,
            "rl_improvement": self._rl_improvement,
            "aco_improvement": self._aco_improvement,
            "rl_start_score": self.episode_rl_start_score,
            "rl_end_score": self.episode_rl_end_score,
            "rl_actual_end_score": self.episode_rl_actual_end_score,
            "rl_best_step": self.episode_rl_best_step,
            "aco_seed_score": self.episode_aco_seed_score,
            "aco_end_score": self.episode_aco_end_score,
            "local_search": "aco",
        }

        return observation, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self):
        return {
            "node_features": self._node_feats,
            "col_features": self._col_feats,
            "k": self._k,
        }

    def _initialize_obs(self):
        node_feats = np.zeros((self._n, 3), dtype=np.float32)
        col_feats = np.zeros((self._n, self._k, 3), dtype=np.float32)

        for node in self._graph.nodes():
            conflicts = 0
            groups = set()
            for adj in self._graph.neighbors(node):
                if self._solution[node] == self._solution[adj]:
                    conflicts += 1
                groups.add(self._solution[adj])
            node_feats[node, :] = [self._graph.degree(node), conflicts, len(groups)]

        nodes_with_color = np.zeros(self._k, dtype=np.int32)
        for col in self._solution:
            nodes_with_color[col] += 1

        for node in self._graph.nodes():
            for col in range(self._k):
                n_adjacent = 0
                for adj in self._graph.neighbors(node):
                    if self._solution[adj] == col:
                        n_adjacent += 1
                col_feats[node, col, :] = [
                    n_adjacent,
                    nodes_with_color[col],
                    self._n,
                ]

        self._node_feats = node_feats
        self._col_feats = col_feats

    def _calculate_score(self):
        score = 0
        for node in self._graph.nodes():
            for neighbor in self._graph.neighbors(node):
                if self._solution[node] == self._solution[neighbor]:
                    score += 1
        return score // 2
