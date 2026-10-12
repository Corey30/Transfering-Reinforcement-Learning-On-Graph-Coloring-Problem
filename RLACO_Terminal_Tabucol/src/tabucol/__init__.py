import importlib.util
import os
import random
from pathlib import Path

import numpy as np


def _load_native_solver():
    if os.environ.get("RLTABUCOL_FORCE_PYTHON") == "1":
        return None, "python-fallback-forced"
    package_dir = Path(__file__).resolve().parent / "python" / "tabucol"
    candidates = sorted(package_dir.glob("tabucol*.pyd")) + sorted(
        package_dir.glob("tabucol*.so")
    )
    for candidate in candidates:
        try:
            spec = importlib.util.spec_from_file_location("tabucol.tabucol", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.TabuColSolver, str(candidate)
        except (ImportError, OSError, AttributeError):
            continue
    return None, "python-fallback"


_NATIVE_SOLVER, BACKEND_ORIGIN = _load_native_solver()
IS_NATIVE = _NATIVE_SOLVER is not None


class _PythonTabuColSolver:
    """Python fallback for the Rust/PyO3 TabuColSolver.

    The original project expects a compiled `tabucol` extension. The checked-in
    binary is CPython 3.13-specific, so this fallback keeps the same public API
    usable on the current Python runtime.
    """

    def __init__(
        self,
        graph,
        k,
        max_iterations=10000,
        tabu_a=10,
        tabu_alpha=1.2,
        beta=0.2,
        seed=None,
    ):
        self.graph = [list(neighbors) for neighbors in graph]
        self.k = int(k)
        self.n = len(self.graph)
        self.max_iters = int(max_iterations)
        self.tabu_a = int(tabu_a)
        self.tabu_alpha = float(tabu_alpha)
        self.beta = float(beta)
        self.rng = random.Random(seed)

        self.solution = np.random.randint(0, self.k, size=self.n, dtype=np.int32)
        self.best_solution = self.solution.copy()
        self.delta = np.zeros((self.n, self.k), dtype=np.int32)
        self.tabu = np.zeros((self.n, self.k), dtype=np.int64)
        self.node_conflicts = np.zeros(self.n, dtype=np.int32)
        self.score = 0
        self.best_score = 0
        self.iters = 0
        self.initialize()

    def initialize(self):
        self.delta.fill(0)
        self.node_conflicts.fill(0)

        score2 = 0
        for node in range(self.n):
            node_color = self.solution[node]
            conflicts = 0
            for neighbor in self.graph[node]:
                if self.solution[neighbor] == node_color:
                    conflicts += 1
            self.node_conflicts[node] = conflicts
            score2 += conflicts

            for col in range(self.k):
                change = 0
                for neighbor in self.graph[node]:
                    neighbor_color = self.solution[neighbor]
                    change += int(neighbor_color == col) - int(neighbor_color == node_color)
                self.delta[node, col] = change

        self.score = score2 // 2
        self.best_score = self.score
        self.best_solution = self.solution.copy()
        self.iters = 0
        self.tabu.fill(0)

    def is_tabu(self, node, col):
        return self.tabu[node, col] > self.iters

    def set_solution(self, solution):
        self.solution = np.asarray(solution, dtype=np.int32).copy()
        self.best_solution = self.solution.copy()
        self.initialize()

    def solve(self):
        initial_score = int(self.score)

        while self.iters < self.max_iters:
            conflicting_nodes = np.flatnonzero(self.node_conflicts > 0)
            if len(conflicting_nodes) == 0:
                break

            best_moves = []
            best_delta = None
            for node in conflicting_nodes:
                old_col = int(self.solution[node])
                for col in range(self.k):
                    if col == old_col:
                        continue
                    change = int(self.delta[node, col])
                    if self.is_tabu(node, col) and self.score + change >= self.best_score:
                        continue
                    if best_delta is None or change < best_delta:
                        best_delta = change
                        best_moves = [(int(node), int(col))]
                    elif change == best_delta:
                        best_moves.append((int(node), int(col)))

            if not best_moves:
                self.iters += 1
                continue

            move_node, new_col = self.rng.choice(best_moves)
            old_col = int(self.solution[move_node])

            for neighbor in self.graph[move_node]:
                neighbor_col = int(self.solution[neighbor])
                if neighbor_col != old_col:
                    self.delta[neighbor, old_col] -= 1
                if neighbor_col != new_col:
                    self.delta[neighbor, new_col] += 1
                if neighbor_col == new_col:
                    for col in range(self.k):
                        if col != new_col:
                            self.delta[neighbor, col] -= 1
                        self.delta[move_node, col] -= 1
                if neighbor_col == old_col:
                    for col in range(self.k):
                        if col != old_col:
                            self.delta[neighbor, col] += 1
                        self.delta[move_node, col] += 1

            tabu_time = self.rng.randrange(max(self.tabu_a, 1)) + int(
                round(self.tabu_alpha * self.score)
            )
            self.tabu[move_node, old_col] = self.iters + tabu_time

            new_node_conflicts = 0
            for neighbor in self.graph[move_node]:
                neighbor_col = int(self.solution[neighbor])
                if neighbor_col == old_col:
                    self.node_conflicts[neighbor] -= 1
                if neighbor_col == new_col:
                    self.node_conflicts[neighbor] += 1
                    new_node_conflicts += 1

            self.solution[move_node] = new_col
            self.node_conflicts[move_node] = new_node_conflicts
            self.score += int(best_delta)

            if self.score <= self.best_score:
                self.best_score = int(self.score)
                self.best_solution = self.solution.copy()

            if self.score == 0:
                break

            self.iters += 1

        if self.rng.random() < self.beta:
            return self.solution.astype(int).tolist(), initial_score - int(self.score)
        return self.best_solution.astype(int).tolist(), initial_score - int(self.best_score)

    def get_iters(self):
        return int(self.iters)

    def get_best_score(self):
        return int(self.best_score)

    def get_best_solution(self):
        return self.best_solution.astype(int).tolist()


TabuColSolver = _NATIVE_SOLVER if IS_NATIVE else _PythonTabuColSolver
