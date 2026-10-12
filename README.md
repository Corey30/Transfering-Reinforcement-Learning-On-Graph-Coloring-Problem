# RLTabucol Transfer

Reproducible experiments for reinforcement-learning-guided graph coloring. The project evaluates a PPO recoloring policy with Tabucol, transfers the learned policy to ant colony optimization (RLACO), and evaluates a separate RLACO-to-terminal-Tabucol pipeline.

## Experiments

| Component | Purpose | Main entry point |
|---|---|---|
| [`RLTCol-main`](RLTCol-main/) | Train and evaluate RLTabucol, plus RLTabucol/Tabucol/ACO controls | `RLTCol-main/src/trainer.py`, `RLTCol-main/src/runner.py` |
| [`RLACO_transfer_test`](RLACO_transfer_test/) | Transfer a frozen RLTabucol policy into ACO with soft pheromone injection | `RLACO_transfer_test/runner.py`, `batch_runner.py` |
| [`RLACO_Terminal_Tabucol`](RLACO_Terminal_Tabucol/) | Run the complete RLACO stage, then terminal Tabucol refinement | `experiments/rlaco_terminal_tabucol/run_campaign.py` |
| [`All_results`](All_results/) | Archived logs and summaries from all experiment families | Data only; no execution required |

The objective is the graph-coloring conflict count: the number of edges whose endpoints have the same color. A solution with score `0` is feasible. The archived results are observations from finite budgets; they do not by themselves prove a graph's chromatic number.

## Repository layout

```text
RLTabucol_transfer/
├── RLTCol-main/                 # Source RLTabucol training and evaluation
├── RLACO_transfer_test/         # Standalone policy-to-ACO transfer
├── RLACO_Terminal_Tabucol/      # RLACO followed by terminal Tabucol
├── All_results/                 # Archived experimental records
├── requirements.txt             # Common Python dependencies for RLTabucol/RLACO
└── .gitignore


Generated environments, compiled Rust/PyO3 artifacts, node modules, caches, and new experiment outputs are ignored by Git. Existing checkpoints (policy.pt) and DIMACS .col instances are kept so the supplied evaluations can run immediately after setup.
Requirements
- Python 3.10–3.13.
- Git.
- Rust and Cargo.
- MSVC C++ Build Tools on Windows when building the native Tabucol extension.
- A GPU is optional; all runners have CPU paths.
The components intentionally use separate virtual environments because their historical Tianshou and Gymnasium versions differ. The root requirements.txt covers the standalone transfer and source workflows. The terminal experiment has its own pinned file:
RLACO_Terminal_Tabucol/experiments/rlaco_terminal_tabucol/requirements-experiment.txt
Quick start: standalone RLACO transfer
From the repository root:
cd RLACO_transfer_test
python -m venv .venv
Activate the environment:
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux or macOS
source .venv/bin/activate
Install dependencies:
python -m pip install -r requirements.txt
Run the learned-policy transfer on a bundled DIMACS instance:
python runner.py policy.pt ../RLTCol-main/src/DSJC250.5.col 28 \
  --RL True --inject-mode soft --time-limit 3600 \
  --aco-ants 40 --aco-iter 50
Run the random-policy control:
python runner.py policy.pt ../RLTCol-main/src/DSJC250.5.col 28 \
  --RL False --inject-mode hard --time-limit 3600 \
  --aco-ants 40 --aco-iter 50
Run ten independent experiments:
python batch_runner.py policy.pt ../RLTCol-main/src/DSJC250.5.col 28 \
  results/DSJC250.5_k28_soft \
  --RL True \
  --inject-mode soft \
  --time-limit 14400 \
  --num-jobs 10 \
  --concurrency 2
Each completed run ends with [DONE] and a JSON summary. Keep the complete logs when comparing results.
Quick start: source RLTabucol
The source workflow uses the code and instances under RLTCol-main/src:
cd RLTCol-main/src
python -m venv .venv
python -m pip install -r ../../requirements.txt
Train a policy:
python trainer.py trained_policy.pt \
  --nodes 250 \
  --probability 0.5 \
  --colors 24 \
  --max-steps 300 \
  --tabucol-iters 5000 \
  --epochs 50
Evaluate the supplied checkpoint:
python runner.py policy.pt DSJC250.5.col 28 \
  --RL True \
  --time-limit 18000
Run the random-policy control:
python runner.py policy.pt DSJC250.5.col 28 \
  --RL False \
  --time-limit 18000
Run the actor-only diagnostic:
python runner.py policy.pt le450_15d.col 15 \
  --RL True \
  --disable-tabucol True \
  --max-steps 300 \
  --time-limit 18000
Run the conflict-prior control:
python runner.py policy.pt DSJC250.5.col 28 \
  --RL True \
  --use-conflict-prior \
  --time-limit 14400
Run the pure ACO baseline:
python runner.py policy.pt DSJC250.5.col 28 \
  --pure-aco \
  --time-limit 14400
The --RL False command is a random-policy control. It is not the pure-Tabucol baseline.
Quick start: RLACO + terminal Tabucol
Read the locked protocol in:
[`RLACO_Terminal_Tabucol/experiments/rlaco_terminal_tabucol/README.md`](RLACO_Terminal_Tabucol/experiments/rlaco_terminal_tabucol/README.md)
From RLACO_Terminal_Tabucol, install and build the native backend.
Windows PowerShell:
Set-ExecutionPolicy -Scope Process Bypass
.\experiments\rlaco_terminal_tabucol\setup_windows.ps1
.\.venv\Scripts\python.exe experiments\rlaco_terminal_tabucol\verify_install.py
Linux:
bash experiments/rlaco_terminal_tabucol/setup_linux.sh
./.venv/bin/python experiments/rlaco_terminal_tabucol/verify_install.py
Run the formal campaign:
.venv/bin/python experiments/rlaco_terminal_tabucol/run_campaign.py \
  --instances all \
  --num-seeds 10 \
  --concurrency 2 \
  --output-root experiments/rlaco_terminal_tabucol/results_formal
On Windows, replace .venv/bin/python with:
.venv\Scripts\python.exe
The formal campaign may run for several days. For a short integration check, use the smoke-test command documented in the component README. Smoke-test output is not benchmark data.
The campaign writes:
- protocol metadata;
- per-seed JSON and CSV summaries;
- RLACO logs;
- terminal Tabucol checkpoints;
- resumable intermediate results.
Reuse the same output directory to resume an interrupted campaign. Use a new output directory when changing the experiment protocol.
Reproducing All_results
All_results is a read-only archive grouped by experiment family. It includes:
- source RLTabucol comparisons;
- pure Tabucol controls;
- pure ACO controls;
- RLACO transfer comparisons;
- conflict-prior controls;
- actor-only diagnostics;
- terminal Tabucol campaigns.
To reproduce a particular result, identify the following information from the archived logs or protocol files:
- graph instance;
- number of colors;
- random seed;
- number of runs;
- time limit;
- iteration limit;
- checkpoint;
- injection mode.
Do not infer the complete protocol from a filename alone.
A fresh run will not necessarily be bit-for-bit identical. Search trajectories can change with random seeds, hardware, PyTorch versions, CUDA versions, and wall-clock scheduling.
For a defensible comparison, report the exact command, checkpoint hash, seed, dependency versions, and complete output logs.
Citation and license
No license has been declared for this repository yet. Add a LICENSE file before distributing the code for reuse.
If you publish results from this repository, cite the accompanying manuscript or report included in the release and preserve the archived protocol metadata.
Contributing
Keep experiment protocols and generated results separate from source changes.
When changing an entry point, dependency, checkpoint, or output format, update the corresponding component README.
Run the relevant smoke test before opening a pull request.
```