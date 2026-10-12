# RLACO — RL-Guided Ant Colony Optimization for Graph Coloring

This project takes a PPO policy that was trained for graph coloring on
random `g(n, p)` graphs and uses it to bias an MMAS-style Ant Colony
Optimization (ACO) solver on hard DIMACS instances. The project compares:

- **RLACO (soft)** — trained policy + ACO with **soft pheromone injection**
  (the policy's per-vertex color distribution shapes pheromone).
- **RLACO0** — uniform-random policy + ACO with **hard injection**
  (the random solution's argmax color is used). Baseline.

Both methods use the same ACO loop, ants count, and time budget — they
differ only in the prior fed into the pheromone matrix.

## Layout

```
RLACO_project/
├── runner.py            # Single-run inference entry
├── batch_runner.py      # Spawns N parallel runner.py processes
├── network.py           # Actor / Critic / GCPPPOPolicy
├── policy.pt            # Trained PPO weights
├── gcp_env/
│   ├── __init__.py
│   ├── gcp_env.py       # Gymnasium env (RL phase + ACO phase per episode)
│   └── ACO.py           # MMAS-style ACO with bias_from_solution / bias_from_distribution
├── instances/           # DIMACS .col benchmark graphs
├── analysis/            # Convergence / plateau analysis scripts
├── results/             # Experimental logs (10 runs × 6 (instance × method))
└── REPORT.md            # Project report
```

## Setup

Python 3.10–3.12 is recommended. Create a virtual env and install
dependencies:

```bash
python -m venv venv
. venv/Scripts/activate         # PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU is not required — inference is fast on CPU.

## Single run

```bash
# RLACO (soft injection) on DSJC250.5 at k=28, 1-hour budget
python runner.py policy.pt instances/DSJC250.5.col 28 \
    --RL True --inject-mode soft --time-limit 3600 \
    --aco-ants 40 --aco-iter 50

# RLACO0 baseline (random policy + hard injection)
python runner.py policy.pt instances/DSJC250.5.col 28 \
    --RL False --inject-mode hard --time-limit 3600 \
    --aco-ants 40 --aco-iter 50
```

The output ends with `[DONE]` followed by a JSON record containing the
best score, episode traces, ACO inner curves, and the final coloring.

## Batch run

`batch_runner.py` launches `--num-jobs` independent runs of `runner.py`,
running `--concurrency` of them at a time. Each job's output goes to a
timestamped `.txt` under `--output_dir`.

```bash
python batch_runner.py policy.pt instances/DSJC250.5.col 28 \
    main_experiment/DSJC250.5_k28_rlaco_soft \
    --RL True --inject-mode soft \
    --time-limit 14400 --num-jobs 10 --concurrency 2 \
    --aco-ants 40 --aco-iter 50
```

## Reproducing the experiment

The 6 batches (3 instances × 2 methods) used in `REPORT.md` were
launched with `time-limit 14400` (4 hours) and `num-jobs 10`. See
`results/` for the 10 logs per cell. Files in `results/<batch>/_contaminated/`
are runs whose wall-clock was distorted by host-side power events; they
were discarded and rerun (final analysis uses only the 10 files in each
batch's main directory).

## Analysis

```bash
python analysis/analyze_convergence.py results/DSJC250.5_k28_rlaco_soft
python analysis/analyze_rlaco0_plateau.py results/DSJC250.5_k28_rlaco0
```
