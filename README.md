# LIPS: Learning Interacting Particle Systems from Unlabeled Snapshot Data

[[Project Page](https://viskawei.github.io/ips_unlabeled_learning_web/)]

Code for reproducing all experiments in the paper.

## Setup

```bash
pip install -e .
```

## Project Structure

```
lips_unlabeled_data/
├── config/              # Central configuration (base.yaml)
├── core/                # Core modules
│   ├── potentials.py    # Potential functions (V, Phi)
│   ├── sde_simulator.py # Euler-Maruyama SDE simulator
│   ├── nn_models.py     # Neural network architectures
│   └── selftest_loss.py # Self-test loss (Algorithm 1)
├── lib/                 # Shared utilities
│   ├── basis.py         # Oracle and RBF basis functions
│   ├── solvers.py       # MLE, Sinkhorn, Self-test solvers
│   ├── eval.py          # Evaluation API with KDE weighting
│   ├── config.py        # Data paths and experiment constants
│   └── plot.py          # Publication-quality figure style
└── scripts/             # Experiment and plotting scripts
    ├── generate_data_elephant6.py  # Data generation
    ├── run_nn.py                   # Neural network training
    ├── run_m_scaling*.py           # Sample size scaling
    ├── run_error_bars.py           # Error bar estimation
    ├── plot_*.py                   # Figure generation
    └── ...
```

## Data Generation

Generate synthetic IPS trajectory data:

```bash
python scripts/generate_data_elephant6.py --model a --d 2 --M 2000 --L 100
```

## Running Experiments

### Neural Network Training (Table 2, Figure 5)

```bash
python scripts/run_nn.py --models a b lj morse --dims 2 5 10 20 --epochs 200 --init random
```

### M-Scaling (Figure 2, Table 6)

```bash
python scripts/run_m_scaling.py --M_values 200 500 1000 2000 --models a b lj morse
```

### Error Bars

```bash
python scripts/run_error_bars.py --models a b lj morse --n_trials 10 --M_sub 2000
```

### Condition Numbers (Figure 4, Table 5)

```bash
python scripts/run_n_scaling_cond.py
python scripts/run_full_cond_table.py
```

## Plotting

All plotting scripts are in `scripts/plot_*.py`. They read from `results/` and save figures to the current directory or a specified output path.

```bash
python scripts/plot_m_scaling_zerogap.py
python scripts/plot_m_scaling_discrete.py
python scripts/plot_condition_numbers.py
python scripts/plot_boundary_recovery.py
python scripts/plot_all_methods_appendix.py
```

## Configuration

Edit `config/base.yaml` to change default parameters (data paths, models, dimensions, etc.).
