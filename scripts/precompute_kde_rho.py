#!/usr/bin/env python3
"""Precompute KDE-based evaluation grids for all models and dimensions.

Uses the largest available dataset to estimate empirical distributions
rho_V (particle radial distances) and rho_Phi (pairwise distances),
then creates fixed evaluation grids. All subsequent error evaluations
can use these grids for deterministic, reproducible results.

Usage:
    # All models, d=2 (default)
    python scripts/precompute_kde_rho.py

    # Specific models and dimensions
    python scripts/precompute_kde_rho.py --models a b --dims 2 5 10

    # With specific dt_obs
    python scripts/precompute_kde_rho.py --dt_obs 0.01

    # Custom data root (e.g. M=20000 data) with tagged output
    python scripts/precompute_kde_rho.py --models a --dims 2 5 10 20 --dt_obs 0.001 \
        --data_root ./data/M20000_dt0.001 --kde_tag M20000

Output:
    results/kde_grids/dt_obs_{val}/d{d}/{model}/kde_grids.npz
    (or results/kde_grids/{tag}_dt_obs_{val}/d{d}/{model}/kde_grids.npz with --kde_tag)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.config import MODELS, DIMS, RESULTS_ROOT, load_experiment_data
from lib.eval import precompute_kde, save_kde, get_kde_path, is_radial


def main():
    p = argparse.ArgumentParser(description='Precompute KDE evaluation grids')
    p.add_argument('--models', nargs='+', default=['a', 'b', 'lj', 'morse'],
                   help='Model suffixes (default: a b lj morse)')
    p.add_argument('--dims', nargs='+', type=int, default=[2],
                   help='Dimensions (default: 2)')
    p.add_argument('--dt_obs', type=float, default=None,
                   help='Observation dt (default: None = legacy)')
    p.add_argument('--data_root', type=str, default=None,
                   help='Override data root directory (for M=20000 etc.)')
    p.add_argument('--kde_tag', type=str, default=None,
                   help='Custom tag for KDE output dir (e.g. M20000)')
    args = p.parse_args()

    for d in args.dims:
        for model_suffix in args.models:
            model_name = f'model_{model_suffix}'
            print(f"\n{'=' * 60}")
            print(f"  KDE grids: {model_name}, d={d}" +
                  (f", dt_obs={args.dt_obs}" if args.dt_obs else ""))
            print(f"{'=' * 60}")

            # Load data
            t0 = time.time()
            if args.data_root:
                data_root = Path(args.data_root)
                if d == 2:
                    data_dir = data_root / model_name
                else:
                    data_dir = data_root / f'd{d}' / model_name
                try:
                    data = np.load(str(data_dir / 'data_unlabeled.npy'))
                    print(f"  Loaded from custom root: {data_dir}")
                except FileNotFoundError as e:
                    print(f"  SKIP: data not found — {e}")
                    continue
            else:
                try:
                    data, t_obs, config = load_experiment_data(
                        model_name, d, labeled=False, dt_obs=args.dt_obs)
                except FileNotFoundError as e:
                    print(f"  SKIP: data not found — {e}")
                    continue

            M, L, N, d_actual = data.shape
            print(f"  Data shape: M={M}, L={L}, N={N}, d={d_actual}")

            # Compute KDE grids (uses current API: data, model_name, d)
            kde_grids = precompute_kde(data, model_name, d=d_actual)
            elapsed = time.time() - t0

            # Save
            if args.kde_tag:
                tag_prefix = f"{args.kde_tag}_"
                kde_dir = RESULTS_ROOT / 'kde_grids' / f'{tag_prefix}dt_obs_{args.dt_obs}' / f'd{d}' / model_name
                out_path = kde_dir / 'kde_grids.npz'
            else:
                out_path = Path(get_kde_path(model_name, d, dt_obs=args.dt_obs))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_kde(kde_grids, str(out_path))

            print(f"  V grid: [{kde_grids['r_V_grid'][0]:.4f}, {kde_grids['r_V_grid'][-1]:.4f}] "
                  f"({len(kde_grids['r_V_grid'])} pts, {kde_grids['n_samples_V']} samples)")
            if is_radial(model_name) and 'r_Phi_grid' in kde_grids:
                print(f"  Phi grid: [{kde_grids['r_Phi_grid'][0]:.4f}, {kde_grids['r_Phi_grid'][-1]:.4f}] "
                      f"({len(kde_grids['r_Phi_grid'])} pts, {kde_grids['n_samples_Phi']} samples)")
            print(f"  Saved to: {out_path}")
            print(f"  Time: {elapsed:.1f}s")

            del data  # free memory

    print(f"\n{'=' * 60}")
    print("KDE grid precomputation complete.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
