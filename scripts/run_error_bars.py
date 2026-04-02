#!/usr/bin/env python3
"""Error bar experiments: sequential non-overlapping splits for basis regression.

Shuffle M_full trajectories once, then split into n_trials non-overlapping blocks
of M_sub each.  Run Oracle/RBF × {selftest, MLE, sinkhorn} on each block,
evaluate with KDE.  Requires M_full >= n_trials × M_sub.

Usage:
    python scripts/run_error_bars.py --data_dir ./data/M20000_dt0.001 \
        --models a --n_trials 10 --M_sub 2000

    # Subsample dt_obs=0.001 data to dt_obs=0.01 (every 10th observation):
    python scripts/run_error_bars.py --data_dir ./data/M20000_dt0.001 \
        --models a b lj morse --subsample_stride 10 \
        --results_tag B1_M20000_dt0.01

    python scripts/run_error_bars.py --models a --n_trials 3 --M_sub 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.config import (
    RESULTS_ROOT, DEFAULT_SIGMA,
    load_experiment_data, get_dt, compute_r_max,
)
from lib.basis import get_basis
from lib.solvers import solve_mle, solve_selftest, solve_sinkhorn
from lib.eval import evaluate_kde, _resolve_kde_path, precompute_kde, save_kde, load_kde


def _load_data_from_dir(data_dir, model_name, labeled=False):
    """Load data from an explicit directory."""
    data_dir = Path(data_dir) / model_name
    fname = 'data_labeled.npy' if labeled else 'data_unlabeled.npy'
    data = np.load(str(data_dir / fname))
    t_obs = np.load(str(data_dir / 't_obs.npy'))
    with open(data_dir / 'config.json') as f:
        config = json.load(f)
    return data, t_obs, config


def run_one_trial(model_name, d, data_unlabeled, data_labeled, t_obs, config,
                  M_sub, method, basis_type, sigma, kde_grids, trial_idx,
                  n_trials, reg='auto'):
    """Run one trial: sequential non-overlapping block from full dataset.

    Block i uses indices [i*M_sub : (i+1)*M_sub].  Data is pre-shuffled
    once so blocks are i.i.d. without overlap.
    """
    start = trial_idx * M_sub
    end = start + M_sub
    indices = slice(start, end)

    data_sub_unl = data_unlabeled[indices]
    r_max_V, r_max_Phi = compute_r_max(data_sub_unl)

    build_V, build_Phi, kv, kp, basis_info = get_basis(
        basis_type, model_name,
        r_max_V=r_max_V, r_max_Phi=r_max_Phi,
    )

    t0 = time.time()
    if method == 'selftest':
        alpha, beta, info = solve_selftest(
            data_sub_unl, t_obs, sigma, build_V, build_Phi, kv, kp,
            M_max=M_sub, reg=reg,
        )
    elif method == 'mle':
        data_sub_lab = data_labeled[indices]
        alpha, beta, info = solve_mle(
            data_sub_lab, t_obs, build_V, build_Phi, kv, kp,
            M_max=M_sub, reg=reg,
        )
    elif method == 'sinkhorn':
        alpha, beta, info = solve_sinkhorn(
            data_sub_unl, t_obs, build_V, build_Phi, kv, kp,
            eps_factor=0.01, M_max=M_sub, reg=reg,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    elapsed = time.time() - t0

    # KDE evaluation (deterministic)
    v_err, phi_err = evaluate_kde(
        model_name, d, alpha, beta, build_V, build_Phi,
        kde_grids=kde_grids,
    )

    return {
        'trial_idx': int(trial_idx),
        'block': f'{trial_idx * M_sub}:{(trial_idx + 1) * M_sub}',
        'M_sub': M_sub,
        'V_error_pct': float(v_err * 100),
        'Phi_error_pct': float(phi_err * 100),
        'time_s': elapsed,
        'reg_effective': float(info.get('reg_effective', 0)),
        'reg_method': info.get('reg_method', 'unknown'),
        'alpha': alpha.tolist(),
        'beta': beta.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description='Error bar experiments (bootstrap)')
    p.add_argument('--models', nargs='+', default=['a', 'b', 'lj', 'morse'])
    p.add_argument('--methods', nargs='+', default=['selftest', 'mle', 'sinkhorn'],
                   choices=['selftest', 'mle', 'sinkhorn'])
    p.add_argument('--basis_types', nargs='+', default=['oracle'],
                   choices=['oracle', 'rbf'])
    p.add_argument('--d', type=int, default=2)
    p.add_argument('--n_trials', type=int, default=10)
    p.add_argument('--M_sub', type=int, default=2000)
    p.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    p.add_argument('--seed_base', type=int, default=42,
                   help='Seed for one-time shuffle before sequential split')
    p.add_argument('--data_M', type=int, default=None,
                   help='Load from ablation_M/M{data_M}/ instead of default dataset')
    p.add_argument('--data_dir', type=str, default=None,
                   help='Explicit data directory (overrides --data_M)')
    p.add_argument('--results_tag', type=str, default='error_bars',
                   help='Tag for results subdirectory')
    p.add_argument('--subsample_stride', type=int, default=1,
                   help='Subsample observations: stride=10 turns dt_obs=0.001→0.01')
    p.add_argument('--reg', type=str, default='auto',
                   help='Regularization: "auto" (Hansen L-curve) or float (e.g. "1e-4")')
    p.add_argument('--no_shuffle', action='store_true',
                   help='Skip shuffle: use sequential blocks [0:M_sub], [M_sub:2*M_sub], ...')
    args = p.parse_args()
    # Parse reg: 'auto' stays as string, else convert to float
    if args.reg != 'auto':
        args.reg = float(args.reg)

    all_results = {}
    results_tag = args.results_tag

    for model_suffix in args.models:
        model_name = f'model_{model_suffix}'
        print(f"\n{'='*70}")
        print(f"  Model: {model_name}")
        print(f"{'='*70}")

        # Load data
        try:
            if args.data_dir:
                data_unl, t_obs, config = _load_data_from_dir(
                    args.data_dir, model_name, labeled=False)
                data_lab, _, _ = _load_data_from_dir(
                    args.data_dir, model_name, labeled=True)
            else:
                data_unl, t_obs, config = load_experiment_data(
                    model_name, args.d, labeled=False, M=args.data_M)
                data_lab, _, _ = load_experiment_data(
                    model_name, args.d, labeled=True, M=args.data_M)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Subsample observations along time axis (e.g., stride=10: dt_obs 0.001→0.01)
        stride = args.subsample_stride
        if stride > 1:
            L_orig = data_unl.shape[1]
            data_unl = data_unl[:, ::stride]
            data_lab = data_lab[:, ::stride]
            t_obs = t_obs[::stride]
            L_new = data_unl.shape[1]
            dt_obs_new = config['dt_obs'] * stride
            config = dict(config)
            config['dt_obs'] = dt_obs_new
            config['L'] = L_new
            print(f"  Subsampled: stride={stride}, L={L_orig}→{L_new}, "
                  f"dt_obs={config['dt_obs']}")

        M_full = data_unl.shape[0]
        M_sub = args.M_sub
        n_trials = args.n_trials
        needed = n_trials * M_sub
        if needed > M_full:
            raise ValueError(
                f"Need {n_trials}×{M_sub}={needed} trajectories but "
                f"M_full={M_full}. Increase M_full or reduce n_trials/M_sub.")

        print(f"  Data shape: {data_unl.shape} (M_full={M_full}, M_sub={M_sub})")
        print(f"  Config: dt_obs={config.get('dt_obs')}, dt_fine={config.get('dt_fine')}, L={config.get('L')}")
        print(f"  Sequential split: {n_trials} non-overlapping blocks of {M_sub}")

        # Shuffle once so sequential blocks are i.i.d. (skip with --no_shuffle)
        if not args.no_shuffle:
            shuffle_rng = np.random.RandomState(args.seed_base)
            perm = shuffle_rng.permutation(M_full)
            data_unl = data_unl[perm]
            data_lab = data_lab[perm]
        else:
            print(f"  No shuffle: using sequential blocks")

        # Precompute KDE from full data
        kde_dir = RESULTS_ROOT / results_tag / 'kde' / f'd{args.d}' / model_name
        kde_path = kde_dir / 'kde_grids.npz'
        if kde_path.exists():
            print(f"  KDE grids found at {kde_path}")
            kde_grids = load_kde(str(kde_path))
        else:
            print(f"  Precomputing KDE from data...")
            kde_grids = precompute_kde(data_unl, model_name)
            kde_dir.mkdir(parents=True, exist_ok=True)
            save_kde(kde_grids, str(kde_path))
            print(f"  Saved KDE to {kde_path}")

        for basis_type in args.basis_types:
            for method in args.methods:
                label = f"{basis_type}/{method}"
                print(f"\n  {label}")
                print(f"  {'─'*50}")

                trial_results = []
                for trial in range(n_trials):
                    blk_start = trial * M_sub
                    blk_end = blk_start + M_sub
                    print(f"    Trial {trial+1}/{n_trials} "
                          f"[{blk_start}:{blk_end}] ...",
                          end=' ', flush=True)

                    result = run_one_trial(
                        model_name, args.d, data_unl, data_lab, t_obs, config,
                        M_sub, method, basis_type, args.sigma, kde_grids,
                        trial, n_trials, reg=args.reg,
                    )
                    trial_results.append(result)
                    print(f"V={result['V_error_pct']:.2f}% "
                          f"Phi={result['Phi_error_pct']:.2f}% "
                          f"({result['time_s']:.1f}s)")

                # Compute statistics
                v_errs = [r['V_error_pct'] for r in trial_results]
                phi_errs = [r['Phi_error_pct'] for r in trial_results]

                stats = {
                    'model': model_name,
                    'basis': basis_type,
                    'method': method,
                    'd': args.d,
                    'M_sub': M_sub,
                    'M_full': M_full,
                    'n_trials': n_trials,
                    'split': 'sequential_nonoverlap',
                    'shuffle_seed': args.seed_base,
                    'dt_obs': config.get('dt_obs'),
                    'dt_fine': config.get('dt_fine'),
                    'subsample_stride': stride if args.subsample_stride > 1 else 1,
                    'V_mean': float(np.mean(v_errs)),
                    'V_std': float(np.std(v_errs)),
                    'Phi_mean': float(np.mean(phi_errs)),
                    'Phi_std': float(np.std(phi_errs)),
                    'trials': trial_results,
                }

                key = f"{model_name}_{basis_type}_{method}"
                all_results[key] = stats

                print(f"\n    >>> V = {stats['V_mean']:.2f} +/- {stats['V_std']:.2f}%")
                print(f"    >>> Phi = {stats['Phi_mean']:.2f} +/- {stats['Phi_std']:.2f}%")

                # Save per-model-method results
                rdir = RESULTS_ROOT / results_tag / basis_type / f'd{args.d}' / model_name / method
                rdir.mkdir(parents=True, exist_ok=True)
                with open(rdir / 'results.json', 'w') as f:
                    json.dump(stats, f, indent=2)

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"ERROR BAR SUMMARY  (M_sub={args.M_sub}, {args.n_trials} non-overlapping blocks, KDE eval)")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Basis':<8} {'Method':<10} {'V% (mean±std)':>20} {'Phi% (mean±std)':>20}")
    print('─' * 75)
    for key in sorted(all_results):
        s = all_results[key]
        v_str = f"{s['V_mean']:.2f} +/- {s['V_std']:.2f}"
        p_str = f"{s['Phi_mean']:.2f} +/- {s['Phi_std']:.2f}"
        print(f"{s['model']:<12} {s['basis']:<8} {s['method']:<10} {v_str:>20} {p_str:>20}")

    # Save aggregated
    out_path = RESULTS_ROOT / results_tag / 'summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
