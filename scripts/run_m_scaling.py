#!/usr/bin/env python3
"""M-scaling study: error vs number of trajectories (basis methods only).

For M < M_full, subsamples from the full dataset (no separate data generation).

Usage:
    python scripts/run_m_scaling.py --M_values 200 500 1000 2000 --models a b lj morse --d 2
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
    DATA_ROOT, RESULTS_ROOT, DEFAULT_SIGMA, DEFAULT_REG,
    load_experiment_data, get_dt, compute_r_max,
)
from lib.basis import get_basis, TRUE_PARAMS
from lib.solvers import solve_mle, solve_sinkhorn, solve_selftest
from lib.eval import evaluate_kde, load_kde, get_kde_path


METHODS = ['mle', 'sinkhorn', 'selftest']
BASIS_TYPES = ['oracle', 'rbf']


def _load_data(model_name, d, M_target, labeled=False, dt_obs=None, L_max=None):
    """Load data for M-scaling: try dt_obs-specific, ablation_M, then default with subsampling."""
    if dt_obs is not None:
        # Try dt_obs-specific directory first, then M20000 variant
        for loader in [
            lambda: load_experiment_data(model_name, d, labeled=labeled, dt_obs=dt_obs),
            lambda: _load_m20k_data(model_name, d, labeled=labeled, dt_obs=dt_obs),
        ]:
            try:
                data, t_obs, config = loader()
                M_full = data.shape[0]
                if M_target <= M_full:
                    data = data[:M_target]
                    if L_max is not None and data.shape[1] > L_max:
                        data = data[:, :L_max]
                        t_obs = t_obs[:L_max]
                    return data, t_obs, config
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"Need M={M_target} for {model_name} d={d} dt_obs={dt_obs}")
    try:
        data, t_obs, config = load_experiment_data(
            model_name, d, labeled=labeled, M=M_target)
        return data, t_obs, config
    except FileNotFoundError:
        pass
    # Fall back to default dataset with subsampling
    data, t_obs, config = load_experiment_data(model_name, d, labeled=labeled)
    M_full = data.shape[0]
    if M_target <= M_full:
        return data[:M_target], t_obs, config
    raise FileNotFoundError(
        f"Need M={M_target} but only M={M_full} available for {model_name} d={d}")


def _load_m20k_data(model_name, d, labeled=False, dt_obs=None):
    """Load from M20000_dt{dt_obs} directory."""
    dt_str = f'{dt_obs}'.rstrip('0').rstrip('.')  # 0.001 -> 0.001
    m20k_dir = DATA_ROOT / f'M20000_dt{dt_str}' / model_name
    if not m20k_dir.exists():
        raise FileNotFoundError(f"No M20000 data at {m20k_dir}")
    fname = 'data_labeled.npy' if labeled else 'data_unlabeled.npy'
    data = np.load(str(m20k_dir / fname))
    t_obs = np.load(str(m20k_dir / 't_obs.npy'))
    config = {}
    cfg_path = m20k_dir / 'config.json'
    if cfg_path.exists():
        import json as _json
        with open(cfg_path) as f:
            config = _json.load(f)
    return data, t_obs, config


def run_one_M(model_name, d, M_target, method, basis_type,
              K_V=20, K_Phi=20, reg=1e-4, sigma=1.0, eps_factor=0.01,
              dt_obs=None, L_max=None, quadrature='left'):
    """Run one (model, M, method, basis) experiment."""
    needs_labeled = (method == 'mle')
    try:
        data_sub, t_obs, config = _load_data(
            model_name, d, M_target, labeled=needs_labeled, dt_obs=dt_obs, L_max=L_max)
    except FileNotFoundError:
        return None

    M_use = data_sub.shape[0]
    dt = get_dt(config)

    r_max_V, r_max_Phi = compute_r_max(data_sub)
    build_V, build_Phi, kv, kp, basis_info = get_basis(
        basis_type, model_name,
        K_V=K_V, K_Phi=K_Phi,
        r_max_V=r_max_V, r_max_Phi=r_max_Phi,
    )

    if method == 'mle':
        alpha, beta, info = solve_mle(
            data_sub, t_obs, build_V, build_Phi, kv, kp,
            reg=reg, M_max=M_use,
        )
    elif method == 'sinkhorn':
        data_unl, t_obs_u, _ = _load_data(
            model_name, d, M_target, labeled=False, dt_obs=dt_obs, L_max=L_max)
        alpha, beta, info = solve_sinkhorn(
            data_unl, t_obs_u, build_V, build_Phi, kv, kp,
            reg=reg, eps_factor=eps_factor, M_max=M_use,
        )
        del data_unl
    elif method == 'selftest':
        data_unl, t_obs_u, _ = _load_data(
            model_name, d, M_target, labeled=False, dt_obs=dt_obs, L_max=L_max)
        alpha, beta, info = solve_selftest(
            data_unl, t_obs_u, sigma, build_V, build_Phi, kv, kp,
            reg=reg, M_max=M_use, quadrature=quadrature,
        )
        del data_unl
    else:
        raise ValueError(f"Unknown method: {method}")

    # Evaluate using precomputed KDE grids (deterministic, rho-weighted)
    v_err, phi_err = evaluate_kde(
        model_name, d, alpha, beta, build_V, build_Phi,
    )

    del data_sub

    result = {
        'model': model_name,
        'd': d,
        'M': M_use,
        'method': method,
        'basis': basis_type,
        'V_error_pct': float(v_err * 100),
        'Phi_error_pct': float(phi_err * 100),
        'reg_effective': float(info.get('reg_effective', 0)),
        'reg_method': info.get('reg_method', 'unknown'),
        'time_s': info['time_s'],
        'alpha': alpha.tolist(),
        'beta': beta.tolist(),
    }
    if 'lcurve' in info:
        result['lcurve'] = info['lcurve']
    return result


def main():
    p = argparse.ArgumentParser(description='M-scaling experiment')
    p.add_argument('--M_values', nargs='+', type=int, default=[200, 500, 1000, 2000])
    p.add_argument('--models', nargs='+', default=['a', 'b', 'lj', 'morse'])
    p.add_argument('--d', type=int, default=2)
    p.add_argument('--basis_types', nargs='+', default=BASIS_TYPES, choices=BASIS_TYPES)
    p.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    p.add_argument('--K_V', type=int, default=20)
    p.add_argument('--K_Phi', type=int, default=20)
    p.add_argument('--reg', type=float, default=1e-4)
    p.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    p.add_argument('--eps_factor', type=float, default=0.01)
    p.add_argument('--dt_obs', type=float, default=None,
                   help='Load data from dt_obs-specific directory (e.g. 0.001)')
    p.add_argument('--L_max', type=int, default=None,
                   help='Truncate trajectories to L_max snapshots (speed up large L)')
    p.add_argument('--eval_method', choices=['mc', 'kde'], default='kde',
                   help='Evaluation method (default: kde)')
    p.add_argument('--quadrature', choices=['left', 'trapezoid'], default='left',
                   help='Quadrature rule for selftest (default: left)')
    args = p.parse_args()

    # Output directory prefix
    dt_prefix = f'dt_obs_{args.dt_obs}' if args.dt_obs else ''

    all_results = {}

    for model_suffix in args.models:
        model_name = f'model_{model_suffix}'
        for M_val in args.M_values:
            for basis_type in args.basis_types:
                for method in args.methods:
                    key = f"{model_name}_M{M_val}_{basis_type}_{method}"
                    print(f"\n{'=' * 60}")
                    print(f"  M={M_val} | {basis_type} | {model_name} | {method}"
                          f" | dt_obs={args.dt_obs or 'default'}")
                    print(f"{'=' * 60}")

                    result = run_one_M(
                        model_name, args.d, M_val, method, basis_type,
                        K_V=args.K_V, K_Phi=args.K_Phi,
                        reg=args.reg, sigma=args.sigma,
                        eps_factor=args.eps_factor,
                        dt_obs=args.dt_obs, L_max=args.L_max,
                        quadrature=args.quadrature,
                    )
                    if result:
                        all_results[key] = result
                        print(f"  V={result['V_error_pct']:.2f}% "
                              f"Phi={result['Phi_error_pct']:.2f}% "
                              f"({result['time_s']:.1f}s)")

                        # Save individual result
                        if dt_prefix:
                            rdir = (RESULTS_ROOT / 'm_scaling' / dt_prefix
                                    / f'M{M_val}' / basis_type / model_name / method)
                        else:
                            rdir = (RESULTS_ROOT / 'm_scaling' / f'M{M_val}'
                                    / basis_type / model_name / method)
                        rdir.mkdir(parents=True, exist_ok=True)
                        metrics_name = 'metrics_kde.json' if args.eval_method == 'kde' else 'metrics.json'
                        with open(rdir / metrics_name, 'w') as f:
                            json.dump(result, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("M-SCALING SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<12} {'M':>5} {'Basis':<8} {'Method':<10} {'V%':>8} {'Phi%':>8}")
    print('-' * 55)
    for key in sorted(all_results):
        r = all_results[key]
        print(f"{r['model']:<12} {r['M']:>5} {r['basis']:<8} {r['method']:<10} "
              f"{r['V_error_pct']:>8.2f} {r['Phi_error_pct']:>8.2f}")

    # Save aggregated
    out_path = RESULTS_ROOT / 'm_scaling_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
