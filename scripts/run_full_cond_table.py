#!/usr/bin/env python3
"""Generate N×d condition number table for Section 4.2.2.

For each (N, d) combo:
  1. Check if data exists; if not, generate it
  2. Build self-test matrix A, compute eigenvalues
  3. Extract kappa(A_*), kappa(A_VV), kappa(A_PhiPhi)

Output: JSON + LaTeX-ready table rows.

Usage:
    python scripts/run_full_cond_table.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.basis import get_basis
from lib.config import DATA_ROOT
from scripts.check_condition_number import load_data, build_A, compute_block_cond

MODEL = 'model_e'
SIGMA = 1.0
M_GEN = 2000        # ensembles to generate
M_USE = 2000         # ensembles to use for condition number
DT_FINE = 0.0001
DT_OBS = 0.01
T = 1.0
L = 100

N_VALUES = [5, 10, 20, 50, 100]
D_VALUES = [2, 5, 10]

# Known data paths for d=2 N-scaling (existing, heterogeneous conventions)
D2_PATHS = {
    5:   DATA_ROOT / 'N_scaling' / 'N5'  / MODEL,
    10:  DATA_ROOT / 'dt_obs_0.01' / MODEL,
    20:  DATA_ROOT / 'N_scaling' / 'N20' / MODEL,
    50:  DATA_ROOT / 'N_scaling' / 'N50' / MODEL,
    100: DATA_ROOT / 'N100' / MODEL,
}


def get_data_path(N, d):
    """Return data path for (N, d). Uses existing paths for d=2."""
    if d == 2:
        return D2_PATHS.get(N)
    # New convention: N_scaling/d{d}_N{N}/model_e/
    return DATA_ROOT / 'N_scaling' / f'd{d}_N{N}' / MODEL


def generate_data(N, d, out_dir):
    """Generate IPS data using the parallel generator."""
    from generate_data_elephant6 import generate_data_parallel

    print(f"\n  Generating data: N={N}, d={d} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_unl, data_lab, t_obs, elapsed = generate_data_parallel(
        model_name=MODEL, N=N, d=d, M=M_GEN, T=T,
        dt_fine=DT_FINE, dt_obs=DT_OBS, sigma=SIGMA,
        seed=42, shuffle_labels=True,
    )

    np.save(out_dir / 'data_unlabeled.npy', data_unl)
    np.save(out_dir / 'data_labeled.npy', data_lab)
    np.save(out_dir / 't_obs.npy', t_obs)

    config = {
        'model': MODEL, 'N': N, 'd': d, 'M': M_GEN,
        'T': T, 'dt_fine': DT_FINE, 'dt_obs': DT_OBS,
        'sigma': SIGMA, 'seed': 42, 'L': len(t_obs),
        'n_fine_steps': int(round(T / DT_FINE)),
        'shuffle_labels': True,
        'elapsed_seconds': elapsed,
        'data_shape': list(data_unl.shape),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Done in {elapsed:.1f}s  shape={data_unl.shape}")


def compute_cond(data_path, N, d):
    """Compute condition numbers and eigenvalue bounds from data."""
    build_V, build_Phi, K_V, K_Phi, _ = get_basis('oracle', MODEL)
    data, t_obs = load_data(data_path, M_use=M_USE)
    A = build_A(data, t_obs, build_V, build_Phi, K_V, K_Phi)
    eigvals = np.linalg.eigvalsh(A)
    lam_min = float(eigvals[0])
    lam_max = float(eigvals[-1])
    kappa = lam_max / max(lam_min, 1e-20)
    cond_VV, cond_PP = compute_block_cond(A, K_V, K_Phi)
    return float(kappa), float(cond_VV), float(cond_PP), lam_max, lam_min


def main():
    results = {}  # (N, d) -> {kappa, kappa_VV, kappa_PP}

    for d in D_VALUES:
        for N in N_VALUES:
            label = f"N={N}, d={d}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            data_path = get_data_path(N, d)
            if data_path is None:
                print(f"  ERROR: no path defined for {label}")
                continue

            # Generate if missing
            if not data_path.exists() or not (data_path / 'data_labeled.npy').exists():
                generate_data(N, d, data_path)

            # Compute condition numbers
            t0 = time.time()
            kappa, kappa_VV, kappa_PP, lam_max, lam_min = compute_cond(data_path, N, d)
            elapsed = time.time() - t0
            print(f"  kappa={kappa:.1f}  kappa_VV={kappa_VV:.1f}  kappa_PP={kappa_PP:.1f}  lam_max={lam_max:.4e}  ({elapsed:.1f}s)")

            results[(N, d)] = {
                'kappa': round(kappa, 1),
                'kappa_VV': round(kappa_VV, 1),
                'kappa_PP': round(kappa_PP, 1),
                'lam_max': lam_max,
                'lam_min': lam_min,
            }

    # ── Save JSON ─────────────────────────────────────────────────
    out_dir = ROOT / 'results' / 'full_cond_table'
    out_dir.mkdir(parents=True, exist_ok=True)
    json_results = [
        {'N': N, 'd': d, **results[(N, d)]}
        for d in D_VALUES for N in N_VALUES
        if (N, d) in results
    ]
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved JSON to {out_dir / 'results.json'}")

    # ── Print table ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FULL CONDITION NUMBER TABLE")
    print(f"{'='*80}")
    header = f"{'N':>5}"
    for d in D_VALUES:
        header += f"  | {'kappa':>8} {'kVV':>8} {'kPP':>8}"
    print(header)
    print(f"{'':>5}  | {'d=2':^26}  | {'d=5':^26}  | {'d=10':^26}")
    print("-" * 80)

    for N in N_VALUES:
        row = f"{N:>5}"
        for d in D_VALUES:
            r = results.get((N, d))
            if r:
                row += f"  | {r['kappa']:>8.1f} {r['kappa_VV']:>8.1f} {r['kappa_PP']:>8.1f}"
            else:
                row += f"  | {'---':>8} {'---':>8} {'---':>8}"
        print(row)

    # ── Print LaTeX rows ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print("LaTeX rows:")
    print(f"{'='*80}")
    for N in N_VALUES:
        parts = [f"{N:>3}"]
        for d in D_VALUES:
            r = results.get((N, d))
            if r:
                parts.append(f"${r['kappa']:.0f}$ & ${r['kappa_VV']:.1f}$ & ${r['kappa_PP']:.1f}$")
            else:
                parts.append(r"\multicolumn{3}{c}{---}")
        print(" & ".join(parts) + r" \\")


if __name__ == '__main__':
    main()
