#!/usr/bin/env python3
"""T5 — Condition number table for paper.

Computes the selftest A matrix eigenvalue spectrum at oracle parameters
from REAL experimental data files. Output can be pasted directly into
the paper as empirical support for Proposition (Coercivity condition).

Run from repo root:
    python scripts/check_condition_number.py
"""

import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.basis import get_basis
from lib.config import DATA_ROOT

# N-scaling data paths (d=2, dt_obs=0.01, model_e)
N_SCALING_PATHS = {
    5:   DATA_ROOT / 'N_scaling' / 'N5'  / 'model_e',
    10:  DATA_ROOT / 'dt_obs_0.01' / 'model_e',
    20:  DATA_ROOT / 'N_scaling' / 'N20' / 'model_e',
    50:  DATA_ROOT / 'N_scaling' / 'N50' / 'model_e',
    100: DATA_ROOT / 'N100' / 'model_e',
}

# Dimension-scaling data paths (N=10, dt_obs=0.001, model_e)
D_SCALING_PATHS = {
    2:  DATA_ROOT / 'dt_obs_0.001' / 'model_e',
    5:  DATA_ROOT / 'dt_obs_0.001' / 'd5' / 'model_e',
    10: DATA_ROOT / 'dt_obs_0.001' / 'd10' / 'model_e',
    20: DATA_ROOT / 'dt_obs_0.001' / 'd20' / 'model_e',
}

# All-models condition number (d=2, dt_obs=0.01)
MODEL_PATHS = {
    'model_e':     DATA_ROOT / 'dt_obs_0.01' / 'model_e',
    'model_a':     DATA_ROOT / 'dt_obs_0.01' / 'model_a',
    'model_b':     DATA_ROOT / 'dt_obs_0.01' / 'model_b',
    'model_lj':    DATA_ROOT / 'dt_obs_0.01' / 'model_lj',
    'model_morse': DATA_ROOT / 'dt_obs_0.01' / 'model_morse',
}


def load_data(data_dir, M_use=2000, L_use=10):
    """Load trajectory data from experimental data files.

    Returns (data, t_obs) where data has shape (M, L, N, d).
    Subsamples to at most M_use ensembles and L_use time steps
    (evenly spaced) to keep computation tractable.
    """
    data_dir = Path(data_dir)
    data = np.load(data_dir / 'data_labeled.npy')  # (M, L, N, d)
    t_obs = np.load(data_dir / 't_obs.npy')
    # Subsample ensembles
    if data.shape[0] > M_use:
        data = data[:M_use]
    # Subsample time steps (evenly spaced)
    L_total = data.shape[1]
    if L_total > L_use:
        idx = np.linspace(0, L_total - 1, L_use, dtype=int)
        data = data[:, idx]
        t_obs = t_obs[idx]
    return data, t_obs


def build_A(data, t_obs, build_V_fn, build_Phi_fn, K_V, K_Phi):
    M, L, N, d = data.shape
    K_total = K_V + K_Phi
    mask = 1.0 - np.eye(N)
    A = np.zeros((K_total, K_total))

    for m in range(M):
        for ell in range(L - 1):
            dt = t_obs[ell + 1] - t_obs[ell]
            X_curr = data[m, ell]

            psi, psi_grad, _, r_x = build_V_fn(X_curr)
            r_x_safe = np.maximum(r_x, 1e-10)
            x_hat = X_curr / r_x_safe[:, None]
            grad_V_coeff = psi_grad[:, None, :] * x_hat[:, :, None]

            diff = X_curr[:, None, :] - X_curr[None, :, :]
            r_pairs_mat = np.sqrt(np.sum(diff**2, axis=-1))
            r_pairs_flat = r_pairs_mat.reshape(-1)

            _, phi_grad_flat, _ = build_Phi_fn(r_pairs_flat, d)
            phi_grad_mat = phi_grad_flat.reshape(N, N, K_Phi)
            r_pairs_safe = np.maximum(r_pairs_mat, 1e-10)
            unit_diff = diff / r_pairs_safe[:, :, None]

            gP = unit_diff[:, :, :, None] * phi_grad_mat[:, :, None, :]
            gP *= mask[:, :, None, None]
            grad_Phi_coeff = gP.sum(axis=1) / N

            F = np.concatenate([grad_V_coeff, grad_Phi_coeff], axis=-1)
            A += (dt / N) * np.einsum('ndk,ndl->kl', F, F)

    return A / (M * (L - 1))


def compute_block_cond(A, K_V, K_Phi):
    """Compute condition numbers of A_VV and A_PhiPhi diagonal blocks."""
    A_VV = A[:K_V, :K_V]
    A_PP = A[K_V:, K_V:]
    eigvals_VV = np.linalg.eigvalsh(A_VV)
    eigvals_PP = np.linalg.eigvalsh(A_PP)
    cond_VV = eigvals_VV[-1] / max(eigvals_VV[0], 1e-20)
    cond_PP = eigvals_PP[-1] / max(eigvals_PP[0], 1e-20)
    return cond_VV, cond_PP


def compute_cond_from_data(model, data_dir, M_use=2000):
    """Compute condition number from real experimental data."""
    build_V, build_Phi, K_V, K_Phi, _ = get_basis('oracle', model)
    data, t_obs = load_data(data_dir, M_use=M_use)
    A = build_A(data, t_obs, build_V, build_Phi, K_V, K_Phi)
    eigvals = np.linalg.eigvalsh(A)
    lmin, lmax = eigvals[0], eigvals[-1]
    cond = lmax / max(lmin, 1e-20)
    cond_VV, cond_PP = compute_block_cond(A, K_V, K_Phi)
    M, L, N, d = data.shape
    return lmin, lmax, cond, cond_VV, cond_PP, N, d


def main():
    M_use = 2000

    # ── Part 1: N-scaling (model_e, d=2, dt_obs=0.01) ──────────────────────
    print("=" * 70)
    print("N-SCALING — model_e (reference), d=2, dt_obs=0.01")
    print(f"Using real experimental data, M={M_use}")
    print("=" * 70)
    print(f"{'N':>6}  {'λ_min':>12}  {'λ_max':>12}  {'κ':>10}  {'κ/N':>8}  {'κ(A_VV)':>10}  {'κ(A_ΦΦ)':>10}")
    print(f"{'─'*55}")

    N_values = sorted(N_SCALING_PATHS.keys())
    conds_n = []
    conds_VV = []
    conds_PP = []
    for N in N_values:
        path = N_SCALING_PATHS[N]
        if not path.exists():
            print(f"{N:>6}  (data not found: {path})")
            continue
        lmin, lmax, cond, cond_VV, cond_PP, _, _ = compute_cond_from_data('model_e', path, M_use)
        conds_n.append(cond)
        conds_VV.append(cond_VV)
        conds_PP.append(cond_PP)
        print(f"{N:>6}  {lmin:>12.4e}  {lmax:>12.4e}  {cond:>10.1f}  {cond/N:>8.2f}  {cond_VV:>10.1f}  {cond_PP:>10.1f}")

    if len(conds_n) == len(N_values):
        log_N = np.log(np.array(N_values))
        log_cond = np.log(np.array(conds_n))
        slope, _ = np.polyfit(log_N, log_cond, 1)
        cond_over_N = np.array(conds_n) / np.array(N_values)
        print(f"\n  log-log slope: {slope:.3f}  (theory: ≤ 1)")
        print(f"  κ/N (N≥10): mean={np.mean(cond_over_N[1:]):.2f}")
        log_cond_VV = np.log(np.array(conds_VV))
        log_cond_PP = np.log(np.array(conds_PP))
        slope_VV, _ = np.polyfit(log_N, log_cond_VV, 1)
        slope_PP, _ = np.polyfit(log_N, log_cond_PP, 1)
        print(f"  κ(A_VV) slope: {slope_VV:.3f}")
        print(f"  κ(A_ΦΦ) slope: {slope_PP:.3f}")

    # ── Part 2: All models (d=2, dt_obs=0.01, N=10) ────────────────────────
    print("\n\n" + "=" * 70)
    print("ALL MODELS — d=2, dt_obs=0.01, N=10 (Table 6 data)")
    print("=" * 70)
    print(f"{'Model':>12}  {'λ_min':>12}  {'λ_max':>12}  {'κ':>10}  {'κ(A_VV)':>10}  {'κ(A_ΦΦ)':>10}")
    print(f"{'─'*50}")

    for model, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"{model:>12}  (data not found)")
            continue
        lmin, lmax, cond, cond_VV, cond_PP, N, d = compute_cond_from_data(model, path, M_use)
        print(f"{model:>12}  {lmin:>12.4e}  {lmax:>12.4e}  {cond:>10.1f}  {cond_VV:>10.1f}  {cond_PP:>10.1f}  (N={N})")

    # ── Part 3: Dimension scaling (model_e, N=10, dt_obs=0.001) ─────────────
    print("\n\n" + "=" * 70)
    print("DIMENSION SCAN — model_e, N=10, dt_obs=0.001")
    print("=" * 70)
    print(f"{'d':>4}  {'λ_min':>12}  {'λ_max':>12}  {'κ':>10}  {'κ/N':>8}  {'κ(A_VV)':>10}  {'κ(A_ΦΦ)':>10}")
    print(f"{'─'*50}")

    for d_val, path in sorted(D_SCALING_PATHS.items()):
        if not path.exists():
            print(f"{d_val:>4}  (data not found: {path})")
            continue
        lmin, lmax, cond, cond_VV, cond_PP, N, d = compute_cond_from_data('model_e', path, M_use)
        print(f"{d_val:>4}  {lmin:>12.4e}  {lmax:>12.4e}  {cond:>10.1f}  {cond/N:>8.2f}  {cond_VV:>10.1f}  {cond_PP:>10.1f}  (N={N})")


if __name__ == '__main__':
    main()
