#!/usr/bin/env python3
"""Dimension-scaling table for Section 4.2.2 (Table 3, lower panel).

Computes for d=2,5,10 (model_e, N=10, dt_obs=0.001):
  - kappa(A_*), kappa/N, kappa(A_VV), kappa(A_PhiPhi)  [from build_A]
  - ST nabla_Phi error %  [from solve_selftest + evaluate_kde]

Usage:
    python scripts/run_d_scaling_table.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.basis import get_basis
from lib.solvers import solve_selftest
from lib.eval import evaluate_kde, precompute_kde
from lib.config import DEFAULT_SIGMA
from scripts.check_condition_number import (
    load_data, build_A, compute_block_cond, D_SCALING_PATHS,
)

MODEL = "model_e"
N_FIXED = 10
M_USE = 2000
D_VALUES = [2, 5, 10]


def run():
    results = []

    for d_val in D_VALUES:
        path = D_SCALING_PATHS[d_val]
        print(f"\n{'='*60}")
        print(f"  d={d_val}  —  {path}")
        print(f"{'='*60}")

        if not path.exists():
            print(f"  ERROR: data not found at {path}")
            continue

        # ── Condition numbers (from build_A) ──────────────────────
        build_V, build_Phi, K_V, K_Phi, _ = get_basis("oracle", MODEL)
        data_lab, t_obs = load_data(path, M_use=M_USE)
        A = build_A(data_lab, t_obs, build_V, build_Phi, K_V, K_Phi)

        eigvals = np.linalg.eigvalsh(A)
        kappa = eigvals[-1] / max(eigvals[0], 1e-20)
        cond_VV, cond_PP = compute_block_cond(A, K_V, K_Phi)

        print(f"  kappa(A_*) = {kappa:.1f},  kappa/N = {kappa/N_FIXED:.1f}")
        print(f"  kappa(A_VV) = {cond_VV:.1f},  kappa(A_PhiPhi) = {cond_PP:.1f}")

        # ── Self-test error ───────────────────────────────────────
        data_unl = np.load(str(path / "data_unlabeled.npy"))
        t_obs_full = np.load(str(path / "t_obs.npy"))
        M_actual = min(M_USE, data_unl.shape[0])

        with open(path / "config.json") as f:
            cfg = json.load(f)
        sigma = cfg.get("sigma", DEFAULT_SIGMA)
        d_actual = cfg["d"]

        # Precompute KDE grids
        kde_grids = precompute_kde(data_unl[:M_actual], MODEL, d_actual)

        t0 = time.time()
        alpha_st, beta_st, info_st = solve_selftest(
            data_unl[:M_actual], t_obs_full, sigma,
            build_V, build_Phi, K_V, K_Phi, M_max=M_actual,
        )
        v_err, p_err = evaluate_kde(
            MODEL, d_actual, alpha_st, beta_st, build_V, build_Phi,
            kde_grids=kde_grids,
        )
        elapsed = time.time() - t0
        st_phi_pct = p_err * 100
        st_v_pct = v_err * 100
        print(f"  ST: V={st_v_pct:.2f}%, Phi={st_phi_pct:.2f}% ({elapsed:.1f}s)")

        results.append({
            "d": d_val,
            "kappa": round(float(kappa), 1),
            "kappa_over_N": round(float(kappa / N_FIXED), 1),
            "kappa_VV": round(float(cond_VV), 1),
            "kappa_PP": round(float(cond_PP), 1),
            "ST_V_pct": round(st_v_pct, 2),
            "ST_Phi_pct": round(st_phi_pct, 2),
        })

    # ── Save results ──────────────────────────────────────────────
    out_dir = ROOT / "results" / "d_scaling_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ── Print LaTeX-ready rows ────────────────────────────────────
    print(f"\n{'='*60}")
    print("LaTeX rows (paste into Table 3 d-scaling panel):")
    print(f"{'='*60}")
    for r in results:
        print(
            f"$d{{=}}{r['d']}$  & ${r['kappa']:.0f}$  & ${r['kappa_over_N']:.1f}$ "
            f"& ${r['kappa_VV']:.1f}$ & ${r['kappa_PP']:.1f}$ "
            f"& ${r['ST_Phi_pct']:.2f}$ \\\\"
        )


if __name__ == "__main__":
    run()
