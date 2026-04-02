#!/usr/bin/env python3
"""N-scaling + condition number for Section 4.2.2.

Uses standard-grid data (dt_fine=0.0001, dt_obs=0.01) at varying N.
Extracts eigenvalues of the self-test matrix A and runs oracle MLE+ST.

Usage:
    python scripts/run_n_scaling_cond.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib.basis import get_basis
from lib.solvers import solve_selftest, solve_mle
from lib.eval import evaluate_kde, precompute_kde
from lib.config import DEFAULT_SIGMA, DATA_ROOT


def _find_data(N, model_name="model_e"):
    """Find data for given N. Try N{N}/ then M20000_dt0.01_split/ for N=10."""
    candidates = [
        DATA_ROOT / f"N{N}" / model_name,
        DATA_ROOT / "M20000_dt0.01_split" / model_name,  # N=10 default
    ]
    for p in candidates:
        if (p / "config.json").exists():
            with open(p / "config.json") as f:
                cfg = json.load(f)
            if cfg["N"] == N:
                return p, cfg
    raise FileNotFoundError(f"No data for {model_name} N={N}")


def run():
    model = "model_e"
    N_values = [5, 10, 20, 50, 100]
    build_V, build_Phi, kv, kp, binfo = get_basis("oracle", model)

    results = []
    for N in N_values:
        print(f"\n{'='*60}")
        print(f"  N={N}")
        print(f"{'='*60}")

        data_dir, config = _find_data(N, model)
        d = config["d"]
        dt_obs = config["dt_obs"]
        print(f"  Data: {data_dir}")
        print(f"  dt_fine={config['dt_fine']}, dt_obs={dt_obs}, L={config['L']}, M={config['M']}")

        data_unl = np.load(str(data_dir / "data_unlabeled.npy"))
        data_lab = np.load(str(data_dir / "data_labeled.npy"))
        t_obs = np.load(str(data_dir / "t_obs.npy"))
        M_use = min(2000, data_unl.shape[0])

        # Precompute KDE
        kde_grids = precompute_kde(data_unl[:M_use], model, d)

        sigma = config.get("sigma", DEFAULT_SIGMA)

        # --- Self-test (returns alpha, beta, info) ---
        t0 = time.time()
        alpha_st, beta_st, info_st = solve_selftest(
            data_unl[:M_use], t_obs, sigma,
            build_V, build_Phi, kv, kp, M_max=M_use)
        v_err, p_err = evaluate_kde(model, d, alpha_st, beta_st, build_V, build_Phi,
                                    kde_grids=kde_grids)
        st_time = time.time() - t0
        st_V = v_err * 100
        st_Phi = p_err * 100
        print(f"  ST:  V={st_V:.2f}%, Phi={st_Phi:.2f}% ({st_time:.1f}s)")

        # --- Condition number from A matrix eigenvalues ---
        eigvals = info_st.get("eigvals_A", [])
        if eigvals:
            lam_min = min(eigvals)
            lam_max = max(eigvals)
            kappa = lam_max / max(lam_min, 1e-30)
        else:
            lam_min = lam_max = kappa = float('nan')
        print(f"  Eigenvalues: min={lam_min:.6e}, max={lam_max:.6e}, kappa={kappa:.1f}")

        # --- MLE ---
        t0 = time.time()
        alpha_mle, beta_mle, info_mle = solve_mle(
            data_lab[:M_use], t_obs,
            build_V, build_Phi, kv, kp, M_max=M_use)
        v_err_m, p_err_m = evaluate_kde(model, d, alpha_mle, beta_mle, build_V, build_Phi,
                                       kde_grids=kde_grids)
        mle_time = time.time() - t0
        mle_V = v_err_m * 100
        mle_Phi = p_err_m * 100
        print(f"  MLE: V={mle_V:.2f}%, Phi={mle_Phi:.2f}% ({mle_time:.1f}s)")

        results.append({
            "N": N,
            "lam_min": float(lam_min),
            "lam_max": float(lam_max),
            "kappa": float(kappa),
            "kappa_over_N": float(kappa / N),
            "ST_V_pct": round(st_V, 2),
            "ST_Phi_pct": round(st_Phi, 2),
            "MLE_V_pct": round(mle_V, 2),
            "MLE_Phi_pct": round(mle_Phi, 2),
            "dt_obs": dt_obs,
            "dt_fine": config["dt_fine"],
            "M_used": M_use,
        })

    # Save
    out_dir = ROOT / "results" / "n_scaling_cond"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary tables
    print(f"\n{'='*60}")
    print("CONDITION NUMBER TABLE")
    print(f"{'N':>5} {'lam_min':>12} {'lam_max':>12} {'kappa':>10} {'kappa/N':>10}")
    for r in results:
        print(f"{r['N']:>5} {r['lam_min']:>12.6e} {r['lam_max']:>12.6e} {r['kappa']:>10.1f} {r['kappa_over_N']:>10.1f}")

    print(f"\nERROR TABLE (dt_obs={results[0]['dt_obs']})")
    print(f"{'N':>5} {'ST V%':>8} {'ST Phi%':>8} {'MLE V%':>8} {'MLE Phi%':>8}")
    for r in results:
        print(f"{r['N']:>5} {r['ST_V_pct']:>8.2f} {r['ST_Phi_pct']:>8.2f} {r['MLE_V_pct']:>8.2f} {r['MLE_Phi_pct']:>8.2f}")


if __name__ == "__main__":
    run()
