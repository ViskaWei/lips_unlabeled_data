#!/usr/bin/env python3
"""Run 10-trial M-scaling experiments for async table (dt_fine=1e-4 fixed).

Usage:
    python3 scripts/run_m_scaling_10trials.py --dt_obs 0.01
    python3 scripts/run_m_scaling_10trials.py --dt_obs 0.0001 --n_workers 10
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.config import compute_r_max
from lib.basis import get_basis
from lib.solvers import solve_selftest
from lib.eval import evaluate_kde
from lib.config import DATA_ROOT
SIGMA, REG = 1.0, 1e-4
N_TRIALS = 10
M_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000]
_QUADRATURE = "trapezoid"  # set by --quadrature flag


def run_one_trial(args):
    """Run a single (M, seed) trial. Returns (M, seed, V_err, Phi_err, time_s)."""
    dt_obs, M, seed = args
    ddir = DATA_ROOT / f"dt_obs_{dt_obs}" / "model_e"
    data_full = np.load(str(ddir / "data_unlabeled.npy"), mmap_mode="r")
    t_obs = np.load(str(ddir / "t_obs.npy"))
    M_full = data_full.shape[0]

    rng = np.random.RandomState(seed)
    idx = rng.choice(M_full, size=M, replace=False)
    idx.sort()
    data_sub = np.array(data_full[idx])

    r_max_V, r_max_Phi = compute_r_max(data_sub)
    build_V, build_Phi, kv, kp, _ = get_basis(
        "oracle", "model_e", r_max_V=r_max_V, r_max_Phi=r_max_Phi)

    t0 = time.time()
    alpha, beta, info = solve_selftest(
        data_sub, t_obs, SIGMA, build_V, build_Phi, kv, kp,
        reg=REG, M_max=M, quadrature=_QUADRATURE)
    elapsed = time.time() - t0

    v_err, phi_err = evaluate_kde("model_e", 2, alpha, beta, build_V, build_Phi)
    return (M, seed, float(v_err * 100), float(phi_err * 100), elapsed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt_obs", type=float, required=True)
    p.add_argument("--n_workers", type=int, default=10)
    p.add_argument("--M_values", nargs="+", type=int, default=None)
    p.add_argument("--quadrature", choices=["left", "trapezoid"], default="trapezoid",
                   help="Quadrature rule: 'left' (Riemann, O(Δt)) or 'trapezoid' (O(Δt²))")
    args = p.parse_args()

    global _QUADRATURE
    _QUADRATURE = args.quadrature

    M_vals = args.M_values or M_VALUES
    dt_obs = args.dt_obs

    print(f"=== dt_obs={dt_obs}, quadrature={_QUADRATURE}, M_values={M_vals}, {N_TRIALS} trials, {args.n_workers} workers ===")
    sys.stdout.flush()

    all_results = {}
    for M in M_vals:
        tasks = [(dt_obs, M, seed) for seed in range(N_TRIALS)]
        t0 = time.time()

        if args.n_workers > 1:
            with Pool(args.n_workers) as pool:
                results = pool.map(run_one_trial, tasks)
        else:
            results = [run_one_trial(t) for t in tasks]

        wall = time.time() - t0
        V_errs = [r[2] for r in results]
        Phi_errs = [r[3] for r in results]

        mean_V, std_V = np.mean(V_errs), np.std(V_errs)
        mean_Phi, std_Phi = np.mean(Phi_errs), np.std(Phi_errs)

        print(f"  M={M:>5d}: V={mean_V:.1f}±{std_V:.1f}  Phi={mean_Phi:.1f}±{std_Phi:.1f}  "
              f"({wall:.0f}s wall)", flush=True)

        all_results[str(M)] = {
            "V_mean": mean_V, "V_std": std_V,
            "Phi_mean": mean_Phi, "Phi_std": std_Phi,
            "V_all": V_errs, "Phi_all": Phi_errs,
            "wall_s": wall,
        }

    # Save
    suffix = "_riemann" if _QUADRATURE == "left" else ""
    out_dir = ROOT / "results" / f"m_scaling_async_10trials{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dt_obs_{dt_obs}.json"
    with open(out_path, "w") as f:
        json.dump({"dt_obs": dt_obs, "n_trials": N_TRIALS,
                   "quadrature": _QUADRATURE, "results": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
