#!/usr/bin/env python3
"""M-scaling on zero-gap data for Section 4.2.1.

Zero-gap means dt_fine = dt_obs (stride=1), isolating the O(Dt) Riemann-sum
bias in the self-test loss from any velocity discretization artifacts.

Section 4.2.1 is SELFTEST ONLY by design.

Usage:
    python scripts/run_m_scaling_zerogap.py --dt_obs 0.001
    python scripts/run_m_scaling_zerogap.py --dt_obs 0.1 --M_values 200 500 1000 2000 5000
"""

import argparse
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
from lib.config import DEFAULT_SIGMA, DATA_ROOT, get_obs_stride, require_zero_gap


def _resolve_zero_gap_data_dir(dt_str, model_name):
    """Accept both historical zero-gap directory names."""
    for dirname in (f"zerogap_dt{dt_str}", f"nogap_dt{dt_str}"):
        data_dir = DATA_ROOT / dirname / model_name
        if data_dir.exists():
            return data_dir
    raise FileNotFoundError(
        f"No zero-gap data for {model_name} at dt={dt_str} under "
        f"{DATA_ROOT}/zerogap_dt{dt_str} or {DATA_ROOT}/nogap_dt{dt_str}"
    )


def run(dt_obs, M_values, model_name="model_e"):
    dt_str = str(dt_obs).rstrip("0").rstrip(".")
    data_dir = _resolve_zero_gap_data_dir(dt_str, model_name)

    data_unl = np.load(str(data_dir / "data_unlabeled.npy"))
    t_obs = np.load(str(data_dir / "t_obs.npy"))
    with open(data_dir / "config.json") as f:
        config = json.load(f)
    require_zero_gap(config, label=str(data_dir))

    d = config["d"]
    M_full = data_unl.shape[0]
    sep = "=" * 60

    build_V, build_Phi, kv, kp, basis_info = get_basis("oracle", model_name)

    print(f"Zero-gap M-scaling: {model_name}, dt_obs={dt_obs}, d={d}")
    print(
        f"  tag=ZERO_GAP_ONLY dt_fine={config['dt_fine']}, dt_obs={config['dt_obs']}, "
        f"stride={get_obs_stride(config)}, L={config['L']}, T={config['T']}"
    )
    print(f"  M_full={M_full}, M_values={M_values}")

    # Precompute KDE
    print("Precomputing KDE...")
    kde_grids = precompute_kde(data_unl, model_name, d)

    results = []
    for M in M_values:
        if M > M_full:
            print(f"  SKIP M={M} > M_full={M_full}")
            continue
        for method in ["selftest"]:
            print(f"\n{sep}")
            print(f"  M={M} | {method} | dt_obs={dt_obs} (zero-gap)")
            print(sep)
            t0 = time.time()

            sub_unl = data_unl[:M]
            alpha, beta, info = solve_selftest(
                sub_unl, t_obs, config.get("sigma", DEFAULT_SIGMA),
                build_V, build_Phi, kv, kp, M_max=M)

            v_err, p_err = evaluate_kde(
                model_name, d, alpha, beta, build_V, build_Phi,
                kde_grids=kde_grids)
            elapsed = time.time() - t0

            v_pct = v_err * 100
            p_pct = p_err * 100
            print(f"  V={v_pct:.2f}% Phi={p_pct:.2f}% ({elapsed:.1f}s)")
            results.append({
                "M": M, "method": method,
                "V_pct": round(v_pct, 4), "Phi_pct": round(p_pct, 4),
                "time_s": round(elapsed, 1),
                "reg_effective": float(info.get("reg_effective", 0.0)),
                "reg_method": info.get("reg_method", "unknown"),
            })

    # Save
    out_dir = ROOT / "results" / f"m_scaling_zerogap_dt{dt_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    metadata = {
        "experiment_tag": "ZERO_GAP_ONLY",
        "section_tag": "SECTION_4_2_1_ST_ONLY",
        "model": model_name,
        "data_dir": str(data_dir),
        "dt_obs": config["dt_obs"],
        "dt_fine": config["dt_fine"],
        "obs_stride": get_obs_stride(config),
        "zero_gap_verified": True,
        "methods": ["selftest"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved to {out_dir / 'results.json'}")

    # Print summary table
    print(f"\n{'M':>6}  {'ST V%':>8} {'ST P%':>8}")
    for M in M_values:
        st = next((r for r in results if r["M"] == M and r["method"] == "selftest"), None)
        sv = f"{st['V_pct']:.2f}" if st else "---"
        sp = f"{st['Phi_pct']:.2f}" if st else "---"
        print(f"{M:>6}  {sv:>8} {sp:>8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zero-gap M-scaling (Section 4.2.1)")
    p.add_argument("--dt_obs", type=float, required=True)
    p.add_argument("--M_values", nargs="+", type=int,
                   default=[200, 500, 1000, 2000, 5000])
    p.add_argument("--model", default="model_e")
    args = p.parse_args()
    run(args.dt_obs, args.M_values, args.model)
