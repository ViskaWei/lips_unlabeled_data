#!/usr/bin/env python3
"""Collect Table 4 (boundary test) results and generate LaTeX.

Reads oracle error bar results + NN error bar results,
computes mean±std, and outputs the LaTeX table for the paper.

Usage:
    python scripts/collect_table4_results.py
    python scripts/collect_table4_results.py --nn_only   # just NN results
    python scripts/collect_table4_results.py --oracle_only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / 'results'

# Model display names matching the paper
MODEL_MAP = {
    'model_a': 'Smoothness',
    'model_b': 'Conditioning',
    'model_lj': 'Singularity',
    'model_morse': 'Smooth ctrl',
}

DT_OBS_LIST = ['0.001', '0.01', '0.1']
DT_DISPLAY = {'0.001': '10^{-3}', '0.01': '10^{-2}', '0.1': '10^{-1}'}

METHODS_ORACLE = ['mle', 'selftest', 'sinkhorn']
METHOD_DISPLAY = {
    'mle': 'Labeled MLE',
    'selftest': 'Self-Test',
    'sinkhorn': 'Sinkhorn',
}


def load_oracle_results():
    """Load oracle error bar results for all boundary models."""
    results = {}

    for dt in DT_OBS_LIST:
        tag = f'table4_oracle_dt{dt}'
        tag_dir = RESULTS_ROOT / tag

        for model_key, model_display in MODEL_MAP.items():
            for method in METHODS_ORACLE:
                rpath = tag_dir / 'oracle' / 'd2' / model_key / method / 'results.json'
                if not rpath.exists():
                    print(f"  WARN: missing {rpath}")
                    continue

                with open(rpath) as f:
                    data = json.load(f)

                key = (model_key, dt, method)
                results[key] = {
                    'V_mean': data['V_mean'],
                    'V_std': data['V_std'],
                    'Phi_mean': data['Phi_mean'],
                    'Phi_std': data['Phi_std'],
                    'n_trials': data['n_trials'],
                    'reg_trials': [t.get('reg_effective', None) for t in data.get('trials', [])],
                }

    return results


def load_nn_results():
    """Load NN error bar results for all boundary models.

    NN results path (when --dt_obs is specified):
      results/dt_obs_{dt}/nn_errbar_table4/trial_{i}/d2/model_{x}/metrics.json

    Also checks legacy paths without dt_obs prefix.
    """
    results = {}

    for model_key in MODEL_MAP:
        for dt in DT_OBS_LIST:
            v_errs, phi_errs = [], []

            for trial in range(10):  # check up to 10 trials
                # Primary path: results/dt_obs_{dt}/nn_errbar_table4/trial_{i}/d2/model/
                metrics_path = (RESULTS_ROOT / f'dt_obs_{dt}' /
                                f'nn_errbar_table4/trial_{trial}' /
                                f'd2' / model_key / 'metrics.json')

                if not metrics_path.exists():
                    # Legacy: results/nn_errbar_table4/trial_{i}/d2/model/
                    metrics_path = (RESULTS_ROOT / f'nn_errbar_table4/trial_{trial}' /
                                    f'd2' / model_key / 'metrics.json')

                if not metrics_path.exists():
                    continue

                with open(metrics_path) as f:
                    m = json.load(f)

                # Check dt_obs matches
                m_dt = m.get('dt_obs')
                if m_dt is not None and abs(float(m_dt) - float(dt)) > 1e-6:
                    continue

                v_errs.append(m['V_error_pct'])
                phi_errs.append(m['Phi_error_pct'])

            if v_errs:
                key = (model_key, dt)
                results[key] = {
                    'V_mean': float(np.mean(v_errs)),
                    'V_std': float(np.std(v_errs)),
                    'Phi_mean': float(np.mean(phi_errs)),
                    'Phi_std': float(np.std(phi_errs)),
                    'n_trials': len(v_errs),
                    'V_all': v_errs,
                    'Phi_all': phi_errs,
                }

    return results


def fmt_err(mean, std, best_v=None, best_p=None, is_v=True):
    """Format error as mean±std with \\best{} if it's the best."""
    if mean is None:
        return '--'

    if std is not None and std > 0.005:
        s = f'{mean:.2f}{{\\pm}}{std:.2f}'
    else:
        s = f'{mean:.2f}'

    best = best_v if is_v else best_p
    if best is not None and abs(mean - best) < 0.005:
        return f'\\best{{{s}}}'
    return s


def generate_latex(oracle_results, nn_results):
    """Generate LaTeX table for Table 4."""
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Boundary test results: gradient errors (\%) across three observation regimes. '
        r'Oracle: mean $\pm$ std, 10 trials. '
        r'NN: MLP $[64{,}64{,}64]$, Softplus, mean $\pm$ std over 5 trials. '
        r'{\best{Green}} = best per (model, $\Delta t$).}',
        r'\label{tab:boundary_results}',
        r'\vspace{2mm}',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{cl cc cc cc cc}',
        r'\toprule',
        r'\multicolumn{10}{c}{\scriptsize $d{=}2$,\; $N{=}10$,\; '
        r'$M{=}2{,}000$,\; $T{=}1$,\; $\sigma{=}1$,\; $\delta t{=}10^{-4}$} \\',
        r'\midrule',
        r'& & \multicolumn{2}{c}{Labeled MLE} & \multicolumn{2}{c}{Self-Test} '
        r'& \multicolumn{2}{c}{Sinkhorn} & \multicolumn{2}{c}{NN} \\',
        r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
        r'Model & $\Delta t$ & $\nabla V$ & $\nabla \Phi$ & $\nabla V$ & $\nabla \Phi$ '
        r'& $\nabla V$ & $\nabla \Phi$ & $\nabla V$ & $\nabla \Phi$ \\',
        r'\midrule',
    ]

    for model_key, model_display in MODEL_MAP.items():
        first_row = True
        for dt in DT_OBS_LIST:
            # Collect all V and Phi means for this (model, dt) to find best
            all_v, all_p = [], []
            for method in METHODS_ORACLE:
                key = (model_key, dt, method)
                if key in oracle_results:
                    all_v.append(oracle_results[key]['V_mean'])
                    all_p.append(oracle_results[key]['Phi_mean'])
            nn_key = (model_key, dt)
            if nn_key in nn_results:
                all_v.append(nn_results[nn_key]['V_mean'])
                all_p.append(nn_results[nn_key]['Phi_mean'])

            best_v = min(all_v) if all_v else None
            best_p = min(all_p) if all_p else None

            # Build row
            model_col = model_display if first_row else ''
            dt_col = f'$10^{{{int(np.log10(float(dt)))}}}$'

            cells = []
            for method in METHODS_ORACLE:
                key = (model_key, dt, method)
                if key in oracle_results:
                    r = oracle_results[key]
                    v_str = fmt_err(r['V_mean'], r['V_std'], best_v, best_p, is_v=True)
                    p_str = fmt_err(r['Phi_mean'], r['Phi_std'], best_v, best_p, is_v=False)
                    cells.append(f'${v_str}$ & ${p_str}$')
                else:
                    cells.append('-- & --')

            # NN
            if nn_key in nn_results:
                r = nn_results[nn_key]
                v_str = fmt_err(r['V_mean'], r['V_std'], best_v, best_p, is_v=True)
                p_str = fmt_err(r['Phi_mean'], r['Phi_std'], best_v, best_p, is_v=False)
                cells.append(f'${v_str}$ & ${p_str}$')
            else:
                cells.append('-- & --')

            row = f'{model_col} & {dt_col} & ' + ' & '.join(cells) + r' \\'
            lines.append(row)
            first_row = False

        lines.append(r'\midrule')

    # Remove last \midrule, replace with \bottomrule
    lines[-1] = r'\bottomrule'

    lines.extend([
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def print_summary(oracle_results, nn_results):
    """Print a readable summary of all results."""
    print(f"\n{'='*90}")
    print("TABLE 4 RESULTS SUMMARY")
    print(f"{'='*90}")

    for model_key, model_display in MODEL_MAP.items():
        print(f"\n  {model_display} ({model_key})")
        print(f"  {'─'*80}")
        print(f"  {'dt':>6}  {'MLE V%':>12} {'MLE Phi%':>12}  "
              f"{'ST V%':>12} {'ST Phi%':>12}  "
              f"{'Sink V%':>12} {'Sink Phi%':>12}  "
              f"{'NN V%':>12} {'NN Phi%':>12}")

        for dt in DT_OBS_LIST:
            row_parts = [f'  {dt:>6}']

            for method in METHODS_ORACLE:
                key = (model_key, dt, method)
                if key in oracle_results:
                    r = oracle_results[key]
                    row_parts.append(f'{r["V_mean"]:5.2f}±{r["V_std"]:.2f}')
                    row_parts.append(f'{r["Phi_mean"]:5.2f}±{r["Phi_std"]:.2f}')
                else:
                    row_parts.extend(['   --   ', '   --   '])

            nn_key = (model_key, dt)
            if nn_key in nn_results:
                r = nn_results[nn_key]
                row_parts.append(f'{r["V_mean"]:5.2f}±{r["V_std"]:.2f}')
                row_parts.append(f'{r["Phi_mean"]:5.2f}±{r["Phi_std"]:.2f}')
            else:
                row_parts.extend(['   --   ', '   --   '])

            print('  '.join(row_parts))

    # Count coverage
    oracle_total = len(MODEL_MAP) * len(DT_OBS_LIST) * len(METHODS_ORACLE)
    nn_total = len(MODEL_MAP) * len(DT_OBS_LIST)
    oracle_found = len(oracle_results)
    nn_found = len(nn_results)
    print(f"\n  Coverage: oracle {oracle_found}/{oracle_total}, NN {nn_found}/{nn_total}")


def main():
    p = argparse.ArgumentParser(description='Collect Table 4 results')
    p.add_argument('--oracle_only', action='store_true')
    p.add_argument('--nn_only', action='store_true')
    p.add_argument('--latex', action='store_true',
                   help='Output LaTeX table to papers/ directory')
    args = p.parse_args()

    oracle_results = {} if args.nn_only else load_oracle_results()
    nn_results = {} if args.oracle_only else load_nn_results()

    if not oracle_results and not nn_results:
        print("ERROR: No results found. Run experiments first:")
        print("  bash scripts/run_table4_oracle_errbar.sh")
        print("  bash scripts/run_table4_nn_errbar.sh")
        sys.exit(1)

    print_summary(oracle_results, nn_results)

    if args.latex or (oracle_results and nn_results):
        latex = generate_latex(oracle_results, nn_results)
        out_path = ROOT / 'papers' / 'ips_unlabeled_learning' / 'table4_boundary_generated.tex'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(latex)
        print(f"\n  LaTeX table written to {out_path}")

    # Also save raw data as JSON
    raw = {
        'oracle': {f'{k[0]}_{k[1]}_{k[2]}': v for k, v in oracle_results.items()},
        'nn': {f'{k[0]}_{k[1]}': v for k, v in nn_results.items()},
    }
    out_json = RESULTS_ROOT / 'table4_collected.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"  Raw data written to {out_json}")


if __name__ == '__main__':
    main()
