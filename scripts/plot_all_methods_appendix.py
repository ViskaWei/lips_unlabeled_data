#!/usr/bin/env python3
"""Appendix figures: all 7 methods + True for each model — COMBINED figure.

Generates 1 combined PDF with 2x4 subplots:
  Top row:    nabla V(r) for each model
  Bottom row: Phi'(r) for each model
  Single shared legend row at top.

Methods plotted:
  1. True                (solid black, lw=2.0)
  2. Labeled MLE         (solid blue,  lw=1.3)
  3. Sinkhorn MLE        (solid cyan,  lw=1.3)
  4. Self-test LSE       (solid orange,lw=1.3)
  5. RBF MLE            (dashed green, lw=1.1, alpha=0.7)
  6. RBF Sinkhorn       (dashed #98df8a, lw=1.1, alpha=0.7)
  7. RBF Self-Test      (dashed red,   lw=1.1, alpha=0.7)
  8. NN self-test       (solid magenta,lw=1.8)

Output:
  papers/ips_unlabeled_learning/figures/all_methods_combined.pdf

Usage:
    python scripts/plot_all_methods_appendix.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.nn_models import RadialNet, RadialInteractionNet
from lib.plot import (
    apply_style, savefig, SINGLE_COL, PANEL_HEIGHT, MODEL_LABELS, RADIAL_MODELS,
    eval_nn_grad_V, eval_nn_dPhi_dr, add_kde_background,
    load_metrics_safe, eval_true_grad_V_radial, eval_true_dPhi_dr_grid,
)
from lib.eval import (
    make_oracle_grad_V, make_oracle_dPhi_dr,
    make_rbf_grad_V, make_rbf_dPhi_dr,
    get_r_V_grid, get_r_Phi_grid,
)

apply_style()
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Paths ──────────────────────────────────────────────────────
FIGURES_DIR = ROOT / 'papers' / 'ips_unlabeled_learning' / 'figures'
RESULTS_DIR = ROOT / 'results' / 'dt_obs_0.001'
ORACLE_FIXREG_DIR = ROOT / 'results' / 'B1_M20000_dt0.001_fixreg' / 'oracle'
KDE_DIR = ROOT / 'results' / 'kde_grids' / 'd2'
D = 2
N_GRID = 500

# ── Combined figure dimensions ─────────────────────────────────
FIG_WIDTH = 6.8      # match boundary_recovery width
FIG_HEIGHT = 3.6     # 2 rows

# ── Model suffix map ──────────────────────────────────────────
MODEL_SUFFIXES = {
    'model_a': 'a',
    'model_b': 'b',
    'model_lj': 'lj',
    'model_morse': 'morse',
}

# ── Method styling (aligned with Figure 7 / boundary_recovery) ──────
# Fig7: True=#888888(--), MLE=#E69F00(--), Selftest=#009E73(-), NN=#0072B2(:)
# Extra methods not in Fig7 get distinct styles.
# (label_prefix, color, linestyle, linewidth, alpha)
METHOD_STYLES = {
    'oracle_mle':      ('Labeled MLE',     '#E69F00', '--', 1.0, 1.0),   # orange dashed (=Fig7 MLE)
    'oracle_sinkhorn': ('Sinkhorn MLE',    '#CC79A7', '--', 1.0, 0.8),   # pink dashed
    'oracle_selftest': ('Self-test LSE',   '#009E73', '-',  1.2, 1.0),   # green solid (=Fig7 Selftest)
    'rbf_mle':         ('RBF MLE',         '#E69F00', ':',  1.0, 0.5),   # orange dotted (RBF of MLE)
    'rbf_sinkhorn':    ('RBF Sinkhorn',    '#CC79A7', ':',  1.0, 0.5),   # pink dotted (RBF of Sinkhorn)
    'rbf_selftest':    ('RBF Self-Test',   '#009E73', ':',  1.0, 0.5),   # green dotted (RBF of Selftest)
    'nn':              ('NN self-test',     '#0072B2', '-',  1.8, 1.0),   # blue solid (=Fig7 NN)
}


def load_nn_model(model_path, d=2):
    """Load V and Phi networks from a checkpoint."""
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    hidden = list(ckpt.get('hidden_dims', [64, 64, 64]))

    V_net = RadialNet(hidden_dims=hidden)
    V_net.load_state_dict(ckpt['V_state_dict'])
    V_net.eval()

    Phi_net = RadialInteractionNet(hidden_dims=hidden)
    Phi_net.load_state_dict(ckpt['Phi_state_dict'])
    Phi_net.eval()

    return V_net, Phi_net


# ── Basis evaluation helpers (oracle + RBF) ──────────────────

def eval_oracle_grad_V(model_name, alpha, r_grid, d=2):
    """Evaluate oracle grad_V on grid."""
    gV = make_oracle_grad_V(model_name, alpha)
    x = np.zeros((len(r_grid), d))
    x[:, 0] = r_grid
    return gV(x)[:, 0]


def eval_oracle_dPhi_dr(model_name, beta, r_grid):
    """Evaluate oracle dPhi/dr on grid."""
    dPhi = make_oracle_dPhi_dr(model_name, beta)
    return dPhi(r_grid)


def eval_rbf_grad_V(alpha, r_max_V, K_V, r_grid, d=2):
    """Evaluate RBF grad_V on grid."""
    gV = make_rbf_grad_V(alpha, r_max_V, K_V)
    x = np.zeros((len(r_grid), d))
    x[:, 0] = r_grid
    return gV(x)[:, 0]


def eval_rbf_dPhi_dr(beta, r_max_Phi, K_Phi, r_grid):
    """Evaluate RBF dPhi/dr on grid."""
    dPhi = make_rbf_dPhi_dr(beta, r_max_Phi, K_Phi)
    return dPhi(r_grid)


def plot_model(model_name, ax_v, ax_phi, show_title=True):
    """Plot V and Phi recovery for a single model into given axes.

    Parameters
    ----------
    model_name : str
    ax_v : matplotlib Axes for nabla V
    ax_phi : matplotlib Axes for Phi'
    show_title : bool — whether to add model title above columns
    """
    label = MODEL_LABELS[model_name]
    print(f"\n{'='*60}")
    print(f"Plotting {label} ({model_name})")
    print(f"{'='*60}")

    # Grids
    r_V = get_r_V_grid(model_name, n_points=N_GRID)
    r_Phi = get_r_Phi_grid(model_name, n_points=N_GRID)

    # True gradients
    true_gV = eval_true_grad_V_radial(model_name, r_V, d=D)
    true_dPhi = eval_true_dPhi_dr_grid(model_name, r_Phi, d=D)

    # ── Load KDE density ──────────────────────────────────────
    kde_path = KDE_DIR / model_name / 'kde_grids.npz'
    r_kde_V, rho_V = None, None
    r_kde_Phi, rho_Phi = None, None
    if kde_path.exists():
        try:
            kde_data = np.load(kde_path)
            if 'r_V_grid' in kde_data and 'rho_V' in kde_data:
                r_kde_V = kde_data['r_V_grid']
                rho_V = kde_data['rho_V']
            if 'r_Phi_grid' in kde_data and 'rho_Phi' in kde_data:
                r_kde_Phi = kde_data['r_Phi_grid']
                rho_Phi = kde_data['rho_Phi']
            print(f"  Loaded KDE from {kde_path}")
        except Exception as e:
            print(f"  Warning: failed to load KDE: {e}")
    else:
        print(f"  KDE not found at {kde_path}")

    # ── Rho drawn AFTER ylim is set (in main), stored for later ──
    plot_model._kde_cache = (r_kde_V, rho_V, r_kde_Phi, rho_Phi)

    # ── Plot True (gray dashed, matching Fig7) ──────────────────
    ax_v.plot(r_V, true_gV, '--', color='#888888', linewidth=1.5, label='True', zorder=10)
    ax_phi.plot(r_Phi, true_dPhi, '--', color='#888888', linewidth=1.5, label='True', zorder=10)

    # ── Collect methods ───────────────────────────────────────
    method_map = {
        'oracle_mle':      ('oracle', 'mle'),
        'oracle_sinkhorn': ('oracle', 'sinkhorn'),
        'oracle_selftest': ('oracle', 'selftest'),
        'rbf_mle':         ('rbf',    'mle'),
        'rbf_sinkhorn':    ('rbf',    'sinkhorn'),
        'rbf_selftest':    ('rbf',    'selftest'),
    }

    for method_key, (basis_type, method_name) in method_map.items():
        style_label, color, ls, lw, alpha = METHOD_STYLES[method_key]

        if basis_type == 'oracle':
            metrics_path = ORACLE_FIXREG_DIR / f'd{D}' / model_name / method_name / 'results.json'
            metrics = load_metrics_safe(metrics_path)
            if metrics is None:
                print(f"  Skipping {method_key}: fixreg results not found at {metrics_path}")
                continue
            trials = metrics.get('trials', [])
            if not trials:
                print(f"  Skipping {method_key}: no trials in results.json")
                continue
            t0 = trials[0]
            alpha_coeffs = t0.get('alpha', None)
            beta_coeffs = t0.get('beta', None)
            V_err_pct = metrics.get('V_mean', None)
            Phi_err_pct = metrics.get('Phi_mean', None)
        else:
            metrics_path = RESULTS_DIR / basis_type / f'd{D}' / model_name / method_name / 'metrics.json'
            metrics = load_metrics_safe(metrics_path)
            if metrics is None:
                print(f"  Skipping {method_key}: metrics not found at {metrics_path}")
                continue
            V_err_pct = metrics.get('V_error_pct', None)
            Phi_err_pct = metrics.get('Phi_error_pct', None)
            alpha_coeffs = metrics.get('alpha', None)
            beta_coeffs = metrics.get('beta', None)

        if alpha_coeffs is None or beta_coeffs is None:
            print(f"  Skipping {method_key}: missing alpha/beta in metrics")
            continue

        # Labels without error pct (shared legend uses clean names)
        v_label = style_label
        phi_label = style_label

        try:
            if basis_type == 'oracle':
                gV_vals = eval_oracle_grad_V(model_name, alpha_coeffs, r_V, d=D)
                dPhi_vals = eval_oracle_dPhi_dr(model_name, beta_coeffs, r_Phi)
            else:
                r_max_V = metrics.get('r_max_V')
                r_max_Phi = metrics.get('r_max_Phi')
                K_V = metrics.get('K_V', 20)
                K_Phi = metrics.get('K_Phi', 20)
                if r_max_V is None or r_max_Phi is None:
                    print(f"  Skipping {method_key}: missing r_max_V/r_max_Phi")
                    continue
                gV_vals = eval_rbf_grad_V(alpha_coeffs, r_max_V, K_V, r_V, d=D)
                dPhi_vals = eval_rbf_dPhi_dr(beta_coeffs, r_max_Phi, K_Phi, r_Phi)

            ax_v.plot(r_V, gV_vals, ls, color=color, linewidth=lw, alpha=alpha,
                      label=v_label, zorder=5)
            ax_phi.plot(r_Phi, dPhi_vals, ls, color=color, linewidth=lw, alpha=alpha,
                        label=phi_label, zorder=5)
            print(f"  Plotted {method_key}: V_err={V_err_pct:.1f}%, Phi_err={Phi_err_pct:.1f}%")

        except Exception as e:
            print(f"  Error plotting {method_key}: {e}")
            continue

    # ── Plot NN Self-Test ─────────────────────────────────────
    nn_model_path = RESULTS_DIR / 'nn' / f'd{D}' / model_name / 'model.pt'
    nn_metrics_path = RESULTS_DIR / 'nn' / f'd{D}' / model_name / 'metrics.json'
    nn_style_label, nn_color, nn_ls, nn_lw, nn_alpha = METHOD_STYLES['nn']

    nn_metrics = load_metrics_safe(nn_metrics_path)
    nn_V_err_pct = nn_metrics.get('V_error_pct', None) if nn_metrics else None
    nn_Phi_err_pct = nn_metrics.get('Phi_error_pct', None) if nn_metrics else None

    if nn_model_path.exists():
        try:
            V_net, Phi_net = load_nn_model(nn_model_path, d=D)
            nn_gV = eval_nn_grad_V(V_net, r_V, d=D)
            nn_dPhi = eval_nn_dPhi_dr(Phi_net, r_Phi, d=D)

            ax_v.plot(r_V, nn_gV, nn_ls, color=nn_color, linewidth=nn_lw, alpha=nn_alpha,
                      label=nn_style_label, zorder=8)
            ax_phi.plot(r_Phi, nn_dPhi, nn_ls, color=nn_color, linewidth=nn_lw, alpha=nn_alpha,
                        label=nn_style_label, zorder=8)
            print(f"  Plotted NN: V_err={nn_V_err_pct:.1f}%, Phi_err={nn_Phi_err_pct:.1f}%")

        except Exception as e:
            print(f"  Error plotting NN: {e}")
    else:
        print(f"  Skipping NN: model not found at {nn_model_path}")

    # ── Axis formatting (boundary_recovery style) ─────────────
    if show_title:
        ax_v.set_title(label, fontsize=9, pad=3)

    ax_v.set_xlabel('$r$', fontsize=8)
    ax_phi.set_xlabel('$r$', fontsize=8)

    for ax in [ax_v, ax_phi]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelsize=6, length=2, pad=1)


def main():
    """Generate combined 2x4 figure with shared legend."""
    models = list(RADIAL_MODELS)

    # Shared Y-axis: V row shares ylim, Phi row shares ylim
    V_YLIM = (-15, 15)
    PHI_YLIM = (-60, 60)   # covers most models; LJ singularity clipped

    fig, axes = plt.subplots(2, len(models), figsize=(FIG_WIDTH, FIG_HEIGHT),
                             sharey='row')

    kde_caches = []
    for col, model_name in enumerate(models):
        ax_v = axes[0, col]
        ax_phi = axes[1, col]
        plot_model(model_name, ax_v, ax_phi, show_title=True)
        kde_caches.append(plot_model._kde_cache)

        # Hide Y tick labels for non-leftmost columns (sharey handles this)
        if col > 0:
            ax_v.tick_params(labelleft=False)
            ax_phi.tick_params(labelleft=False)

    # ── Set shared ylims ──────────────────────────────────────
    axes[0, 0].set_ylim(V_YLIM)
    axes[1, 0].set_ylim(PHI_YLIM)

    # ── Draw rho AFTER ylim is set (anchored to bottom) ───────
    for col, model_name in enumerate(models):
        r_kde_V, rho_V, r_kde_Phi, rho_Phi = kde_caches[col]
        for ax, r_kde, rho, ylim in [
            (axes[0, col], r_kde_V, rho_V, V_YLIM),
            (axes[1, col], r_kde_Phi, rho_Phi, PHI_YLIM),
        ]:
            if rho is not None and len(rho) > 0 and rho.max() > 1e-20:
                rho_norm = rho / rho.max()
                d_scale = (ylim[1] - ylim[0]) * 0.35
                ax.fill_between(r_kde, ylim[0], ylim[0] + rho_norm * d_scale,
                                color='gray', alpha=0.2, zorder=0)

    # Y-axis labels only on leftmost column
    axes[0, 0].set_ylabel(r'$\nabla V(r)$', fontsize=8)
    axes[1, 0].set_ylabel(r"$\Phi'(r)$", fontsize=8)

    # ── Shared legend at top ──────────────────────────────────
    handles = [
        mlines.Line2D([], [], color='#888888', lw=1.5, ls='--', label='Ground truth'),
        mlines.Line2D([], [], color='#E69F00', lw=1.0, ls='--', label='Labeled MLE'),
        mlines.Line2D([], [], color='#CC79A7', lw=1.0, ls='--', label='Sinkhorn MLE'),
        mlines.Line2D([], [], color='#009E73', lw=1.2, ls='-',  label='Self-test LSE'),
        mlines.Line2D([], [], color='#0072B2', lw=1.8, ls='-',  label='NN self-test'),
        mlines.Line2D([], [], color='#E69F00', lw=1.0, ls=':',  alpha=0.5, label='RBF MLE'),
        mlines.Line2D([], [], color='#CC79A7', lw=1.0, ls=':',  alpha=0.5, label='RBF Sinkhorn'),
        mlines.Line2D([], [], color='#009E73', lw=1.0, ls=':',  alpha=0.5, label='RBF Self-Test'),
        mlines.Line2D([], [], color='gray', lw=0, marker='s', markersize=6,
                       alpha=0.3, label=r'$\rho$'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5,
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handletextpad=0.4)

    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.82,
                        wspace=0.08, hspace=0.35)

    # ── Save combined figure ──────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / 'all_methods_combined.pdf'
    savefig(fig, out_path)
    print(f"\nSaved combined figure: {out_path}")


if __name__ == '__main__':
    main()
