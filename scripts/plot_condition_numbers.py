#!/usr/bin/env python3
"""Plot condition numbers of the self-test matrix (Table 2).

Publication-quality 1×3 log-log figure with three information layers:
  Layer 1: Theory reference lines (O(N))
  Layer 2: Physical mechanism annotations
  Layer 3: Numerical slope labels at N=100 right margin

Layout follows Fig 3 (M-scaling): tight wspace, legend single row on top,
inset bold labels, y-ticks only on Panel 1.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.plot import apply_style, savefig, COLORS, DOUBLE_COL

apply_style()

# ── Data from Table 2 ──────────────────────────────────────────
N = np.array([5, 10, 20, 50, 100])

data = {
    2:  {'kappa_star': [37, 53, 80, 123, 155],
         'kappa_VV':   [19.7, 20.2, 20.5, 20.8, 20.9],
         'kappa_PhiPhi': [12.8, 6.8, 3.7, 1.9, 1.5],
         'slope_star': 0.5, 'slope_VV': 0.0, 'slope_Phi': -0.7},
    5:  {'kappa_star': [42, 44, 50, 69, 98],
         'kappa_VV':   [35.4, 32.4, 31.4, 30.7, 30.4],
         'kappa_PhiPhi': [6.2, 4.7, 4.5, 6.1, 9.1],
         'slope_star': 0.3, 'slope_VV': 0.0, 'slope_Phi': 0.1},
    10: {'kappa_star': [170, 232, 533, 1629, 3426],
         'kappa_VV':   [156, 166, 171, 177, 179],
         'kappa_PhiPhi': [3.0, 6.9, 14, 31, 50],
         'slope_star': 1.1, 'slope_VV': 0.0, 'slope_Phi': 0.9},
}

# ── Style ───────────────────────────────────────────────────────
dim_colors = {2: COLORS['blue'], 5: COLORS['orange'], 10: COLORS['red']}
dim_markers = {2: 'o', 5: 's', 10: '^'}
ANNOT_COLOR = '#555555'
THEORY_COLOR = COLORS['green']

panels = [
    ('kappa_star',    r'$\kappa_*$',          'slope_star',   'joint matrix'),
    ('kappa_VV',      r'$\kappa_{VV}$',       'slope_VV',     'confinement block'),
    ('kappa_PhiPhi',  r'$\kappa_{\Phi\Phi}$', 'slope_Phi',    'interaction block'),
]

# ── Figure (tight layout like Fig 3) ───────────────────────────
fig_w = DOUBLE_COL
fig_h = 2.0

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

N_theory = np.linspace(4, 130, 200)

for idx, (ax, (key, symbol, slope_key, subtitle)) in enumerate(zip(axes, panels)):
    # --- Data lines (zorder=5) ---
    for d in [2, 5, 10]:
        y = np.array(data[d][key], dtype=float)
        ax.plot(N, y, marker=dim_markers[d], color=dim_colors[d],
                markersize=4.0, linewidth=1.3, zorder=5)

    # --- Layer 1: Theory reference lines ---
    if idx == 0:  # κ_*: O(N) line anchored at d=2, N=5 → κ=37
        C_linear = 37.0 / 5.0
        y_ON = C_linear * N_theory
        ax.plot(N_theory, y_ON, ls='--', color=THEORY_COLOR,
                alpha=0.7, linewidth=1.0, zorder=1)

    if idx == 2:  # κ_ΦΦ: O(N) reference line only (O(1/N) removed)
        C_lin_phi = 3.0 / 5.0
        y_lin_phi = C_lin_phi * N_theory
        ax.plot(N_theory, y_lin_phi, ls='--', color=THEORY_COLOR,
                alpha=0.7, linewidth=1.0, zorder=1)

    # --- Axes setup ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])
    ax.minorticks_off()
    ax.tick_params(which='both', direction='in')
    ax.grid(True, alpha=0.15, which='major', linewidth=0.4)

    # Inset panel label: bold symbol + gray subtitle
    # κ_VV panel: place at top-right to avoid overlap with data lines
    if key == 'kappa_VV':
        label_x, label_y, sub_x, sub_y = 0.05, 0.92, 0.05, 0.82
        label_ha = 'left'
    else:
        label_x, label_y, sub_x, sub_y = 0.05, 0.92, 0.05, 0.82
        label_ha = 'left'
    ax.text(label_x, label_y, symbol, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha=label_ha)
    ax.text(sub_x, sub_y, subtitle, transform=ax.transAxes,
            fontsize=5.5, color=ANNOT_COLOR, va='top', ha=label_ha)

    # --- Layer 3: Slope labels at N=100 right margin ---
    # Skip Panel 1 (κ_VV) where all slopes are 0.0 — annotated separately
    if idx != 1:
        for d in [2, 5, 10]:
            y_end = data[d][key][-1]
            slope_val = data[d][slope_key]
            ax.annotate(f'{slope_val:+.1f}',
                        xy=(100, y_end), xytext=(4, 0),
                        textcoords='offset points',
                        fontsize=5.5, fontstyle='italic',
                        color=dim_colors[d], va='center', ha='left',
                        clip_on=False)

# --- Layer 2: Physical mechanism annotations ---
# Panel 0 (κ_*): c_N mechanism near d=10 line
ax0 = axes[0]
ax0.annotate(r'$c_N \!\sim\! 1/N$',
             xy=(20, 533), xytext=(-20, 8),
             textcoords='offset points',
             fontsize=5.5, color=ANNOT_COLOR, ha='center', va='bottom')

# Panel 1 (κ_VV): flat slope + mechanism
ax1 = axes[1]
ax1.text(0.50, 0.50, r'slope $\approx 0$',
         transform=ax1.transAxes,
         fontsize=6, color=ANNOT_COLOR, ha='center', va='center',
         fontstyle='italic')
ax1.text(0.50, 0.42, '$N$ indep. samples',
         transform=ax1.transAxes,
         fontsize=5.5, color=ANNOT_COLOR, ha='center', va='center')

ax2 = axes[2]

# --- Legend (single row, top center, like Fig 3) ─────────────
dim_handles = [Line2D([0], [0], marker=dim_markers[d], color=dim_colors[d],
                      markersize=4, linewidth=1.2, label=f'$d={d}$')
               for d in [2, 5, 10]]
theory_handles = [
    Line2D([0], [0], ls='--', color=THEORY_COLOR, alpha=0.7,
           linewidth=1.0, label=r'$O(N)$'),
]
all_handles = dim_handles + theory_handles
all_labels = [h.get_label() for h in all_handles]

fig.legend(all_handles, all_labels,
           loc='upper center', ncol=4, fontsize=6.5, frameon=False,
           bbox_to_anchor=(0.52, 0.99),
           columnspacing=1.0, handletextpad=0.3)

# --- Per-panel y-limits ──────────────────────────────────────
axes[1].set_ylim(18, 800)  # lower=18 so orange line clears κ_VV label at top-left

# --- Layout: tight like Fig 3 ────────────────────────────────
axes[0].set_ylabel(r'Condition number $\kappa$')
axes[1].tick_params(labelleft=False)
axes[2].tick_params(labelleft=False)

plt.subplots_adjust(left=0.08, right=0.96, bottom=0.17, top=0.90, wspace=0.05)

# ── Save (both formats before closing) ──────────────────────────
out_dir = os.path.join(os.path.dirname(__file__),
                       '../papers/ips_unlabeled_learning/figures')
os.makedirs(out_dir, exist_ok=True)
out_pdf = os.path.join(out_dir, 'condition_numbers_1x3.pdf')
out_png = os.path.join(out_dir, 'condition_numbers_1x3.png')
fig.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f'Saved: {out_pdf}')
fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f'Saved: {out_png}')
plt.close(fig)
