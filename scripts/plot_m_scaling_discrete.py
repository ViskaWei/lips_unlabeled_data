#!/usr/bin/env python3
# NEW-CODE-VERIFIED: searched="plot_m_scaling_discrete plot_discrete" found="none" reason="no existing script for zero-gap discrete-model figure"
"""Discrete-model bias figure for §5.2: 1×2 layout showing zero-gap (δt = Δt).

Shows the intrinsic O(Δt) bias floor when the system is a discrete time series
model rather than an SDE discretization.

Usage:
    python3 scripts/plot_m_scaling_discrete.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.plot import apply_style, savefig, COLORS, DOUBLE_COL

apply_style()

# ── Data: zero-gap (δt = Δt), mean ± std, 10 trials ──────────
M = np.array([20, 50, 200, 1000, 2000, 5000])

ZG = {
    1e-4: dict(
        V=np.array([18.0, 11.8, 4.7, 2.2, 1.3, 1.2]),  Vs=np.array([8.7, 6.1, 3.6, 1.3, 0.8, 0.5]),
        P=np.array([8.3, 7.6, 3.6, 1.6, 1.2, 0.8]),     Ps=np.array([5.7, 1.5, 2.0, 0.5, 0.3, 0.4]),
    ),
    1e-3: dict(
        V=np.array([17.2, 12.8, 5.9, 2.5, 1.6, 1.2]),   Vs=np.array([8.3, 4.7, 3.3, 1.1, 0.8, 0.5]),
        P=np.array([10.1, 7.1, 7.2, 6.1, 6.1, 6.2]),    Ps=np.array([4.4, 3.5, 3.2, 1.3, 0.9, 0.4]),
    ),
    1e-2: dict(
        V=np.array([15.6, 15.5, 14.1, 13.7, 13.6, 13.7]), Vs=np.array([6.4, 3.4, 1.4, 0.6, 0.5, 0.3]),
        P=np.array([47.3, 47.0, 46.6, 46.8, 46.7, 46.7]), Ps=np.array([3.0, 1.6, 1.2, 0.3, 0.3, 0.2]),
    ),
    1e-1: dict(
        V=np.array([20.0, 20.2, 20.0, 19.8, 19.9, 19.8]),   Vs=np.array([2.0, 0.8, 0.5, 0.2, 0.1, 0.1]),
        P=np.array([100.8, 100.5, 100.2, 100.4, 100.3, 100.4]), Ps=np.array([1.7, 1.5, 0.5, 0.3, 0.2, 0.1]),
    ),
}

# ── Styles ────────────────────────────────────────────────────
STYLES = {
    1e-4: dict(color='#1b2838', marker='o', ms=3.5, ls='-',  lw=1.2, zorder=5),
    1e-3: dict(color='#2a6496', marker='s', ms=3.3, ls='-',  lw=1.0, zorder=4),
    1e-2: dict(color='#6aafe6', marker='^', ms=3.3, ls='--', lw=0.9, zorder=3),
    1e-1: dict(color='#a8d0f0', marker='D', ms=3.0, ls=':',  lw=0.9, zorder=2),
}

DT_ORDER = [1e-1, 1e-2, 1e-3, 1e-4]
THEORY_COLOR = COLORS['green']

# ── Figure: 1×2 compact ─────────────────────────────────────
fig_w = DOUBLE_COL * 0.52
fig_h = 2.0

fig, (ax_v, ax_p) = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)
plt.subplots_adjust(left=0.14, right=0.97, bottom=0.16, top=0.88, wspace=0.06)

# ── Plot ─────────────────────────────────────────────────────
for dt_val in DT_ORDER:
    d = ZG[dt_val]; s = STYLES[dt_val]
    exp = int(np.log10(dt_val))
    lbl = rf'$\Delta t = 10^{{{exp}}}$'
    eb_kw = dict(fmt=s['marker'], color=s['color'], markersize=s['ms'],
                 linestyle=s['ls'], linewidth=s['lw'],
                 capsize=1.5, elinewidth=0.4, capthick=0.4, zorder=s['zorder'])
    ax_v.errorbar(M, d['V'], yerr=d['Vs'], label=lbl, **eb_kw)
    ax_p.errorbar(M, d['P'], yerr=d['Ps'], **eb_kw)

# ── Theory M^{-1/2} ─────────────────────────────────────────
M_ref = np.array([12, 10000])
ref_y = 85 * M_ref**(-0.5)
for ax in [ax_v, ax_p]:
    ax.plot(M_ref, ref_y, '-', color=THEORY_COLOR, linewidth=1.2, alpha=0.7, zorder=1)
ax_v.plot([], [], '-', color=THEORY_COLOR, linewidth=1.2, alpha=0.7,
          label=r'$O(M^{-1/2})$')

# ── Formatting ───────────────────────────────────────────────
for ax in [ax_v, ax_p]:
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$M$')
    ax.set_xlim(15, 6000); ax.set_ylim(0.5, 150)
    ax.tick_params(which='both', direction='in')
    ax.grid(True, which='major', alpha=0.15, linewidth=0.4)

ax_v.set_ylabel(r'Relative $L^2(\rho)$ error (\%)')
ax_p.tick_params(labelleft=False)

# ── Panel labels ─────────────────────────────────────────────
lbl_kw = dict(fontsize=8, fontweight='bold', va='bottom', ha='left')
ax_v.text(0.05, 0.04, r'$\nabla V$', transform=ax_v.transAxes, **lbl_kw)
ax_p.text(0.05, 0.04, r'$\nabla\Phi$', transform=ax_p.transAxes, **lbl_kw)

# ── Bias floor annotations ──────────────────────────────────
ax_v.annotate(r'$\theta^*_V\Delta t/2 = 20\%$',
              xy=(5000, 19.8), xytext=(800, 28),
              fontsize=5, color='#555',
              arrowprops=dict(arrowstyle='->', color='#888', lw=0.5))
ax_p.annotate(r'$\theta^*_\Phi\Delta t/2 = 100\%$',
              xy=(5000, 100.4), xytext=(800, 130),
              fontsize=5, color='#555',
              arrowprops=dict(arrowstyle='->', color='#888', lw=0.5))

# ── Legend ────────────────────────────────────────────────────
handles, labels = ax_v.get_legend_handles_labels()
handles = [handles[-1]] + handles[:-1][::-1]
labels = [labels[-1]] + labels[:-1][::-1]

fig.legend(handles, labels,
           loc='upper center', ncol=5, fontsize=5.5, frameon=False,
           bbox_to_anchor=(0.55, 0.99),
           columnspacing=0.6, handletextpad=0.3)

outpath = os.path.join(os.path.dirname(__file__),
                       '../papers/ips_unlabeled_learning/figures/m_scaling_discrete_bias.pdf')
savefig(fig, outpath)
print("[OK] Discrete-model bias 1x2 figure saved.")
