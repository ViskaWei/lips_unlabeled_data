#!/usr/bin/env python3
"""M-scaling figure for §5.2: 1×4 layout comparing Riemann sum vs Trapezoid.

Both use δt = 10⁻⁴ (with gap), isolating the quadrature effect:
Left pair:  left-endpoint Riemann sum — shows O(Δt) bias floor at large Δt
Right pair: trapezoidal rule — O(Δt²) bias, floor eliminated

Usage:
    python3 scripts/plot_m_scaling_zerogap.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.plot import apply_style, savefig, COLORS, DOUBLE_COL

apply_style()

# ── Data: Riemann sum with gap (δt = 10⁻⁴), mean ± std, 10 trials ──
M_rs = np.array([20, 50, 200, 1000, 2000, 5000])

# dt_obs=0.0001: Riemann = Trapezoid (no gap), use Trapezoid data
RS = {
    1e-4: dict(
        V=np.array([6.6, 4.1, 2.2, 1.4, 1.1, 0.6]),   Vs=np.array([2.2, 1.9, 1.1, 0.6, 0.5, 0.4]),
        P=np.array([2.8, 2.2, 1.2, 0.7, 0.6, 0.6]),    Ps=np.array([1.5, 1.4, 0.6, 0.3, 0.2, 0.1]),
    ),
    1e-3: dict(
        V=np.array([6.4, 4.2, 2.2, 0.8, 0.8, 0.3]),   Vs=np.array([2.9, 1.9, 1.2, 0.5, 0.3, 0.2]),
        P=np.array([3.1, 1.8, 1.3, 0.9, 0.9, 0.7]),    Ps=np.array([1.0, 0.8, 0.6, 0.2, 0.2, 0.1]),
    ),
    1e-2: dict(
        V=np.array([5.4, 4.9, 1.9, 1.0, 0.8, 0.5]),   Vs=np.array([3.1, 1.3, 1.0, 0.5, 0.4, 0.3]),
        P=np.array([3.6, 2.2, 1.5, 0.8, 0.8, 0.8]),    Ps=np.array([1.9, 0.8, 0.6, 0.3, 0.3, 0.2]),
    ),
    1e-1: dict(
        V=np.array([12.2, 5.8, 5.3, 3.9, 3.7, 3.8]),  Vs=np.array([7.2, 2.8, 1.8, 1.1, 1.0, 0.4]),
        P=np.array([6.0, 4.1, 2.6, 1.0, 0.8, 0.6]),    Ps=np.array([3.5, 2.8, 0.9, 0.7, 0.4, 0.3]),
    ),
}

# ── Data: trapezoid (δt = 10⁻⁴ fixed), mean ± std, 10 trials ─
M_tp = np.array([20, 50, 200, 1000, 2000, 5000])

TP = {
    1e-4: dict(
        V=np.array([6.6, 4.1, 2.2, 1.4, 1.1, 0.6]),   Vs=np.array([2.2, 1.9, 1.1, 0.6, 0.5, 0.4]),
        P=np.array([2.8, 2.2, 1.2, 0.7, 0.6, 0.6]),    Ps=np.array([1.5, 1.4, 0.6, 0.3, 0.2, 0.1]),
    ),
    1e-3: dict(
        V=np.array([6.4, 4.2, 2.2, 0.8, 0.8, 0.4]),   Vs=np.array([2.9, 1.9, 1.2, 0.5, 0.4, 0.2]),
        P=np.array([3.1, 1.7, 1.2, 0.9, 0.9, 0.7]),    Ps=np.array([1.0, 0.8, 0.6, 0.2, 0.2, 0.1]),
    ),
    1e-2: dict(
        V=np.array([5.9, 5.1, 2.1, 1.0, 0.7, 0.5]),   Vs=np.array([3.2, 1.7, 0.9, 0.5, 0.3, 0.2]),
        P=np.array([3.5, 2.2, 1.4, 0.7, 0.6, 0.7]),    Ps=np.array([1.9, 0.7, 0.6, 0.2, 0.3, 0.2]),
    ),
    1e-1: dict(
        V=np.array([12.3, 5.5, 4.2, 2.0, 1.3, 0.8]),  Vs=np.array([6.7, 2.4, 2.0, 1.0, 0.6, 0.4]),
        P=np.array([5.7, 3.8, 2.4, 1.0, 0.7, 0.5]),    Ps=np.array([3.4, 2.9, 0.8, 0.6, 0.3, 0.3]),
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

# ── Figure: 1×4 panels ──────────────────────────────────────
fig_w = DOUBLE_COL
fig_h = 2.0

fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), sharey=True)
ax_rs_v, ax_rs_p, ax_tp_v, ax_tp_p = axes

plt.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.90, wspace=0.06)

# ── Plot Riemann sum (with gap) ──────────────────────────────
for dt_val in DT_ORDER:
    d = RS[dt_val]; s = STYLES[dt_val]
    exp = int(np.log10(dt_val))
    lbl = rf'$\Delta t = 10^{{{exp}}}$'
    eb_kw = dict(fmt=s['marker'], color=s['color'], markersize=s['ms'],
                 linestyle=s['ls'], linewidth=s['lw'],
                 capsize=1.5, elinewidth=0.4, capthick=0.4, zorder=s['zorder'])
    ax_rs_v.errorbar(M_rs, d['V'], yerr=d['Vs'], label=lbl, **eb_kw)
    ax_rs_p.errorbar(M_rs, d['P'], yerr=d['Ps'], **eb_kw)

# ── Plot trapezoid ───────────────────────────────────────────
for dt_val in DT_ORDER:
    d = TP[dt_val]; s = STYLES[dt_val]
    eb_kw = dict(fmt=s['marker'], color=s['color'], markersize=s['ms'],
                 linestyle=s['ls'], linewidth=s['lw'],
                 capsize=1.5, elinewidth=0.4, capthick=0.4, zorder=s['zorder'])
    ax_tp_v.errorbar(M_tp, d['V'], yerr=d['Vs'], **eb_kw)
    ax_tp_p.errorbar(M_tp, d['P'], yerr=d['Ps'], **eb_kw)

# ── Theory M^{-1/2} (identical on all 4 panels) ─────────────
M_ref = np.array([12, 10000])
ref_y = 85 * M_ref**(-0.5)
for ax in axes:
    ax.plot(M_ref, ref_y, '-', color=THEORY_COLOR, linewidth=1.2, alpha=0.7, zorder=1)
ax_rs_v.plot([], [], '-', color=THEORY_COLOR, linewidth=1.2, alpha=0.7,
             label=r'$O(M^{-1/2})$')

# ── Main panels: identical formatting ────────────────────────
for ax in axes:
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$M$')
    ax.set_xlim(15, 6000); ax.set_ylim(0.2, 25)
    ax.tick_params(which='both', direction='in')
    ax.grid(True, which='major', alpha=0.15, linewidth=0.4)

ax_rs_v.set_ylabel(r'Relative $L^2(\rho)$ error (\%)')
for ax in [ax_rs_p, ax_tp_v, ax_tp_p]:
    ax.tick_params(labelleft=False)

# ── Panel labels ─────────────────────────────────────────────
lbl_kw = dict(fontsize=8, fontweight='bold', va='bottom', ha='left')
sub_kw = dict(fontsize=6, va='bottom', ha='left', color='#444')
ax_rs_v.text(0.05, 0.12, r'$\nabla V$', transform=ax_rs_v.transAxes, **lbl_kw)
ax_rs_v.text(0.05, 0.04, r'Riemann $O(\Delta t)$', transform=ax_rs_v.transAxes, **sub_kw)
ax_rs_p.text(0.05, 0.12, r'$\nabla\Phi$', transform=ax_rs_p.transAxes, **lbl_kw)
ax_rs_p.text(0.05, 0.04, r'Riemann $O(\Delta t)$', transform=ax_rs_p.transAxes, **sub_kw)
ax_tp_v.text(0.05, 0.12, r'$\nabla V$', transform=ax_tp_v.transAxes, **lbl_kw)
ax_tp_v.text(0.05, 0.04, r'Trapezoidal $O((\Delta t)^2)$', transform=ax_tp_v.transAxes, **sub_kw)
ax_tp_p.text(0.05, 0.12, r'$\nabla\Phi$', transform=ax_tp_p.transAxes, **lbl_kw)
ax_tp_p.text(0.05, 0.04, r'Trapezoidal $O((\Delta t)^2)$', transform=ax_tp_p.transAxes, **sub_kw)

# ── Δt bias arrow on Riemann V panel ─────────────────────────
ax_rs_v.annotate('', xy=(5000, 3.8), xytext=(5000, 0.5),
                 arrowprops=dict(arrowstyle='<->', color='#555', lw=0.7))
ax_rs_v.text(4000, 1.3, r'$\Delta t$' + '\nbias', fontsize=5,
             color='#555', ha='right', va='center')

# ── Legend ────────────────────────────────────────────────────
handles, labels = ax_rs_v.get_legend_handles_labels()
handles = [handles[-1]] + handles[:-1][::-1]
labels = [labels[-1]] + labels[:-1][::-1]

fig.legend(handles, labels,
           loc='upper center', ncol=5, fontsize=6, frameon=False,
           bbox_to_anchor=(0.52, 0.99),
           columnspacing=0.8, handletextpad=0.3)

outpath = os.path.join(os.path.dirname(__file__),
                       '../papers/ips_unlabeled_learning/figures/m_scaling_zerogap_multi_dt.pdf')
savefig(fig, outpath)
print("[OK] M-scaling 1x4 figure (Riemann vs Trapezoid, both with gap) saved.")
