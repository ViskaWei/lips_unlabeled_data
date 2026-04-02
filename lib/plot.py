"""Publication-quality figure style for matplotlib.

Usage:
    from lib.paper_style import apply_style, cleanup_log_axes, COLORS, MARKERS

References:
    - Nature: 89mm (single col) / 183mm (double col), min 7pt font, 300+ DPI
    - IEEE/APA: similar constraints
    - https://github.com/jbmouret/matplotlib_for_papers
    - https://allanchain.github.io/blog/post/mpl-paper-tips/
    - https://www.bastibl.net/publication-quality-plots/
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Column widths (inches) ──────────────────────────────────────
SINGLE_COL = 3.5    # ~89mm (Nature/IEEE single column)
DOUBLE_COL = 7.2    # ~183mm (Nature/IEEE double column)
PANEL_HEIGHT = 2.8  # good height for 1-row panels

# ── Colors: colorblind-safe palette (Okabe-Ito) ────────────────
COLORS = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'cyan':    '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
}

# ── Semantic colors for this project ────────────────────────────
C_V   = COLORS['blue']     # ∇V
C_PHI = COLORS['red']      # ∇Φ
C_LOSS = COLORS['green']   # loss

# ── Markers ─────────────────────────────────────────────────────
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# ── Line styles for method categories ───────────────────────────
LS_NN     = '-'     # solid  → neural network
LS_ORACLE = '--'    # dashed → oracle basis
LS_RBF    = ':'     # dotted → RBF basis


# ── Model display names ────────────────────────────────────
MODEL_LABELS = {
    'model_a': 'Smoothness',
    'model_b': 'Conditioning',
    'model_lj': 'Singularity (LJ)',
    'model_morse': 'Smooth control (Morse)',
    'model_aniso': 'Anisotropic',
    'model_dipole': 'Dipole',
}

# Canonical model list for iteration
RADIAL_MODELS = ['model_a', 'model_b', 'model_lj', 'model_morse']

# ── Method display labels ──────────────────────────────────
METHOD_LABELS = {
    'mle': 'MLE',
    'sinkhorn': 'Sinkhorn',
    'selftest': 'Self-Test',
}

# ── Method colors (consistent across all figures) ──────────
METHOD_COLORS = {
    'mle': COLORS['blue'],
    'sinkhorn': COLORS['cyan'],
    'selftest': COLORS['orange'],
}


def apply_style():
    """Apply publication rcParams. Call once at script start."""
    plt.rcParams.update({
        # Font: match document body (serif for math papers)
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'cmr10', 'Times'],
        'font.size': 8,
        'mathtext.fontset': 'cm',

        # Axes
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.linewidth': 0.6,
        'axes.formatter.use_mathtext': True,  # ×10⁻⁵ instead of 1e-5

        # Ticks — inward, compact
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,

        # Lines
        'lines.linewidth': 1.2,
        'lines.markersize': 4,

        # Legend — frameless, compact
        'legend.fontsize': 6,
        'legend.frameon': False,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,
        'legend.labelspacing': 0.3,
        'legend.columnspacing': 1.0,

        # Grid — subtle
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,

        # Layout
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def fig_1xN(N, height=PANEL_HEIGHT, width_per_panel=SINGLE_COL):
    """Create 1×N figure. Returns (fig, axes) where axes is always a list."""
    fig, axes = plt.subplots(1, N, figsize=(width_per_panel * N, height))
    if N == 1:
        axes = [axes]
    return fig, axes


def fig_2xN(N, row_height=PANEL_HEIGHT, width_per_panel=SINGLE_COL):
    """Create 2×N figure. Returns (fig, axes) as 2D array."""
    fig, axes = plt.subplots(2, N, figsize=(width_per_panel * N, row_height * 2))
    return fig, axes


def cleanup_log_axes(ax, which='both'):
    """Fix log-scale tick labels: use compact ScalarFormatter.

    Converts verbose '2×10⁻⁵' ticks to '2.0' with '×10⁻⁵' offset at top.
    Works for both linear and log-scale axes.

    Parameters
    ----------
    ax : matplotlib Axes
    which : 'x', 'y', or 'both'
    """
    def _apply(axis):
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))  # use offset for 10⁻³..10³
        axis.set_major_formatter(fmt)
        # Force update
        ax.figure.canvas.draw_idle()

    if which in ('x', 'both'):
        _apply(ax.xaxis)
    if which in ('y', 'both'):
        _apply(ax.yaxis)


def cleanup_log_ticks(ax, which='y', minor=True):
    """For log-scale axes: show only major decade ticks (10⁰, 10¹, ...)
    with clean formatting, and optionally show minor ticks without labels.

    This avoids verbose labels like '2×10¹' and just shows '10⁰', '10¹', etc.
    """
    def _apply(axis):
        axis.set_major_formatter(ticker.LogFormatterMathtext())
        if minor:
            axis.set_minor_formatter(ticker.NullFormatter())

    if which in ('x', 'both'):
        _apply(ax.xaxis)
    if which in ('y', 'both'):
        _apply(ax.yaxis)


def shared_ylabel(fig, axes_row, label, **kwargs):
    """Add a single y-label for a row of axes (remove individual labels)."""
    for ax in axes_row[1:]:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)
    axes_row[0].set_ylabel(label, **kwargs)


def shared_xlabel(fig, axes_col, label, **kwargs):
    """Add a single x-label for a column of axes."""
    for ax in axes_col[:-1]:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    axes_col[-1].set_xlabel(label, **kwargs)


def savefig(fig, path, **kwargs):
    """Save with publication defaults."""
    defaults = dict(dpi=300, bbox_inches='tight', pad_inches=0.02)
    defaults.update(kwargs)
    fig.savefig(path, **defaults)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Error bar constants ──────────────────────────────────────────
ERRORBAR_CAPSIZE = 2.5
ERRORBAR_LINEWIDTH = 0.8
ERRORBAR_CAPTHICK = 0.8
BAND_ALPHA = 0.2        # shaded ±std band opacity


def errorbar_kwargs(color=None):
    """Standard error bar kwargs for consistency across all figures."""
    kw = dict(
        capsize=ERRORBAR_CAPSIZE,
        elinewidth=ERRORBAR_LINEWIDTH,
        capthick=ERRORBAR_CAPTHICK,
    )
    if color is not None:
        kw['ecolor'] = color
    return kw


def plot_with_band(ax, x, y_mean, y_std, color, label=None, marker='o',
                   linestyle='-', fill=True):
    """Plot line with shaded ±std error band.

    Parameters
    ----------
    ax : matplotlib Axes
    x : array-like  — x values
    y_mean : array-like — mean values
    y_std : array-like — standard deviation values
    color : str — line/fill color
    label : str — legend label
    marker : str — marker style
    linestyle : str — line style
    fill : bool — if True, draw shaded band; if False, use error bars
    """
    x = np.asarray(x, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    line, = ax.plot(x, y_mean, linestyle=linestyle, marker=marker, color=color,
                    label=label)
    if fill:
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                         color=color, alpha=BAND_ALPHA)
    else:
        ax.errorbar(x, y_mean, yerr=y_std, fmt='none', color=color,
                    **errorbar_kwargs(color))
    return line


def grouped_bars(ax, categories, groups, values, errors=None, colors=None,
                 bar_width=0.22, value_fontsize=5.5, y_cap=None):
    """Grouped bar chart with optional error bars and value labels.

    Parameters
    ----------
    ax : matplotlib Axes
    categories : list[str] — x-axis category labels (e.g., model names)
    groups : list[str] — group labels within each category (e.g., method names)
    values : dict[str, list[float]] — {group_name: [val_per_category]}
    errors : dict[str, list[float]] or None — {group_name: [std_per_category]}
    colors : dict[str, str] or None — {group_name: color}
    bar_width : float
    value_fontsize : float
    y_cap : float or None — clip y-axis at this value
    """
    n_cats = len(categories)
    n_groups = len(groups)
    x = np.arange(n_cats)
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2,
                          n_groups) * bar_width

    if colors is None:
        default_colors = list(COLORS.values())
        colors = {g: default_colors[i % len(default_colors)]
                  for i, g in enumerate(groups)}

    # Determine y_cap if not given
    if y_cap is None:
        all_tops = []
        for g in groups:
            for i in range(n_cats):
                v = values[g][i]
                e = errors[g][i] if errors else 0
                all_tops.append(v + e)
        sorted_tops = sorted(all_tops)
        if len(sorted_tops) >= 2:
            y_cap = min(sorted_tops[-2] * 2.5, max(sorted_tops[-1] * 1.1, 30))
        else:
            y_cap = max(sorted_tops) * 1.2

    for j, g in enumerate(groups):
        yerr = errors[g] if errors else None
        bars = ax.bar(x + offsets[j], values[g], bar_width,
                      yerr=yerr, label=g,
                      color=colors.get(g, COLORS['black']), alpha=0.85,
                      edgecolor='white', linewidth=0.5,
                      error_kw=errorbar_kwargs(colors.get(g)),
                      capsize=ERRORBAR_CAPSIZE)
        # Value labels
        for i, bar in enumerate(bars):
            v = values[g][i]
            e = errors[g][i] if errors else 0
            if v > 0:
                if v > y_cap * 0.85:
                    ax.annotate(f'{v:.0f}', xy=(bar.get_x() + bar.get_width() / 2,
                                y_cap * 0.95), fontsize=value_fontsize,
                                ha='center', va='top',
                                color=colors.get(g), fontweight='bold')
                else:
                    y_pos = min(v + e, y_cap * 0.88) + y_cap * 0.015
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                            f'{v:.1f}', ha='center', va='bottom',
                            fontsize=value_fontsize)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(bottom=0, top=y_cap)
    ax.grid(axis='y', alpha=0.3)
    return ax


def dt_obs_ticks(ax):
    """Set standard dt_obs tick locations and labels for log-scale x-axis."""
    ax.set_xticks([1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$'])
    ax.set_xlabel(r'$\Delta t$')


# ── NN evaluation utilities ──────────────────────────────────
# Functions for evaluating trained neural network models on grids.
# These accept net objects as arguments — callers import model classes
# from core.nn_models or scripts/run_nn.py.

def eval_nn_grad_V(V_net, r_grid, d=2):
    """Evaluate radial dV/dr from a trained NN on a 1D grid.

    V_net takes x in R^d, computes |x|, passes through MLP.
    For radial eval: x = (r, 0, ..., 0), dV/dr = grad_x V . e_1.

    Parameters
    ----------
    V_net : nn.Module — trained confinement network
    r_grid : array-like — radial distances to evaluate
    d : int — spatial dimension

    Returns
    -------
    np.ndarray — gradient values at each r
    """
    import torch
    grads = []
    for r in r_grid:
        x = torch.zeros(1, d, requires_grad=True)
        x.data[0, 0] = float(r)
        V = V_net(x)
        V.backward()
        grads.append(x.grad[0, 0].item())
    return np.array(grads)


def eval_nn_dPhi_dr(Phi_net, r_grid, d=2):
    """Evaluate dPhi/dr from a trained NN on a 1D grid.

    Phi_net takes x in R^d, computes |x|, passes through MLP.
    For radial eval: x = (r, 0, ..., 0), dPhi/dr = grad_x Phi . e_1.

    Parameters
    ----------
    Phi_net : nn.Module — trained interaction network
    r_grid : array-like — radial distances to evaluate
    d : int — spatial dimension

    Returns
    -------
    np.ndarray — gradient values at each r
    """
    import torch
    grads = []
    for r in r_grid:
        x = torch.zeros(1, d, requires_grad=True)
        x.data[0, 0] = float(r)
        Phi = Phi_net(x)
        Phi.backward()
        grads.append(x.grad[0, 0].item())
    return np.array(grads)


# ── KDE visualization ────────────────────────────────────────
# Functions for KDE density background shading on gradient plots.

def add_kde_background(ax, r_kde, rho, y_ref, label_prefix=""):
    """Add normalized KDE density as gray background shading.

    Normalizes rho to 30% of the y-axis range inferred from y_ref.

    Parameters
    ----------
    ax : matplotlib Axes
    r_kde : array-like — radial grid for KDE
    rho : array-like — KDE density values
    y_ref : array-like — reference y-values (used to determine scale)
    label_prefix : str — if "first", add legend label for rho
    """
    if rho is None or len(rho) == 0 or rho.max() < 1e-20:
        return
    y_range = max(abs(y_ref.max()), abs(y_ref.min()))
    if y_range < 1e-20:
        y_range = 1.0
    rho_norm = rho / rho.max() * y_range * 0.3
    ax.fill_between(r_kde, 0, rho_norm, color='gray', alpha=0.15, zorder=0,
                    label=r'$\rho$ (KDE)' if label_prefix == "first" else None)


# ── Data loading utilities ───────────────────────────────────
# Functions for loading metrics, fitting slopes, etc.

def load_metrics_safe(path):
    """Load a JSON metrics file, return None on missing/corrupt.

    Parameters
    ----------
    path : str or Path — path to JSON file

    Returns
    -------
    dict or None
    """
    import json
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        return None
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: failed to load {p}: {e}")
        return None


def fit_loglog_slope(x, y):
    """Fit slope in log-log space via least squares.

    Parameters
    ----------
    x : array-like — independent variable (e.g. M values)
    y : array-like — dependent variable (e.g. error values)

    Returns
    -------
    (slope, intercept) : tuple of float — NaN if insufficient valid data
    """
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask = (y_arr > 0) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return np.nan, np.nan
    log_x = np.log10(x_arr[mask])
    log_y = np.log10(y_arr[mask])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return slope, intercept


# ── True gradient evaluation on radial grids ─────────────────
# Thin wrappers around lib.eval functions for radial 1D plotting.

def eval_true_grad_V_radial(model_name, r_grid, d=2):
    """Evaluate true grad_V on a 1D radial grid.

    Places each r as x = (r, 0, ..., 0) and returns the first component
    of grad_V(x).

    Parameters
    ----------
    model_name : str — e.g. 'model_a'
    r_grid : array-like — radial distances
    d : int — spatial dimension

    Returns
    -------
    np.ndarray — grad_V values at each r (1D)
    """
    from lib.eval import get_true_grad_V
    grad_fn = get_true_grad_V(model_name, d=d)
    x = np.zeros((len(r_grid), d))
    x[:, 0] = r_grid
    return grad_fn(x)[:, 0]


def eval_true_dPhi_dr_grid(model_name, r_grid, d=2):
    """Evaluate true dPhi/dr on a 1D radial grid.

    Parameters
    ----------
    model_name : str — e.g. 'model_a'
    r_grid : array-like — radial distances
    d : int — spatial dimension

    Returns
    -------
    np.ndarray — dPhi/dr values at each r (1D)
    """
    from lib.eval import get_true_dPhi_dr
    dphi_fn = get_true_dPhi_dr(model_name, d=d)
    return dphi_fn(r_grid)
