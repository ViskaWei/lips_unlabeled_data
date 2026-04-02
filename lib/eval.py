"""Canonical evaluation API for IPS unlabeled learning.

Single module combining:
- Model registry and grid ranges
- True gradient references (delegating to core/potentials.py classes)
- KDE density estimation
- All evaluation and error computation

Scripts import from here: `from lib.eval import ...`
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

from core.potentials import (
    QuadraticConfinement, DoubleWellPotential, HarmonicPotential,
    PiecewiseInteraction, InverseInteraction, LennardJonesPotential,
    MorsePotential, AnisotropicConfinement, AnisotropicGaussianInteraction,
    DipolarInteraction, GaussianInteraction, GaussianBumpInteraction,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = _PROJECT_ROOT / 'results'


# ══════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════

RADIAL_MODELS = ['model_a', 'model_b', 'model_lj', 'model_morse', 'model_e']
NONRADIAL_MODELS = {'model_aniso', 'model_dipole'}
ALL_MODELS = RADIAL_MODELS + list(NONRADIAL_MODELS)

MODEL_LABELS = {
    'model_a': 'Model A',
    'model_b': 'Model B',
    'model_lj': 'Model C (LJ)',
    'model_morse': 'Model D (Morse)',
    'model_e': 'Model E (Gaussian Bumps)',
    'model_aniso': 'Model Aniso',
    'model_dipole': 'Model Dipole',
}

# ══════════════════════════════════════════════════════════════
# Grid ranges for radial evaluation
# ══════════════════════════════════════════════════════════════

R_V_RANGE = {
    'model_a':     (0.01, 1.5),
    'model_b':     (0.01, 2.0),
    'model_lj':    (0.01, 1.5),
    'model_morse': (0.01, 2.0),
    'model_e':     (0.01, 1.5),
}

R_PHI_RANGE = {
    'model_a':     (0.05, 2.5),
    'model_b':     (0.05, 2.5),
    'model_lj':    (0.35, 2.5),
    'model_morse': (0.05, 2.5),
    'model_e':     (0.05, 2.5),
}

R_PHI_RANGE_FIG4 = {
    'model_lj':    (0.50, 2.5),
    'model_morse': (0.20, 2.5),
}

# ══════════════════════════════════════════════════════════════
# KDE parameters
# ══════════════════════════════════════════════════════════════

KDE_BANDWIDTH = 0.15
GRID_SIZE = 2000
MIN_KDE_SAMPLES = 100


# ══════════════════════════════════════════════════════════════
# Grid helper functions
# ══════════════════════════════════════════════════════════════

def get_r_V_grid(model_name, n_points=GRID_SIZE):
    """Return the r_V evaluation grid for a given model."""
    assert model_name in R_V_RANGE, (
        f"Unknown model '{model_name}'. Available: {list(R_V_RANGE.keys())}"
    )
    lo, hi = R_V_RANGE[model_name]
    return np.linspace(lo, hi, n_points)


def get_r_Phi_grid(model_name, n_points=GRID_SIZE):
    """Return the r_Phi evaluation grid for a given model."""
    assert model_name in R_PHI_RANGE, (
        f"Unknown model '{model_name}'. Available: {list(R_PHI_RANGE.keys())}"
    )
    lo, hi = R_PHI_RANGE[model_name]
    return np.linspace(lo, hi, n_points)


def is_radial(model_name):
    """Return True if the model uses radial (scalar) interaction potential."""
    return model_name not in NONRADIAL_MODELS


# ══════════════════════════════════════════════════════════════
# Consistency assertions
# ══════════════════════════════════════════════════════════════

def validate_data_shape(data, model_name=None):
    """Assert data has the expected (M, L, N, d) shape."""
    tag = f" for {model_name}" if model_name else ""
    assert data.ndim == 4, (
        f"Expected 4D data (M, L, N, d){tag}, got shape {data.shape}"
    )
    M, L, N, d = data.shape
    assert M >= 1 and L >= 1 and N >= 2 and d >= 1, (
        f"Invalid data dimensions{tag}: M={M}, L={L}, N={N}, d={d}"
    )
    return M, L, N, d


def validate_grid(r_grid, name="grid"):
    """Assert the grid is 1D, sorted, and of expected length."""
    assert r_grid.ndim == 1, f"{name} must be 1D, got shape {r_grid.shape}"
    assert len(r_grid) >= 10, f"{name} too short: {len(r_grid)} points"
    assert np.all(np.diff(r_grid) > 0), f"{name} must be strictly increasing"


# ══════════════════════════════════════════════════════════════
# Model potentials factory — single source of truth
# Uses classes from core/potentials.py (no duplicate implementations)
# ══════════════════════════════════════════════════════════════

def _get_model_potentials(model_name, d=2):
    """Return (V, Phi) potential instances for a given model.

    Args:
        model_name: model identifier
        d: spatial dimension (only needed for model_dipole)

    Returns:
        (V, Phi) tuple of Potential instances
    """
    if model_name == 'model_a':
        return QuadraticConfinement(alpha1=-1, alpha2=2), PiecewiseInteraction(beta1=-3, beta2=2)
    elif model_name == 'model_b':
        return DoubleWellPotential(), InverseInteraction(gamma=0.5)
    elif model_name == 'model_lj':
        return HarmonicPotential(k=2), LennardJonesPotential(epsilon=0.5, sigma_lj=0.5)
    elif model_name == 'model_morse':
        return DoubleWellPotential(), MorsePotential(D=0.5, a=2, r0=0.8)
    elif model_name == 'model_e':
        return QuadraticConfinement(alpha1=-1, alpha2=2), GaussianBumpInteraction()
    elif model_name == 'model_aniso':
        return AnisotropicConfinement(a=(1, 4)), AnisotropicGaussianInteraction(A=2, s=(0.5, 1.5))
    elif model_name == 'model_dipole':
        a = tuple(0.01 / k for k in range(1, d + 1))
        return AnisotropicConfinement(a=a), DipolarInteraction(mu=0.5, d=d, r_safe=0.15)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════
# True gradient function registry
# ══════════════════════════════════════════════════════════════

def get_true_grad_V(model_name, d=2):
    """Return the true grad_V function: callable(X) -> (n, d) array."""
    V, _ = _get_model_potentials(model_name, d)
    return V.gradient


def get_true_dPhi_dr(model_name, d=2):
    """Return the true dPhi/dr function for radial models: callable(r) -> (n,)."""
    assert is_radial(model_name), (
        f"dPhi/dr not defined for non-radial model '{model_name}'. "
        f"Use get_true_grad_Phi_vec() instead."
    )
    _, Phi = _get_model_potentials(model_name, d)
    return Phi.gradient


def get_true_grad_Phi_vec(model_name, d=2):
    """Return the true vector grad_Phi function for non-radial models."""
    assert not is_radial(model_name), (
        f"Vector grad_Phi not defined for radial model '{model_name}'. "
        f"Use get_true_dPhi_dr() instead."
    )
    _, Phi = _get_model_potentials(model_name, d)
    return Phi.gradient


# ══════════════════════════════════════════════════════════════
# True gradient functions — backward-compat names
# These delegate to core/potentials.py classes (single source of truth)
# ══════════════════════════════════════════════════════════════

def true_grad_V_model_a(x, alpha1=-1.0, alpha2=2.0):
    """nabla V for Model A: V = alpha1/2 |x| + alpha2 |x|^2."""
    return QuadraticConfinement(alpha1=alpha1, alpha2=alpha2).gradient(x)


def true_grad_V_model_b(x):
    """nabla V for Model B: V = (|x|^2-1)^2/4."""
    return DoubleWellPotential().gradient(x)


def true_grad_V_model_lj(x, k=2.0):
    """nabla V for Model LJ: V = 0.5*k*|x|^2."""
    return HarmonicPotential(k=k).gradient(x)


def true_grad_V_model_morse(x):
    """nabla V for Model Morse: same as Model B (Double Well)."""
    return DoubleWellPotential().gradient(x)


def true_grad_V_model_aniso(x, a=(1.0, 4.0)):
    """nabla V for Model Aniso: V(x) = a1*x1^2 + a2*x2^2."""
    return AnisotropicConfinement(a=a).gradient(x)


def true_grad_V_model_dipole(x):
    """nabla V for Model Dipole: V(x) = sum_k (0.01/k)*x_k^2."""
    d = x.shape[-1]
    a = tuple(0.01 / k for k in range(1, d + 1))
    return AnisotropicConfinement(a=a).gradient(x)


def true_grad_V_model_e(x, alpha1=-1.0, alpha2=2.0):
    """nabla V for Model E: same as Model A."""
    return QuadraticConfinement(alpha1=alpha1, alpha2=alpha2).gradient(x)


def true_dPhi_dr_model_e(r):
    """dPhi/dr for Model E (Gaussian bumps)."""
    return GaussianBumpInteraction().gradient(r)


def true_dPhi_dr_model_a(r, beta1=-3.0, beta2=2.0, eps=0.05):
    """dPhi/dr for Model A (smoothed piecewise)."""
    return PiecewiseInteraction(beta1=beta1, beta2=beta2, eps=eps).gradient(r)


def true_dPhi_dr_model_b(r, gamma=0.5):
    """dPhi/dr for Model B: Phi = gamma/(r+1)."""
    return InverseInteraction(gamma=gamma).gradient(r)


def true_dPhi_dr_model_lj(r, epsilon=0.5, sigma_lj=0.5, r_cut=2.5, r_safe_factor=0.7):
    """dPhi/dr for Model LJ: truncated & shifted Lennard-Jones."""
    return LennardJonesPotential(epsilon=epsilon, sigma_lj=sigma_lj,
                                  r_cut=r_cut, r_safe_factor=r_safe_factor).gradient(r)


def true_dPhi_dr_model_morse(r, D=0.5, a=2.0, r0=0.8):
    """dPhi/dr for Model Morse: Phi = D*(1-exp(-a(r-r0)))^2."""
    return MorsePotential(D=D, a=a, r0=r0).gradient(r)


def true_grad_Phi_model_aniso(z, A=2.0, s=(0.5, 1.5)):
    """nabla_z Phi for Model Aniso (vector gradient)."""
    return AnisotropicGaussianInteraction(A=A, s=s).gradient(z)


def true_grad_Phi_model_dipole(z, mu=0.5, n_hat=None, r_safe_clamp=0.15):
    """nabla_z Phi for Model Dipole (vector gradient)."""
    d = z.shape[-1]
    return DipolarInteraction(mu=mu, d=d, r_safe=r_safe_clamp).gradient(z)


def true_V_harmonic(x, k=1.0):
    """True kinetic potential: V(x) = 0.5 * k * x^2."""
    return HarmonicPotential(k=k)(x)


def true_grad_V_harmonic(x, k=1.0):
    """Gradient of harmonic potential: grad V = k * x."""
    return HarmonicPotential(k=k).gradient(x)


def true_Phi_gaussian(r, A=1.0, sigma=1.0):
    """True interaction potential: Phi(r) = A * exp(-r^2 / (2*sigma^2))."""
    return GaussianInteraction(A=A, sigma=sigma)(r)


def true_grad_Phi_gaussian(r, A=1.0, sigma=1.0):
    """Gradient of Gaussian interaction."""
    return GaussianInteraction(A=A, sigma=sigma).gradient(r)


def true_V_harmonic_torch(x, k=1.0):
    """True kinetic potential (torch version)."""
    return 0.5 * k * (x**2).sum(dim=-1)


def true_Phi_gaussian_torch(r, A=1.0, sigma=1.0):
    """True interaction potential (torch version)."""
    return A * torch.exp(-r**2 / (2 * sigma**2))


# ══════════════════════════════════════════════════════════════
# KDE density estimation
# ══════════════════════════════════════════════════════════════

def fast_kde_1d(samples, grid, bw_factor=KDE_BANDWIDTH):
    """FFT-based 1D KDE on a uniform grid.

    ~2000x faster than scipy.stats.gaussian_kde for large sample counts
    because it bins first (O(n)), then convolves via FFT (O(m log m)).
    """
    m = len(grid)
    dr = grid[1] - grid[0]
    lo, hi = grid[0], grid[-1]

    data_std = np.std(samples)
    h = bw_factor * data_std
    if h < 1e-15:
        return np.zeros(m)

    bin_edges = np.concatenate([
        [lo - dr / 2],
        (grid[:-1] + grid[1:]) / 2,
        [hi + dr / 2],
    ])
    counts, _ = np.histogram(samples, bins=bin_edges)
    counts = counts.astype(np.float64)

    k_half = min(m - 1, int(5 * h / dr) + 1)
    k_range = np.arange(-k_half, k_half + 1) * dr
    kernel = np.exp(-0.5 * (k_range / h) ** 2) / (h * np.sqrt(2 * np.pi))

    density = fftconvolve(counts, kernel, mode='same')

    total = np.trapz(density, grid)
    if total > 0:
        density = density / total

    return density


def precompute_kde(data, model_name, d=None):
    """Compute KDE densities on FULL data (no subsampling).

    For rho_V: compute norms |x_i| for ALL particles across ALL ensembles
               and ALL time steps.
    For rho_Phi (radial models): compute pairwise distances |x_i - x_j|
               (upper triangle) for ALL snapshots.

    Args:
        data: (M, L, N, d) particle trajectory data
        model_name: model identifier
        d: dimension override (unused, inferred from data; kept for API compat)

    Returns:
        dict with 'r_V_grid', 'rho_V', 'r_Phi_grid', 'rho_Phi', etc.
    """
    M, L, N, d_actual = validate_data_shape(data, model_name)

    result = {
        'model_name': model_name,
        'd': d_actual,
        'kde_bw': KDE_BANDWIDTH,
    }

    # rho_V: density of particle norms |x_i|
    r_V_grid = get_r_V_grid(model_name, n_points=GRID_SIZE)
    all_pos = data.reshape(-1, d_actual)
    r_norms = np.linalg.norm(all_pos, axis=1)

    mask_V = (r_norms >= r_V_grid[0]) & (r_norms <= r_V_grid[-1])
    r_norms_in_range = r_norms[mask_V]

    if len(r_norms_in_range) >= MIN_KDE_SAMPLES:
        rho_V = fast_kde_1d(r_norms_in_range, r_V_grid, bw_factor=KDE_BANDWIDTH)
    else:
        rho_V = np.zeros_like(r_V_grid)

    result['r_V_grid'] = r_V_grid
    result['rho_V'] = rho_V
    result['n_samples_V'] = int(len(r_norms_in_range))

    # rho_Phi: density of pairwise distances (radial models only)
    if is_radial(model_name):
        r_Phi_grid = get_r_Phi_grid(model_name, n_points=GRID_SIZE)

        snapshots = data.reshape(-1, N, d_actual)
        n_snapshots = snapshots.shape[0]
        triu_i, triu_j = np.triu_indices(N, k=1)
        n_pairs_per_snap = len(triu_i)

        max_bytes = 2 * 1024**3
        chunk_size = max(1, int(max_bytes / (n_pairs_per_snap * 8)))

        all_dists_list = []
        for start in range(0, n_snapshots, chunk_size):
            end = min(start + chunk_size, n_snapshots)
            chunk = snapshots[start:end]
            diffs = chunk[:, triu_i, :] - chunk[:, triu_j, :]
            dists = np.linalg.norm(diffs, axis=-1)
            all_dists_list.append(dists.ravel())

        all_dists = np.concatenate(all_dists_list)

        mask_Phi = (all_dists >= r_Phi_grid[0]) & (all_dists <= r_Phi_grid[-1])
        dists_in_range = all_dists[mask_Phi]

        if len(dists_in_range) >= MIN_KDE_SAMPLES:
            rho_Phi = fast_kde_1d(dists_in_range, r_Phi_grid, bw_factor=KDE_BANDWIDTH)
        else:
            rho_Phi = np.zeros_like(r_Phi_grid)

        result['r_Phi_grid'] = r_Phi_grid
        result['rho_Phi'] = rho_Phi
        result['n_samples_Phi'] = int(len(dists_in_range))

    print(f"[KDE] {model_name} d={d_actual}: "
          f"n_V={result['n_samples_V']}"
          + (f", n_Phi={result.get('n_samples_Phi', 'N/A')}" if is_radial(model_name) else ""))

    return result


def save_kde(kde_data, path):
    """Save precomputed KDE data to .npz file + sidecar JSON for scalars."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arrays = {}
    scalars = {}
    for k, v in kde_data.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            scalars[k] = v

    np.savez_compressed(path, **arrays)

    json_path = path.replace('.npz', '_meta.json')
    with open(json_path, 'w') as f:
        json.dump(scalars, f, indent=2)

    print(f"[KDE] Saved to {path} + {json_path}")


def load_kde(path):
    """Load precomputed KDE data from .npz file + sidecar JSON."""
    npz = np.load(path)
    result = {}
    for k in npz.files:
        result[k] = npz[k]

    json_path = path.replace('.npz', '_meta.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            scalars = json.load(f)
        result.update(scalars)

    return result


def get_kde_path(model_name, d, dt_obs=None, base_dir=None):
    """Return the canonical KDE grid file path."""
    if base_dir is None:
        base_dir = str(_PROJECT_ROOT / 'results' / 'kde_grids')
    if dt_obs is not None:
        return os.path.join(base_dir, f'dt_obs_{dt_obs}', f'd{d}', model_name, 'kde_grids.npz')
    return os.path.join(base_dir, f'd{d}', model_name, 'kde_grids.npz')


# Backward-compat aliases for old KDE names
load_kde_grids = load_kde
compute_kde_grids = precompute_kde
save_kde_grids = save_kde


# ══════════════════════════════════════════════════════════════
# Gradient function constructors
# ══════════════════════════════════════════════════════════════

def make_grad_funcs(alpha, beta, build_V_fn, build_Phi_fn):
    """Construct grad_V and dPhi_dr callables from estimated coefficients.

    Works for both oracle and RBF bases since the build functions have
    identical signatures.
    """
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)

    def grad_V_func(x):
        """Compute nabla V(x) = sum_k alpha_k * psi'_k(|x|) * x/|x|."""
        _, psi_grad, _, r = build_V_fn(x)
        r_safe = np.maximum(r, 1e-10)
        dV_dr = psi_grad @ alpha
        return dV_dr[:, None] * (x / r_safe[:, None])

    def dPhi_dr_func(r_arr):
        """Compute dPhi/dr = sum_l beta_l * phi'_l(r)."""
        _, phi_grad, _ = build_Phi_fn(r_arr, 2)
        return phi_grad @ beta

    return grad_V_func, dPhi_dr_func


# ══════════════════════════════════════════════════════════════
# Oracle / RBF coefficient -> gradient function constructors
# ══════════════════════════════════════════════════════════════

def make_oracle_grad_V(model_name, alpha):
    """Construct predicted grad_V from Oracle basis coefficients."""
    alpha = np.array(alpha)

    if model_name in ('model_a', 'model_e'):
        def gV(x):
            r = np.sqrt(np.sum(x**2, axis=-1, keepdims=True))
            r_safe = np.maximum(r, 1e-10)
            return alpha[0] * x / r_safe + 2.0 * alpha[1] * x
        return gV

    elif model_name in ('model_b', 'model_morse'):
        def gV(x):
            r_sq = np.sum(x**2, axis=-1, keepdims=True)
            return 4.0 * alpha[0] * r_sq * x + 2.0 * alpha[1] * x
        return gV

    elif model_name == 'model_lj':
        def gV(x):
            return 2.0 * alpha[0] * x
        return gV

    else:
        raise ValueError(f"Unknown model for oracle grad_V: {model_name}")


def make_oracle_dPhi_dr(model_name, beta):
    """Construct predicted dPhi/dr from Oracle basis coefficients."""
    beta = np.array(beta)

    if model_name == 'model_a':
        eps = 0.05
        def dPhi(r):
            def sech2(z):
                return 1.0 / np.cosh(np.clip(z, -500, 500))**2
            d_ind1 = 0.5 / eps * (sech2((r - 0.5) / eps) - sech2((r - 1.0) / eps))
            d_ind2 = 0.5 / eps * (sech2((r - 1.0) / eps) - sech2((r - 2.0) / eps))
            return beta[0] * d_ind1 + beta[1] * d_ind2
        return dPhi

    elif model_name == 'model_b':
        def dPhi(r):
            return -beta[0] / (r + 1.0)**2
        return dPhi

    elif model_name == 'model_lj':
        sigma_lj = 0.5
        r_cut = 2.5
        r_safe_factor = 0.7
        def dPhi(r):
            r_safe = r_safe_factor * sigma_lj
            r_c = np.maximum(r, r_safe)
            sr6 = (sigma_lj / r_c)**6
            sr12 = sr6**2
            dphi = beta[0] * (-12.0 * sr12 / r_c) + beta[1] * (-6.0 * sr6 / r_c)
            return np.where(r < r_cut, dphi, 0.0)
        return dPhi

    elif model_name == 'model_morse':
        a = 2.0
        r0 = 0.8
        def dPhi(r):
            exp1 = np.exp(-a * (r - r0))
            exp2 = np.exp(-2.0 * a * (r - r0))
            return -a * beta[0] * exp1 + (-2.0 * a) * beta[1] * exp2
        return dPhi

    elif model_name == 'model_e':
        centers = np.array([0.75, 1.5])
        widths = np.array([0.125, 0.25])
        def dPhi(r):
            result = np.zeros_like(r)
            for k in range(len(beta)):
                z = (r - centers[k]) / widths[k]
                g = np.exp(-0.5 * z**2)
                result += beta[k] * (-z / widths[k]) * g
            return result
        return dPhi

    else:
        raise ValueError(f"Unknown model for oracle dPhi/dr: {model_name}")


def make_rbf_grad_V(alpha, r_max_V, K_V=20):
    """Construct predicted grad_V from RBF coefficients."""
    from lib.basis import rbf_grad_V as _rbf_grad_V
    alpha = np.array(alpha)
    centers_V = np.linspace(0, r_max_V, K_V)
    width_V = r_max_V / K_V

    def gV(X):
        return _rbf_grad_V(X, alpha, centers_V, width_V)
    return gV


def make_rbf_dPhi_dr(beta, r_max_Phi, K_Phi=20):
    """Construct predicted dPhi/dr from RBF coefficients."""
    from lib.basis import rbf_dPhi_dr as _rbf_dPhi_dr
    beta = np.array(beta)
    centers_Phi = np.linspace(0, r_max_Phi, K_Phi)
    width_Phi = r_max_Phi / K_Phi

    def dPhi(r):
        return _rbf_dPhi_dr(r, beta, centers_Phi, width_Phi)
    return dPhi


# ══════════════════════════════════════════════════════════════
# Grid-based L2(rho) quadrature
# ══════════════════════════════════════════════════════════════

def l2_rho_error_quadrature(f_pred, f_true, rho, r_grid):
    """Compute rho-weighted L2 norm via trapezoidal quadrature.

    ||f_pred - f_true||^2_{L2(rho)} ~ sum_k (f_pred(r_k) - f_true(r_k))^2
                                           * rho(r_k) * Delta_r_k
    """
    validate_grid(r_grid, "r_grid")
    assert f_pred.shape == f_true.shape == rho.shape == r_grid.shape, (
        f"Shape mismatch: f_pred={f_pred.shape}, f_true={f_true.shape}, "
        f"rho={rho.shape}, r_grid={r_grid.shape}"
    )

    dr = np.zeros_like(r_grid)
    dr[0] = (r_grid[1] - r_grid[0]) / 2.0
    dr[-1] = (r_grid[-1] - r_grid[-2]) / 2.0
    dr[1:-1] = (r_grid[2:] - r_grid[:-2]) / 2.0

    rho_safe = np.maximum(rho, 0.0)

    diff_sq = (f_pred - f_true) ** 2
    true_sq = f_true ** 2

    numerator = np.sum(diff_sq * rho_safe * dr)
    denominator = np.sum(true_sq * rho_safe * dr)

    if denominator < 1e-20:
        relative_error = float('inf')
    else:
        relative_error = np.sqrt(numerator / denominator)

    return numerator, denominator, relative_error


# ══════════════════════════════════════════════════════════════
# Gradient functions on the grid
# ══════════════════════════════════════════════════════════════

def grad_V_on_grid(grad_V_func, r_V_grid, d=2):
    """Apply a gradient function at radial grid points.

    Creates points x = (r, 0, ..., 0) in R^d, computes grad_V(x),
    and extracts the radial component (index 0).
    """
    n = len(r_V_grid)
    X = np.zeros((n, d))
    X[:, 0] = r_V_grid
    grad_vals = grad_V_func(X)
    dV_dr = grad_vals[:, 0]
    return dV_dr


def dPhi_dr_on_grid(dPhi_dr_func, r_Phi_grid):
    """Apply dPhi/dr on the pairwise distance grid."""
    return dPhi_dr_func(r_Phi_grid)


# ══════════════════════════════════════════════════════════════
# KDE error computation
# ══════════════════════════════════════════════════════════════

def compute_kde_errors(kde_data, grad_V_func, dPhi_dr_func, model_name, d=2):
    """Compute rho-weighted L2 errors using precomputed KDE."""
    assert is_radial(model_name), (
        f"compute_kde_errors only supports radial models, got '{model_name}'"
    )

    r_V_grid = kde_data['r_V_grid']
    rho_V = kde_data['rho_V']
    r_Phi_grid = kde_data['r_Phi_grid']
    rho_Phi = kde_data['rho_Phi']

    true_gV = get_true_grad_V(model_name, d)
    true_dPhi = get_true_dPhi_dr(model_name, d)

    dV_dr_pred = grad_V_on_grid(grad_V_func, r_V_grid, d=d)
    dV_dr_true = grad_V_on_grid(true_gV, r_V_grid, d=d)

    V_num, V_den, V_err = l2_rho_error_quadrature(
        dV_dr_pred, dV_dr_true, rho_V, r_V_grid
    )

    dPhi_dr_pred = dPhi_dr_on_grid(dPhi_dr_func, r_Phi_grid)
    dPhi_dr_true = dPhi_dr_on_grid(true_dPhi, r_Phi_grid)

    Phi_num, Phi_den, Phi_err = l2_rho_error_quadrature(
        dPhi_dr_pred, dPhi_dr_true, rho_Phi, r_Phi_grid
    )

    return {
        'V_error': float(V_err),
        'V_error_pct': float(V_err * 100),
        'Phi_error': float(Phi_err),
        'Phi_error_pct': float(Phi_err * 100),
        'V_numerator': float(V_num),
        'V_denominator': float(V_den),
        'Phi_numerator': float(Phi_num),
        'Phi_denominator': float(Phi_den),
    }


# ══════════════════════════════════════════════════════════════
# High-level evaluation API
# ══════════════════════════════════════════════════════════════

def _resolve_kde_path(model_name, d, dt_obs=None):
    """Resolve path to precomputed KDE grids."""
    return Path(get_kde_path(model_name, d, dt_obs=dt_obs))


def evaluate_kde(model_name, d, alpha, beta, build_V_fn, build_Phi_fn,
                 dt_obs=None, kde_grids=None):
    """Evaluate gradient-based errors using precomputed KDE grids.

    Deterministic evaluation on fixed grid weighted by KDE density.
    For radial models only.
    """
    if kde_grids is None:
        kde_path = _resolve_kde_path(model_name, d, dt_obs)
        if not kde_path.exists():
            raise FileNotFoundError(
                f"KDE grids not found at {kde_path}. "
                f"Run: python scripts/precompute_kde_rho.py --models {model_name.replace('model_', '')} --dims {d}"
            )
        kde_grids = load_kde(str(kde_path))

    grad_V_func, dPhi_dr_func = make_grad_funcs(
        alpha, beta, build_V_fn, build_Phi_fn
    )
    errs = compute_kde_errors(kde_grids, grad_V_func, dPhi_dr_func,
                              model_name, d=d)
    return errs['V_error'], errs['Phi_error']
