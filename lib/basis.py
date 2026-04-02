"""Unified basis function interface for V and Phi estimation.

Supports oracle (true functional form) and RBF (generic Gaussian RBF) bases.
Each basis provides (val, grad, laplacian) tuples for use in MLE, self-test,
and Sinkhorn solvers.

Unified interface:
    build_V_fn(x)            -> (psi, psi_grad_r, psi_lap, r)
    build_Phi_fn(r_pairs, d) -> (phi, phi_grad_r, phi_lap)
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# RBF basis primitives
# ══════════════════════════════════════════════════════════════

def rbf_val(r, centers, width):
    """Gaussian RBF: phi_k(r) = exp(-(r-c_k)^2/(2w^2)). Returns (n, K)."""
    return np.exp(-((r[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))


def rbf_grad(r, centers, width):
    """d/dr of RBF. Returns (n, K)."""
    return -(r[:, None] - centers[None, :]) / (width ** 2) * rbf_val(r, centers, width)


def _rbf_lap_1d(r, centers, width):
    """d^2/dr^2 of RBF. Returns (n, K)."""
    z = (r[:, None] - centers[None, :]) / width
    return (z ** 2 - 1) / (width ** 2) * rbf_val(r, centers, width)


def _rbf_lap_radial(r, d, centers, width):
    """Radial Laplacian in d dims: f''(r) + (d-1)/r * f'(r). Returns (n, K)."""
    r_safe = np.maximum(r, 1e-10)
    return (_rbf_lap_1d(r, centers, width)
            + (d - 1) / r_safe[:, None] * rbf_grad(r, centers, width))


def rbf_grad_V(x, alpha, centers_V, width_V):
    """Compute ∇V̂(x) for RBF basis: ∇V = Σ αₖ ψₖ'(|x|)·x/|x|.
    x: (n, d), returns: (n, d)
    """
    alpha = np.asarray(alpha)
    r = np.sqrt(np.sum(x**2, axis=-1))  # (n,)
    r_safe = np.maximum(r, 1e-10)
    psi_g = rbf_grad(r, centers_V, width_V)  # (n, K_V)
    dV_dr = psi_g @ alpha  # (n,)
    return dV_dr[:, None] * (x / r_safe[:, None])  # (n, d)


def rbf_dPhi_dr(r, beta, centers_Phi, width_Phi):
    """Compute dΦ̂/dr for RBF basis: dΦ/dr = Σ βₗ φₗ'(r).
    r: (n,), returns: (n,)
    """
    beta = np.asarray(beta)
    phi_g = rbf_grad(r, centers_Phi, width_Phi)  # (n, K_Phi)
    return phi_g @ beta  # (n,)


def _build_V_rbf(x, centers, width):
    """RBF V basis evaluated at particle positions."""
    d = x.shape[-1]
    r = np.sqrt(np.sum(x ** 2, axis=-1))
    return (rbf_val(r, centers, width),
            rbf_grad(r, centers, width),
            _rbf_lap_radial(r, d, centers, width),
            r)


def _build_Phi_rbf(r_pairs, d, centers, width):
    """RBF Phi basis evaluated at pairwise distances."""
    return (rbf_val(r_pairs, centers, width),
            rbf_grad(r_pairs, centers, width),
            _rbf_lap_radial(r_pairs, d, centers, width))


# ══════════════════════════════════════════════════════════════
# Oracle basis: Model A
#   V = alpha1/2 * |x| + alpha2 * |x|^2
#   Phi = beta1 * I_{[0.5,1]} + beta2 * I_{[1,2]}
# ══════════════════════════════════════════════════════════════

def _build_V_oracle_a(x):
    """Oracle V for model_a: {|x|, |x|^2}. True alpha=(-0.5, 2.0)."""
    d = x.shape[-1]
    r = np.sqrt(np.sum(x ** 2, axis=-1))
    r_safe = np.maximum(r, 1e-10)
    psi = np.column_stack([r, r ** 2])
    psi_grad = np.column_stack([np.ones_like(r), 2 * r])
    # Lap(|x|) = (d-1)/r,  Lap(|x|^2) = 2d
    psi_lap = np.column_stack([(d - 1) / r_safe, 2 * d * np.ones_like(r)])
    return psi, psi_grad, psi_lap, r


def _build_Phi_oracle_a(r_pairs, d, eps=0.05):
    """Oracle Phi for model_a: {I_{[0.5,1]}, I_{[1,2]}}. True beta=(-3.0, 2.0)."""
    clip = 50.0

    def _si(r, lo, hi):
        return 0.5 * (np.tanh(np.clip((r - lo) / eps, -clip, clip))
                       - np.tanh(np.clip((r - hi) / eps, -clip, clip)))

    def _si_g(r, lo, hi):
        sech2_lo = 1.0 / np.cosh(np.clip((r - lo) / eps, -clip, clip)) ** 2
        sech2_hi = 1.0 / np.cosh(np.clip((r - hi) / eps, -clip, clip)) ** 2
        return 0.5 / eps * (sech2_lo - sech2_hi)

    def _si_g2(r, lo, hi):
        t_lo = np.clip((r - lo) / eps, -clip, clip)
        t_hi = np.clip((r - hi) / eps, -clip, clip)
        sech2_lo = 1.0 / np.cosh(t_lo) ** 2
        sech2_hi = 1.0 / np.cosh(t_hi) ** 2
        return -1.0 / eps ** 2 * (sech2_lo * np.tanh(t_lo) - sech2_hi * np.tanh(t_hi))

    phi = np.column_stack([_si(r_pairs, 0.5, 1.0), _si(r_pairs, 1.0, 2.0)])
    phi_grad = np.column_stack([_si_g(r_pairs, 0.5, 1.0), _si_g(r_pairs, 1.0, 2.0)])
    phi_g2 = np.column_stack([_si_g2(r_pairs, 0.5, 1.0), _si_g2(r_pairs, 1.0, 2.0)])

    r_safe = np.maximum(r_pairs, 1e-10)
    phi_lap = phi_g2 + (d - 1) / r_safe[:, None] * phi_grad
    return phi, phi_grad, phi_lap


# ══════════════════════════════════════════════════════════════
# Oracle basis: Model B
#   V = (|x|^2 - 1)^2 / 4  =>  V = 0.25*|x|^4 - 0.5*|x|^2 + const
#   Phi = 0.5 / (r + 1)
# ══════════════════════════════════════════════════════════════

def _build_V_oracle_b(x):
    """Oracle V for model_b: {|x|^4, |x|^2}. True alpha=(0.25, -0.5)."""
    d = x.shape[-1]
    r = np.sqrt(np.sum(x ** 2, axis=-1))
    psi = np.column_stack([r ** 4, r ** 2])
    psi_grad = np.column_stack([4 * r ** 3, 2 * r])
    # Lap(r^4) = 12r^2 + 4(d-1)r^2 = (4d+8)r^2
    # Lap(r^2) = 2 + 2(d-1) = 2d
    psi_lap = np.column_stack([(4 * d + 8) * r ** 2, 2 * d * np.ones_like(r)])
    return psi, psi_grad, psi_lap, r


def _build_Phi_oracle_b(r_pairs, d):
    """Oracle Phi for model_b: {1/(r+1)}. True beta=(0.5,)."""
    rp1 = r_pairs + 1.0
    r_safe = np.maximum(r_pairs, 1e-10)
    phi = (1.0 / rp1)[:, None]
    phi_grad = (-1.0 / rp1 ** 2)[:, None]
    phi_g2 = (2.0 / rp1 ** 3)[:, None]
    phi_lap = phi_g2 + (d - 1) / r_safe[:, None] * phi_grad
    return phi, phi_grad, phi_lap


# ══════════════════════════════════════════════════════════════
# Oracle basis: Model LJ
#   V = 0.5 * k * |x|^2  with k=2  =>  V = |x|^2
#   Phi = 4*eps*[(sigma/r)^12 - (sigma/r)^6]  (truncated at r_cut)
# ══════════════════════════════════════════════════════════════

def _build_V_oracle_lj(x):
    """Oracle V for model_lj: {|x|^2}. True alpha=(1.0,)."""
    d = x.shape[-1]
    r = np.sqrt(np.sum(x ** 2, axis=-1))
    psi = r[:, None] ** 2
    psi_grad = (2 * r)[:, None]
    psi_lap = (2 * d * np.ones_like(r))[:, None]
    return psi, psi_grad, psi_lap, r


def _build_Phi_oracle_lj(r_pairs, d, sigma_lj=0.5, r_safe_min=0.35):
    """Oracle Phi for model_lj: {(sigma/r)^12, (sigma/r)^6}. True beta=(2.0, -2.0)."""
    r_c = np.maximum(r_pairs, r_safe_min)
    r_safe = np.maximum(r_pairs, 1e-10)

    sr = sigma_lj / r_c
    sr6 = sr ** 6
    sr12 = sr ** 12

    phi = np.column_stack([sr12, sr6])
    # d/dr (sigma/r)^n = -n * sigma^n / r^(n+1) = -n/r * (sigma/r)^n
    phi_grad = np.column_stack([-12.0 / r_c * sr12, -6.0 / r_c * sr6])
    # d^2/dr^2 (sigma/r)^n = n(n+1)/r^2 * (sigma/r)^n
    phi_g2 = np.column_stack([156.0 / r_c ** 2 * sr12, 42.0 / r_c ** 2 * sr6])

    phi_lap = phi_g2 + (d - 1) / r_safe[:, None] * phi_grad
    return phi, phi_grad, phi_lap


# ══════════════════════════════════════════════════════════════
# Oracle basis: Model Morse
#   V = (|x|^2 - 1)^2 / 4  (same as Model B)
#   Phi = D * (1 - exp(-a*(r-r0)))^2
#       = D - 2D*exp(-a(r-r0)) + D*exp(-2a(r-r0))
# ══════════════════════════════════════════════════════════════

def _build_V_oracle_morse(x):
    """Oracle V for model_morse: {|x|^4, |x|^2}. Same as model_b."""
    return _build_V_oracle_b(x)


def _build_Phi_oracle_morse(r_pairs, d, a=2.0, r0=0.8):
    """Oracle Phi for model_morse: {exp(-a(r-r0)), exp(-2a(r-r0))}.
    True beta=(-1.0, 0.5)."""
    r_safe = np.maximum(r_pairs, 1e-10)

    e1 = np.exp(-a * (r_pairs - r0))
    e2 = np.exp(-2 * a * (r_pairs - r0))

    phi = np.column_stack([e1, e2])
    phi_grad = np.column_stack([-a * e1, -2 * a * e2])
    phi_g2 = np.column_stack([a ** 2 * e1, 4 * a ** 2 * e2])
    phi_lap = phi_g2 + (d - 1) / r_safe[:, None] * phi_grad
    return phi, phi_grad, phi_lap


# ══════════════════════════════════════════════════════════════
# Oracle basis: Model E
#   V = same as Model A
#   Phi = beta1*g1(r) + beta2*g2(r),  g_k(r) = exp(-(r-c_k)^2/(2*sigma_k^2))
# ══════════════════════════════════════════════════════════════

def _build_V_oracle_e(x):
    """Oracle V for model_e: same as model_a. True alpha=(-0.5, 2.0)."""
    return _build_V_oracle_a(x)


def _build_Phi_oracle_e(r_pairs, d, centers=(0.75, 1.5), widths=(0.125, 0.25)):
    """Oracle Phi for model_e: Gaussian bumps. True beta=(-3.0, 2.0)."""
    centers = np.array(centers)
    widths = np.array(widths)
    r_safe = np.maximum(r_pairs, 1e-10)

    phi_cols, grad_cols, g2_cols = [], [], []
    for c, w in zip(centers, widths):
        z = (r_pairs - c) / w
        g = np.exp(-0.5 * z**2)
        phi_cols.append(g)
        grad_cols.append(-z / w * g)           # dg/dr
        g2_cols.append((z**2 - 1) / w**2 * g)  # d²g/dr²

    phi = np.column_stack(phi_cols)
    phi_grad = np.column_stack(grad_cols)
    phi_g2 = np.column_stack(g2_cols)
    phi_lap = phi_g2 + (d - 1) / r_safe[:, None] * phi_grad
    return phi, phi_grad, phi_lap


# ══════════════════════════════════════════════════════════════
# Registry and unified interface
# ══════════════════════════════════════════════════════════════

# (builder_fn, K) for each model
ORACLE_V_BUILDERS = {
    'model_a': (_build_V_oracle_a, 2),
    'model_b': (_build_V_oracle_b, 2),
    'model_lj': (_build_V_oracle_lj, 1),
    'model_morse': (_build_V_oracle_morse, 2),
    'model_e': (_build_V_oracle_e, 2),
}

ORACLE_PHI_BUILDERS = {
    'model_a': (_build_Phi_oracle_a, 2),
    'model_b': (_build_Phi_oracle_b, 1),
    'model_lj': (_build_Phi_oracle_lj, 2),
    'model_morse': (_build_Phi_oracle_morse, 2),
    'model_e': (_build_Phi_oracle_e, 2),
}

TRUE_PARAMS = {
    'model_a': {'alpha': [-0.5, 2.0], 'beta': [-3.0, 2.0]},
    'model_b': {'alpha': [0.25, -0.5], 'beta': [0.5]},
    'model_lj': {'alpha': [1.0], 'beta': [2.0, -2.0]},
    'model_morse': {'alpha': [0.25, -0.5], 'beta': [-1.0, 0.5]},
    'model_e': {'alpha': [-0.5, 2.0], 'beta': [-3.0, 2.0]},
}


def get_basis(basis_type, model_name, K_V=20, K_Phi=20,
              r_max_V=5.0, r_max_Phi=5.0):
    """Unified basis interface.

    Args:
        basis_type: 'oracle' or 'rbf'
        model_name: e.g. 'model_a'
        K_V, K_Phi: number of basis functions (RBF only)
        r_max_V, r_max_Phi: upper radius for centers (RBF only)

    Returns:
        build_V_fn:  callable(x) -> (psi, psi_grad_r, psi_lap, r)
        build_Phi_fn: callable(r_pairs, d) -> (phi, phi_grad_r, phi_lap)
        K_V:  actual number of V basis functions
        K_Phi: actual number of Phi basis functions
        basis_info: dict with metadata for evaluation
    """
    if basis_type == 'oracle':
        if model_name not in ORACLE_V_BUILDERS:
            raise ValueError(f"No oracle basis for {model_name}")

        build_V_fn, K_V_actual = ORACLE_V_BUILDERS[model_name]
        build_Phi_raw, K_Phi_actual = ORACLE_PHI_BUILDERS[model_name]

        # Wrap to match unified signature (r_pairs, d) -> (phi, grad, lap)
        def build_Phi_fn(r_pairs, d):
            return build_Phi_raw(r_pairs, d)

        basis_info = {
            'type': 'oracle',
            'model': model_name,
            'true_params': TRUE_PARAMS[model_name],
        }
        return build_V_fn, build_Phi_fn, K_V_actual, K_Phi_actual, basis_info

    elif basis_type == 'rbf':
        centers_V = np.linspace(0.01, r_max_V, K_V)
        width_V = 1.5 * r_max_V / K_V
        centers_Phi = np.linspace(0.01, r_max_Phi, K_Phi)
        width_Phi = 1.5 * r_max_Phi / K_Phi

        def build_V_fn(x):
            return _build_V_rbf(x, centers_V, width_V)

        def build_Phi_fn(r_pairs, d):
            return _build_Phi_rbf(r_pairs, d, centers_Phi, width_Phi)

        basis_info = {
            'type': 'rbf',
            'centers_V': centers_V,
            'width_V': width_V,
            'centers_Phi': centers_Phi,
            'width_Phi': width_Phi,
        }
        return build_V_fn, build_Phi_fn, K_V, K_Phi, basis_info

    else:
        raise ValueError(f"Unknown basis_type: {basis_type}")
