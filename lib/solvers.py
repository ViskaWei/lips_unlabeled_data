"""MLE, Sinkhorn+MLE, and self-test solvers with generic basis interface.

All solvers accept build_V_fn and build_Phi_fn from lib.basis.get_basis().
"""

import time

import numpy as np

# ══════════════════════════════════════════════════════════════
# Sinkhorn label imputation
# ══════════════════════════════════════════════════════════════

def sinkhorn_matching(X_curr, X_next, eps_factor=0.01, max_iter=100, tol=1e-9):
    """Find permutation matching particles between consecutive snapshots.

    Uses the Sinkhorn-Knopp algorithm to solve the entropy-regularised
    optimal transport problem, then extracts a hard permutation via argmax.

    Args:
        X_curr: (N, d) current particle positions
        X_next: (N, d) next particle positions
        eps_factor: regularisation as fraction of median cost
        max_iter: maximum Sinkhorn iterations
        tol: convergence tolerance

    Returns:
        perm: (N,) integer permutation such that X_next[perm] ≈ matched X_curr
    """
    # Cost matrix: squared Euclidean distances
    diff = X_curr[:, None, :] - X_next[None, :, :]
    C = np.sum(diff ** 2, axis=-1)  # (N, N)

    # Regularisation parameter
    eps = eps_factor * np.median(C)
    eps = max(eps, 1e-10)

    # Gibbs kernel
    K = np.exp(-C / eps)

    # Sinkhorn iterations (log-domain not needed for N~10)
    u = np.ones(K.shape[0])
    for _ in range(max_iter):
        v = 1.0 / (K.T @ u)
        u_new = 1.0 / (K @ v)
        if np.max(np.abs(u_new - u)) < tol:
            u = u_new
            break
        u = u_new

    # Transport plan
    P = u[:, None] * K * v[None, :]

    # Extract hard permutation (row-wise argmax)
    perm = np.argmax(P, axis=1)

    return perm


# ══════════════════════════════════════════════════════════════
# Hansen L-curve: SVD-based optimal Tikhonov parameter
# ══════════════════════════════════════════════════════════════

def _block_sqrt_transform(A, K_V, K_Phi):
    """Build C^{-1/2} = diag(A_VV^{1/2}, A_ΦΦ^{1/2}) for block-inverse Tikhonov.

    Given C = diag(A_VV^{-1}, A_ΦΦ^{-1}), the substitution θ = C^{-1/2} z
    transforms (A + λC)θ = b into standard form (Ã + λI)z = b̃ where
    Ã = C^{-1/2} A C^{-1/2} and b̃ = C^{-1/2} b.

    Returns:
        T: (K, K) block-diagonal matrix C^{-1/2}
        block_info: dict with per-block condition numbers
    """
    K = K_V + K_Phi
    A_VV = A[:K_V, :K_V]
    A_PP = A[K_V:, K_V:]

    eigV, QV = np.linalg.eigh(A_VV)
    eigP, QP = np.linalg.eigh(A_PP)
    eigV = np.maximum(eigV, 1e-12)
    eigP = np.maximum(eigP, 1e-12)

    T = np.zeros((K, K))
    T[:K_V, :K_V] = QV @ np.diag(np.sqrt(eigV)) @ QV.T
    T[K_V:, K_V:] = QP @ np.diag(np.sqrt(eigP)) @ QP.T

    block_info = {
        'cond_V_block': float(eigV[-1] / eigV[0]),
        'cond_Phi_block': float(eigP[-1] / eigP[0]),
        'eig_V_range': [float(eigV[0]), float(eigV[-1])],
        'eig_Phi_range': [float(eigP[0]), float(eigP[-1])],
    }
    return T, block_info


def _hansen_lcurve_full_inverse(A, b, n_lambdas=200):
    """L-curve for (A + λA^{-1})θ = b using SVD (no matrix inversion needed).

    If A = UΣU^T, then A + λA^{-1} = U(Σ + λΣ^{-1})U^T.
    Solution: θ = U diag(1/(σ_i + λ/σ_i)) U^T b.

    Regularization norm: ||θ||²_{A^{-1}} = θ^T A^{-1} θ = Σ θ_i²/σ_i.
    This penalizes directions with little data information (small σ_i) more,
    which is natural for learning nonlocal kernels from sparse pairwise data.

    Returns:
        lam_opt, info (same format as _hansen_lcurve_lambda)
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    beta = U.T @ b
    s = np.maximum(s, 1e-12)  # clamp for stability

    lam_lo = s.min() ** 2 * 1e-4
    lam_hi = s.max() ** 2 * 10
    lambdas = np.logspace(np.log10(max(lam_lo, 1e-16)), np.log10(lam_hi), n_lambdas)

    kappa = np.full(n_lambdas, -np.inf)
    for i, lam in enumerate(lambdas):
        # Effective eigenvalues: D_i = σ_i + λ/σ_i = (σ_i² + λ)/σ_i
        p = s ** 2 + lam  # σ_i² + λ

        # Residual: (Aθ - b)_i = -λ β_i / (σ_i² + λ)
        rho2 = np.sum(lam ** 2 * beta ** 2 / p ** 2)

        # Solution norm: θ_i = β_i σ_i / (σ_i² + λ)
        eta2 = np.sum(s ** 2 * beta ** 2 / p ** 2)

        if rho2 < 1e-30 or eta2 < 1e-30:
            continue

        # Analytical derivatives w.r.t. λ
        # Helper sum: Σ σ²β²/(σ²+λ)³
        S3 = np.sum(s ** 2 * beta ** 2 / p ** 3)

        # dρ²/dλ = 2λ Σ σ²β²/(σ²+λ)³
        d_rho2 = 2 * lam * S3

        # d²ρ²/dλ² = 2 Σ σ²β²(σ² - 2λ)/(σ²+λ)⁴
        d2_rho2 = 2 * np.sum(s ** 2 * beta ** 2 * (s ** 2 - 2 * lam) / p ** 4)

        # dη²/dλ = -2 Σ σ²β²/(σ²+λ)³
        d_eta2 = -2 * S3

        # d²η²/dλ² = 6 Σ σ²β²/(σ²+λ)⁴
        d2_eta2 = 6 * np.sum(s ** 2 * beta ** 2 / p ** 4)

        # Log-space derivatives: ξ = log(ρ), η_l = log(η)
        xi_p = d_rho2 / (2 * rho2)
        eta_p = d_eta2 / (2 * eta2)

        xi_pp = (d2_rho2 * rho2 - d_rho2 ** 2 / rho2) / (4 * rho2 ** 2)
        eta_pp = (d2_eta2 * eta2 - d_eta2 ** 2 / eta2) / (4 * eta2 ** 2)

        num = xi_p * eta_pp - eta_p * xi_pp
        den = (xi_p ** 2 + eta_p ** 2) ** 1.5
        kappa[i] = num / den if den > 1e-30 else 0

    LAMBDA_FLOOR = 1e-10

    max_kappa = np.max(kappa)
    idx = np.argmax(kappa)

    if max_kappa > 0:
        lam_opt = float(lambdas[idx])
    elif abs(max_kappa) > 0.01:
        lam_opt = float(lambdas[idx])
    else:
        lam_opt = float(s[-1] ** 2 * 1e-6)

    below_floor = lam_opt < LAMBDA_FLOOR
    if below_floor:
        lam_opt = LAMBDA_FLOOR

    return lam_opt, {
        'singular_values': s.tolist(),
        'lambda_range': [float(lambdas[0]), float(lambdas[-1])],
        'lambda_selected': lam_opt,
        'curvature_max': float(max_kappa),
        'has_corner': bool(max_kappa > 0),
        'below_floor': bool(below_floor),
    }


def _solve_full_inverse(A, b, reg='auto'):
    """Solve (A + λA^{-1})θ = b via SVD. No matrix inversion needed.

    A = UΣU^T → θ = U diag(σ_i/(σ_i² + λ)) U^T b
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.maximum(s, 1e-12)
    beta = U.T @ b

    if reg == 'auto':
        lam, lcurve_info = _hansen_lcurve_full_inverse(A, b)
        reg_method = 'hansen_lcurve_full_inverse'
    else:
        # For fixed reg, use adaptive scaling like selftest
        reg_scale = np.trace(A) / len(s)
        lam = float(reg) * reg_scale
        lcurve_info = None
        reg_method = 'adaptive_fixed_full_inverse'

    # Solve in eigenbasis: θ_i = β_i / (σ_i + λ/σ_i) = β_i σ_i / (σ_i² + λ)
    D = s + lam / s
    theta_eig = beta / D
    theta = U @ theta_eig

    info = {
        'reg_effective': float(lam),
        'reg_method': reg_method,
        'cond_number': float(D.max() / D.min()),
        'effective_eigenvalues': D.tolist(),
    }
    if lcurve_info:
        info['lcurve'] = lcurve_info
    return theta, info


def _hansen_lcurve_lambda(A, b, n_lambdas=200):
    """Select Tikhonov parameter via L-curve maximum curvature (Hansen 1992).

    Computes the curvature of the (log||r||, log||θ||) curve analytically
    using the SVD of A, and returns the λ at maximum curvature.

    Args:
        A: (K, K) symmetric positive semi-definite matrix
        b: (K,) right-hand side
        n_lambdas: number of λ candidates to evaluate

    Returns:
        lam_opt: optimal regularisation parameter
        info: dict with diagnostics (singular values, chosen index, etc.)
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    beta = U.T @ b  # Fourier coefficients

    # Search range: from well below σ_min² to above σ_max²
    lam_lo = s.min() ** 2 * 1e-4
    lam_hi = s.max() ** 2 * 10
    lambdas = np.logspace(np.log10(max(lam_lo, 1e-16)), np.log10(lam_hi), n_lambdas)

    kappa = np.full(n_lambdas, -np.inf)
    for i, lam in enumerate(lambdas):
        f = s ** 2 / (s ** 2 + lam)

        rho2 = np.sum((1 - f) ** 2 * beta ** 2)
        eta2 = np.sum(f ** 2 * beta ** 2 / s ** 2)
        if rho2 < 1e-30 or eta2 < 1e-30:
            continue

        # Analytical derivatives (Hansen 2001, §4.3)
        phi = np.sum(beta ** 2 * s ** 2 / (s ** 2 + lam) ** 3)
        psi = np.sum(beta ** 2 * s ** 4 / (s ** 2 + lam) ** 3)
        phi_p = np.sum(beta ** 2 * s ** 2 / (s ** 2 + lam) ** 4)
        psi_p = np.sum(beta ** 2 * s ** 4 / (s ** 2 + lam) ** 4)

        # ξ = log(ρ), η_l = log(η)  — derivatives w.r.t. λ
        xi_p = lam * phi / rho2
        eta_p = -psi / eta2

        d_rho2 = 2 * lam * phi
        d2_rho2 = 2 * phi - 6 * lam * phi_p
        xi_pp = (d2_rho2 * rho2 - d_rho2 ** 2 / rho2) / (4 * rho2 ** 2)

        d_eta2 = -2 * psi
        d2_eta2 = 6 * psi_p
        eta_pp = (d2_eta2 * eta2 - d_eta2 ** 2 / eta2) / (4 * eta2 ** 2)

        num = xi_p * eta_pp - eta_p * xi_pp
        den = (xi_p ** 2 + eta_p ** 2) ** 1.5
        kappa[i] = num / den if den > 1e-30 else 0

    max_kappa = np.max(kappa)
    idx = np.argmax(kappa)

    # Machine-precision floor: if L-curve picks λ below this,
    # regularization is numerically meaningless — use lstsq instead.
    LAMBDA_FLOOR = 1e-10

    if max_kappa > 0:
        # Clear L-curve corner
        lam_opt = float(lambdas[idx])
    else:
        # No positive corner: fall back to numerical-stability floor.
        lam_opt = float(s[-1] ** 2 * 1e-6)

    # Clamp to machine-precision floor
    below_floor = lam_opt < LAMBDA_FLOOR
    if below_floor:
        lam_opt = LAMBDA_FLOOR

    return lam_opt, {
        'singular_values': s.tolist(),
        'lambda_range': [float(lambdas[0]), float(lambdas[-1])],
        'lambda_selected': lam_opt,
        'curvature_max': float(max_kappa),
        'has_corner': bool(max_kappa > 0),
        'below_floor': bool(below_floor),
    }


# Public alias for the L-curve function
hansen_lcurve_lambda = _hansen_lcurve_lambda


# ══════════════════════════════════════════════════════════════
# Shared: vectorized normal equations for velocity regression
# ══════════════════════════════════════════════════════════════

def _accumulate_snapshot(X, vel, N, d, K_V, K_Phi, K_total,
                         build_V_fn, build_Phi_fn, ATA, ATb):
    """Accumulate normal equations from one snapshot.

    The velocity model is:
        v_i = -grad_V(x_i) - (1/N) sum_{j!=i} grad_Phi(x_i - x_j)
            = F_i^T @ theta

    where theta = (alpha, beta) and F_i encodes the basis gradients.
    """
    # V basis gradients at particle positions
    _, psi_grad, _, r = build_V_fn(X)  # (N, K_V), ..., (N,)
    r_safe = np.maximum(r, 1e-10)

    # F_V[i, d_dim, k] = -psi'_k(r_i) * X_i[d_dim] / r_i
    x_hat = X / r_safe[:, None]  # (N, d) unit radial directions
    F_V = -psi_grad[:, None, :] * x_hat[:, :, None]  # (N, d, K_V)

    # Pairwise distances and Phi basis gradients
    diff = X[:, None, :] - X[None, :, :]  # (N, N, d)
    r_pairs = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, N)
    r_pairs_flat = r_pairs.reshape(-1)

    _, phi_grad_flat, _ = build_Phi_fn(r_pairs_flat, d)  # (N*N, K_Phi)
    phi_grad_mat = phi_grad_flat.reshape(N, N, K_Phi)

    r_pairs_safe = np.maximum(r_pairs, 1e-10)
    unit_diff = diff / r_pairs_safe[:, :, None]  # (N, N, d)
    mask = 1.0 - np.eye(N)

    # F_Phi[i, d_dim, l] = -(1/N) sum_{j!=i} phi'_l(r_ij) * unit_ij[d_dim]
    F_Phi_all = unit_diff[:, :, :, None] * phi_grad_mat[:, :, None, :]  # (N,N,d,K_Phi)
    F_Phi_all *= mask[:, :, None, None]
    F_Phi = -F_Phi_all.sum(axis=1) / N  # (N, d, K_Phi)

    # Total force design: (N, d, K_total)
    F = np.concatenate([F_V, F_Phi], axis=-1)

    # Accumulate normal equations: ATA += sum_i F_i^T @ F_i, ATb += sum_i F_i^T @ v_i
    ATA += np.einsum('ndk,ndl->kl', F, F)
    ATb += np.einsum('ndk,nd->k', F, vel)


# ══════════════════════════════════════════════════════════════
# Solver 1: MLE (labeled trajectories -> velocity regression)
# ══════════════════════════════════════════════════════════════

def solve_mle(data_labeled, t_obs, build_V_fn, build_Phi_fn, K_V, K_Phi,
              reg='auto', M_max=None, reg_type='identity'):
    """MLE velocity regression using labeled (trajectory) data.

    Args:
        data_labeled: (M, L, N, d)
        t_obs: (L,) observation times
        build_V_fn, build_Phi_fn: from get_basis()
        K_V, K_Phi: basis sizes
        reg: Tikhonov regularisation. 'auto' (default) uses Hansen L-curve;
             a float value uses fixed regularisation.
        M_max: use at most this many trajectories

    Returns:
        alpha: (K_V,) V coefficients
        beta: (K_Phi,) Phi coefficients
        info: dict with timing and metadata
    """
    M_data, L, N, d = data_labeled.shape
    M = min(M_data, M_max) if M_max else M_data
    K_total = K_V + K_Phi
    dt = t_obs[1] - t_obs[0]

    ATA = np.zeros((K_total, K_total))
    ATb = np.zeros(K_total)

    t0 = time.time()
    for m in range(M):
        if m % 200 == 0 and m > 0:
            print(f"  MLE: trajectory {m}/{M}")
        for ell in range(L - 1):
            X = data_labeled[m, ell]
            vel = (data_labeled[m, ell + 1] - data_labeled[m, ell]) / dt
            _accumulate_snapshot(X, vel, N, d, K_V, K_Phi, K_total,
                                 build_V_fn, build_Phi_fn, ATA, ATb)

    # Normalise so that L-curve sees O(1) entries (same as selftest)
    n_samples = M * (L - 1)
    ATA /= n_samples
    ATb /= n_samples

    # ── Regularise and solve ──
    if reg_type == 'full_inverse':
        theta, solve_info = _solve_full_inverse(ATA, ATb, reg=reg)
        reg_method = solve_info['reg_method']
        reg_effective = solve_info['reg_effective']
        lcurve_info = solve_info.get('lcurve')
    elif reg_type == 'block_inverse':
        T, block_info = _block_sqrt_transform(ATA, K_V, K_Phi)
        A_work = T @ ATA @ T
        b_work = T @ ATb
        if reg == 'auto':
            reg_effective, lcurve_info = _hansen_lcurve_lambda(A_work, b_work)
            reg_method = 'hansen_lcurve_block_inverse'
        else:
            reg_effective = float(reg) / n_samples
            lcurve_info = None
            reg_method = 'fixed_block_inverse'
        z = np.linalg.solve(A_work + reg_effective * np.eye(K_total), b_work)
        theta = T @ z
    else:  # identity
        if reg == 'auto':
            reg_effective, lcurve_info = _hansen_lcurve_lambda(ATA, ATb)
            reg_method = 'hansen_lcurve'
        else:
            reg_effective = float(reg) / n_samples
            lcurve_info = None
            reg_method = 'fixed'
        theta = np.linalg.solve(ATA + reg_effective * np.eye(K_total), ATb)

    elapsed = time.time() - t0
    alpha = theta[:K_V]
    beta = theta[K_V:]

    info = {'time_s': elapsed, 'M_used': M, 'method': 'mle',
            'reg_effective': float(reg_effective), 'reg_method': reg_method,
            'reg_type': reg_type}
    if lcurve_info:
        info['lcurve'] = lcurve_info
    return alpha, beta, info


# ══════════════════════════════════════════════════════════════
# Solver 2: Sinkhorn + MLE (unlabeled -> match -> regression)
# ══════════════════════════════════════════════════════════════

def solve_sinkhorn(data_unlabeled, t_obs, build_V_fn, build_Phi_fn, K_V, K_Phi,
                   reg='auto', eps_factor=0.01, M_max=None, reg_type='identity'):
    """Sinkhorn label imputation followed by MLE velocity regression.

    Args:
        data_unlabeled: (M, L, N, d)
        t_obs: (L,) observation times
        build_V_fn, build_Phi_fn: from get_basis()
        K_V, K_Phi: basis sizes
        reg: Tikhonov regularisation. 'auto' (default) uses Hansen L-curve;
             a float value uses fixed regularisation.
        eps_factor: Sinkhorn regularisation
        M_max: use at most this many trajectories

    Returns:
        alpha, beta, info
    """
    M_data, L, N, d = data_unlabeled.shape
    M = min(M_data, M_max) if M_max else M_data
    K_total = K_V + K_Phi
    dt = t_obs[1] - t_obs[0]

    ATA = np.zeros((K_total, K_total))
    ATb = np.zeros(K_total)

    t0 = time.time()
    for m in range(M):
        if m % 100 == 0:
            print(f"  Sinkhorn+MLE: trajectory {m}/{M}")
        for ell in range(L - 1):
            X_curr = data_unlabeled[m, ell]
            X_next = data_unlabeled[m, ell + 1]

            perm = sinkhorn_matching(X_curr, X_next, eps_factor=eps_factor)
            X_matched = X_next[perm]
            vel = (X_matched - X_curr) / dt

            _accumulate_snapshot(X_curr, vel, N, d, K_V, K_Phi, K_total,
                                 build_V_fn, build_Phi_fn, ATA, ATb)

    # Normalise so that L-curve sees O(1) entries (same as selftest)
    n_samples = M * (L - 1)
    ATA /= n_samples
    ATb /= n_samples

    # ── Regularise and solve ──
    if reg_type == 'full_inverse':
        theta, solve_info = _solve_full_inverse(ATA, ATb, reg=reg)
        reg_method = solve_info['reg_method']
        reg_effective = solve_info['reg_effective']
        lcurve_info = solve_info.get('lcurve')
    elif reg_type == 'block_inverse':
        T, _ = _block_sqrt_transform(ATA, K_V, K_Phi)
        A_work = T @ ATA @ T
        b_work = T @ ATb
        if reg == 'auto':
            reg_effective, lcurve_info = _hansen_lcurve_lambda(A_work, b_work)
            reg_method = 'hansen_lcurve_block_inverse'
        else:
            reg_effective = float(reg) / n_samples
            lcurve_info = None
            reg_method = 'fixed_block_inverse'
        z = np.linalg.solve(A_work + reg_effective * np.eye(K_total), b_work)
        theta = T @ z
    else:  # identity
        if reg == 'auto':
            reg_effective, lcurve_info = _hansen_lcurve_lambda(ATA, ATb)
            reg_method = 'hansen_lcurve'
        else:
            reg_effective = float(reg) / n_samples
            lcurve_info = None
            reg_method = 'fixed'
        theta = np.linalg.solve(ATA + reg_effective * np.eye(K_total), ATb)

    elapsed = time.time() - t0

    info = {
        'time_s': elapsed, 'M_used': M, 'method': 'sinkhorn',
        'eps_factor': eps_factor,
        'reg_effective': float(reg_effective), 'reg_method': reg_method,
        'reg_type': reg_type,
    }
    if lcurve_info:
        info['lcurve'] = lcurve_info
    return theta[:K_V], theta[K_V:], info


# ══════════════════════════════════════════════════════════════
# Solver 3: Self-test (energy balance, unlabeled data)
# ══════════════════════════════════════════════════════════════

def solve_selftest(data_unlabeled, t_obs, sigma, build_V_fn, build_Phi_fn,
                   K_V, K_Phi, reg='auto', M_max=None, reg_type='identity',
                   quadrature='left'):
    """Self-test loss solver via energy dissipation balance.

    The loss L(theta) = (1/2) theta^T A theta - b^T theta + C
    is minimised by solving A theta = b.

    The three contributions per (m, ell) pair:
      - Dissipation (quadratic -> A): J_diss = (1/N) sum_i |force_i|^2
      - Diffusion (linear -> b):     J_diff = (1/N) sum_i [Lap_V + (1/N) sum_j Lap_Phi]
      - Energy change (linear -> b):  dE = E(t+1) - E(t)

    Args:
        data_unlabeled: (M, L, N, d)
        t_obs: (L,) observation times
        sigma: noise coefficient
        build_V_fn, build_Phi_fn: from get_basis()
        K_V, K_Phi: basis sizes
        reg: Tikhonov regularisation. 'auto' (default) uses Hansen L-curve;
             a float value uses fixed adaptive regularisation.
        M_max: use at most this many trajectories
        quadrature: 'left' (default, O(Δt) bias) or 'trapezoid' (O(Δt²) bias).
                    Trapezoid evaluates dissipation/diffusion at BOTH endpoints
                    of each interval and averages, reducing discretization bias.

    Returns:
        alpha, beta, info
    """
    M_data, L, N, d = data_unlabeled.shape
    M_use = min(M_data, M_max) if M_max else M_data
    K_total = K_V + K_Phi
    sigma_sq_half = sigma ** 2 / 2

    A = np.zeros((K_total, K_total))
    b = np.zeros(K_total)
    mask = 1.0 - np.eye(N)

    t0 = time.time()

    def _eval_force_and_lap(X):
        """Evaluate force design matrix and Laplacian coefficients at snapshot X.
        Returns (F_coeff, lap_coeff, psi_vals, phi_vals_flat)."""
        psi, psi_grad, psi_lap, r_x = build_V_fn(X)
        r_x_safe = np.maximum(r_x, 1e-10)
        x_hat = X / r_x_safe[:, None]
        grad_V_coeff = psi_grad[:, None, :] * x_hat[:, :, None]

        diffs = X[:, None, :] - X[None, :, :]
        r_pairs_mat = np.sqrt(np.sum(diffs ** 2, axis=-1))
        r_pairs_flat = r_pairs_mat.reshape(-1)
        phi_flat, phi_grad_flat, phi_lap_flat = build_Phi_fn(r_pairs_flat, d)
        phi_grad_mat = phi_grad_flat.reshape(N, N, K_Phi)
        r_pairs_safe = np.maximum(r_pairs_mat, 1e-10)
        unit_diff = diffs / r_pairs_safe[:, :, None]
        gP = unit_diff[:, :, :, None] * phi_grad_mat[:, :, None, :]
        gP *= mask[:, :, None, None]
        grad_Phi_coeff = gP.sum(axis=1) / N

        F_coeff = np.concatenate([grad_V_coeff, grad_Phi_coeff], axis=-1)

        lap_V_coeff = psi_lap.mean(axis=0)
        phi_lap_mat = phi_lap_flat.reshape(N, N, K_Phi)
        phi_lap_mat_masked = phi_lap_mat * mask[:, :, None]
        lap_Phi_coeff = phi_lap_mat_masked.sum(axis=1).mean(axis=0) / N
        lap_coeff = np.concatenate([lap_V_coeff, lap_Phi_coeff])

        return F_coeff, lap_coeff, psi, phi_flat

    use_trapezoid = (quadrature == 'trapezoid')

    for m in range(M_use):
        if m % 100 == 0 and m > 0:
            print(f"  Self-test: trajectory {m}/{M_use}")
        for ell in range(L - 1):
            dt = t_obs[ell + 1] - t_obs[ell]
            X_curr = data_unlabeled[m, ell]
            X_next = data_unlabeled[m, ell + 1]

            F_curr, lap_curr, psi_curr, phi_curr_flat = _eval_force_and_lap(X_curr)

            if use_trapezoid:
                F_next, lap_next, psi_next, phi_next_flat = _eval_force_and_lap(X_next)
                # ── DISSIPATION: trapezoidal average ──
                A += (dt / (2 * N)) * (
                    np.einsum('ndk,ndl->kl', F_curr, F_curr) +
                    np.einsum('ndk,ndl->kl', F_next, F_next))
                # ── DIFFUSION: trapezoidal average ──
                b += sigma_sq_half * dt * (lap_curr + lap_next) / 2
            else:
                # ── DISSIPATION: left-endpoint ──
                A += (dt / N) * np.einsum('ndk,ndl->kl', F_curr, F_curr)
                # ── DIFFUSION: left-endpoint ──
                b += sigma_sq_half * dt * lap_curr

            # ── ENERGY CHANGE (exact from data — no quadrature needed) ──
            psi_curr_mean = psi_curr.mean(axis=0)
            if use_trapezoid:
                # psi_next and phi_next_flat already computed above
                psi_next_mean = psi_next.mean(axis=0)
                phi_next_mat = phi_next_flat.reshape(N, N, K_Phi) * mask[:, :, None]
                phi_next_mean = phi_next_mat.sum(axis=(0, 1)) / (2 * N ** 2)
            else:
                psi_next_vals, _, _, _ = build_V_fn(X_next)
                psi_next_mean = psi_next_vals.mean(axis=0)
                diff_next = X_next[:, None, :] - X_next[None, :, :]
                r_next_flat = np.sqrt(np.sum(diff_next ** 2, axis=-1)).reshape(-1)
                phi_next_flat_e, _, _ = build_Phi_fn(r_next_flat, d)
                phi_next_mat = phi_next_flat_e.reshape(N, N, K_Phi) * mask[:, :, None]
                phi_next_mean = phi_next_mat.sum(axis=(0, 1)) / (2 * N ** 2)

            phi_curr_mat = phi_curr_flat.reshape(N, N, K_Phi) * mask[:, :, None]
            phi_curr_mean = phi_curr_mat.sum(axis=(0, 1)) / (2 * N ** 2)

            dE_V = psi_next_mean - psi_curr_mean
            dE_Phi = phi_next_mean - phi_curr_mean
            dE_coeff = np.concatenate([dE_V, dE_Phi])
            b -= dE_coeff

    # Normalise
    n_pairs = M_use * (L - 1)
    A /= n_pairs
    b /= n_pairs

    # ── Regularise and solve ──
    block_info = None
    lcurve_info = None
    if reg_type == 'full_inverse':
        theta, solve_info = _solve_full_inverse(A, b, reg=reg)
        reg_method = solve_info['reg_method']
        reg_effective = solve_info['reg_effective']
        lcurve_info = solve_info.get('lcurve')
    elif reg_type == 'block_inverse':
        T, block_info = _block_sqrt_transform(A, K_V, K_Phi)
        A_work = T @ A @ T
        b_work = T @ b
        if reg == 'auto':
            reg_effective, lcurve_info = _hansen_lcurve_lambda(A_work, b_work)
            reg_method = 'hansen_lcurve_block_inverse'
        else:
            reg_scale = np.trace(A_work) / K_total
            reg_effective = reg * reg_scale
            reg_method = 'adaptive_fixed_block_inverse'
        z = np.linalg.solve(A_work + reg_effective * np.eye(K_total), b_work)
        theta = T @ z
    else:  # identity
        if reg == 'auto':
            reg_effective, lcurve_info = _hansen_lcurve_lambda(A, b)
            reg_method = 'hansen_lcurve'
        else:
            reg_scale = np.trace(A) / K_total
            reg_effective = reg * reg_scale
            reg_method = 'adaptive_fixed'
        try:
            theta = np.linalg.solve(A + reg_effective * np.eye(K_total), b)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(A + reg_effective * np.eye(K_total), b, rcond=None)[0]

    # Compute loss at minimizer: L(θ*) = (1/2) θ*ᵀ A θ* - bᵀ θ*
    loss_value = float(0.5 * theta @ A @ theta - b @ theta)

    elapsed = time.time() - t0
    alpha = theta[:K_V]
    beta = theta[K_V:]

    # Eigenvalues of A for condition number analysis (Section 4.2.2)
    eigvals_A = np.linalg.eigvalsh(A).tolist()

    info = {
        'time_s': elapsed,
        'M_used': M_use,
        'method': 'selftest',
        'reg_effective': float(reg_effective),
        'reg_method': reg_method,
        'reg_type': reg_type,
        'loss_value': loss_value,
        'eigvals_A': eigvals_A,
    }
    if lcurve_info:
        info['lcurve'] = lcurve_info
    if block_info:
        info['block_info'] = block_info

    return alpha, beta, info
