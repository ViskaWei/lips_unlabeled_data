"""Potential functions for IPS simulation."""

import numpy as np
from abc import ABC, abstractmethod


class Potential(ABC):
    """Base class for potential functions.

    Attributes:
        radial: If True (default), interaction potentials take scalar r = |z|
                and gradient() returns dPhi/dr (scalar).
                If False, interaction potentials take vector z in R^d
                and gradient() returns nabla_z Phi (vector in R^d).
    """
    radial: bool = True

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate potential at x."""
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of potential at x."""
        pass


class HarmonicPotential(Potential):
    """Harmonic potential V(x) = 0.5 * k * x^2."""

    def __init__(self, k: float = 1.0):
        self.k = k

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.k * np.sum(x**2, axis=-1)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.k * x


class ZeroPotential(Potential):
    """Zero potential (no interaction)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[:-1]) if x.ndim > 1 else 0.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class GaussianInteraction(Potential):
    """Gaussian interaction potential Phi(r) = A * exp(-r^2 / (2*sigma^2))."""

    def __init__(self, A: float = 1.0, sigma: float = 1.0):
        self.A = A
        self.sigma = sigma

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.A * np.exp(-r**2 / (2 * self.sigma**2))

    def gradient(self, r: np.ndarray) -> np.ndarray:
        """Gradient w.r.t. r (scalar distance)."""
        return -self.A * r / (self.sigma**2) * np.exp(-r**2 / (2 * self.sigma**2))


class QuadraticConfinement(Potential):
    """Model A confining potential: V(x) = alpha1/2 * |x| + alpha2 * |x|^2.

    From paper Section 3.3 Model A with alpha=(-1, 2).
    Note: gradient of |x| at x is x/|x|, undefined at 0.
    """

    def __init__(self, alpha1: float = -1.0, alpha2: float = 2.0):
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(x**2, axis=-1))
        return 0.5 * self.alpha1 * r + self.alpha2 * r**2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(x**2, axis=-1, keepdims=True))
        r_safe = np.maximum(r, 1e-10)
        # grad(|x|) = x/|x|, grad(|x|^2) = 2x
        return 0.5 * self.alpha1 * x / r_safe + 2.0 * self.alpha2 * x


class DoubleWellPotential(Potential):
    """Model B confining potential: V(x) = (|x|^2 - 1)^2 / 4.

    From paper Section 3.3 Model B.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r_sq = np.sum(x**2, axis=-1)
        return 0.25 * (r_sq - 1.0)**2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        r_sq = np.sum(x**2, axis=-1, keepdims=True)
        return (r_sq - 1.0) * x


class PiecewiseInteraction(Potential):
    """Model A interaction: Phi(r) = beta1 * 1_{[0.5,1]}(r) + beta2 * 1_{[1,2]}(r).

    From paper Section 3.3 Model A with beta=(-3, 2).
    Uses smoothed indicators for numerical stability.
    """

    def __init__(self, beta1: float = -3.0, beta2: float = 2.0, eps: float = 0.05):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def _smooth_indicator(self, r, lo, hi):
        """Smoothed indicator function on [lo, hi] using sigmoid transitions."""
        eps = self.eps
        return 0.5 * (np.tanh((r - lo) / eps) - np.tanh((r - hi) / eps))

    def _smooth_indicator_grad(self, r, lo, hi):
        """Gradient of smoothed indicator."""
        eps = self.eps
        sech2_lo = 1.0 / np.cosh((r - lo) / eps)**2
        sech2_hi = 1.0 / np.cosh((r - hi) / eps)**2
        return 0.5 / eps * (sech2_lo - sech2_hi)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return (self.beta1 * self._smooth_indicator(r, 0.5, 1.0)
                + self.beta2 * self._smooth_indicator(r, 1.0, 2.0))

    def gradient(self, r: np.ndarray) -> np.ndarray:
        return (self.beta1 * self._smooth_indicator_grad(r, 0.5, 1.0)
                + self.beta2 * self._smooth_indicator_grad(r, 1.0, 2.0))


class GaussianBumpInteraction(Potential):
    """Model E: Phi(r) = sum_k beta_k * exp(-(r-c_k)^2/(2*sigma_k^2)).
    C∞ smooth counterpart of Model A's piecewise indicator interaction."""

    def __init__(self, beta=(-3.0, 2.0), centers=(0.75, 1.5), widths=(0.125, 0.25)):
        self.beta = np.array(beta, dtype=float)
        self.centers = np.array(centers, dtype=float)
        self.widths = np.array(widths, dtype=float)

    def __call__(self, r):
        result = np.zeros_like(r, dtype=float)
        for k in range(len(self.beta)):
            result += self.beta[k] * np.exp(-(r - self.centers[k])**2 / (2 * self.widths[k]**2))
        return result

    def gradient(self, r):
        result = np.zeros_like(r, dtype=float)
        for k in range(len(self.beta)):
            g = np.exp(-(r - self.centers[k])**2 / (2 * self.widths[k]**2))
            result += self.beta[k] * (-(r - self.centers[k]) / self.widths[k]**2) * g
        return result


class InverseInteraction(Potential):
    """Model B interaction: Phi(r) = gamma / (r + 1).

    From paper Section 3.3 Model B with gamma=0.5.
    """

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.gamma / (r + 1.0)

    def gradient(self, r: np.ndarray) -> np.ndarray:
        return -self.gamma / (r + 1.0)**2


class MorsePotential(Potential):
    """Morse potential Phi(r) = D * (1 - exp(-a*(r-r0)))^2."""

    def __init__(self, D: float = 1.0, a: float = 1.0, r0: float = 1.0):
        self.D = D
        self.a = a
        self.r0 = r0

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.D * (1 - np.exp(-self.a * (r - self.r0)))**2

    def gradient(self, r: np.ndarray) -> np.ndarray:
        exp_term = np.exp(-self.a * (r - self.r0))
        return 2 * self.D * self.a * (1 - exp_term) * exp_term


class LennardJonesPotential(Potential):
    """Truncated & shifted Lennard-Jones: Phi(r) = 4*eps*[(sig/r)^12 - (sig/r)^6] - shift.

    Standard molecular dynamics potential with:
    - Equilibrium distance r_min = 2^(1/6) * sigma_lj
    - Truncated at r_cut (set to zero beyond)
    - Shifted so Phi(r_cut) = 0 (continuous)
    - Clamped at r_safe to prevent force blowup in Euler-Maruyama
    """

    def __init__(self, epsilon: float = 0.5, sigma_lj: float = 0.5,
                 r_cut: float = 2.5, r_safe_factor: float = 0.7):
        self.epsilon = epsilon
        self.sigma_lj = sigma_lj
        self.r_cut = r_cut
        self.r_safe = r_safe_factor * sigma_lj
        # Shift so Phi(r_cut) = 0
        sr6_cut = (sigma_lj / r_cut) ** 6
        self.shift = 4.0 * epsilon * (sr6_cut**2 - sr6_cut)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r_c = np.maximum(r, self.r_safe)
        sr6 = (self.sigma_lj / r_c) ** 6
        phi = 4.0 * self.epsilon * (sr6**2 - sr6) - self.shift
        return np.where(r < self.r_cut, phi, 0.0)

    def gradient(self, r: np.ndarray) -> np.ndarray:
        r_c = np.maximum(r, self.r_safe)
        sr6 = (self.sigma_lj / r_c) ** 6
        # dPhi/dr = (4*eps/r) * (-12*sr6^2 + 6*sr6)
        dphi = (4.0 * self.epsilon / r_c) * (-12.0 * sr6**2 + 6.0 * sr6)
        return np.where(r < self.r_cut, dphi, 0.0)


# ====================================================
# Non-radial potentials (radial = False)
# ====================================================

class AnisotropicConfinement(Potential):
    """Non-radial confining potential: V(x) = sum_k a_k * x_k^2.

    When a_k differ across dimensions, this breaks rotational symmetry.
    Example: a=(0.01, 0.005) gives weak elliptical confinement.
    """
    radial = False

    def __init__(self, a=(0.01, 0.005)):
        self.a = np.array(a, dtype=float)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (..., d)
        return np.sum(self.a * x**2, axis=-1)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * self.a * x


class AnisotropicGaussianInteraction(Potential):
    """Non-radial interaction: Phi(z) = A * exp(-(z_1^2/s1^2 + z_2^2/s2^2) / 2).

    Even symmetry Phi(-z) = Phi(z) holds because z appears as z^2.
    When s1 != s2, the interaction is anisotropic (non-radial).

    gradient(z) returns nabla_z Phi (vector), NOT dPhi/dr.
    """
    radial = False

    def __init__(self, A: float = 2.0, s=(0.5, 1.5)):
        self.A = A
        self.s = np.array(s, dtype=float)
        self.s_sq = self.s ** 2

    def __call__(self, z: np.ndarray) -> np.ndarray:
        # z: (..., d)
        return self.A * np.exp(-0.5 * np.sum(z**2 / self.s_sq, axis=-1))

    def gradient(self, z: np.ndarray) -> np.ndarray:
        # nabla_z Phi = -Phi(z) * z / s^2
        phi_vals = self.__call__(z)
        return -phi_vals[..., None] * z / self.s_sq


class DipolarInteraction(Potential):
    """Dipolar interaction: Phi(x) = mu/(4*pi*||x||^3) * (1 - 3*cos^2(theta)).

    cos(theta) = (x . n_hat) / ||x|| where n_hat is the fixed dipole axis.
    Symmetric: Phi(-x) = Phi(x) since cos^2(-theta) = cos^2(theta).
    Attractive at poles (theta=0,pi), repulsive at equator (theta=pi/2).

    gradient(x) returns nabla_x Phi (vector), NOT dPhi/dr.
    """
    radial = False

    def __init__(self, mu: float = 0.1, n_hat=None, d: int = 3, r_safe: float = 0.05):
        self.mu = mu
        if n_hat is None:
            n_hat = np.zeros(d)
            n_hat[-1] = 1.0  # dipole along last axis
        self.n_hat = np.array(n_hat, dtype=float)
        self.n_hat = self.n_hat / np.linalg.norm(self.n_hat)
        self.r_safe = r_safe
        self.prefactor = mu / (4.0 * np.pi)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (..., d)
        r = np.sqrt(np.sum(x**2, axis=-1))
        r_c = np.maximum(r, self.r_safe)
        cos_theta = np.sum(x * self.n_hat, axis=-1) / r_c
        return self.prefactor / r_c**3 * (1.0 - 3.0 * cos_theta**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        # nabla Phi = (3*mu)/(4*pi*||x||^5) * [(5*cos^2(theta)-1)*x - 2*||x||*cos(theta)*n_hat]
        r = np.sqrt(np.sum(x**2, axis=-1))
        r_c = np.maximum(r, self.r_safe)
        cos_theta = np.sum(x * self.n_hat, axis=-1) / r_c
        coeff = 3.0 * self.prefactor / r_c**5
        term1 = (5.0 * cos_theta**2 - 1.0)[..., None] * x
        term2 = (2.0 * r_c * cos_theta)[..., None] * self.n_hat
        return coeff[..., None] * (term1 - term2)
