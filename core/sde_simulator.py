"""SDE simulator using Euler-Maruyama method."""

import numpy as np
from typing import Tuple, Optional
from .potentials import Potential, ZeroPotential


class SDESimulator:
    """Simulate Interacting Particle System SDE using Euler-Maruyama.

    dX_t^i = -grad_V(X_t^i) dt - (1/N) sum_j grad_Phi(X_t^i - X_t^j) dt + sigma dW_t^i
    """

    def __init__(
        self,
        V: Potential,
        Phi: Optional[Potential] = None,
        sigma: float = 0.1,
        dt: float = 0.01,
    ):
        self.V = V
        self.Phi = Phi if Phi is not None else ZeroPotential()
        self.sigma = sigma
        self.dt = dt

    def compute_interaction_gradient(self, X: np.ndarray) -> np.ndarray:
        """Compute mean-field interaction gradient for each particle (vectorized).

        Args:
            X: Particle positions, shape (N, d)

        Returns:
            Interaction gradient, shape (N, d)
        """
        N, d = X.shape
        diff = X[:, None, :] - X[None, :, :]  # (N, N, d)
        mask = 1.0 - np.eye(N)

        if getattr(self.Phi, 'radial', True):
            # Radial: grad_Phi = dPhi/dr * (x_i - x_j) / |x_i - x_j|
            r = np.sqrt(np.sum(diff**2, axis=-1))  # (N, N)
            r_safe = np.maximum(r, 1e-10)
            unit = diff / r_safe[:, :, None]  # (N, N, d)
            grad_r = self.Phi.gradient(r)  # (N, N)
            grad_interaction = np.sum(
                (grad_r * mask)[:, :, None] * unit, axis=1
            )  # (N, d)
        else:
            # Non-radial: grad_Phi = nabla_z Phi(z) directly
            grad_z = self.Phi.gradient(diff)  # (N, N, d)
            grad_interaction = np.sum(
                grad_z * mask[:, :, None], axis=1
            )  # (N, d)

        return grad_interaction / N

    def step(self, X: np.ndarray) -> np.ndarray:
        """Perform one Euler-Maruyama step.

        Args:
            X: Current particle positions, shape (N, d)

        Returns:
            Updated particle positions, shape (N, d)
        """
        N, d = X.shape

        # Drift from kinetic potential
        grad_V = self.V.gradient(X)  # shape (N, d)

        # Drift from interaction
        grad_Phi = self.compute_interaction_gradient(X)  # shape (N, d)

        # Total drift
        drift = -grad_V - grad_Phi

        # Diffusion (Brownian noise)
        noise = self.sigma * np.sqrt(self.dt) * np.random.randn(N, d)

        # Euler-Maruyama update
        X_new = X + drift * self.dt + noise

        return X_new

    def simulate(
        self,
        N: int,
        d: int,
        T: float,
        L: int,
        M: int,
        seed: int = 42,
        init_std: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate IPS and record snapshots.

        Args:
            N: Number of particles
            d: Spatial dimension
            T: Total simulation time
            L: Number of snapshots to record
            M: Number of independent samples
            seed: Random seed
            init_std: Std of initial Gaussian distribution

        Returns:
            data: Snapshot data, shape (M, L, N, d)
            t_snapshots: Time points of snapshots, shape (L,)
        """
        np.random.seed(seed)

        n_steps = int(T / self.dt)
        save_interval = max(1, n_steps // L)

        # Actual number of snapshots
        actual_L = n_steps // save_interval

        # Output containers
        data = np.zeros((M, actual_L, N, d))
        t_snapshots = np.array([(i + 1) * save_interval * self.dt for i in range(actual_L)])

        for m in range(M):
            # Initialize particles from Gaussian
            X = init_std * np.random.randn(N, d)

            snapshot_idx = 0
            for step in range(n_steps):
                X = self.step(X)

                # Record snapshot
                if (step + 1) % save_interval == 0 and snapshot_idx < actual_L:
                    data[m, snapshot_idx] = X.copy()
                    snapshot_idx += 1

        return data, t_snapshots


def simulate_ou_process(
    N: int = 10,
    d: int = 1,
    L: int = 20,
    M: int = 200,
    dt: float = 0.01,
    T: float = 2.0,
    sigma: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Convenience function to simulate Ornstein-Uhlenbeck process (no interaction).

    Returns:
        data: shape (M, L, N, d)
        t_snapshots: shape (L,)
        config: Configuration dict
    """
    from .potentials import HarmonicPotential, ZeroPotential

    V = HarmonicPotential(k=1.0)
    Phi = ZeroPotential()

    simulator = SDESimulator(V=V, Phi=Phi, sigma=sigma, dt=dt)
    data, t_snapshots = simulator.simulate(N=N, d=d, T=T, L=L, M=M, seed=seed)

    config = {
        'N': N,
        'd': d,
        'L': L,
        'M': M,
        'dt': dt,
        'T': T,
        'sigma': sigma,
        'seed': seed,
        'V': 'Harmonic(k=1.0)',
        'Phi': 'Zero (no interaction)',
    }

    return data, t_snapshots, config
