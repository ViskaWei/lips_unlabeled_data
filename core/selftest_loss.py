"""Self-test loss for learning IPS potentials from unlabeled snapshots.

This implements Algorithm 1 from the paper. The self-test loss is derived
from Ito's lemma applied to the energy functional of the SDE:

    dX_i = b_i dt + sigma dW_i,   b_i = -nabla V(X_i) - (1/N) sum_j nabla Phi(X_i - X_j)

Ito's lemma on E = <V,mu> + (1/2)<Phi*mu, mu> gives:

    L = (1/2) J_diss dt - (sigma^2/2) J_diff dt + Delta E

where:
    J_diss = (1/N) sum_i |nabla V(X_i) + (1/N) sum_{j!=i} nabla Phi(X_i - X_j)|^2
    J_diff = (1/N) sum_i [Delta V(X_i) + (1/N) sum_{j!=i} Delta Phi(X_i - X_j)]
    Delta E = E(t+dt) - E(t)
    E_Phi   = (1/(2N^2)) sum_{i!=j} Phi(|X_i - X_j|)

Key properties:
    - L is NOT squared; the minimum is negative (not zero).
    - At true parameters: E[L] = -(1/2) E[J_diss dt] < 0.
    - At V=Phi=0: L = 0.  The (1/2) factor on J_diss breaks the degeneracy.
"""

import torch

from .nn_models import compute_laplacian


def compute_selftest_loss_batch(V_net, Phi_net, X_curr, X_next, dt, sigma):
    """Vectorized self-test loss over a batch of (m, ell) pairs.

    Args:
        V_net:   confinement potential network, maps (*, d) -> (*)
        Phi_net: interaction potential network, maps (*, d) -> (*)
        X_curr:  particle positions at t,      shape (B, N, d)
        X_next:  particle positions at t+dt,   shape (B, N, d)
        dt:      time step(s),                 scalar or shape (B,)
        sigma:   diffusion coefficient,        scalar

    Returns:
        Per-sample self-test residuals, shape (B,).
        Training objective: residuals.mean()  (minimize toward negative).
    """
    B, N, d = X_curr.shape
    mask = ~torch.eye(N, dtype=torch.bool, device=X_curr.device)

    # ── V terms at current time ──────────────────────────────
    X_flat = X_curr.reshape(B * N, d)
    V_vals, grad_V, lap_V = compute_laplacian(V_net, X_flat)
    V_vals = V_vals.reshape(B, N)
    grad_V = grad_V.reshape(B, N, d)
    lap_V = lap_V.reshape(B, N)

    # ── Phi pairwise terms at current time ───────────────────
    diff = X_curr.unsqueeze(2) - X_curr.unsqueeze(1)       # (B, N, N, d)
    diff_pairs = diff[:, mask].reshape(B, N * (N - 1), d)   # exclude diagonal
    diff_flat = diff_pairs.reshape(B * N * (N - 1), d)

    Phi_vals, grad_Phi, lap_Phi = compute_laplacian(Phi_net, diff_flat)
    Phi_vals = Phi_vals.reshape(B, N, N - 1)
    grad_Phi = grad_Phi.reshape(B, N, N - 1, d)
    lap_Phi = lap_Phi.reshape(B, N, N - 1)

    # ── J_diss: dissipation ──────────────────────────────────
    # force_i = grad_V(X_i) + (1/N) sum_{j!=i} grad_Phi(X_i - X_j)
    mean_grad_Phi = grad_Phi.sum(dim=2) / N
    force = grad_V + mean_grad_Phi
    J_diss = (force ** 2).sum(dim=-1).mean(dim=-1)          # (B,)

    # ── J_diff: diffusion (Laplacian) ────────────────────────
    mean_lap_Phi = lap_Phi.sum(dim=2) / N
    J_diff = (lap_V + mean_lap_Phi).mean(dim=-1)            # (B,)

    # ── Energy at current time ───────────────────────────────
    E_V_curr = V_vals.mean(dim=-1)                           # (1/N) sum_i V(X_i)
    E_Phi_curr = Phi_vals.sum(dim=(1, 2)) / (2 * N ** 2)    # (1/(2N^2)) sum_{i!=j} Phi
    E_curr = E_V_curr + E_Phi_curr

    # ── Energy at next time ──────────────────────────────────
    X_next_flat = X_next.reshape(B * N, d)
    V_next = V_net(X_next_flat).reshape(B, N)
    E_V_next = V_next.mean(dim=-1)

    diff_next = X_next.unsqueeze(2) - X_next.unsqueeze(1)
    diff_next_pairs = diff_next[:, mask].reshape(B * N * (N - 1), d)
    Phi_next = Phi_net(diff_next_pairs).reshape(B, N, N - 1)
    E_Phi_next = Phi_next.sum(dim=(1, 2)) / (2 * N ** 2)
    E_next = E_V_next + E_Phi_next

    # ── Self-test loss (Algorithm 1) ─────────────────────────
    return 0.5 * J_diss * dt - (sigma ** 2 / 2) * J_diff * dt + (E_next - E_curr)
