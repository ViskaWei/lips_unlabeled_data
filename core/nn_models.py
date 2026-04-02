"""Neural network models for potential learning.

Canonical definitions for all NN architectures used in IPS experiments.
All scripts should import from here — do NOT duplicate these classes.

Active architectures:
  - RadialNet: V(x) = f(|x|), radial potential
  - RadialInteractionNet: Phi(x) = g(|x|), radial interaction
  - GeneralNet: V(x) = f(x1,...,xd), non-radial potential
  - SymmetricInteractionNet: Phi(z) = 0.5*(f(z)+f(-z)), non-radial interaction

Differential operators:
  - compute_laplacian: AD-based f, grad_f, laplacian_f

Utility functions:
  - compute_pairwise_distances: |X_i - X_j| matrix
  - compute_pairwise_diff: (X_i - X_j) tensor
"""

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════
# Activation helpers
# ══════════════════════════════════════════════════════════════

ACTIVATION_MAP = {
    'softplus': nn.Softplus,
    'tanh': nn.Tanh,
}


def _make_activation(name='softplus'):
    """Create activation module from string name."""
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATION_MAP.keys())}")
    return ACTIVATION_MAP[name]


# ══════════════════════════════════════════════════════════════
# Network architectures
# ══════════════════════════════════════════════════════════════

class RadialNet(nn.Module):
    """V(x) = f(|x|) via MLP."""
    def __init__(self, hidden_dims=(64, 64, 64), activation='softplus'):
        super().__init__()
        act_cls = _make_activation(activation)
        layers = []
        in_dim = 1
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_cls()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # clamp avoids NaN in AD gradient when ||x|| → 0
        r = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        return self.net(r).squeeze(-1)


class RadialInteractionNet(nn.Module):
    """Phi(x) = g(|x|) via MLP."""
    def __init__(self, d=2, hidden_dims=(64, 64, 64), activation='softplus'):
        super().__init__()
        act_cls = _make_activation(activation)
        layers = []
        in_dim = 1
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_cls()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # clamp avoids NaN in AD gradient when ||x|| → 0
        r = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        return self.net(r).squeeze(-1)


class GeneralNet(nn.Module):
    """V(x) = f(x_1,...,x_d) via MLP (non-radial)."""
    def __init__(self, d=2, hidden_dims=(64, 64, 64), activation='softplus'):
        super().__init__()
        act_cls = _make_activation(activation)
        layers = []
        in_dim = d
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_cls()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SymmetricInteractionNet(nn.Module):
    """Phi(z) = 0.5*(f(z) + f(-z)) via MLP (non-radial, symmetric)."""
    def __init__(self, d=2, hidden_dims=(64, 64, 64), activation='softplus'):
        super().__init__()
        act_cls = _make_activation(activation)
        layers = []
        in_dim = d
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_cls()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return 0.5 * (self.net(x).squeeze(-1) + self.net(-x).squeeze(-1))


# ══════════════════════════════════════════════════════════════
# AD-based differential operators
# ══════════════════════════════════════════════════════════════

def compute_laplacian(net, x):
    """f(x), grad_f(x), laplacian_f(x) via AD."""
    x = x.requires_grad_(True)
    f = net(x)
    grad = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
    laplacian = torch.zeros(x.shape[0], device=x.device)
    for k in range(x.shape[1]):
        grad2_k = torch.autograd.grad(
            grad[:, k].sum(), x, create_graph=True
        )[0][:, k]
        laplacian = laplacian + grad2_k
    return f, grad, laplacian


# ══════════════════════════════════════════════════════════════
# Pairwise distance utilities
# ══════════════════════════════════════════════════════════════

def compute_pairwise_distances(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances between particles.

    Args:
        X: Particle positions, shape (N, d)
    Returns:
        Distance matrix, shape (N, N)
    """
    diff = X.unsqueeze(0) - X.unsqueeze(1)  # shape (N, N, d)
    distances = torch.norm(diff, dim=-1)  # shape (N, N)
    return distances


def compute_pairwise_diff(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise differences X_i - X_j.

    Args:
        X: Particle positions, shape (N, d)
    Returns:
        Difference tensor, shape (N, N, d)
    """
    return X.unsqueeze(0) - X.unsqueeze(1)
