#!/usr/bin/env python3
"""Parallel IPS data generation optimized for multi-core servers.

Faithful reimplementation of generate_paper_data.py with:
1. Multi-process parallelism across M ensembles
2. Batch-vectorized SDE stepping (multiple ensembles per NumPy call)
3. Pre-computed constants to reduce per-step overhead

Physics: dX_t^i = -grad_V(X_t^i) dt - (1/N) sum_j grad_Phi(|X_t^i - X_t^j|) * (X_t^i - X_t^j)/|...| dt + sigma dW_t^i
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.potentials import (
    QuadraticConfinement, DoubleWellPotential,
    PiecewiseInteraction, InverseInteraction,
    HarmonicPotential, LennardJonesPotential, MorsePotential,
    AnisotropicConfinement, AnisotropicGaussianInteraction,
    DipolarInteraction, GaussianBumpInteraction,
)

# ---------------------------------------------------------------------------
# Batch-vectorized SDE worker
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'model_a': {
        'V': lambda: QuadraticConfinement(alpha1=-1.0, alpha2=2.0),
        'Phi': lambda: PiecewiseInteraction(beta1=-3.0, beta2=2.0),
        'desc': 'Model A (Quadratic Confinement + Piecewise Interaction)',
    },
    'model_b': {
        'V': lambda: DoubleWellPotential(),
        'Phi': lambda: InverseInteraction(gamma=0.5),
        'desc': 'Model B (Double Well + Inverse Interaction)',
    },
    'model_lj': {
        'V': lambda: HarmonicPotential(k=2.0),
        'Phi': lambda: LennardJonesPotential(epsilon=0.5, sigma_lj=0.5, r_cut=2.5),
        'desc': 'Model LJ (Harmonic + Lennard-Jones)',
    },
    'model_morse': {
        'V': lambda: DoubleWellPotential(),
        'Phi': lambda: MorsePotential(D=0.5, a=2.0, r0=0.8),
        'desc': 'Model Morse (Double Well + Morse Interaction)',
    },
    'model_e': {
        'V': lambda: QuadraticConfinement(alpha1=-1.0, alpha2=2.0),
        'Phi': lambda: GaussianBumpInteraction(),
        'desc': 'Model E (Quadratic Confinement + Gaussian Bump Interaction)',
    },
    'model_aniso': {
        'V': lambda: AnisotropicConfinement(a=(1.0, 4.0)),
        'Phi': lambda: AnisotropicGaussianInteraction(A=2.0, s=(0.5, 1.5)),
        'desc': 'Model Aniso (Anisotropic Confinement + Anisotropic Gaussian)',
    },
    'model_dipole': {
        # V(x) = sum_k a_k * x_k^2, a_k = 0.01/k (weak confinement, sigma=0.1)
        # Phi = dipolar with mu=0.5, r_safe=0.15
        'V': lambda d=3: AnisotropicConfinement(a=tuple(0.01/k for k in range(1, d+1))),
        'Phi': lambda d=3: DipolarInteraction(mu=0.5, d=d, r_safe=0.15),
        'desc': 'Model Dipole (Weak Confinement + Dipolar mu=0.5)',
    },
}


def _create_potentials(model_name, d=2):
    """Create V, Phi potentials for a given model name.

    Some models (e.g. model_dipole) have d-dependent potentials.
    Factories that accept d will receive it; others are called without args.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    spec = MODEL_REGISTRY[model_name]
    try:
        V = spec['V'](d)
    except TypeError:
        V = spec['V']()
    try:
        Phi = spec['Phi'](d)
    except TypeError:
        Phi = spec['Phi']()
    return V, Phi


def _batch_interaction_gradient(X, Phi, mask_eye):
    """Vectorized interaction gradient for a batch of ensembles.

    Args:
        X: (B, N, d) particle positions
        Phi: Potential object with .gradient() supporting broadcast
        mask_eye: (N, N) pre-computed 1 - eye(N)

    Returns:
        (B, N, d) interaction gradient
    """
    # Pairwise differences: (B, N, N, d)
    diff = X[:, :, None, :] - X[:, None, :, :]
    N = X.shape[1]

    if getattr(Phi, 'radial', True):
        # Radial: dPhi/dr * unit vector
        r = np.sqrt(np.sum(diff * diff, axis=-1))  # (B, N, N)
        r_safe = np.maximum(r, 1e-10)
        unit = diff / r_safe[..., None]  # (B, N, N, d)
        grad_r = Phi.gradient(r)  # (B, N, N)
        grad_interaction = np.sum((grad_r * mask_eye)[..., None] * unit, axis=2)
    else:
        # Non-radial: nabla_z Phi(z) directly, shape (B, N, N, d)
        grad_z = Phi.gradient(diff)
        grad_interaction = np.sum(grad_z * mask_eye[..., None], axis=2)

    return grad_interaction / N


def _worker(args):
    """Worker: simulate a batch of ensembles with batch-vectorized stepping.

    Returns (data_labeled, data_unlabeled) each of shape (B, L, N, d).
    """
    (model_name, ensemble_indices, N, d, T, dt_fine, dt_obs,
     sigma, base_seed, shuffle_labels) = args

    V, Phi = _create_potentials(model_name, d)

    B = len(ensemble_indices)
    n_fine_steps = int(round(T / dt_fine))
    obs_interval = int(round(dt_obs / dt_fine))
    L = n_fine_steps // obs_interval

    # Seed this worker deterministically
    worker_seed = base_seed + ensemble_indices[0]
    rng = np.random.RandomState(worker_seed)

    # Initialize particles: (B, N, d)
    X = 0.5 * rng.randn(B, N, d)

    data_labeled = np.zeros((B, L, N, d))
    data_unlabeled = np.zeros((B, L, N, d))

    # Pre-compute constants
    sqrt_dt = np.sqrt(dt_fine)
    mask_eye = 1.0 - np.eye(N)

    snap_idx = 0
    for step in range(n_fine_steps):
        # --- Batch V gradient: (B, N, d) ---
        grad_V = V.gradient(X)

        # --- Batch interaction gradient: (B, N, d) ---
        grad_Phi = _batch_interaction_gradient(X, Phi, mask_eye)

        # --- Euler-Maruyama update ---
        noise = sigma * sqrt_dt * rng.randn(B, N, d)
        X = X + (-grad_V - grad_Phi) * dt_fine + noise

        # --- Record snapshot ---
        if (step + 1) % obs_interval == 0 and snap_idx < L:
            data_labeled[:, snap_idx] = X.copy()
            if shuffle_labels:
                for i in range(B):
                    perm = rng.permutation(N)
                    data_unlabeled[i, snap_idx] = X[i, perm]
            else:
                data_unlabeled[:, snap_idx] = X.copy()
            snap_idx += 1

    return data_labeled, data_unlabeled


# ---------------------------------------------------------------------------
# Main parallel driver
# ---------------------------------------------------------------------------

def generate_data_parallel(
    model_name, N, d, M, T, dt_fine, dt_obs, sigma,
    seed=42, shuffle_labels=True, n_workers=None, batch_size=None,
):
    """Generate IPS data using multi-process + batch-vectorized simulation.

    Returns:
        data_unlabeled: (M, L, N, d)
        data_labeled:   (M, L, N, d)
        t_obs:          (L,)
        elapsed:        float (seconds)
    """
    n_fine_steps = int(round(T / dt_fine))
    obs_interval = int(round(dt_obs / dt_fine))
    L = n_fine_steps // obs_interval

    if n_workers is None:
        n_workers = min(60, cpu_count() // 2)
    if batch_size is None:
        # Each worker handles batch_size ensembles simultaneously via vectorization
        batch_size = max(1, M // n_workers)
        # Cap batch size to limit memory per worker
        batch_size = min(batch_size, 100)

    print(f"  Model:        {model_name}")
    print(f"  N={N}, d={d}, M={M}, sigma={sigma}")
    print(f"  dt_fine={dt_fine}, dt_obs={dt_obs}, T={T}")
    print(f"  n_fine_steps={n_fine_steps}, obs_interval={obs_interval}, L={L}")
    print(f"  Workers: {n_workers}, batch_size: {batch_size}")

    # Build work items
    work_items = []
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        indices = list(range(start, end))
        work_items.append((
            model_name, indices, N, d, T, dt_fine, dt_obs,
            sigma, seed, shuffle_labels,
        ))

    print(f"  Total batches: {len(work_items)}")
    print(f"\n  Simulating...", flush=True)

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(_worker, work_items)):
            results.append(result)
            done = sum(len(r[0]) for r in results)
            elapsed_so_far = time.time() - t0
            rate = done / elapsed_so_far if elapsed_so_far > 0 else 0
            remaining = (M - done) / rate if rate > 0 else 0
            print(f"\r  Progress: {done}/{M} ensembles "
                  f"({elapsed_so_far:.1f}s elapsed, ~{remaining:.0f}s remaining)",
                  end="", flush=True)
    print()

    elapsed = time.time() - t0

    # Concatenate and sort by ensemble index
    # imap_unordered may return out of order, but we don't need exact ordering
    # since all ensembles are statistically equivalent
    data_labeled = np.concatenate([r[0] for r in results], axis=0)[:M]
    data_unlabeled = np.concatenate([r[1] for r in results], axis=0)[:M]
    t_obs = np.array([(i + 1) * dt_obs for i in range(L)])

    assert data_labeled.shape == (M, L, N, d), \
        f"Shape mismatch: expected ({M},{L},{N},{d}), got {data_labeled.shape}"

    print(f"  Done in {elapsed:.1f}s ({M/elapsed:.1f} ensembles/sec)")
    return data_unlabeled, data_labeled, t_obs, elapsed


def parse_args():
    p = argparse.ArgumentParser(
        description='Parallel IPS data generation'
    )
    p.add_argument('--model', choices=['a', 'b', 'both', 'lj', 'morse', 'e', 'aniso', 'dipole', 'all'],
                   default='both', help='Model A, B, both (A+B), lj, morse, aniso, dipole, or all')
    p.add_argument('--N', type=int, default=10, help='Number of particles')
    p.add_argument('--d', type=int, default=2, help='Spatial dimension')
    p.add_argument('--M', type=int, default=2000, help='Number of ensembles')
    p.add_argument('--L', type=int, default=100, help='Number of observation snapshots')
    p.add_argument('--dt_fine', type=float, default=0.001,
                   help='Fine simulation timestep (default: 0.001, matching dt_obs for zero discretization bias)')
    p.add_argument('--dt_obs', type=float, default=0.001,
                   help='Observation interval (default: 0.001, must equal dt_fine for zero discretization bias)')
    p.add_argument('--sigma', type=float, default=1.0, help='Diffusion coefficient')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--n_workers', type=int, default=None,
                   help='Number of parallel workers (default: auto)')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Ensembles per worker batch (default: auto)')
    p.add_argument('--output_dir', type=str,
                   default='./data',
                   help='Output directory')
    p.add_argument('--no_shuffle', action='store_true',
                   help='Disable label shuffling')
    return p.parse_args()


def main():
    args = parse_args()

    # Compute T from L and dt_obs
    T = args.L * args.dt_obs

    print("=" * 70)
    print("Parallel IPS Data Generation")
    print("=" * 70)
    print(f"\nServer: {cpu_count()} CPUs available")
    print(f"Target: M={args.M}, L={args.L}, N={args.N}, d={args.d}")
    print(f"Physics: dt_fine={args.dt_fine}, dt_obs={args.dt_obs}, T={T}, sigma={args.sigma}")
    n_fine_steps = int(round(T / args.dt_fine))
    print(f"Fine steps per ensemble: {n_fine_steps}")
    print(f"Total fine steps: {args.M * n_fine_steps:,}")

    model_map = {
        'a': ['model_a'], 'b': ['model_b'], 'lj': ['model_lj'],
        'morse': ['model_morse'], 'e': ['model_e'], 'aniso': ['model_aniso'],
        'dipole': ['model_dipole'],
        'both': ['model_a', 'model_b'],
        'all': list(MODEL_REGISTRY.keys()),
    }
    models = {k: MODEL_REGISTRY[k]['desc'] for k in model_map[args.model]}

    total_elapsed = 0.0

    for model_name, desc in models.items():
        print(f"\n{'='*70}")
        print(f"Generating {desc}")
        print(f"{'='*70}\n")

        if args.d == 2:
            data_dir = Path(args.output_dir) / model_name
        else:
            data_dir = Path(args.output_dir) / f'd{args.d}' / model_name
        data_dir.mkdir(parents=True, exist_ok=True)

        data_unlabeled, data_labeled, t_obs, elapsed = generate_data_parallel(
            model_name=model_name,
            N=args.N, d=args.d, M=args.M, T=T,
            dt_fine=args.dt_fine, dt_obs=args.dt_obs,
            sigma=args.sigma, seed=args.seed,
            shuffle_labels=not args.no_shuffle,
            n_workers=args.n_workers,
            batch_size=args.batch_size,
        )
        total_elapsed += elapsed

        # Save data
        np.save(data_dir / 'data_unlabeled.npy', data_unlabeled)
        np.save(data_dir / 'data_labeled.npy', data_labeled)
        np.save(data_dir / 't_obs.npy', t_obs)

        config = {
            'model': model_name,
            'description': desc,
            'N': args.N, 'd': args.d, 'M': args.M,
            'T': T, 'dt_fine': args.dt_fine,
            'dt_obs': args.dt_obs, 'sigma': args.sigma,
            'seed': args.seed, 'L': len(t_obs),
            'n_fine_steps': int(round(T / args.dt_fine)),
            'shuffle_labels': not args.no_shuffle,
            'elapsed_seconds': elapsed,
            'data_shape': list(data_unlabeled.shape),
        }
        with open(data_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n  Saved to {data_dir}")
        print(f"    data_unlabeled: {data_unlabeled.shape}")
        print(f"    data_labeled:   {data_labeled.shape}")
        print(f"    t_obs:          {t_obs.shape} (t=[{t_obs[0]:.2f}, ..., {t_obs[-1]:.2f}])")

    print(f"\n{'='*70}")
    print(f"ALL DONE — total wall time: {total_elapsed:.1f}s")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
