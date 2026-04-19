#!/usr/bin/env python3
"""Group 3: Neural network self-test training across models and dimensions.

Refactored from train_nn_selftest.py to use lib/ for data loading and
consistent results structure.

Usage:
    python scripts/run_nn.py --models a b lj morse --dims 2 5 10 20 --epochs 200
    python scripts/run_nn.py --models a --dims 2 --epochs 100 --init random
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.config import (
    MODELS, DIMS, RESULTS_ROOT, DEFAULT_SIGMA,
    load_experiment_data, get_results_dir,
)
from lib.eval import (
    NONRADIAL_MODELS, is_radial, compute_kde_errors,
    load_kde, get_kde_path,
)
from core.nn_models import (
    RadialNet, RadialInteractionNet, GeneralNet, SymmetricInteractionNet,
)
from core.selftest_loss import compute_selftest_loss_batch


# ══════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════

def train_epoch(V_net, Phi_net, opt_V, opt_Phi, data_t, dt_all, sigma,
                batch_size=64, grad_clip=1.0, scaler=None, frac_missing=0.0,
                grad_accum_steps=1, penalty_lambda=0.0):
    """One epoch over all (m, ell) pairs.

    Args:
        data_t: (M, L, N, d) tensor on device (pre-allocated)
        dt_all: (L-1,) tensor of time deltas on device (pre-allocated)
        scaler: optional GradScaler for AMP
        frac_missing: fraction of particles to drop per snapshot (0-0.5).
            Works because unlabeled data has particles pre-shuffled at each
            (m, l), so slicing [:N_keep] is equivalent to random subsampling.
        grad_accum_steps: accumulate gradients over this many mini-batches
            before stepping. Effective batch size = batch_size * grad_accum_steps.
        penalty_lambda: coefficient for positive-loss penalty λ·[max(0,L)]².
            When > 0, adds quadratic penalty on positive loss values to
            accelerate convergence (loss minimum is known to be negative).
    """
    M, L, N, d = data_t.shape
    N_keep = max(2, int(round(N * (1 - frac_missing)))) if frac_missing > 0 else N

    m_idx = np.repeat(np.arange(M), L - 1)
    ell_idx = np.tile(np.arange(L - 1), M)
    perm = np.random.permutation(len(m_idx))
    m_idx, ell_idx = m_idx[perm], ell_idx[perm]

    total_loss, n_batches = 0.0, 0
    opt_V.zero_grad()
    opt_Phi.zero_grad()

    for start in range(0, len(m_idx), batch_size):
        end = min(start + batch_size, len(m_idx))
        mb, eb = m_idx[start:end], ell_idx[start:end]
        X_curr = data_t[mb, eb][:, :N_keep, :]
        X_next = data_t[mb, eb + 1][:, :N_keep, :]

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                residuals = compute_selftest_loss_batch(
                    V_net, Phi_net, X_curr, X_next,
                    dt_all[eb], sigma,
                )
                loss_raw = residuals.mean()
                if penalty_lambda > 0:
                    loss_raw = loss_raw + penalty_lambda * torch.relu(loss_raw) ** 2
                loss = loss_raw / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            residuals = compute_selftest_loss_batch(
                V_net, Phi_net, X_curr, X_next,
                dt_all[eb], sigma,
            )
            loss_raw = residuals.mean()
            if penalty_lambda > 0:
                loss_raw = loss_raw + penalty_lambda * torch.relu(loss_raw) ** 2
            loss = loss_raw / grad_accum_steps
            loss.backward()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

        if n_batches % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(opt_V)
                scaler.unscale_(opt_Phi)
                torch.nn.utils.clip_grad_norm_(V_net.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(Phi_net.parameters(), grad_clip)
                scaler.step(opt_V)
                scaler.step(opt_Phi)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(V_net.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(Phi_net.parameters(), grad_clip)
                opt_V.step()
                opt_Phi.step()
            opt_V.zero_grad()
            opt_Phi.zero_grad()

    # Handle remaining accumulated gradients
    if n_batches % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(opt_V)
            scaler.unscale_(opt_Phi)
            torch.nn.utils.clip_grad_norm_(V_net.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(Phi_net.parameters(), grad_clip)
            scaler.step(opt_V)
            scaler.step(opt_Phi)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(V_net.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(Phi_net.parameters(), grad_clip)
            opt_V.step()
            opt_Phi.step()

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════
# NN evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_nn(V_net, Phi_net, model_name, d, kde_grids, device='cpu'):
    """KDE-based gradient errors using AD-based gradient functions.

    Uses precomputed KDE densities for deterministic, rho-weighted L2 evaluation.
    For radial models only; non-radial models return (inf, inf).

    Args:
        V_net: trained V network
        Phi_net: trained Phi network
        model_name: model identifier
        d: spatial dimension
        kde_grids: precomputed KDE grids (from load_kde)
        device: torch device
    """
    if kde_grids is None or not is_radial(model_name):
        return float('inf'), float('inf')

    V_net.eval()
    Phi_net.eval()

    def grad_V_func(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=device).requires_grad_(True)
        f = V_net(x)
        g = torch.autograd.grad(f.sum(), x)[0]
        return g.detach().cpu().numpy()

    def dPhi_dr_func(r_np):
        x = torch.zeros(len(r_np), d, dtype=torch.float32, device=device)
        x[:, 0] = torch.tensor(r_np, dtype=torch.float32, device=device)
        x.requires_grad_(True)
        f = Phi_net(x)
        g = torch.autograd.grad(f.sum(), x)[0]
        return g[:, 0].detach().cpu().numpy()

    errs = compute_kde_errors(kde_grids, grad_V_func, dPhi_dr_func,
                              model_name, d=d)

    V_net.train()
    Phi_net.train()
    return errs['V_error'], errs['Phi_error']


# ══════════════════════════════════════════════════════════════
# Basis pre-training
# ══════════════════════════════════════════════════════════════

def _rbf_basis(r, centers, width):
    return np.exp(-((r[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))

def _rbf_grad(r, centers, width):
    return -(r[:, None] - centers[None, :]) / (width ** 2) * _rbf_basis(r, centers, width)


def pretrain_from_basis(V_net, Phi_net, model_name, data, device,
                        pretrain_epochs=200, pretrain_lr=1e-3):
    """Supervised pre-training from basis function coefficients."""
    d = data.shape[-1]
    coeffs_path = ROOT / 'results' / f'paper_{model_name}' / 'coefficients.json'
    if not coeffs_path.exists():
        # Try new results structure
        coeffs_path = RESULTS_ROOT / 'rbf' / 'd2' / model_name / 'selftest' / 'metrics.json'
    if not coeffs_path.exists():
        print(f"  No basis coefficients found, skipping pre-training")
        return False

    with open(coeffs_path) as f:
        coeffs = json.load(f)

    alpha = np.array(coeffs['alpha'])
    beta = np.array(coeffs['beta'])

    # Try to get RBF parameters
    if 'centers_V' not in coeffs:
        print(f"  No RBF parameters in coefficients, skipping pre-training")
        return False

    centers_V = np.array(coeffs['centers_V'])
    centers_Phi = np.array(coeffs['centers_Phi'])
    width_V = coeffs['width_V']
    width_Phi = coeffs['width_Phi']

    M, L, N, _ = data.shape
    X_flat = data.reshape(-1, d)
    r_particles = np.sqrt(np.sum(X_flat ** 2, axis=-1))

    n_snap = min(200, M * L)
    snap_idx = np.random.choice(M * L, n_snap, replace=False)
    X_snaps = data.reshape(M * L, N, d)[snap_idx]
    r_pairs_list = []
    for snap in X_snaps:
        diff_s = snap[:, None, :] - snap[None, :, :]
        r_pw = np.sqrt(np.sum(diff_s ** 2, axis=-1))
        i_up, j_up = np.triu_indices(N, k=1)
        r_pairs_list.append(r_pw[i_up, j_up])
    r_pairs = np.concatenate(r_pairs_list)

    V_vals = _rbf_basis(r_particles, centers_V, width_V) @ alpha
    V_derivs = _rbf_grad(r_particles, centers_V, width_V) @ alpha
    Phi_vals = _rbf_basis(r_pairs, centers_Phi, width_Phi) @ beta
    Phi_derivs = _rbf_grad(r_pairs, centers_Phi, width_Phi) @ beta

    x_V = torch.zeros(len(r_particles), d, dtype=torch.float32, device=device)
    x_V[:, 0] = torch.tensor(r_particles, dtype=torch.float32)
    y_V_val = torch.tensor(V_vals, dtype=torch.float32, device=device)
    y_V_der = torch.tensor(V_derivs, dtype=torch.float32, device=device)

    x_Phi = torch.zeros(len(r_pairs), d, dtype=torch.float32, device=device)
    x_Phi[:, 0] = torch.tensor(r_pairs, dtype=torch.float32)
    y_Phi_val = torch.tensor(Phi_vals, dtype=torch.float32, device=device)
    y_Phi_der = torch.tensor(Phi_derivs, dtype=torch.float32, device=device)

    print(f"  Pre-training: {len(r_particles)} V, {len(r_pairs)} Phi samples")

    params = list(V_net.parameters()) + list(Phi_net.parameters())
    optimizer = torch.optim.Adam(params, lr=pretrain_lr)
    batch_size = 4096

    for epoch in range(pretrain_epochs):
        idx_v = np.random.choice(len(x_V), min(batch_size, len(x_V)), replace=False)
        xv = x_V[idx_v].detach().requires_grad_(True)
        pred_V = V_net(xv)
        loss_V_val = ((pred_V - y_V_val[idx_v]) ** 2).mean()
        grd_V = torch.autograd.grad(pred_V.sum(), xv, create_graph=True)[0]
        loss_V_grad = ((grd_V[:, 0] - y_V_der[idx_v]) ** 2).mean()

        idx_p = np.random.choice(len(x_Phi), min(batch_size, len(x_Phi)), replace=False)
        xp = x_Phi[idx_p].detach().requires_grad_(True)
        pred_Phi = Phi_net(xp)
        loss_Phi_val = ((pred_Phi - y_Phi_val[idx_p]) ** 2).mean()
        grd_Phi = torch.autograd.grad(pred_Phi.sum(), xp, create_graph=True)[0]
        loss_Phi_grad = ((grd_Phi[:, 0] - y_Phi_der[idx_p]) ** 2).mean()

        loss = loss_V_val + loss_V_grad + loss_Phi_val + loss_Phi_grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Pre-train [{epoch+1}/{pretrain_epochs}]: "
                  f"V={loss_V_val.item():.4f}+{loss_V_grad.item():.4f}, "
                  f"Phi={loss_Phi_val.item():.4f}+{loss_Phi_grad.item():.4f}")

    return True


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def train_one(model_name, d, args, device):
    """Train NN self-test for one (model, d) pair."""
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not (data_dir / 'data_unlabeled.npy').exists():
            print(f"  SKIP: no data in {data_dir}")
            return None
        data = np.load(data_dir / 'data_unlabeled.npy')
        t_obs = np.load(data_dir / 't_obs.npy')
        config = {}
        if (data_dir / 'config.json').exists():
            with open(data_dir / 'config.json') as f:
                config = json.load(f)
    else:
        try:
            data, t_obs, config = load_experiment_data(model_name, d, labeled=False,
                                                        dt_obs=args.dt_obs)
        except FileNotFoundError:
            print(f"  SKIP: no data for {model_name} d={d} dt_obs={args.dt_obs}")
            return None

    M, L, N, d_actual = data.shape
    print(f"  Data: M={M}, L={L}, N={N}, d={d_actual}")

    # Load precomputed KDE grids for evaluation
    kde_grids = None
    if is_radial(model_name):
        try:
            if args.kde_dir:
                # Custom KDE dir: skip dt_obs prefix (user provides full base path)
                kde_path = get_kde_path(model_name, d_actual, dt_obs=None,
                                        base_dir=args.kde_dir)
            else:
                kde_path = get_kde_path(model_name, d_actual, dt_obs=args.dt_obs)
            kde_grids = load_kde(kde_path)
            print(f"  KDE grids loaded from {kde_path}")
        except (FileNotFoundError, Exception) as e:
            print(f"  WARNING: KDE grids not found ({e}). Eval will return inf.")

    M_use = min(args.M_train, M)
    data_train = data[:M_use]
    N_keep = max(2, int(round(N * (1 - args.frac_missing)))) if args.frac_missing > 0 else N
    if args.frac_missing > 0:
        print(f"  Missing particles: frac={args.frac_missing}, N_keep={N_keep}/{N}")
    if args.grad_accum > 1:
        print(f"  Gradient accumulation: {args.grad_accum} steps (effective batch={args.batch_size * args.grad_accum})")

    hdims = tuple(args.hidden_dims)
    act = args.activation
    is_nonradial = model_name in NONRADIAL_MODELS
    if is_nonradial:
        V_net = GeneralNet(d=d_actual, hidden_dims=hdims, activation=act).to(device)
        Phi_net = SymmetricInteractionNet(d=d_actual, hidden_dims=hdims, activation=act).to(device)
        print(f"  Architecture: GeneralNet + SymmetricInteractionNet (non-radial, {act})")
    else:
        V_net = RadialNet(hidden_dims=hdims, activation=act).to(device)
        Phi_net = RadialInteractionNet(d=d_actual, hidden_dims=hdims, activation=act).to(device)
        print(f"  Architecture: RadialNet + RadialInteractionNet ({act})")
    n_V = sum(p.numel() for p in V_net.parameters())
    n_Phi = sum(p.numel() for p in Phi_net.parameters())
    print(f"  V: {n_V} params, Phi: {n_Phi} params, hidden={list(hdims)}")

    if args.init == 'basis':
        pretrained = pretrain_from_basis(V_net, Phi_net, model_name, data, device,
                                          pretrain_epochs=args.pretrain_epochs)
        if pretrained:
            v_e, p_e = evaluate_nn(V_net, Phi_net, model_name, d_actual, kde_grids, device)
            print(f"  After pre-train: V={v_e*100:.1f}%, Phi={p_e*100:.1f}%")

    v_init, phi_init = evaluate_nn(V_net, Phi_net, model_name, d_actual, kde_grids, device)
    best_err = v_init + phi_init
    best_state = {
        'V': {k: v.cpu().clone() for k, v in V_net.state_dict().items()},
        'Phi': {k: v.cpu().clone() for k, v in Phi_net.state_dict().items()},
    }

    # Pre-allocate data tensors on device (once, not per epoch)
    data_t = torch.tensor(data_train, dtype=torch.float32, device=device)
    dt_all = torch.tensor(np.diff(t_obs), dtype=torch.float32, device=device)

    lr_phi = args.lr_phi if args.lr_phi else 5.0 * args.lr
    opt_V = torch.optim.Adam(V_net.parameters(), lr=args.lr)
    opt_Phi = torch.optim.Adam(Phi_net.parameters(), lr=lr_phi)
    sched_V = torch.optim.lr_scheduler.CosineAnnealingLR(opt_V, T_max=args.epochs, eta_min=args.lr * 0.01)
    sched_Phi = torch.optim.lr_scheduler.CosineAnnealingLR(opt_Phi, T_max=args.epochs, eta_min=lr_phi * 0.01)

    # AMP scaler for mixed precision (GH200 BF16 ~1.5-2× speedup)
    scaler = torch.amp.GradScaler('cuda') if args.amp and device == 'cuda' else None
    if scaler:
        print(f"  AMP enabled (mixed precision)")
    if args.penalty_lambda > 0:
        print(f"  Positive-loss penalty: lambda={args.penalty_lambda}")

    losses = []
    patience_ctr = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        if args.warmup > 0 and epoch < args.warmup:
            scale = 0.1 + 0.9 * epoch / args.warmup
            for pg in opt_V.param_groups:
                pg['lr'] = args.lr * scale
            for pg in opt_Phi.param_groups:
                pg['lr'] = lr_phi * scale

        loss = train_epoch(V_net, Phi_net, opt_V, opt_Phi, data_t, dt_all,
                           sigma=args.sigma, batch_size=args.batch_size,
                           grad_clip=args.grad_clip, scaler=scaler,
                           frac_missing=args.frac_missing,
                           grad_accum_steps=args.grad_accum,
                           penalty_lambda=args.penalty_lambda)
        if epoch >= args.warmup:
            sched_V.step()
            sched_Phi.step()
        losses.append(loss)

        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            v_err, phi_err = evaluate_nn(V_net, Phi_net, model_name, d_actual, kde_grids, device)
            total = v_err + phi_err
            tag = ''
            if total < best_err:
                best_err = total
                best_state = {
                    'V': {k: v.cpu().clone() for k, v in V_net.state_dict().items()},
                    'Phi': {k: v.cpu().clone() for k, v in Phi_net.state_dict().items()},
                }
                patience_ctr = 0
                tag = ' *'
            else:
                patience_ctr += 1

            neg = ' [NEG]' if loss < 0 else ''
            print(f"  [{epoch+1:3d}/{args.epochs}] loss={loss:.6f}{neg} "
                  f"V={v_err*100:.1f}% Phi={phi_err*100:.1f}%{tag}")

            # Validate save pipeline on first evaluation: write a real metrics.json
            if epoch == 0:
                _ckpt_dir = get_results_dir(args.results_name, model_name, d, dt_obs=args.dt_obs)
                _ckpt_dir.mkdir(parents=True, exist_ok=True)
                try:
                    _test_metrics = {
                        'method': 'NN Self-Test', 'model': model_name,
                        'd': d_actual, 'N': int(N), 'M_train': M_use,
                        'V_error': float(v_err), 'Phi_error': float(phi_err),
                        'V_error_pct': float(v_err * 100),
                        'Phi_error_pct': float(phi_err * 100),
                        'training_time_s': 0.0, 'epochs': 1,
                        'init': args.init, 'status': 'epoch1_test',
                    }
                    with open(_ckpt_dir / 'metrics.json', 'w') as _f:
                        json.dump(_test_metrics, _f, indent=2)
                    torch.save({'V': V_net.state_dict(), 'Phi': Phi_net.state_dict()},
                               _ckpt_dir / 'checkpoint_test.pt')
                    (_ckpt_dir / 'checkpoint_test.pt').unlink()
                    print(f"  Save pipeline OK ({_ckpt_dir})")
                except Exception as e:
                    print(f"  FATAL: save pipeline broken — {e}")
                    raise

            # Periodic checkpoint saves (best model so far)
            _CKPT_EPOCHS = {10, 50, 100, 150}
            if (epoch + 1) in _CKPT_EPOCHS and best_state is not None:
                _ckpt_dir = get_results_dir(args.results_name, model_name, d, dt_obs=args.dt_obs)
                _ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'V_state_dict': best_state['V'],
                    'Phi_state_dict': best_state['Phi'],
                    'hidden_dims': list(hdims),
                    'd': d_actual,
                    'epoch': epoch + 1,
                }, _ckpt_dir / f'model_epoch{epoch+1}.pt')
                print(f"  Checkpoint saved: model_epoch{epoch+1}.pt")

            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    V_net.load_state_dict({k: v.to(device) for k, v in best_state['V'].items()})
    Phi_net.load_state_dict({k: v.to(device) for k, v in best_state['Phi'].items()})
    v_final, phi_final = evaluate_nn(V_net, Phi_net, model_name, d_actual, kde_grids, device)
    elapsed = time.time() - t0

    print(f"  FINAL: V={v_final*100:.2f}%, Phi={phi_final*100:.2f}%, "
          f"time={elapsed:.0f}s")

    results_dir = get_results_dir(args.results_name, model_name, d, dt_obs=args.dt_obs)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'method': 'NN Self-Test',
        'model': model_name,
        'd': d_actual,
        'N': int(N),
        'M_train': M_use,
        'V_error': float(v_final),
        'Phi_error': float(phi_final),
        'V_error_pct': float(v_final * 100),
        'Phi_error_pct': float(phi_final * 100),
        'training_time_s': elapsed,
        'epochs': args.epochs,
        'init': args.init,
        'lr': args.lr,
        'lr_phi': lr_phi,
        'hidden_dims': list(hdims),
        'batch_size': args.batch_size,
        'warmup': args.warmup,
        'grad_clip': args.grad_clip,
        'sigma': args.sigma,
        'dt_obs': args.dt_obs,
        'seed': args.seed,
        'amp': args.amp,
        'activation': args.activation,
        'results_name': args.results_name,
        'frac_missing': args.frac_missing,
        'N_observed': N_keep,
        'min_loss': float(min(losses)),
        'final_loss': float(losses[-1]),
        'loss_negative': bool(any(l < 0 for l in losses)),
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    torch.save({
        'V_state_dict': V_net.state_dict(),
        'Phi_state_dict': Phi_net.state_dict(),
        'hidden_dims': list(hdims),
        'activation': args.activation,
        'd': d_actual,
    }, results_dir / 'model.pt')

    np.save(results_dir / 'loss_history.npy', np.array(losses))
    print(f"  Saved to {results_dir}")
    return metrics


def main():
    p = argparse.ArgumentParser(description='NN self-test training')
    p.add_argument('--models', nargs='+', default=['a', 'b', 'lj', 'morse'])
    p.add_argument('--dims', nargs='+', type=int, default=DIMS)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_phi', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--M_train', type=int, default=2000)
    p.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    p.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 64])
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--grad_accum', type=int, default=1,
                   help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
    p.add_argument('--print_every', type=int, default=5)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--init', choices=['random', 'basis'], default='basis')
    p.add_argument('--pretrain_epochs', type=int, default=200)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--amp', action='store_true',
                   help='Enable mixed precision (AMP) for GPU training')
    p.add_argument('--dt_obs', type=float, default=None,
                   help='Observation timestep for new dt_obs grid (None = legacy)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--results_name', type=str, default='nn',
                   help='Results subdirectory name (e.g. nn, nn_b64)')
    p.add_argument('--frac_missing', type=float, default=0.0,
                   help='Fraction of particles to randomly drop per snapshot (0-0.5)')
    p.add_argument('--activation', choices=['tanh', 'softplus'], default='softplus',
                   help='Hidden layer activation (default: softplus)')
    p.add_argument('--kde_dir', type=str, default=None,
                   help='Override KDE grids base directory (e.g. results/kde_grids/M20000_dt_obs_0.001)')
    p.add_argument('--data_dir', type=str, default=None,
                   help='Override data directory (bypasses auto-detection)')
    p.add_argument('--penalty_lambda', type=float, default=0.0,
                   help='Positive-loss penalty coefficient. Adds lambda*[max(0,L)]^2 to loss.')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    all_metrics = {}
    for suffix in args.models:
        model_name = f'model_{suffix}'
        for d in args.dims:
            print(f"\n{'=' * 60}")
            print(f"  NN Self-Test | {model_name} | d={d}")
            print(f"{'=' * 60}")
            metrics = train_one(model_name, d, args, device)
            if metrics:
                all_metrics[f"{model_name}_d{d}"] = metrics

    print(f"\n{'=' * 60}")
    print("NN SELF-TEST SUMMARY")
    print(f"{'=' * 60}")
    for key in sorted(all_metrics):
        m = all_metrics[key]
        print(f"  {m['model']} d={m['d']}: V={m['V_error_pct']:.2f}% Phi={m['Phi_error_pct']:.2f}% "
              f"({m['training_time_s']:.0f}s)")


if __name__ == '__main__':
    main()
