"""Re-evaluate saved NN checkpoints with correct KDE grids."""
import sys
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.nn_models import RadialNet, RadialInteractionNet
from lib.eval import compute_kde_errors, load_kde
from lib.config import RESULTS_ROOT


def evaluate(model_pt, kde_grids, device='cpu'):
    ckpt = torch.load(model_pt, map_location=device)
    d = ckpt.get('d', 2)
    hidden = ckpt.get('hidden_dims', [64, 64, 64])
    act = ckpt.get('activation', 'softplus')

    V_net = RadialNet(hidden_dims=hidden, activation=act).to(device)
    Phi_net = RadialInteractionNet(d=d, hidden_dims=hidden, activation=act).to(device)
    V_net.load_state_dict(ckpt['V_state_dict'])
    Phi_net.load_state_dict(ckpt['Phi_state_dict'])
    V_net.eval(); Phi_net.eval()

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

    errs = compute_kde_errors(kde_grids, grad_V_func, dPhi_dr_func, 'model_e', d=d)
    return errs['V_error'], errs['Phi_error']


results_base = RESULTS_ROOT

for dt in ['0.1', '0.01']:
    kde_path = str(results_base / f'model_e_dt{dt}_lcurve/kde/d2/model_e/kde_grids.npz')
    kde_grids = load_kde(kde_path)

    trial_dir = results_base / f'dt_obs_{dt}/nn_errbar_table5'
    v_errs, p_errs = [], []
    for t in range(10):
        pt = trial_dir / f'trial_{t}/d2/model_e/model.pt'
        if not pt.exists():
            continue
        v, p = evaluate(pt, kde_grids)
        v_errs.append(v); p_errs.append(p)
        print('  dt=%s trial %d: V=%.2f%% Phi=%.2f%%' % (dt, t, v, p))

    if v_errs:
        print('dt=%s: V=%.2f+/-%.2f%% Phi=%.2f+/-%.2f%% (n=%d)' % (
            dt, np.mean(v_errs), np.std(v_errs), np.mean(p_errs), np.std(p_errs), len(v_errs)))
    print()
