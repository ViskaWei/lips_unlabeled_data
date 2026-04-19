"""Model registry, data paths, and experiment constants."""

import json
from pathlib import Path

import numpy as np

# ── Config Spine (single source of truth) ────────────────────
from config.model import Config as _Cfg
_cfg = _Cfg()

MODELS = _cfg.models
DIMS = _cfg.dims
DT_OBS_VALUES = _cfg.dt_obs_values
METHODS = _cfg.methods
BASIS_TYPES = _cfg.basis_types

# ── Data paths ────────────────────────────────────────────────
DATA_ROOT = _cfg.data_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / 'results'

# ── Defaults ──────────────────────────────────────────────────
DEFAULT_SIGMA = _cfg.default_sigma
DEFAULT_REG = _cfg.default_reg
DEFAULT_M_MAX = _cfg.default_m_max


def get_data_dir(model, d, M=None, dt_obs=None):
    """Resolve data directory for a given model, dimension, and optional M.

    New dt_obs grid uses dt_obs_{val}/ prefix.  Legacy paths unchanged.
    """
    if dt_obs is not None:
        base = DATA_ROOT / f'dt_obs_{dt_obs}'
        if d == 2:
            return base / model
        return base / f'd{d}' / model
    if M is not None:
        return DATA_ROOT / 'ablation_M' / f'M{M}' / model
    if d == 2:
        return DATA_ROOT / model
    return DATA_ROOT / f'd{d}' / model


def get_results_dir(basis_or_method, model, d, method=None, dt_obs=None):
    """Resolve results directory, optionally under dt_obs prefix."""
    if dt_obs is not None:
        base = RESULTS_ROOT / f'dt_obs_{dt_obs}'
    else:
        base = RESULTS_ROOT
    parts = [basis_or_method, f'd{d}', model]
    if method is not None:
        parts.append(method)
    result = base
    for p in parts:
        result = result / p
    return result


def load_experiment_data(model, d, labeled=False, M=None, dt_obs=None):
    """Load experiment data and config.

    Returns:
        data: (M, L, N, d) array
        t_obs: (L,) array
        config: dict from config.json
    """
    data_dir = get_data_dir(model, d, M=M, dt_obs=dt_obs)
    fname = 'data_labeled.npy' if labeled else 'data_unlabeled.npy'
    data = np.load(str(data_dir / fname))
    t_obs = np.load(str(data_dir / 't_obs.npy'))

    config = {}
    cfg_path = data_dir / 'config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)

    return data, t_obs, config


def get_dt(config, fallback=0.001):
    """Extract dt_obs from config dict."""
    return config.get('dt_obs', fallback)


def get_dt_fine(config, fallback=None):
    """Extract dt_fine from config dict."""
    return config.get('dt_fine', fallback)


def get_obs_stride(config):
    """Infer observation stride dt_obs / dt_fine from config."""
    dt_obs = config.get('dt_obs')
    dt_fine = config.get('dt_fine')
    if dt_obs is None or dt_fine in (None, 0):
        return None
    stride = dt_obs / dt_fine
    stride_round = round(stride)
    if np.isclose(stride, stride_round, rtol=0.0, atol=1e-12):
        return int(max(1, stride_round))
    return float(stride)


def is_zero_gap_config(config):
    """Return True only when config explicitly records dt_fine == dt_obs."""
    dt_obs = config.get('dt_obs')
    dt_fine = config.get('dt_fine')
    if dt_obs is None or dt_fine is None:
        return False
    return bool(np.isclose(dt_obs, dt_fine, rtol=0.0, atol=1e-12))


def require_zero_gap(config, *, label='dataset', subsample_stride=1):
    """Validate that a dataset is truly zero-gap and not post-subsampled."""
    dt_obs = config.get('dt_obs')
    dt_fine = config.get('dt_fine')
    if dt_obs is None or dt_fine is None:
        raise ValueError(
            f"{label} is missing explicit dt_obs/dt_fine metadata; "
            "cannot certify zero-gap."
        )
    if subsample_stride != 1:
        raise ValueError(
            f"{label} uses subsample_stride={subsample_stride}; "
            "zero-gap must come from the raw data, not post-subsampling."
        )
    if not np.isclose(dt_obs, dt_fine, rtol=0.0, atol=1e-12):
        stride = get_obs_stride(config)
        raise ValueError(
            f"{label} is not zero-gap: dt_fine={dt_fine}, dt_obs={dt_obs}, "
            f"effective stride={stride}."
        )
    return {
        'dt_obs': float(dt_obs),
        'dt_fine': float(dt_fine),
        'obs_stride': 1,
        'zero_gap': True,
    }


def compute_r_max(data, percentile=99):
    """Compute r_max_V and r_max_Phi from particle data.

    Args:
        data: (M, L, N, d) particle data
        percentile: percentile for r_max determination

    Returns:
        r_max_V: float, upper radius for V basis
        r_max_Phi: float, upper radius for Phi basis
    """
    M, L, N, d = data.shape

    # V: radial distances from origin
    all_r = np.linalg.norm(data.reshape(-1, d), axis=1)
    r_max_V = float(np.percentile(all_r, percentile))

    # Phi: pairwise distances (subsample for efficiency)
    n_snap = min(100, M)
    snap = data[:n_snap, L // 2]  # (n_snap, N, d)
    diffs = snap[:, :, None, :] - snap[:, None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)  # (n_snap, N, N)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    r_max_Phi = float(np.percentile(dists[:, mask], percentile))

    return r_max_V, r_max_Phi
