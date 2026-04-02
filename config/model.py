"""Central configuration model — loads from config/base.yaml.

Single source of truth for all experiment parameters.
Both lib/config.py and scripts/lib/config.py import from here.
"""
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / 'base.yaml'


class Config:
    """Lightweight config object loaded from base.yaml."""

    def __init__(self, path=None):
        p = Path(path) if path else _CONFIG_PATH
        with open(p) as f:
            d = yaml.safe_load(f)
        self.weighting_method = d['weighting_method']
        self.default_sigma = d['default_sigma']
        self.default_reg = d['default_reg']
        self.default_m_max = d['default_m_max']
        self.models = d['models']
        self.dims = d['dims']
        self.dt_obs_values = d['dt_obs_values']
        self.methods = d['methods']
        self.basis_types = d['basis_types']
        self.data_root = Path(d['data_root'])
        self.results_subdir = d['results_subdir']

    def __repr__(self):
        return (f"Config(method={self.weighting_method}, "
                f"reg={self.default_reg}, sigma={self.default_sigma}, "
                f"m_max={self.default_m_max})")
