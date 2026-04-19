<div align="center">

# LIPS

### Learning Interacting Particle Systems from Unlabeled Snapshot Data

**Viska Wei** &nbsp;·&nbsp; **Fei Lu**

<p>
<a href="https://arxiv.org/abs/2604.02581"><img src="https://img.shields.io/badge/arXiv-2604.02581-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/></a>
<a href="https://viskawei.github.io/ips_unlabeled_learning_web/"><img src="https://img.shields.io/badge/Project_Page-visit-1f6feb?style=for-the-badge&logo=astro&logoColor=white" alt="Project Page"/></a>
<a href="#-citation"><img src="https://img.shields.io/badge/BibTeX-copy-8b5cf6?style=for-the-badge&logo=latex&logoColor=white" alt="BibTeX"/></a>
<img src="https://img.shields.io/badge/python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
</p>

<p align="center">
<em>Recover the physics — <b>V</b> (confinement) and <b>Φ</b> (interaction) —<br/>
from unordered particle snapshots, <b>without ever knowing which particle is which</b>.</em>
</p>

$$
dX_t^i \;=\; \Big(\underbrace{-\nabla V(X_t^i)}_{\text{confinement}} \;\;\underbrace{-\; \tfrac{1}{N}\!\!\sum_{j\neq i}\!\nabla\Phi(X_t^i - X_t^j)}_{\text{interaction}}\Big)\,dt \;+\; \underbrace{\sigma\,dW_t^i}_{\text{noise}}
$$

</div>

---

## ✨ Highlights

|   |   |
|---|---|
| 🏷️ **No labels.** | No trajectory reconstruction, no velocity estimation, no identity matching. |
| 🧮 **Provably unique.** | A linear self-test loss (Algorithm 1) recovers $(V, \Phi)$ with closed-form uniqueness. |
| 📐 **Basis-driven.** | RBF basis + MLE / Sinkhorn / self-test solvers — converges at $M^{-1/2}$ rates across $d = 2 \ldots 20$. |
| 🔁 **Fully reproducible.** | 17 scripts, one YAML config — every figure and table in the paper. |

---

## 🚀 Quick Start

```bash
git clone https://github.com/ViskaWei/lips_unlabeled_data
cd lips_unlabeled_data
pip install -e .
```

End-to-end smoke test (data → NN → figure):

```bash
python scripts/generate_data_elephant6.py --model a --d 2 --M 2000 --L 100
python scripts/run_nn.py        --models a --dims 2 --epochs 200 --init random
python scripts/plot_m_scaling_zerogap.py
```

---

## 📁 Repository Layout

```
lips_unlabeled_data/
├── config/          # base.yaml — single source of truth
├── core/            # SDE simulator · potentials · NN models · self-test loss (Alg. 1)
├── lib/             # RBF basis · solvers (MLE, Sinkhorn, self-test) · KDE eval · plotting
└── scripts/         # experiments (run_*.py) + figures (plot_*.py)
```

---

## 🔬 Reproducing the Paper

<details>
<summary><b>Figure 2 · Table 6 — sample-size scaling</b></summary>

```bash
python scripts/run_m_scaling.py --M_values 200 500 1000 2000 --models a b lj morse
python scripts/plot_m_scaling_zerogap.py
python scripts/plot_m_scaling_discrete.py
```

</details>

<details>
<summary><b>Figure 4 · Table 5 — condition numbers</b></summary>

```bash
python scripts/run_n_scaling_cond.py
python scripts/run_full_cond_table.py
python scripts/plot_condition_numbers.py
```

</details>

<details>
<summary><b>Figure 5 · Table 2 — neural-network training</b></summary>

```bash
python scripts/run_nn.py --models a b lj morse --dims 2 5 10 20 --epochs 200 --init random
```

</details>

<details>
<summary><b>Error bars (10-trial runs)</b></summary>

```bash
python scripts/run_error_bars.py --models a b lj morse --n_trials 10 --M_sub 2000
```

</details>

<details>
<summary><b>Appendix figures · boundary recovery</b></summary>

```bash
python scripts/plot_boundary_recovery.py
python scripts/plot_all_methods_appendix.py
```

</details>

---

## ⚙️ Configuration

All defaults live in [`config/base.yaml`](config/base.yaml) — data paths, model list, dimensions, RBF settings. Override any of them via CLI flags on the `run_*.py` scripts.

---

## 📖 Citation

If this work is useful to you, please cite:

```bibtex
@article{wei2026learning,
  title   = {Learning interacting particle systems from unlabeled data},
  author  = {Wei, Viska and Lu, Fei},
  journal = {arXiv preprint arXiv:2604.02581},
  year    = {2026}
}
```

---

<div align="center">
<sub>Questions? Open an issue, or explore the <a href="https://viskawei.github.io/ips_unlabeled_learning_web/">interactive project page</a> 🌐</sub>
</div>
