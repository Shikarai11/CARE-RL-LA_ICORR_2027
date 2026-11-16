"""
Smoke test on SYNTHETIC data (not the study data)
=================================================

Checks that the installation works without the participant data, and shows the
core API in a few lines:

    1) build a small Dataset of (s_k, a_k, s_{k+1}, r_k, p_k, p_{k+1}) transitions,
    2) fit OfflinePolicyIteration (the reference implementation, CVXPY),
    3) fit FastOPI (the vectorised implementation used by the controls) and
       check that both learn the same S matrix,
    4) ask the learned policy for an update of the three parameters.

The synthetic gait features are a noisy linear function of the parameters and the
reward simply penalises their squared norm, so the numbers have NO physical meaning.

Run from the repository root (about 20 seconds):
    python examples/synthetic_demo.py
"""
from __future__ import annotations

import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                # OPI_core, elsa_configOPI, ...
sys.path.insert(0, str(ROOT / "controls"))   # fast_opi

from elsa_configOPI import OPIConfig                  # noqa: E402
from OPI_core import Dataset, OfflinePolicyIteration, Transition  # noqa: E402
import fast_opi as F                                  # noqa: E402


def make_synthetic_dataset(cfg: OPIConfig, n_blocks: int = 13, n_cycles: int = 4, seed: int = 0) -> Dataset:
    """Random settings per block, a few gait cycles per block, all-to-all pairing of consecutive blocks."""
    rng = np.random.default_rng(seed)
    p_min, p_max = cfg.prosthesis.p_min, cfg.prosthesis.p_max
    half = (p_max - p_min) / 2.0
    W = rng.normal(size=(cfg.n_features, cfg.n_actions))      # features <- parameters
    p_opt = p_min + half * (1.0 + rng.uniform(-0.5, 0.5, cfg.n_actions))

    P = rng.uniform(p_min, p_max, size=(n_blocks, cfg.n_actions))   # one setting per block
    feats = [((P[b] - p_opt) / half) @ W.T + 0.1 * rng.normal(size=(n_cycles, cfg.n_features))
             for b in range(n_blocks)]                              # (n_cycles, n_features) per block

    ds = Dataset()
    for k in range(n_blocks - 1):
        a = P[k + 1] - P[k]
        for x in feats[k]:                 # every cycle of block k ...
            for x1 in feats[k + 1]:        # ... with every cycle of block k+1
                r = -float(x1 @ x1) - float(((a / half) ** 2).sum())
                ds.add(Transition(x_feat=x, a=a, x1_feat=x1, r=r, p=P[k], p1=P[k + 1]))
    return ds


def main():
    cfg = OPIConfig(n_features=6, n_actions=3, lambda_reg=1e-3, n_iterations=30)
    ds = make_synthetic_dataset(cfg)
    print(f"Synthetic dataset: {len(ds)} pseudo-transitions")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        ref = OfflinePolicyIteration(copy.deepcopy(cfg)).fit(ds)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        fast = F.FastOPI(copy.deepcopy(cfg)).fit(ds)
        t_fast = time.perf_counter() - t0

    a_idx = 1 + cfg.n_features                 # bias + state coordinates come first in mu = [1; s; a]
    eig = np.linalg.eigvalsh(ref.S[a_idx:, a_idx:])
    rel = np.linalg.norm(ref.S - fast.S) / np.linalg.norm(ref.S)
    print(f"OfflinePolicyIteration: {len(ref.history['frobenius'])} iterations, kept iterate {ref.best_iter} ({t_ref:.1f} s)")
    print(f"FastOPI:                {len(fast.history['frobenius'])} iterations, kept iterate {fast.best_iter} ({t_fast:.1f} s)")
    print(f"Relative difference between the two S matrices: {rel:.1e}")
    print(f"Eigenvalues of S_aa (curvature of Q in the action): {np.round(eig, 3)}")

    x, _, _, _, p, _ = ds.arrays(cfg)
    a_k, p_k1 = ref.predict(x[0], p[0])
    names = ["d_dors [N m s/rad]", "theta_trig [rad]", "tau_lim [N m]"]
    print("Recommended update for the first state:")
    for n, cur, da, new in zip(names, p[0], a_k, p_k1):
        print(f"  {n:<20} {cur:8.3f} -> {new:8.3f}   (update {da:+.3f})")


if __name__ == "__main__":
    main()
