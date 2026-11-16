"""
Check that fast_opi.FastOPI reproduces the original OfflinePolicyIteration
==========================================================================

Fits both implementations on the first `--n-pairs` real block pairs of the
all-to-all dataset (default 4 pairs = 6,400 rows, a few minutes) and compares:
  - the Frobenius history ||S(i+1) - S(i)||_F,
  - the final S matrix and the retained iteration,
  - the greedy actions on every state,
  - the saturation and sensitivity diagnostics (original OPIEvaluator vs vectorised).

Run from the repository root:
    python controls/check_equivalence.py --pickle data/all_to_all_opi_dataset.pkl \
        --csv data/all_to_all_transitions.csv --n-pairs 4
"""
from __future__ import annotations

import argparse
import copy
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ (OPI_core, OPI_validation, reward, elsa_configOPI)
sys.path.insert(0, str(Path(__file__).resolve().parent))      # code/controls/

from elsa_configOPI import OPIConfig                     # noqa: E402
from OPI_core import Dataset, OfflinePolicyIteration     # noqa: E402
from OPI_validation import OPIEvaluator                  # noqa: E402
import fast_opi as F                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n-pairs", type=int, default=4, help="at least 4: with fewer pairs the fit is too underdetermined to compare iterates")
    ap.add_argument("--lambda-reg", type=float, default=1e-3)
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        full: Dataset = pickle.load(f)
    pair_ids = pd.read_csv(args.csv, usecols=["pair_id"])["pair_id"].astype(str).to_numpy()
    keep = np.isin(pair_ids, pd.unique(pair_ids)[: args.n_pairs])
    ds = Dataset()
    for t, k in zip(full, keep):
        if k:
            ds.add(copy.deepcopy(t))
    print(f"Subset: {len(ds)} rows from {args.n_pairs} pairs")

    cfg = OPIConfig(n_features=6, n_actions=3, lambda_reg=args.lambda_reg)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = OfflinePolicyIteration(copy.deepcopy(cfg)).fit(ds)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast = F.FastOPI(copy.deepcopy(cfg)).fit(ds)
    t_fast = time.perf_counter() - t0

    fr_ref, fr_fast = np.array(ref.history["frobenius"]), np.array(fast.history["frobenius"])
    n = min(len(fr_ref), len(fr_fast))
    print(f"time original = {t_ref:.1f} s, fast = {t_fast:.1f} s  (x{t_ref / max(t_fast, 1e-9):.0f})")
    print(f"iterations original/fast = {len(fr_ref)}/{len(fr_fast)}, best_iter = {ref.best_iter}/{fast.best_iter}")
    print(f"max |frob_ref - frob_fast| over common iterations = {np.max(np.abs(fr_ref[:n] - fr_fast[:n])):.3e}")
    print(f"||S_ref - S_fast||_F / ||S_ref||_F = {np.linalg.norm(ref.S - fast.S) / np.linalg.norm(ref.S):.3e}")

    x_feat, a, x1, r, p, p1 = ds.arrays(cfg)
    a_ref = np.array([ref.predict(x_feat[i], p[i])[0] for i in range(len(ds))])
    fast_on_ref_S = F.FastOPI(copy.deepcopy(cfg))
    fast_on_ref_S.__dict__.update({k: v for k, v in ref.__dict__.items() if k != "cfg"})
    a_fast, _ = fast_on_ref_S.predict_batch(x_feat, p)
    print(f"greedy actions, same S: max |a_ref - a_fast| = {np.max(np.abs(a_ref - a_fast)):.3e}")

    ev = OPIEvaluator(ref)
    sat_ref = ev.action_distribution(ds).pct_any_action_saturated
    stab_ref = ev.stability_test(ds, seed=0)
    stab_fast = F.stability(fast_on_ref_S, ds, seed=0)
    rmse_ref = ev.bellman_error(ds).rmse
    print(f"sat_any  original = {sat_ref:.3f} %   fast = {F.saturation(fast_on_ref_S, ds)['sat_any_pct']:.3f} %")
    print(f"sens p95 original = {stab_ref.p95:.4f}     fast = {stab_fast['sens_p95']:.4f}")
    print(f"sens p95 (half-range) original = {stab_ref.p95_norm:.4f}     fast = {stab_fast['sens_norm_p95']:.4f}")
    print(f"bellman  original = {rmse_ref:.5f}    fast = {F.bellman_rmse(fast_on_ref_S, ds):.5f}")


if __name__ == "__main__":
    main()
