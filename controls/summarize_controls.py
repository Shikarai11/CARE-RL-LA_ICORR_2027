"""
Summarise the permutation test and the lambda_h sweeps
======================================================

Permutation test, one-sided p-values with the usual +1 correction:
    p = (1 + #{permuted at least as good as observed}) / (1 + n_permuted)
"At least as good" means lower for the overfitting ratio, the validation RMSE,
the saturation and the sensitivity, and higher for the training RMSE (the
direction that lowers the ratio).

Example
-------
    python controls/summarize_controls.py --perm results/controls/permutation.jsonl \
        --sweep results/controls/sweep_ra1.jsonl results/controls/sweep_ra5.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"cv_overfit_ratio": "low", "cv_val_rmse": "low", "cv_train_rmse": "high", "sat_any_pct": "low",
           "sat_tau_lim_pct": "low", "sat_d_plant_pct": "low", "sens_norm_p95": "low"}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def _protocol(df):
    return f"cv_purge = {df['cv_purge'].iloc[0]}, rating z-score = {df['cv_rating_stats'].iloc[0]}"


def summarize_permutation(path: Path):
    df = _read(path)
    obs = df[df.replicate == 0]
    null = df[df.replicate > 0]
    if obs.empty or null.empty:
        print("Permutation file needs replicate 0 (real ratings) and at least one permuted replicate.")
        return
    print(f"\n=== Permutation test: {len(null)} permutations ({_protocol(df)}) ===")
    print(f"{'metric':<18}{'observed':>10}{'null mean':>11}{'null 2.5%':>11}{'null 97.5%':>12}{'p (one-sided)':>15}")
    for m, side in METRICS.items():
        o = float(obs[m].iloc[0])
        v = null[m].to_numpy(float)
        p = (1 + np.sum(v <= o if side == "low" else v >= o)) / (1 + len(v))
        print(f"{m.replace('d_plant', 'd_dors'):<18}{o:>10.3f}{v.mean():>11.3f}{np.percentile(v, 2.5):>11.3f}{np.percentile(v, 97.5):>12.3f}{p:>15.3f}")
    rank = 1 + int(np.sum(null["cv_overfit_ratio"].to_numpy(float) < float(obs["cv_overfit_ratio"].iloc[0])))
    print(f"rank of the real ratings among the {len(df)} overfitting ratios: {rank}")


def summarize_sweep(path: Path):
    df = _read(path).sort_values("lambda_human")
    print(f"\n=== lambda_h sweep (R_a scale on tau_lim = {df['ra_scale_tau_lim'].iloc[0]}, "
          f"action cost = {df['action_cost'].iloc[0]}, {_protocol(df)}) ===")
    print(f"{'metric':<18}" + "".join(f"{'lh=' + format(l, 'g'):>10}" for l in df.lambda_human))
    for m in METRICS:
        print(f"{m.replace('d_plant', 'd_dors'):<18}" + "".join(f"{v:>10.3f}" for v in df[m].to_numpy(float)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perm", type=Path, nargs="*", default=[])
    ap.add_argument("--sweep", type=Path, nargs="*", default=[])
    args = ap.parse_args()
    for path in args.perm:
        summarize_permutation(path)
    for path in args.sweep:
        summarize_sweep(path)


if __name__ == "__main__":
    main()
