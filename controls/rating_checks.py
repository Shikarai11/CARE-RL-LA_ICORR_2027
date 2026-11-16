"""
Checks on the ratings themselves (Section VI of the paper)
====================================================================

  1) trend over the session: Spearman correlation between each rating and the
     block index, over all recorded blocks (the blocks used for the z-score);
  2) correlation between consecutive blocks (lag 1), with a one-sided p-value
     from random reorderings of the blocks: shuffling breaks both 1) and 2),
     so neither can explain a difference between real and shuffled ratings;
  3) Spearman correlation between the ratings of each block pair (block k+1)
     and the three parameters of that block, over the 52 pairs.
The assistance score is coded as in the paper, -|A - 5|.

Example (from the repository root)
------------------------------
    python controls/rating_checks.py --csv data/all_to_all_transitions.csv \
        --score-csv data/ExpS1_Full_scores.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def lag1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--assist-optimum", type=float, default=5.0)
    ap.add_argument("--n-shuffles", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scores = pd.read_csv(args.score_csv).sort_values("condition number")
    blocks = scores["condition number"].to_numpy()
    ratings = {"comfort": scores["score COMF"].to_numpy(float),
               "assistance": -np.abs(scores["score ASSIST"].to_numpy(float) - args.assist_optimum)}
    rng = np.random.default_rng(args.seed)

    print(f"1-2) {len(blocks)} recorded blocks")
    for name, x in ratings.items():
        rho, p = spearmanr(blocks, x)
        r1 = lag1(x)
        null = np.array([lag1(rng.permutation(x)) for _ in range(args.n_shuffles)])
        p1 = (1 + np.sum(null >= r1)) / (1 + args.n_shuffles)
        print(f"  {name:10s} trend rho = {rho:+.2f} (p = {p:.2f})   lag-1 r = {r1:+.2f} (one-sided p = {p1:.3f})")

    df = pd.read_csv(args.csv)
    pairs = df.groupby("pair_id", sort=False).first()
    param_cols = [c for c in pairs.columns if c.startswith("p_k1__")]
    pair_ratings = {"comfort": pairs["comfort_used"].to_numpy(float),
                    "assistance": -np.abs(pairs["assistance_used"].to_numpy(float) - args.assist_optimum)}
    print(f"\n3) {len(pairs)} block pairs, rating of block k+1 against its parameters")
    for name, x in pair_ratings.items():
        cells = [f"{c.replace('p_k1__', '').replace('d_plant', 'd_dors')} rho = {spearmanr(pairs[c], x)[0]:+.2f} (p = {spearmanr(pairs[c], x)[1]:.2f})"
                 for c in param_cols]
        print(f"  {name:10s} " + "   ".join(cells))


if __name__ == "__main__":
    main()
