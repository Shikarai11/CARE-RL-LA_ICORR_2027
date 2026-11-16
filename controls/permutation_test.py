"""
Permutation test of the human feedback
======================================

Question
--------
Does the hybrid reward (lambda_h > 0) lower the overfitting ratio because the
comfort/assistance ratings carry information about the tested configurations,
or simply because adding a second, noisy term to the reward regularises the fit?

Procedure
---------
Replicate 0 uses the real ratings. Replicate j >= 1 shuffles the 52
(comfort, assistance) rating PAIRS across the 52 real block pairs
(seed = base_seed + j), recomputes the rewards, refits OPI and recomputes every
diagnostic with the same folds, hyperparameters and normalisations. Shuffling
whole pairs keeps the 1,600 pseudo-transitions of a block pair on the same
rating and keeps the distribution of the ratings unchanged, so only the link
"configuration -> rating" is broken.

The CV options are those of run.py: purge of the pairs that share a raw block
across the split (--cv-purge) and z-score of the ratings on the training pairs
of each fold (--cv-rating-stats fold). Each replicate is written as one JSON
line, so the job can be split (--start/--stop) and resumed.

Example (from the repository root)
------------------------------
    python controls/permutation_test.py \
        --pickle data/all_to_all_opi_dataset.pkl \
        --csv data/all_to_all_transitions.csv \
        --score-csv data/ExpS1_Full_scores.csv \
        --start 0 --stop 110 --out results/controls/permutation.jsonl
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np

import controls_common as C
import fast_opi as F


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--out", default="results/controls/permutation.jsonl")
    ap.add_argument("--start", type=int, default=0, help="first replicate (0 = real ratings)")
    ap.add_argument("--stop", type=int, default=110, help="last replicate + 1")
    ap.add_argument("--base-seed", type=int, default=20261007)
    ap.add_argument("--lambda-human", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lambda-reg", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--ra-scale-tau-lim", type=float, default=1.0)
    ap.add_argument("--assist-coding", choices=["centered", "raw"], default="centered")
    ap.add_argument("--action-cost", choices=["mixed", "separate"], default="mixed")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-purge", choices=["validation", "training", "none"], default="validation")
    ap.add_argument("--cv-rating-stats", choices=["fold", "global"], default="fold")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    warnings.simplefilter("ignore")

    out = Path(args.out)
    inp = C.load_inputs(args.pickle, args.csv, args.score_csv)
    cfg = C.make_cfg(lambda_human=args.lambda_human, alpha=args.alpha, gamma=args.gamma, lambda_reg=args.lambda_reg,
                     ra_scale=(1.0, 1.0, args.ra_scale_tau_lim), assist_coding=args.assist_coding,
                     action_cost=args.action_cost)
    already = C.done_ids(out, "replicate")
    n_pairs = len(inp.pair_order)

    for j in range(args.start, args.stop):
        if j in already:
            continue
        t0 = time.perf_counter()
        perm = np.arange(n_pairs) if j == 0 else np.random.default_rng(args.base_seed + j).permutation(n_pairs)
        comfort, assist = inp.comfort_pair[perm], inp.assist_pair[perm]
        ds = C.build_dataset(inp, C.rewards_for(inp, cfg, comfort, assist))
        fold_fn = C.fold_reward_fn(inp, cfg, comfort, assist) if args.cv_rating_stats == "fold" else None
        metrics = F.evaluate_all(cfg, ds, inp.pair_ids, k_folds=args.cv_folds, pair_blocks=inp.pair_blocks,
                                 purge=args.cv_purge, fold_reward_fn=fold_fn)
        rec = {"replicate": j, "real_ratings": j == 0, "lambda_human": args.lambda_human,
               "ra_scale_tau_lim": args.ra_scale_tau_lim, "assist_coding": args.assist_coding,
               "action_cost": args.action_cost, "cv_purge": args.cv_purge, "cv_rating_stats": args.cv_rating_stats,
               "seconds": round(time.perf_counter() - t0, 1), **metrics}
        C.append_jsonl(out, rec)
        print(f"[perm {j}] overfit={metrics['cv_overfit_ratio']:.3f} sat={metrics['sat_any_pct']:.1f}% "
              f"p95_norm={metrics['sens_norm_p95']:.2f} ({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
