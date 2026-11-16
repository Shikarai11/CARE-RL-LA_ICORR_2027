"""
lambda_h sweep (Table II and Fig. 3b-c of the paper)
==================================================

One full evaluation (fit on the 52 transitions + diagnostics + 5-fold CV) per
lambda_h value, with the real ratings, for a given action-cost configuration.
The CV options are those of run.py. Each value is written as one JSON line
(with the learned S matrix), and values already in the output file are skipped.

Examples (from the repository root)
-------------------------------
    # R_a = I, reward (4)
    python controls/sweep_lambda.py --pickle data/all_to_all_opi_dataset.pkl \
        --csv data/all_to_all_transitions.csv --score-csv data/ExpS1_Full_scores.csv \
        --out results/controls/sweep_ra1.jsonl
    # R_a = diag(1, 1, 5)
    ... --ra-scale-tau-lim 5 --out results/controls/sweep_ra5.jsonl
    # rating-only reward with the action cost kept, (7)
    ... --lambdas 1 --action-cost separate --out results/controls/sweep_ra1_separate.jsonl
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import controls_common as C
import fast_opi as F


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--out", default="results/controls/sweep_ra1.jsonl")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--ra-scale-tau-lim", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lambda-reg", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.9)
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
    already = C.done_ids(out, "lambda_human")

    for lam in args.lambdas:
        if lam in already:
            continue
        t0 = time.perf_counter()
        cfg = C.make_cfg(lambda_human=lam, alpha=args.alpha, gamma=args.gamma, lambda_reg=args.lambda_reg,
                         ra_scale=(1.0, 1.0, args.ra_scale_tau_lim), assist_coding=args.assist_coding,
                         action_cost=args.action_cost)
        ds = C.build_dataset(inp, C.rewards_for(inp, cfg))
        fold_fn = C.fold_reward_fn(inp, cfg) if args.cv_rating_stats == "fold" else None
        metrics = F.evaluate_all(cfg, ds, inp.pair_ids, k_folds=args.cv_folds, pair_blocks=inp.pair_blocks,
                                 purge=args.cv_purge, fold_reward_fn=fold_fn)
        rec = {"lambda_human": lam, "ra_scale_tau_lim": args.ra_scale_tau_lim, "assist_coding": args.assist_coding,
               "action_cost": args.action_cost, "cv_purge": args.cv_purge, "cv_rating_stats": args.cv_rating_stats,
               "seconds": round(time.perf_counter() - t0, 1), **metrics}
        C.append_jsonl(out, rec)
        print(f"[sweep {args.action_cost} lambda_h={lam}] overfit={metrics['cv_overfit_ratio']:.3f} "
              f"sat={metrics['sat_any_pct']:.1f}% p95_norm={metrics['sens_norm_p95']:.3f} ({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
