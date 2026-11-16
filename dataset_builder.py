"""
Dataset construction utilities for the ELSA Offline Policy Iteration (OPI) pipeline.

- The builder returns both:
    (a) a pandas DataFrame with rich metadata (pair ids, step ids, weights, etc.),
    (b) an OPI_core.Dataset object that can be passed directly to OfflinePolicyIteration.fit().
- Sample weights are computed and exported even if your current OPI implementation does
  not yet consume them. This keeps the construction ready for future weighted fitting.
- Pair identifiers are exported for future grouped train/test splitting, which is crucial
  when pseudo-transitions are created from the same real block pair.
"""


from __future__ import annotations

import argparse
import json
import logging
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from OPI_core import Dataset, Transition
from reward import RewardComputer
from elsa_configOPI import OPIConfig

logger = logging.getLogger(__name__)

# ============================================================================
# |                                                                          |
# |  1) FEATURE REGISTRY                                                     |
# |                                                                          |
# ============================================================================
@dataclass(frozen=True)
class FeatureSpec:
    """
    One feature that can be included in the state and the reward.

    exp_col:
        Column name in the experimental stride-level CSV.
    normative_name:
        Feature name used in the normative reference CSV.
    aliases:
        User-friendly names accepted by the builder. This makes the API easier to use
        when you want to test several feature combinations quickly.
    """
    exp_col: str
    normative_name: str
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


FEATURE_SPECS: Tuple[FeatureSpec, ...] = (
    FeatureSpec(
        exp_col="f1_drop_deg_signed",
        normative_name="f1A_drop_deg_signed",
        aliases=("f1", "f1_signed", "drop_signed"),
        description="Signed plantarflexion drop after heel strike.",
    ),
    FeatureSpec(
        exp_col="f1_hs_to_mpa_time",
        normative_name="f1B_hs_to_mpa_sec",
        aliases=("f1_time", "f1b"),
        description="Heel-strike to maximal plantarflexion time.",
    ),
    FeatureSpec(
        exp_col="f1_max_abs_speed_hs_to_mpa",
        normative_name="f1C_max_pf_speed_deg_s",
        aliases=("f1_speed", "f1c"),
        description="Maximum absolute angular speed from heel strike to MPA.",
    ),
    FeatureSpec(
        exp_col="f2_mpa_to_pushoff_time",
        normative_name="f2_mpa_to_po_start_sec",
        aliases=("f2",),
        description="Time from MPA to push-off start.",
    ),
    FeatureSpec(
        exp_col="f3_max_abs_speed_pushoff",
        normative_name="f3_push_off_max_ang_vel_abs_deg_s",
        aliases=("f3",),
        description="Maximum absolute angular speed during push-off.",
    ),
    FeatureSpec(
        exp_col="f4_pushoff_duration",
        normative_name="f4_push_off_duration_sec",
        aliases=("f4",),
        description="Push-off duration.",
    ),
    FeatureSpec(
        exp_col="f5_peak_torque_pushoff_norm",
        normative_name="f5_peak_pf_torque_norm_nm_kg",
        aliases=("f5", "f5_norm"),
        description="Peak push-off torque, normalised by mass.",
    ),
    FeatureSpec(
        exp_col="f6_generated_mechanical_energy_norm",
        normative_name="f6_positive_push_off_energy_norm_j_kg",
        aliases=("f6", "f6_norm"),
        description="Generated push-off mechanical energy, normalised by mass.",
    ),
)

# I put raw f5/f6 columns in the experimental CSV, but the current normative table is compatible with the normalised versions. 
# I therefore reject raw versions to avoid an inconsistent reward construction.
UNSUPPORTED_EXACT_FEATURES: Dict[str, str] = {
    "f5_peak_torque_pushoff": (
        "Use 'f5_peak_torque_pushoff_norm' (or alias 'f5') instead. "
        "The current normative reward references are in normalised torque units."
    ),
    "f6_generated_mechanical_energy": (
        "Use 'f6_generated_mechanical_energy_norm' (or alias 'f6') instead. "
        "The current normative reward references are in normalised energy units."
    ),
}

_FEATURE_BY_EXP_COL = {spec.exp_col: spec for spec in FEATURE_SPECS}
_FEATURE_ALIAS_TO_SPEC = {}
for spec in FEATURE_SPECS:
    _FEATURE_ALIAS_TO_SPEC[spec.exp_col] = spec
    _FEATURE_ALIAS_TO_SPEC[spec.normative_name] = spec
    for alias in spec.aliases:
        _FEATURE_ALIAS_TO_SPEC[alias] = spec
        
        
def resolve_feature_list(features: Sequence[str]):
    """
    Resolve user-provided feature names / aliases into FeatureSpec objects.

    The returned order matches the user-provided order, because feature ordering matters for the OPI state vectors and for the feature weights in OPIConfig.
    """
    resolved: List[FeatureSpec] = []
    seen_exp_cols = set()

    for name in features:
        if name in UNSUPPORTED_EXACT_FEATURES:
            raise ValueError(f"Feature '{name}' is not supported here. {UNSUPPORTED_EXACT_FEATURES[name]}")

        spec = _FEATURE_ALIAS_TO_SPEC.get(name)
        if spec is None:
            valid = sorted(set(_FEATURE_ALIAS_TO_SPEC.keys()) | set(UNSUPPORTED_EXACT_FEATURES.keys()))
            raise ValueError(
                f"Unknown feature '{name}'. Valid feature names / aliases include: {valid}"
            )

        if spec.exp_col not in seen_exp_cols:
            resolved.append(spec)
            seen_exp_cols.add(spec.exp_col)

    if not resolved:
        raise ValueError("No feature selected.")

    return resolved


# ============================================================================
# |                                                                          |
# |  2)  BUILD CONFIGURATION                                                 |
# |                                                                          |
# ============================================================================

VALID_DATASET_KINDS = ("block_summary", "all_to_all")

@dataclass
class DatasetBuildConfig:
    """
    Configuration of the dataset construction.

    features:
        List of experimental feature column names or aliases.
        Example: ["f1", "f2", "f3"] or
                 ["f1_drop_deg_signed", "f2_mpa_to_pushoff_time", "f6"]

    dataset_kinds:
        One or more among: "block_summary", "all_to_all".
        Use "all" as a shorthand for both.


    block_agg:
        Aggregation used when a block must be summarised into one feature vector (only used by "block_summary"). 
        Default = "median" for robustness to outliers.

    normative_reference_choice:
        How to choose the normative reference value.
        Default = "median" because the normative dataset processing already motivated a robust approach and because medians are less sensitive to outlying participants.

    normative_sigma_choice:
        How to choose the normative dispersion for reward standardisation.
        Default = "robust_iqr" because it is the robust counterpart of the median.
        
    sigma_source:
        How to compute the per-feature scale used in the reward standardisation
            x̃ᵢ = (fᵢ - fᵢ^{ref}) / σᵢ
        Two options:
            - "normative" : use the normative table σ from Moreira et al. 2021 (inter-subject variability in a healthy population). 
                            Preserves the clinical interpretation: "x̃=1 means one healthy population std away from the norm".
            - "empirical" : compute σᵢ as the standard deviation of feature i over ALL stride-level rows of the current subject's feature_csv.
                            Guarantees <x̃²> ~ O(1) after standardisation, avoiding single-feature dominance in the reward.
 
    normative_scope:
        "global" or "by_speed".
        Default = "global" because the current experimental files do not contain an obvious speed label for each condition.
 
    normative_speed_kmh:
        Required only when normative_scope = "by_speed".
 
    reward_from:
        Which block should be used to compute the prosthesis and human reward term for the transition K -> K+1.
        - "next" : use block K+1  (default, most causally consistent)
        - "current" : use block K
 
    parse_param_order:
        Names associated with the tuple values parsed from the C file.
 
    output_param_order:
        Order of the parameter vector p stored in the OPI dataset.
        This order must match the order expected later by your OPI config and action logic.
 
    strict_param_bounds:
        If True, raises an error when the experimental parameters fall outside the bounds stored in OPIConfig. 
        If False, only warns.
    """

    features: Sequence[str]
    dataset_kinds: Sequence[str] = ("block_summary", "all_to_all")
    block_agg: str = "median"

    normative_reference_choice: str = "median" # {"median", "mean"}
    normative_sigma_choice: str = "robust_iqr" # {"robust_iqr", "std"}
    normative_scope: str = "global" # {"global", "by_speed"}
    normative_speed_kmh: Optional[float] = None
    
    sigma_source: str = "normative" # {"normative", "empirical"}

    reward_from: str = "next" # {"next", "current"}

    n_bins: int = 3

    parse_param_order: Tuple[str, ...] = ("tau_lim", "theta", "d_plant")
    output_param_order: Tuple[str, ...] = ("d_plant", "theta", "tau_lim")

    strict_param_bounds: bool = False

    def __post_init__(self):
        invalid = [k for k in self.dataset_kinds if k not in VALID_DATASET_KINDS and k != "all"]
        if invalid:
            raise ValueError(f"Unknown dataset kind(s): {invalid}. Valid: {VALID_DATASET_KINDS} or 'all'.")

        if self.block_agg not in {"median", "mean"}:
            raise ValueError("block_agg must be 'median' or 'mean'.")

        if self.normative_reference_choice not in {"median", "mean"}:
            raise ValueError("normative_reference_choice must be 'median' or 'mean'.")

        if self.normative_sigma_choice not in {"robust_iqr", "std"}:
            raise ValueError("normative_sigma_choice must be 'robust_iqr' or 'std'.")

        if self.sigma_source not in {"normative", "empirical"}:
            raise ValueError("sigma_source must be 'normative' or 'empirical'.")
        
        if self.normative_scope not in {"global", "by_speed"}:
            raise ValueError("normative_scope must be 'global' or 'by_speed'.")

        if self.normative_scope == "by_speed" and self.normative_speed_kmh is None:
            raise ValueError("normative_speed_kmh must be provided when normative_scope='by_speed'.")

        if self.reward_from not in {"next", "current"}:
            raise ValueError("reward_from must be 'next' or 'current'.")

        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1.")

        if sorted(self.parse_param_order) != sorted(self.output_param_order):
            raise ValueError(
                "parse_param_order and output_param_order must contain the same parameter names."
            )

    @property
    def expanded_dataset_kinds(self):
        if "all" in self.dataset_kinds:
            return VALID_DATASET_KINDS
        return tuple(self.dataset_kinds)
    
    
@dataclass
class BuiltDataset:
    """
    Final object returned for one dataset definition.

    dataset:
        OPI_core.Dataset object ready for OfflinePolicyIteration.fit().

    dataframe:
        Rich tabular view of the same transitions, with metadata useful for analysis,
        grouped splitting, and later debugging.

    metadata:
        JSON-serialisable dictionary describing the construction choices.
    """
    name: str
    dataset: Dataset
    dataframe: pd.DataFrame
    metadata: Dict[str, object]
    
    
# ============================================================================
# |                                                                          |
# |  3) LOW-LVL LOADING UTILITIES                                            |
# |                                                                          |
# ============================================================================
def parse_c_experimental_params(c_file: str | Path, *, n_rows: Optional[int] = None, parse_param_order: Sequence[str] = ("tau_lim", "theta", "d_plant")):
    """
    Parse the experimental parameter tuples from the provided C file.

    Expected tuple syntax inside the file:
        {11.0f, 0.113f, 20.708f},

    The first n_rows tuples are retained when n_rows is provided.
    """
    c_file = Path(c_file)
    text = c_file.read_text(encoding="utf-8", errors="ignore")

    tuples = re.findall(r"\{([^{}]+)\}", text)
    values: List[List[float]] = []
    for tup in tuples:
        parts = [p.strip() for p in tup.split(",")]
        if len(parts) != len(parse_param_order):
            continue

        row = []
        for p in parts:
            p = p.replace("f", "")
            row.append(float(p))
        values.append(row)

    if n_rows is not None:
        values = values[:n_rows]

    if not values:
        raise ValueError(f"No experimental parameter tuples could be parsed from '{c_file}'.")

    df = pd.DataFrame(values, columns=list(parse_param_order))
    return df


def load_scores_table(score_csv: str | Path):
    """
    Load and standardise the condition-level human scores table.
    """
    df = pd.read_csv(score_csv).copy()
    rename_map = {
        "condition number": "condition_id",
        "score COMF": "comfort",
        "score ASSIST": "assistance",
    }
    df = df.rename(columns=rename_map)

    required = {"condition_id", "comfort", "assistance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Score CSV is missing columns: {sorted(missing)}")

    df["condition_id"] = df["condition_id"].astype(int)
    return df


def load_feature_table(feature_csv: str | Path):
    """
    Load the stride-level experimental feature table and perform basic checks.
    """
    df = pd.read_csv(feature_csv).copy()

    required = {"subject_id", "condition_id", "cycle_local_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {sorted(missing)}")

    df["condition_id"] = df["condition_id"].astype(int)
    df["cycle_local_idx"] = df["cycle_local_idx"].astype(int)

    return df


# ============================================================================
# |                                                                          |
# |  4)  NORMATIVE REFERENCES                                                |
# |                                                                          |
# ============================================================================
class NormativeReferenceTable:
    """
    Helper around the normative reward reference CSV.

    It provides a one-stop method:
        get_reference(feature_name) -> (ref_value, sigma_value)

    according to:
        - selected scope,
        - selected speed if needed,
        - selected robust or non-robust statistics.
    """

    def __init__(self, ref_csv: str | Path, *, reference_choice: str = "median", sigma_choice: str = "robust_iqr", scope: str = "global", speed_kmh: Optional[float] = None):
        self.df = pd.read_csv(ref_csv).copy()
        self.reference_choice = reference_choice
        self.sigma_choice = sigma_choice
        self.scope = scope
        self.speed_kmh = speed_kmh

    def _ref_col(self):
        return "ref_median" if self.reference_choice == "median" else "ref_mean"

    def _sigma_col(self):
        return "robust_sigma_from_iqr" if self.sigma_choice == "robust_iqr" else "ref_std"

    def get_reference(self, normative_name: str):
        df = self.df[self.df["feature_name"] == normative_name].copy()
        if df.empty:
            raise KeyError(f"No normative reference found for feature '{normative_name}'.")

        if self.scope == "global":
            df = df[df["scope"] == "global"]
        else:
            df = df[(df["scope"] == "by_speed") & (np.isclose(df["speed_kmh"], float(self.speed_kmh)))]

        if df.empty:
            raise KeyError(
                f"No normative row found for feature '{normative_name}' with scope='{self.scope}'"
                + (f" and speed_kmh={self.speed_kmh}" if self.scope == "by_speed" else "")
            )

        # There should be exactly one row now. If not, keep the first and warn.
        if len(df) > 1:
            warnings.warn(
                f"Multiple normative rows found for feature '{normative_name}'. Keeping the first."
            )
        row = df.iloc[0]

        ref_value = float(row[self._ref_col()])
        sigma_value = float(row[self._sigma_col()])
        sigma_value = max(sigma_value, 1e-8)
        return ref_value, sigma_value

    def build_reference_vectors(self, features: Sequence[FeatureSpec]):
        refs = []
        sigmas = []
        for spec in features:
            r, s = self.get_reference(spec.normative_name)
            refs.append(r)
            sigmas.append(s)
        return np.asarray(refs, dtype=float), np.asarray(sigmas, dtype=float)
    
    
# ============================================================================
# |                                                                          |
# |  5)  INTERNAL HELPERS                                                    |
# |                                                                          |
# ============================================================================

def _aggregate_feature_block(block_df: pd.DataFrame, feature_cols: Sequence[str], agg: str):
    """
    Summarise a block or a bin into one feature vector.

    Default = median because it is more robust to atypical strides.
    """
    if agg == "median":
        return block_df.loc[:, feature_cols].median(axis=0).to_numpy(dtype=float)
    if agg == "mean":
        return block_df.loc[:, feature_cols].mean(axis=0).to_numpy(dtype=float)
    raise ValueError(f"Unsupported aggregator '{agg}'.")


def _sort_block(block_df: pd.DataFrame):
    """
    Sort strides inside a block in a deterministic order.
    """
    sort_cols = [c for c in ("cycle_local_idx", "step_number") if c in block_df.columns]
    return block_df.sort_values(sort_cols).reset_index(drop=True)


def _cartesian_pairs(n_a: int, n_b: int):
    for i in range(n_a):
        for j in range(n_b):
            yield i, j


def _condition_order_from_scores(scores_df: pd.DataFrame, available_conditions: Iterable[int]):
    """
    Determine the experimental block order.

    We use the order in the scores file because:
    - it is the explicit condition-level table entered during the experiment,
    - it aligns naturally with the parameter list if both were recorded sequentially.
    """
    order = scores_df["condition_id"].astype(int).tolist()
    missing = sorted(set(order) - set(map(int, available_conditions)))
    if missing:
        logger.warning(
            "Some conditions are present in the score table but missing from "
            "the feature table: %s. They will be kept in condition_order to "
            "preserve the 1-to-1 alignment with the C-parameter file, but any "
            "transition pair involving them will be skipped at build time.",
            missing,
        )
    return order


def _output_param_matrix(parsed_params_df: pd.DataFrame, *, output_param_order: Sequence[str]):
    """
    Reorder parsed parameter columns into the final vector order used in the dataset.
    """
    missing = set(output_param_order) - set(parsed_params_df.columns)
    if missing:
        raise ValueError(f"Parsed parameter table is missing columns required by output_param_order: {sorted(missing)}")
    return parsed_params_df.loc[:, list(output_param_order)].copy()


def _validate_params_against_cfg(params_df: pd.DataFrame, *, output_param_order: Sequence[str], opi_cfg: OPIConfig, strict: bool = False):
    """
    Validate the parameter matrix against the bounds stored in OPIConfig.

    Because your current experimental C file seems to contain a "theta" parameter around 10-28,
    while the current config documents theta bounds [3,11], this check is deliberately explicit.
    """
    expected_order = ("d_plant", "theta", "tau_lim")
    if tuple(output_param_order) != expected_order:
        warnings.warn(
            "Parameter bound validation was skipped because output_param_order != "
            f"{expected_order}. Current order is {tuple(output_param_order)}."
        )
        return

    # Build a bound matrix from the current config naming convention.
    lower = np.array([opi_cfg.prosthesis.d_plant_bounds[0], opi_cfg.prosthesis.theta_trig_bounds[0], opi_cfg.prosthesis.tau_lim_bounds[0]], dtype=float)
    upper = np.array([opi_cfg.prosthesis.d_plant_bounds[1], opi_cfg.prosthesis.theta_trig_bounds[1], opi_cfg.prosthesis.tau_lim_bounds[1]], dtype=float)

    P = params_df.loc[:, list(output_param_order)].to_numpy(dtype=float)
    bad_mask = (P < lower) | (P > upper)
    if bad_mask.any():
        rows, cols = np.where(bad_mask)
        examples = []
        for r, c in zip(rows[:5], cols[:5]):
            examples.append(
                {
                    "row": int(r),
                    "parameter": output_param_order[c],
                    "value": float(P[r, c]),
                    "bounds": [float(lower[c]), float(upper[c])],
                }
            )
        msg = (
            "Some experimental parameters fall outside the bounds stored in your current OPIConfig. "
            "This is not necessarily a dataset-builder bug; it likely means the config bounds "
            "do not yet match the parameter actually tuned experimentally. "
            f"Examples: {examples}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
        
        
# ============================================================================
# |                                                                          |
# |  6) MAIN BUILDER                                                         |
# |                                                                          |
# ============================================================================
 
class ELSADatasetBuilder:
    """
    Main dataset builder class.

    Builds two dataset variants from a set of consecutive experimental conditions:
        - "block_summary" : one transition per condition pair, features aggregated by median (or mean) over the block's strides.
        - "all_to_all" : full stride-level cross-product per pair, yielding O(N_k x N_{k+1}) pseudo-transitions; the dominant variant used for OPI training.

    Typical usage
    -------------
    >>> opi_cfg = OPIConfig(n_features=6)
    >>> build_cfg = DatasetBuildConfig(features=["f1", "f2", "f3", "f4", "f5", "f6"],
    ...                                dataset_kinds=["all_to_all"])
    >>> builder = ELSADatasetBuilder(
    ...     feature_csv="ExpS1_final_features.csv",
    ...     score_csv="ExpS1_scores.csv",
    ...     param_c_file="S1_exp_params.c",
    ...     normative_ref_csv="reward_ready_normative_feature_refs.csv",
    ...     opi_cfg=opi_cfg,
    ...     build_cfg=build_cfg,
    ... )
    >>> built = builder.build_all()
    >>> dataset = built["all_to_all"].dataset
    >>> df      = built["all_to_all"].dataframe
    """

    def __init__(self, *, feature_csv: str | Path, score_csv: str | Path, param_c_file: str | Path, normative_ref_csv: str | Path, opi_cfg: OPIConfig, build_cfg: DatasetBuildConfig):
        self.feature_csv = Path(feature_csv)
        self.score_csv = Path(score_csv)
        self.param_c_file = Path(param_c_file)
        self.normative_ref_csv = Path(normative_ref_csv)
        self.opi_cfg = opi_cfg
        self.build_cfg = build_cfg

        self.feature_specs = resolve_feature_list(build_cfg.features)
        self.feature_cols = [spec.exp_col for spec in self.feature_specs]

        # Keep OPIConfig consistent with the selected feature subset.
        if self.opi_cfg.n_features != len(self.feature_specs):
            raise ValueError(
                f"OPIConfig.n_features={self.opi_cfg.n_features}, but the selected feature subset "
                f"contains {len(self.feature_specs)} features. Please update OPIConfig accordingly."
            )

        # Load the base tables.
        self.features_df = load_feature_table(self.feature_csv)
        self.scores_df = load_scores_table(self.score_csv)

        self.condition_order = _condition_order_from_scores(
            self.scores_df,
            available_conditions=self.features_df["condition_id"].unique(),
        )

        self.parsed_params_df = parse_c_experimental_params(
            self.param_c_file,
            n_rows=len(self.condition_order),
            parse_param_order=self.build_cfg.parse_param_order,
        )
        self.params_df = _output_param_matrix(
            self.parsed_params_df,
            output_param_order=self.build_cfg.output_param_order,
        )
        _validate_params_against_cfg(
            self.params_df,
            output_param_order=self.build_cfg.output_param_order,
            opi_cfg=self.opi_cfg,
            strict=self.build_cfg.strict_param_bounds,
        )

        # Map condition id -> parameter vector and human scores.
        cond_param_df = self.params_df.copy()
        cond_param_df["condition_id"] = self.condition_order
        self.condition_to_params = {
            int(row["condition_id"]): row.loc[list(self.build_cfg.output_param_order)].to_numpy(dtype=float)
            for _, row in cond_param_df.iterrows()
        }

        score_cols = ["condition_id", "comfort", "assistance"]
        self.scores_map = {
            int(row["condition_id"]): {
                "comfort": float(row["comfort"]),
                "assistance": float(row["assistance"]),
            }
            for _, row in self.scores_df.loc[:, score_cols].iterrows()
        }

        # Normative references for the selected subset.
        self.norm_table = NormativeReferenceTable(
            self.normative_ref_csv,
            reference_choice=self.build_cfg.normative_reference_choice,
            sigma_choice=self.build_cfg.normative_sigma_choice,
            scope=self.build_cfg.normative_scope,
            speed_kmh=self.build_cfg.normative_speed_kmh,
        )
        self.ref_vector, self.sigma_vector = self.norm_table.build_reference_vectors(self.feature_specs)
        
        
        # Sigma source selection
        self.sigma_source = self.build_cfg.sigma_source
        if self.sigma_source == "empirical":
            normative_sigma_snapshot = self.sigma_vector.copy()
            empirical = np.asarray(
                self.features_df.loc[:, self.feature_cols].std(axis=0, ddof=1).to_numpy(dtype=float)
            )
            empirical = np.maximum(empirical, 1e-8) # avoid zero std
            self.sigma_vector = empirical
            self.sigma_vector_normative = normative_sigma_snapshot
            self.sigma_vector_empirical = empirical.copy()
            logger.info(
                "sigma_source='empirical': replacing normative sigma by the "
                "stride-level std of the current subject (N=%d strides).",
                len(self.features_df),
            )
            for spec, sig_norm, sig_emp in zip(
                self.feature_specs, normative_sigma_snapshot, empirical
            ):
                ratio = sig_emp / max(sig_norm, 1e-12)
                logger.info(
                    "  %-40s  sigma_ref=%.4g  sigma_emp=%.4g   (x %.2f)",
                    spec.exp_col, sig_norm, sig_emp, ratio,
                )
        else:
            # Legacy path: keep the normative sigma. Mirror attributes for
            # symmetry with the "empirical" branch so downstream code can
            # always introspect both vectors.
            self.sigma_vector_normative = self.sigma_vector.copy()
            self.sigma_vector_empirical = None
            logger.info("sigma_source='normative': using the normative table sigma (Moreira et al. 2021).")
                

        # Reward helper.
        self.reward_computer = RewardComputer(self.opi_cfg)
        self.reward_computer.set_reference_stats(self.sigma_vector)
        self.reward_computer.fit_human_stats(
            self.scores_df["comfort"].to_numpy(dtype=float),
            self.scores_df["assistance"].to_numpy(dtype=float),
        )

        # One block dataframe per condition, sorted deterministically.
        self.blocks: Dict[int, pd.DataFrame] = {}
        for cond_id, block_df in self.features_df.groupby("condition_id"):
            block_sorted = _sort_block(block_df)
            missing_cols = set(self.feature_cols) - set(block_sorted.columns)
            if missing_cols:
                raise ValueError(
                    f"Condition {cond_id}: feature CSV is missing selected columns {sorted(missing_cols)}"
                )
            self.blocks[int(cond_id)] = block_sorted.reset_index(drop=True)

        self._real_pairs: List[Tuple[int, int, int]] = self._compute_real_pairs()


    # Public API
    # -------------

    def build_all(self):
        """
        Build every dataset requested in the config.
        """
        out: Dict[str, BuiltDataset] = {}
        for kind in self.build_cfg.expanded_dataset_kinds:
            out[kind] = self.build_one(kind)
        return out

    def build_one(self, dataset_kind: str):
        if dataset_kind not in VALID_DATASET_KINDS:
            raise ValueError(f"Unknown dataset_kind '{dataset_kind}'. Valid: {VALID_DATASET_KINDS}")

        if dataset_kind == "block_summary":
            df = self._build_block_summary_df()
        elif dataset_kind == "all_to_all":
            df = self._build_all_to_all_df()
        else:
            raise RuntimeError(f"Unsupported dataset_kind '{dataset_kind}'.")

        dataset = self._df_to_opi_dataset(df)
        metadata = self._build_metadata(dataset_kind, df)
        return BuiltDataset(name=dataset_kind, dataset=dataset, dataframe=df, metadata=metadata)

    def export(self, built: Mapping[str, BuiltDataset], output_dir: str | Path, *, write_csv: bool = True, write_json: bool = True, write_pickle: bool = True):
        """
        Export built datasets to disk.

        Files written per dataset kind:
            - <name>_transitions.csv
            - <name>_metadata.json
            - <name>_opi_dataset.pkl

        The CSV is the most useful for inspection and analysis.
        The pickle stores the OPI_core.Dataset object directly.
        """
        import pickle

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written: Dict[str, Dict[str, str]] = {}
        for name, obj in built.items():
            paths = {}

            if write_csv:
                csv_path = output_dir / f"{name}_transitions.csv"
                obj.dataframe.to_csv(csv_path, index=False)
                paths["csv"] = str(csv_path)

            if write_json:
                json_path = output_dir / f"{name}_metadata.json"
                json_path.write_text(json.dumps(obj.metadata, indent=2), encoding="utf-8")
                paths["json"] = str(json_path)

            if write_pickle:
                pkl_path = output_dir / f"{name}_opi_dataset.pkl"
                with pkl_path.open("wb") as f:
                    pickle.dump(obj.dataset, f)
                paths["pickle"] = str(pkl_path)

            written[name] = paths

        return written


    # Pair iteration
    # -----------------

    def _compute_real_pairs(self):
        """
        Build the definitive list of '(pair_index, cond_k, cond_k1)' tuples used to construct transitions.

        Filtering rule
        --------------
        A candidate pair '(cond_k, cond_k1)' is KEPT if and only if both conditions are present in 'self.blocks' (i.e. both have a feature block available). 
        Otherwise the pair is DROPPED.

        Why we drop and do not interpolate/skip-to-next
        -----------------------------------------------
        Transitions between non-adjacent conditions, e.g. '(K-1, K+1)' when 'K' is missing, are NOT valid single-step actions of the RL process:
        the parameter delta between 'K-1' and 'K+1' is the cumulative effect of two consecutive operator adjustments, not one atomic action.
        Including such pairs would silently corrupt the dataset by mislabeling a compound action as atomic, which in turn would bias every quadratic action-cost term and every Bellman update in OPI.

        Index semantics
        ---------------
        'pair_index' is kept as the positional index in 'condition_order'.
        Skipped indices are simply absent from the returned list (they do not shift). 
        This keeps the index semantically tied to the experimental protocol ordering and preserves traceability in metadata / pair ids.
        """
        pairs: List[Tuple[int, int, int]] = []
        skipped: List[Tuple[int, int, int, str]] = []
        for pair_index in range(len(self.condition_order) - 1):
            cond_k = int(self.condition_order[pair_index])
            cond_k1 = int(self.condition_order[pair_index + 1])
            k_ok = cond_k in self.blocks
            k1_ok = cond_k1 in self.blocks
            if k_ok and k1_ok:
                pairs.append((pair_index, cond_k, cond_k1))
            else:
                which = "both" if (not k_ok and not k1_ok) else ("K" if not k_ok else "K+1")
                skipped.append((pair_index, cond_k, cond_k1, which))

        if skipped:
            detail = ", ".join(
                f"pair_index={p} ({k}->{k1}, missing {which})"
                for p, k, k1, which in skipped
            )
            logger.warning(
                "Skipped %d/%d candidate transition pairs due to missing "
                "feature blocks: %s. The final dataset will contain %d real "
                "transition pairs.",
                len(skipped), len(skipped) + len(pairs), detail, len(pairs),
            )
        else:
            logger.info(
                "All %d candidate transition pairs are valid (no missing "
                "feature block).",
                len(pairs),
            )
        return pairs

    def _iter_real_pairs(self) -> Iterable[Tuple[int, int, int]]:
        """
        Yield ``(pair_index, condition_k, condition_k1)`` in experimental order.

        Iterates over the precomputed ``self._real_pairs`` list, which already
        excludes pairs whose endpoints are missing from the feature table.
        See :meth:`_compute_real_pairs` for the filtering logic.
        """
        return iter(self._real_pairs)

    # Dataset variants
    # ------------------

    def _build_block_summary_df(self):
        rows: List[Dict[str, object]] = []

        for pair_index, cond_k, cond_k1 in self._iter_real_pairs():
            block_k = self.blocks[cond_k]
            block_k1 = self.blocks[cond_k1]

            f_k = _aggregate_feature_block(block_k, self.feature_cols, agg=self.build_cfg.block_agg)
            f_k1 = _aggregate_feature_block(block_k1, self.feature_cols, agg=self.build_cfg.block_agg)

            rows.append(
                self._build_transition_row(
                    pair_index=pair_index,
                    cond_k=cond_k,
                    cond_k1=cond_k1,
                    f_current=f_k,
                    f_next=f_k1,
                    step_idx_k=None,
                    step_idx_k1=None,
                    local_idx_k=None,
                    local_idx_k1=None,
                    bin_id=None,
                    pair_kind="block_summary",
                    sample_weight=1.0, # one real transition per pair
                )
            )

        return pd.DataFrame(rows)

    def _build_all_to_all_df(self):
        rows: List[Dict[str, object]] = []

        for pair_index, cond_k, cond_k1 in self._iter_real_pairs():
            block_k = _sort_block(self.blocks[cond_k])
            block_k1 = _sort_block(self.blocks[cond_k1])

            m_pair = len(block_k) * len(block_k1)
            sample_weight = 1.0 / max(m_pair, 1)

            for i, j in _cartesian_pairs(len(block_k), len(block_k1)):
                row_k = block_k.iloc[i]
                row_k1 = block_k1.iloc[j]
                rows.append(
                    self._build_transition_row(
                        pair_index=pair_index,
                        cond_k=cond_k,
                        cond_k1=cond_k1,
                        f_current=row_k.loc[self.feature_cols].to_numpy(dtype=float),
                        f_next=row_k1.loc[self.feature_cols].to_numpy(dtype=float),
                        step_idx_k=int(row_k.get("step_number", i)),
                        step_idx_k1=int(row_k1.get("step_number", j)),
                        local_idx_k=int(row_k["cycle_local_idx"]),
                        local_idx_k1=int(row_k1["cycle_local_idx"]),
                        bin_id=None,
                        pair_kind="all_to_all",
                        sample_weight=sample_weight,
                    )
                )

        return pd.DataFrame(rows)


    # Transition construction
    # --------------------------

    def _build_transition_row(self, *, pair_index: int, cond_k: int, cond_k1: int, f_current: np.ndarray, f_next: np.ndarray, step_idx_k: Optional[int], step_idx_k1: Optional[int], local_idx_k: Optional[int], local_idx_k1: Optional[int], bin_id: Optional[int], pair_kind: str, sample_weight: float):
        """
        Build one transition row and compute its reward.

        Reward design choice
        --------------------
        By default, reward_from = "next", meaning:
            - the prosthesis reward uses the features of block K+1,
            - the human score uses the condition-level scores of block K+1.

        This is the most causally consistent choice for a transition K -> K+1, because the action applied after block K is meant to produce block K+1.
        """
        p_k = self.condition_to_params[int(cond_k)]
        p_k1 = self.condition_to_params[int(cond_k1)]
        a_k = p_k1 - p_k

        delta_current = f_current - self.ref_vector
        delta_next = f_next - self.ref_vector

        x_current = delta_current / self.sigma_vector
        x_next = delta_next / self.sigma_vector

        if self.build_cfg.reward_from == "next":
            delta_reward = delta_next # passed to RewardComputer (centred, not z-scored)
            score_cond = cond_k1
            reward_block_label = "next"
        else:
            delta_reward = delta_current
            score_cond = cond_k
            reward_block_label = "current"

        scores = self.scores_map[int(score_cond)]
        reward = self.reward_computer.compute(
            x_feat=delta_reward, # historical contract: centred, not z-scored
            a=a_k,
            comfort=scores["comfort"],
            assistance=scores["assistance"],
            alpha=self.opi_cfg.alpha,
            lambda_human=self.opi_cfg.lambda_,
        )

        pair_id = f"pair_{pair_index:03d}_cond_{int(cond_k)}_to_{int(cond_k1)}"

        row: Dict[str, object] = {
            "dataset_kind": pair_kind,
            "pair_id": pair_id,
            "pair_index": int(pair_index),
            "subject_id": str(self.features_df["subject_id"].iloc[0]),
            "condition_id_k": int(cond_k),
            "condition_id_k1": int(cond_k1),
            "step_number_k": None if step_idx_k is None else int(step_idx_k),
            "step_number_k1": None if step_idx_k1 is None else int(step_idx_k1),
            "cycle_local_idx_k": None if local_idx_k is None else int(local_idx_k),
            "cycle_local_idx_k1": None if local_idx_k1 is None else int(local_idx_k1),
            "bin_id": None if bin_id is None else int(bin_id),
            "reward_from": reward_block_label,
            "comfort_used": float(scores["comfort"]),
            "assistance_used": float(scores["assistance"]),
            "sample_weight": float(sample_weight),
            "pair_total_weight_target": 1.0,
            "reward": float(reward),
        }

        # Store the parameter vectors and action in flat numeric columns.
        for idx, name in enumerate(self.build_cfg.output_param_order):
            row[f"p_k__{name}"] = float(p_k[idx])
            row[f"p_k1__{name}"] = float(p_k1[idx])
            row[f"a_k__{name}"] = float(a_k[idx])

        # Store the feature vectors both as raw values and as normative deviations.
        for idx, spec in enumerate(self.feature_specs):
            row[f"raw_k__{spec.exp_col}"] = float(f_current[idx])
            row[f"raw_k1__{spec.exp_col}"] = float(f_next[idx])
            row[f"ref__{spec.exp_col}"] = float(self.ref_vector[idx])
            row[f"sigma__{spec.exp_col}"] = float(self.sigma_vector[idx])
            row[f"x_k__{spec.exp_col}"] = float(x_current[idx])
            row[f"x_k1__{spec.exp_col}"] = float(x_next[idx])

        return row


    # Conversion to OPI Dataset
    # -----------------------------

    def _df_to_opi_dataset(self, df: pd.DataFrame):
        """
        Convert the transition dataframe into an OPI_core.Dataset object.
        """
        dataset = Dataset()

        for _, row in df.iterrows():
            x_feat = np.array([row[f"x_k__{spec.exp_col}"] for spec in self.feature_specs], dtype=float)
            x1_feat = np.array([row[f"x_k1__{spec.exp_col}"] for spec in self.feature_specs], dtype=float)
            p = np.array([row[f"p_k__{name}"] for name in self.build_cfg.output_param_order], dtype=float)
            p1 = np.array([row[f"p_k1__{name}"] for name in self.build_cfg.output_param_order], dtype=float)
            a = np.array([row[f"a_k__{name}"] for name in self.build_cfg.output_param_order], dtype=float)
            r = float(row["reward"])

            dataset.add(
                Transition(
                    x_feat=x_feat,
                    a=a,
                    x1_feat=x1_feat,
                    r=r,
                    p=p,
                    p1=p1,
                )
            )

        return dataset


    # Metadata
    # -----------

    def _build_metadata(self, dataset_kind: str, df: pd.DataFrame):
        """
        Build a compact metadata dictionary for JSON export.
        """
        pair_weight_sums = (
            df.groupby("pair_id")["sample_weight"].sum().to_dict()
            if "sample_weight" in df.columns and not df.empty
            else {}
        )

        return {
            "dataset_kind": dataset_kind,
            "selected_feature_exp_cols": self.feature_cols,
            "selected_feature_normative_names": [spec.normative_name for spec in self.feature_specs],
            "block_agg": self.build_cfg.block_agg,
            "normative_reference_choice": self.build_cfg.normative_reference_choice,
            "normative_sigma_choice": self.build_cfg.normative_sigma_choice,
            "normative_scope": self.build_cfg.normative_scope,
            "normative_speed_kmh": self.build_cfg.normative_speed_kmh,
            "reward_from": self.build_cfg.reward_from,
            "parse_param_order": list(self.build_cfg.parse_param_order),
            "output_param_order": list(self.build_cfg.output_param_order),
            "n_rows": int(len(df)),
            "n_real_pairs": int(df["pair_id"].nunique()) if not df.empty else 0,
            "condition_order": list(map(int, self.condition_order)),
            "sample_weight_sum_by_pair": {k: float(v) for k, v in pair_weight_sums.items()},
        }
        
        

# ============================================================================
# |                                                                          |
# |  7) CONVENIENCE WRAPPER                                                  |
# |                                                                          |
# ============================================================================

def build_datasets_from_files(*, feature_csv: str | Path, score_csv: str | Path, param_c_file: str | Path, normative_ref_csv: str | Path, opi_cfg: OPIConfig, build_cfg: DatasetBuildConfig):
    """
    Functional wrapper around ELSADatasetBuilder.

    Returns a dict keyed by dataset kind.
    """
    builder = ELSADatasetBuilder(
        feature_csv=feature_csv,
        score_csv=score_csv,
        param_c_file=param_c_file,
        normative_ref_csv=normative_ref_csv,
        opi_cfg=opi_cfg,
        build_cfg=build_cfg,
    )
    return builder.build_all()


# ============================================================================
# |                                                                          |
# |  8)  CLI (generated by Claude AI)                                        |
# |                                                                          |
# ============================================================================

def _build_argparser():
    parser = argparse.ArgumentParser(description="Build OPI datasets for ELSA from experimental CSV/C files.")
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--param-c-file", required=True)
    parser.add_argument("--normative-ref-csv", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Selected feature names or aliases. Example: --features f1 f2 f6",
    )
    parser.add_argument(
        "--dataset-kinds",
        nargs="+",
        default=["all"],
        help="One or more among block_summary all_to_all, or simply 'all'.",
    )
    parser.add_argument("--block-agg", default="median", choices=["median", "mean"])
    parser.add_argument("--normative-reference-choice", default="median", choices=["median", "mean"])
    parser.add_argument("--normative-sigma-choice", default="robust_iqr", choices=["robust_iqr", "std"])
    parser.add_argument(
        "--sigma-source",
        default="normative",
        choices=["normative", "empirical"],
        help="Source of per-feature standardisation scale sigma_i. "
                "'normative' (default): sigma from Moreira et al. 2021. "
                "'empirical': sigma computed as the std of this subject's stride-level data. "
                "See DatasetBuildConfig docstring for the rationale.",
    )
    parser.add_argument("--normative-scope", default="global", choices=["global", "by_speed"])
    parser.add_argument("--normative-speed-kmh", type=float, default=None)
    parser.add_argument("--reward-from", default="next", choices=["next", "current"])

    parser.add_argument("--n-features", type=int, required=True, help="Must match the number of selected features.")
    parser.add_argument("--lambda-human", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument(
        "--parse-param-order",
        nargs="+",
        default=["tau_lim", "theta", "d_plant"],
    )
    parser.add_argument(
        "--output-param-order",
        nargs="+",
        default=["d_plant", "theta", "tau_lim"],
    )
    parser.add_argument("--strict-param-bounds", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    opi_cfg = OPIConfig(
        n_features=args.n_features,
        lambda_=args.lambda_human,
        alpha=args.alpha,
    )

    build_cfg = DatasetBuildConfig(
        features=args.features,
        dataset_kinds=args.dataset_kinds,
        block_agg=args.block_agg,
        normative_reference_choice=args.normative_reference_choice,
        normative_sigma_choice=args.normative_sigma_choice,
        sigma_source=args.sigma_source,
        normative_scope=args.normative_scope,
        normative_speed_kmh=args.normative_speed_kmh,
        reward_from=args.reward_from,
        parse_param_order=tuple(args.parse_param_order),
        output_param_order=tuple(args.output_param_order),
        strict_param_bounds=args.strict_param_bounds,
    )

    builder = ELSADatasetBuilder(
        feature_csv=args.feature_csv,
        score_csv=args.score_csv,
        param_c_file=args.param_c_file,
        normative_ref_csv=args.normative_ref_csv,
        opi_cfg=opi_cfg,
        build_cfg=build_cfg,
    )
    built = builder.build_all()
    written = builder.export(built, args.output_dir)

    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()