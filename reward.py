""" 
Reward construction for the ELSA ankle prosthesis

Author : Raphaël VASSART
=============================================================
This module is one of the most important of the OPI pipeline.
It implements the shaped reward r_k that blends two complementary objectives: biomechanical quality and subjective human feedback.

Reward formula
---------------
    r_k = (1 - λ) · r^{prosthesis}_k  +  λ · r^{human}_k

  r^{prosthesis}_k = −Σᵢ wᵢ x̃ₖⁱ²  -  aₖᵀ R_a aₖ
      ┌ First term  : penalises deviation from normative biomechanics.
      └ Second term : penalises large / aggressive parameter changes.
      x̃ₖⁱ = (fₖⁱ - fⁱ·ʳᵉᶠ) / σᵢ^{dataset}   (inter-subject standardisation)

  r^{human}_k = α C̃ₖ + (1 - α) Ãₖ
      ┌ C̃ₖ, Ãₖ : per-subject z-scored comfort and assistance scores.
      └ α balances comfort vs. perceived assistance.

  With cfg.assist_coding = "centered", Ãₖ is the z-score of -|Aₖ - 5| (the assistance scale is bipolar).
  With cfg.action_cost = "separate", the action term is kept at full weight:
      r_k = (1 - λ) · (−Σᵢ wᵢ x̃ₖⁱ²)  +  λ · r^{human}_k  -  aₖᵀ R_a aₖ

Usage
------
    cfg = OPIConfig()
    reward = RewardComputer(cfg)
    reward.set_reference_stats(sigma_dataset)
    reward.fit_human_stats(comfort, assistance) # calibrate per-subject z-scores

    r = reward.compute(x_feat, a, comfort=4.0, assistance=3.5)
"""


from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
from elsa_configOPI import OPIConfig

# ============================================================================
# |                                                                          |
# |  REWARD COMPUTER                                                         |
# |                                                                          |
# ============================================================================
class RewardComputer:
    """
    Implements the shaped reward r_k for the ELSA ankle prosthesis.

    Two complementary reward components are combined:
        - r^{prosthesis} : purely data-driven, based on biomechanical features.
        - r^{human} : based on the user's subjective comfort and perceived assistance, z-scored per subject for comparability.

    The RewardComputer must be set up in two steps before use:
        1. set_reference_stats(sigma_dataset) : provide inter-subject stds from a normative/empirical dataset 
        2. fit_human_stats(comfort_scores, assistance_scores) : calibrate the per-subject z-score statistics from collected feedback.
    """

    def __init__(self, cfg: OPIConfig):
        self.cfg = cfg
        
        # Inter-subject std from the normative dataset.
        # Set via set_reference_stats() before computing any reward.
        self.sigma_dataset: Optional[np.ndarray] = None 
        
        # Per-subject z-score parameters for human feedback.
        # Set via fit_human_stats() before calling human_reward().
        self._mu_c: float = 0.0; self._sig_c: float = 1.0
        self._mu_a: float = 0.0; self._sig_a: float = 1.0
        
        
    # -- Setup methods --------------------------------------------------------
    
    def set_reference_stats(self, sigma_dataset: np.ndarray):
        """
        Provide inter-subject standard deviations from the empirical dataset for each biomechanical feature.
        
        These stds are used to compute the standardised feature error:
            x̃ₖⁱ = xₖⁱ / σᵢ^{dataset}
            
        This makes the prosthesis reward comparable across features that have different physical units or magnitudes.
        """
        sigma_dataset = np.asarray(sigma_dataset, dtype=float)
        self.sigma_dataset = np.maximum(sigma_dataset, 1e-8)
        return self
    
    def fit_human_stats(self, comfort_scores: np.ndarray, assistance_scores: np.ndarray):
        """
        Compute per-subject z-score statistics from collected human feedback.
        
        Call once per participant before building the dataset.
        This makes comfort and assistance scores comparable even if their raw scales or distributions differ across subjects.
        """
        comfort_scores = np.asarray(comfort_scores, dtype=float)
        assistance_scores = self.code_assistance(assistance_scores)
        
        self._mu_c = float(np.mean(comfort_scores))
        self._sig_c = max(float(np.std(comfort_scores)), 1e-8)
        self._mu_a = float(np.mean(assistance_scores))
        self._sig_a = max(float(np.std(assistance_scores)), 1e-8)
        
        return self
    
    
    # -- Individual reward components -----------------------------------------
    
    def prosthesis_reward(self, x_feat: np.ndarray, a: np.ndarray):
        """
        Compute the prosthesis-related reward term : 
            r^{prosthesis} = - wᵀ (x̃²) - aᵀ R_a a
        where x̃ is the standardised biomechanical feature vector.
        
        Interpretation:
        ----------------
          - The first term penalises deviation from normative biomechanics.
          - The second term penalises large / aggressive parameter changes.
        """
        sig = self.sigma_dataset if self.sigma_dataset is not None else np.ones_like(x_feat)
        x_tilde = np.asarray(x_feat, dtype=float) / sig
        a = np.asarray(a, dtype=float)
        
        return float(-np.dot(self.cfg.feature_weights, x_tilde**2) - a @ self.cfg.R_a @ a)
    
    def human_reward(self, comfort: float, assistance: float, *, alpha: float = 0.5):
        """
        Compute the human-feedback reward term : 
            r^{human} = α C̃ₖ + (1-α) Ãₖ
            
        C̃ and Ã are z-scored across blocks for the current subject so that scores from different subjects remain comparable.
        """
        c_tilde = (float(comfort) - self._mu_c) / self._sig_c
        a_tilde = (float(self.code_assistance(assistance)) - self._mu_a) / self._sig_a
        return float(alpha * c_tilde + (1.0 - alpha) * a_tilde)
    
    def code_assistance(self, assistance):
        """Assistance score as it enters the z-score: raw, or -|A - A_opt| for the bipolar scale."""
        assistance = np.asarray(assistance, dtype=float)
        if self.cfg.assist_coding == "centered":
            return -np.abs(assistance - self.cfg.assist_optimum)
        return assistance
    
    
    # -- Final Reward Computation -----------------------------------------
    
    def compute(self, x_feat: np.ndarray, a: np.ndarray, comfort: Optional[float] = None, assistance: Optional[float] = None, *, alpha: float = 0.5, lambda_human: float = 0.5):
        """
        Combine prosthesis and human terms into one shaped reward.
            r = (1 - λ) · r^{prosthesis}  +  λ · r^{human}
            
        If no human feedback is provided: 
            r = r^{prosthesis} 
        """
        r_p = self.prosthesis_reward(x_feat, a)
        
        if comfort is None or assistance is None:
            return r_p
        
        r_h = self.human_reward(comfort, assistance, alpha=alpha)
        
        if self.cfg.action_cost == "separate":
            a = np.asarray(a, dtype=float)
            r_act = float(-a @ self.cfg.R_a @ a)
            return (1.0 - lambda_human) * (r_p - r_act) + lambda_human * r_h + r_act
        return (1.0 - lambda_human) * r_p + lambda_human * r_h
    
    def compute_batch(self, x_tilde: np.ndarray, a: np.ndarray, comfort: np.ndarray, assistance: np.ndarray):
        """
        Vectorised version of compute() for N transitions, with the balance (λ, α) of the config.
        x_tilde : (N, n_features) standardised features, as stored in the transitions CSV.
        a : (N, n_actions) actions, comfort / assistance : (N,) raw scores.
        """
        x_tilde, a = np.asarray(x_tilde, dtype=float), np.asarray(a, dtype=float)
        lam, alpha = float(self.cfg.lambda_), float(self.cfg.alpha)
        c_tilde = (np.asarray(comfort, dtype=float) - self._mu_c) / self._sig_c
        a_tilde = (self.code_assistance(assistance) - self._mu_a) / self._sig_a
        r_gait = -((x_tilde * x_tilde) @ self.cfg.feature_weights)
        r_act = -np.einsum("ni,ij,nj->n", a, self.cfg.R_a, a)
        r_h = alpha * c_tilde + (1.0 - alpha) * a_tilde
        if self.cfg.action_cost == "separate":
            return (1.0 - lam) * r_gait + lam * r_h + r_act
        return (1.0 - lam) * (r_gait + r_act) + lam * r_h
    
    # -- Source-CSV factory (used for offline reward recomputation) -------------
 
    @classmethod
    def fit_from_scores_csv(cls, cfg: OPIConfig, score_csv_path: Union[str, Path], *, sigma_dataset: Optional[np.ndarray] = None):
        """
        When an ablation study changes "cfg.alpha" (or "cfg.lambda_") at training time, the rewards frozen into the dataset must be recomputed with the new balance. 
        The correct μ/σ for that recomputation are those used at build time, NOT those recoverable by deduplicating the transitions CSV.

        """
        path = Path(score_csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Score CSV not found: {path}")
 
        df = pd.read_csv(path)
        # Accept both the raw experimental headers and the canonical ones.
        # Rename is idempotent when the canonical columns are already present.
        rename_map = {
            "condition number": "condition_id",
            "score COMF": "comfort",
            "score ASSIST": "assistance",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
 
        missing = {"comfort", "assistance"} - set(df.columns)
        if missing:
            raise ValueError(
                f"Score CSV {path} is missing columns {sorted(missing)} "
                "(accepted raw names: 'score COMF', 'score ASSIST')."
            )
 
        rc = cls(cfg)
        rc.fit_human_stats(
            df["comfort"].to_numpy(dtype=float),
            df["assistance"].to_numpy(dtype=float),
        )
        if sigma_dataset is not None:
            rc.set_reference_stats(sigma_dataset)
        return rc

    
    
    

