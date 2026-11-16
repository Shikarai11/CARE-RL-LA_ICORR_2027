# Data folder

The participant data of the study are **not distributed** (single participant, collected under ethics
approval). Put your own files here: everything in this folder except this README and the normative table
is ignored by git, so the data cannot be committed by mistake.

## Included

### `reward_ready_normative_feature_refs.csv`

Normative references of the gait features, computed from the public dataset of healthy adults walking at
controlled speeds of Moreira et al. (2021), *Sci. Data* 8:103 (participant × speed medians, one participant
of the public dataset excluded). One row per feature and scope:

| Column | Meaning |
|---|---|
| `feature_name` | normative feature name (e.g. `f1A_drop_deg_signed`), matched by `FEATURE_SPECS` in `dataset_builder.py` |
| `description`, `unit` | what the feature measures and its unit |
| `scope`, `speed_kmh` | `global` (all speeds pooled, used in the paper) or `by_speed` (one row per speed) |
| `n_subjects` | number of healthy participants used |
| `ref_median`, `ref_mean` | reference value $f_\mathrm{ref}$ (`--normative-reference-choice median` in the paper) |
| `robust_sigma_from_iqr`, `ref_std` | dispersion of the healthy population (`--normative-sigma-choice`) |

In the paper, the reference is the global median, and the scale $\sigma$ of each feature is the standard
deviation of the participant's own gait cycles (`--sigma-source empirical`); the healthy-population
dispersions are only used with `--sigma-source normative`.

## Expected, not included

### Stride-level features: `ExpS1_Full_final_features.csv`

One row per gait cycle. Required columns:

| Column | Meaning |
|---|---|
| `subject_id` | participant code |
| `condition_id` | block index, 0-based, in the order of the protocol (`int`) |
| `cycle_local_idx` | index of the gait cycle within the block (`int`) |
| `step_number` | step counter of the recording (stored as metadata) |
| `f1_drop_deg_signed` | $f^1$, plantarflexion drop after heel strike (deg) |
| `f2_mpa_to_pushoff_time` | $f^2$, delay from maximal plantarflexion to push-off onset (s) |
| `f3_max_abs_speed_pushoff` | $f^3$, peak ankle angular speed during push-off (deg/s) |
| `f4_pushoff_duration` | $f^4$, push-off duration (s) |
| `f5_peak_torque_pushoff_norm` | $f^5$, peak push-off torque per unit body mass (N·m/kg) |
| `f6_generated_mechanical_energy_norm` | $f^6$, mechanical energy generated during push-off per unit body mass (J/kg) |

Other columns are allowed and ignored. A block missing from this file (as block 22 in the study, excluded
after a recording error) is detected, and the two transitions that involve it are skipped.

### Ratings: `ExpS1_Full_scores.csv`

One row per block:

| Column | Meaning |
|---|---|
| `condition number` | block index, as `condition_id` above |
| `score COMF` | comfort rating, 0–10 |
| `score ASSIST` | perceived-assistance rating, 0–10, bipolar: 0 = no assistance felt, 5 = appropriate, 10 = far too strong |

The columns `condition_id`, `comfort` and `assistance` are also accepted. Any other column is ignored.

### Parameter settings: `S1_exp_params.c`

The settings of the protocol, one brace-enclosed tuple per block, in block order, as in the C file loaded on
the prosthesis. Each tuple is `{tau_lim [N m], theta_trig [rad], d_dors [N m s/rad]}`:

```c
const ExperimentalParams test_params[] = {
  {11.0f, 0.113f, 20.708f},
  {27.0f, 0.171f, 12.387f},
  ...
};
```

The order of the values can be changed with `--parse-param-order`.

## Built by `dataset_builder.py`

| File | Content |
|---|---|
| `all_to_all_transitions.csv` | one row per pseudo-transition: pair id, block ids, cycle indices, ratings used, reward, parameters `p_k__*`, `p_k1__*`, actions `a_k__*`, and for each feature its raw values, reference, scale and standardized values (`x_k__*`, `x_k1__*`) |
| `all_to_all_opi_dataset.pkl` | the same transitions as a pickled `OPI_core.Dataset`, read by `run.py` and the controls |
| `all_to_all_metadata.json` | the construction options and the list of block pairs |

The pickle is a Python object: only load pickles that you built yourself.
