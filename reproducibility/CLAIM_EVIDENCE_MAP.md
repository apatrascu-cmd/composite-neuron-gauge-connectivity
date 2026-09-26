# Claim-to-evidence map

This file maps every central claim of the manuscript to the exact analytical
check or numerical record that supports it. It also records negative results
and claims that the article deliberately does not make. The JSON and CSV files
are ordinary, unencrypted text.

## Analytical claims

| Claim | Status | Direct evidence | Reconstruction |
|---|---|---|---|
| Static local rephasing leaves the dynamics and the three update laws covariant. | Proved and checked. | Proposition 2; `results/diagnostic_controls.json`; `results/phase_routing_controls.json`. | `python3 check_gauge_rule.py`; `python3 check_revision_v2.py`; `python3 check_phase_routing.py`. |
| An arbitrary physical offset change has an exact algebraic transporter compensator. | Proved and checked; this is not a claim that learning always finds it. | Proposition 1; `offset_compensator_positive` in `results/diagnostic_controls.json`. | `python3 check_revision_v2.py`. |
| The original submitted transporter objective had zero gradient, whereas the replacement is non-vacuous. | Corrected and checked. | Remark 1; `results/diagnostic_controls.json`; per-run non-vacuity controls in `results/abm_main_v2.json`. | `python3 check_gauge_rule.py`. |
| A transported composite mean is path independent exactly on a flat connected subgraph. | Proved and checked with positive and negative graph controls. | Proposition 3; `known_flat_positive` and `triangle_free_cycle_negative` in `results/diagnostic_controls.json`. | `python3 check_revision_v2.py`. |
| Cluster elimination gives an exact static Schur complement, while transients retain memory and two poles. | Proved and numerically checked. | `results/reduction_dynamics.json`. | `python3 check_reduction.py`. |

## Agent-based numerical claims

| Claim | Status | Direct evidence | Reconstruction |
|---|---|---|---|
| Adaptive transport suppresses complete-cycle inconsistency and increases the certified-unit fraction relative to delay-only and quenched controls. | Supported for the declared finite campaign. | `results/abm_main_v2.json`; `../manuscript/tables/abm_main_table.csv`. | `python3 make_tables.py --abm_tag main_v2 --abm_mild_tag mild_v2`. |
| Adaptive transport recruits more candidate units. | Supported. | Same primary record and table. | Same command. |
| Fine-scale candidates recover the planted partition more accurately. | Falsified under the pre-stated directional criterion. | Gauge-versus-control adjusted Rand indices in `results/abm_main_v2.json`. | Same command. |
| Active transporters recover after a physical redraw of all fixed offsets. | Supported for coverage-weighted coherence and partition identity against frozen-transporter and delay-only controls. | `tests/delay_shock` in `results/abm_main_v2.json`; `../manuscript/tables/abm_tests_table.csv`. | Same command. |
| The model has generic robustness to arbitrary perturbations. | Not claimed. | Removal and frequency-step outcomes are reported separately in `../manuscript/tables/abm_tests_table.csv`. | Same command. |

## Hierarchical-scale claims

| Claim | Status | Direct evidence | Reconstruction |
|---|---|---|---|
| Observable next-scale information can reconnect certified fine fragments and recover much of a larger planted partition. | Supported in a five-seed held-out two-level experiment. | `results/abm_hierarchy_main.json`; `../manuscript/tables/hierarchy_main_table.csv`; `../manuscript/tables/hierarchy_sensitivity_table.csv`. | `python3 make_hierarchy_tables.py --tag hierarchy_main`; `python3 make_hierarchy_figure.py --tag hierarchy_main`. |
| Gauge transport itself discovers the missing scale labels. | Not supported and not claimed. | Matched frequency-support/no-transporter, random-link and planted-label controls in the same record. | Same commands. |
| Gauge transport contributes complete-cycle consistency once support is proposed. | Supported by the matched same-support control. | Cycle residual and certified-fraction fields in the same record. | Same commands. |
| The label-free selector always finds a new scale. | Falsified: it accepts three held-out realisations and abstains in two. | `scale_selection` in the same record. | Same commands. |
| The two-level experiment establishes a universal scaling law. | Not claimed. | The manuscript reports exactly two levels and the selector's abstentions. | No additional inference is made. |

## Convolutional claims

| Claim | Status | Direct evidence | Reconstruction |
|---|---|---|---|
| The composite penalty makes weight rows more compact. | Supported on the declared CIFAR-10 backbone. | `results/cnn_*.json`; `../manuscript/tables/cnn_table.csv`; `../manuscript/tables/cnn_gamma_table.csv`. | `python3 make_tables.py --abm_tag main_v2 --abm_mild_tag mild_v2`. |
| The penalty improves clean or adversarial accuracy. | Not supported; the reported rows trade accuracy for compactness. | Same records and tables. | Same command. |
| Phase routing outperforms capacity-matched cosine routing in clean accuracy. | Observed in the original matched-capacity comparison; this is an association between complete layer designs, not a causal invariance result. | `results/cnn_adaptive_phase_*.json`; `results/cnn_adaptive_girl_*.json`. | Same command. |
| Increasing read-out invariance causes the phase-routing gain. | Not supported by the pre-stated paired criterion.  The exactly invariant endpoint differs from the broken endpoint by -0.2 +/- 1.3 clean-accuracy points, wins in one of three seeds, and the ordering is monotone in one seed. | `results/cnn_phase_lambda*_g0.01_s*.json`; `../manuscript/tables/phase_ablation_table.csv`; identical per-seed state and split fingerprints; at most 4.9% within-seed initial-RMS spread. | `python3 check_phase_routing.py`; `python3 make_phase_ablation_table.py`. |
| The interpolation changes the declared representation symmetry. | Supported directly.  Prediction agreement under common rephasing rises from 0.630 at lambda 0 to 1.000 at lambda 1; route and logit defects fall from about 0.413 to numerical zero. | Same phase records and `results/phase_routing_controls.json`. | Same commands. |
| The CNN construction provides adversarial robustness. | Falsified on the declared attacks; the earlier claim is withdrawn. | `../manuscript/tables/cnn_table.csv`; `../manuscript/tables/cnn_gamma_table.csv`. | `python3 make_tables.py --abm_tag main_v2 --abm_mild_tag mild_v2`. |

## Interpretation boundaries

- The complex oscillator is a normal form, not a detailed biophysical neuron.
- Static covariance is not time-dependent gauge covariance.
- Offset-redraw recovery is one specified physical robustness test, not generic robustness.
- The hierarchy supplies an observable descriptor; the transporter does not manufacture missing information.
- No claim of broad biological equivalence or universal superiority is made.
