# Reproducibility package

This folder reconstructs the analytical checks, tables and figures for
“Emergent Composite Clustering and Gauge-Inspired Adaptive Connectivity in Agent-Based Neural Networks.”
All commands are run from this folder. No script requires a path outside the unpacked package.
The archive is unencrypted and has no password.  Its SHA-256 manifest is an
integrity check, not an access restriction.

`CLAIM_EVIDENCE_MAP.md` is the quickest audit route: it maps each positive,
negative and deliberately excluded claim to its raw record, control and
reconstruction command.

## What is archived

- Exact source for the agent-based and CIFAR-10 experiments.
- Raw JSON records and logs used by the submitted manuscript.
- Deterministic positive and negative controls for covariance, connection consistency and disconnected-cluster coverage.
- A static and transient cluster-elimination check.
- Scripts that regenerate every manuscript macro, table and figure from the JSON records.
- `requirements.txt` and `ENVIRONMENT.md`.
- `CLAIM_EVIDENCE_MAP.md`, which connects every central claim to its direct evidence.

## Requirements

Create a Python 3.8-or-later environment and install:

```text
pip install -r requirements.txt
```

PyTorch is needed only to repeat the CIFAR-10 training. The archived CNN JSON files are sufficient to regenerate the submitted tables without retraining. CPU execution is sufficient. If `data/cifar-10-batches-py` is absent, the CNN script downloads the public Python-format CIFAR-10 archive from the University of Toronto. `cifar10_input_manifest.json` records both the archive checksum and the checksums of every batch used by the paper; run `python3 check_cifar10_inputs.py` after download or extraction to verify the input.

## Deterministic checks

1. `python3 check_revision_v2.py`
   tests a known flat connected graph, a triangle-free non-flat square, disconnected-cluster coverage, static covariance, the exact offset compensator, and a time-dependent-rephasing negative control. It writes `results/diagnostic_controls.json` and must print `OVERALL PASS`.
2. `python3 check_reduction.py`
   verifies the exact fixed-point Schur complement and fixed points, a deliberately wrong negative control, the frequency-dependent memory kernel and the exact transient poles. It writes `results/reduction_dynamics.json` and must print `OVERALL PASS`.
3. `python3 check_gauge_rule.py`
   verifies that the learning objective printed in the earlier manuscript had zero gradient, that the replacement transporter objective has the stated non-zero gradient, and that the quantities claimed to be invariant are invariant under a static local rephasing. It must print `OVERALL PASS`.
4. `python3 check_hierarchy.py`
   verifies that a certified fine cluster promotes to one covariant coarse state, that a root-dressed boundary link transforms as a coarse transporter, and that the support graph is unchanged by a static rephasing. It writes `results/hierarchy_controls.json` and must print `OVERALL PASS`.
5. `python3 check_phase_routing.py`
   verifies the exact local-input rephasing covariance and common-phase invariance of the phase-routing layer, with uncompensated rephasing and cosine routing as negative controls. It writes `results/phase_routing_controls.json` and must print `OVERALL PASS`. This is a representation-symmetry check, not an adversarial-image test.

## Agent-based campaign

6. `bash run_abm_validation_v2.sh`
   runs the corrected pilot and three sensitivity checks: coupling-weighted transporter adaptation, faster frequency adaptation and a halved integration step. The production parameters were fixed only after these checks were inspected.
7. `bash run_abm_production_v2.sh`
   runs `main_v2` with offsets in `(-pi, pi)` and `mild_v2` with offsets in `(-pi/2, pi/2)`, five seeds each, 1000 units. Each JSON stores all parameters, per-seed measurements, aggregate statistics, schema version and code hash.
8. A short functional smoke test is:
   `python3 abm_composite_gauge.py --quick --seeds 1 --N 250 --tag smoke_local`.

The simulation uses a reciprocal random geometric graph, synchronous Euler--Maruyama updates, coverage-aware transported coherence and all independent cycle constraints (reciprocal two-cycles plus a fundamental cycle basis). Candidate strong-coupling clusters and certified flat clusters are distinct outputs.

## Held-out hierarchical-scale campaign

9. `python3 abm_hierarchy.py --tag hierarchy_main --N 1000 --seeds 5 --seed_start 10`
   repeats the fine-scale gauge organisation on held-out seeds 10--14, promotes only complete-cycle-certified fine composites, and tests a second dynamical layer.  The scale information is observable: composite mean frequencies define a sparse neighbour graph, and the phase of the measured complex inter-composite relation initialises its covariant links.  Planted labels are excluded from construction and are read only afterward to score recovery.
10. `python3 make_hierarchy_tables.py --tag hierarchy_main` and
   `python3 make_hierarchy_figure.py --tag hierarchy_main`
   generate the hierarchy macros, two LaTeX/CSV tables and Figure 5 directly from the held-out JSON.

The scale selector was fixed in the development pilot before the held-out run: among k = 2, 4 and 8, choose the largest k whose gauge arm has complete-cycle residual q95 at most 0.01 rad and places at least 95% of promoted units in certified modules; if none passes, retain physical boundary support only. The same selected k is then applied to a matched non-gauge arm. Fixed-count random links and a planted-label oracle are negative and positive controls. Consequently, a partition gain shared by the matched arms belongs to the scale information, while a cycle-consistency gain belongs to the gauge transporter. The pilot is archived as model-selection provenance and is not pooled with the held-out scores.

## CIFAR-10 campaign

11. `bash run_cnn_batch.sh`
   repeats the five configurations and three seeds plus the regularisation-strength scan. Training lasts eight epochs and the final-epoch model is evaluated; no best-validation checkpoint is selected.
12. `bash run_cnn_provenance_v2.sh`
   repeats the four seed-zero records that predated the JSON code-hash field. The submitted archive retains only result records carrying the producing script hash.
13. `WORKERS=2 THREADS=1 EPOCHS=8 bash run_phase_ablation.sh`
   runs the paired read-out interpolation at lambda 0, 0.5 and 1 for seeds 0--2.  For each seed, the three arms have identical initial tensors, split, batch order, optimiser, architecture and parameter count; the JSON stores fingerprints that verify the pairing.  The factor 2/(1+lambda) equalises the expected squared routed activation, and every record stores its measured pre-training routed RMS.  This experiment tests whether clean accuracy follows the strength of common-phase invariance.  It does not test image-space adversarial robustness.

One three-seed configuration takes roughly one to several CPU-hours on a laptop, depending on load. The table reports both inference parameters and total trainable parameters; regulariser-only centroids and mixing logits are excluded from inference counts and included in trainable counts.

## Reconstruct the paper artifacts

14. `python3 make_tables.py --abm_tag main_v2 --abm_mild_tag mild_v2`
   writes `../manuscript/tables/numbers.tex`, all five LaTeX tables, and matching plain CSV tables.
15. `python3 make_phase_ablation_table.py`
   writes the plain CSV and LaTeX interpolation table plus the numerical macros used by the manuscript.
16. `python3 make_figures.py --abm_tag main_v2`
    writes the four manuscript figures as PDF and PNG.
17. From `../manuscript`, run pdfLaTeX three times:
    `pdflatex composite_neuron_revised.tex`.

The generated `numbers.tex` is the only route from numerical JSON values into manuscript prose. Hand-entered numerical results are avoided.

## Principal files

- `abm_composite_gauge.py`: model, three arms, diagnostics, perturbations and aggregation.
- `abm_hierarchy.py`: promotion of certified fine composites, scale-information graph, matched controls and held-out scale selection.
- `cnn_phase_ablation.py`: paired-seed interpolation of the phase read-out at lambda 0, 0.5 and 1; this isolates the read-out symmetry strength while holding architecture, parameter count, initial weights, split, optimiser and expected initial activation scale fixed.
- `cnn_cifar10_composite.py`: CIFAR-10 models, attacks and result writer.
- `cifar10_input_manifest.json`, `check_cifar10_inputs.py`: source URL and exact input checksums.
- `check_revision_v2.py`, `check_reduction.py`, `check_gauge_rule.py`, `check_hierarchy.py`, `check_phase_routing.py`: PASS/FAIL controls.
- `make_tables.py`, `make_figures.py`, `make_hierarchy_tables.py`, `make_hierarchy_figure.py`: artifact generators.
- `make_phase_ablation_table.py`: plain CSV, LaTeX table and manuscript macros for the phase-read-out interpolation.
- `run_abm_validation_v2.sh`, `run_abm_production_v2.sh`, `run_cnn_batch.sh`, `run_cnn_provenance_v2.sh`, `run_phase_ablation.sh`: exact campaigns.
- `results/`: the submitted raw result JSON and text logs.

## Interpretation boundaries

The proved symmetry is a static local rephasing. A physical redraw of transmission offsets is a perturbation, while the exact transporter compensator is a separate algebraic control. Only the transporter update is derived as the gradient of the normalised mismatch; coupling and frequency adaptation are invariant local rules driven by the same link correlation. Cluster elimination is exact for fixed points and static gains; transient elimination retains memory and has exact poles. The CNN comparison does not establish that covariance causes the observed clean-accuracy difference.

Gauge transport does not create scale information.  In the hierarchy experiment, observable frequency and temporal relations propose which certified composites should be compared; complete-cycle consistency tests whether those pairwise relations can be glued into a valid larger composite.  A two-level construction is evidence for hierarchical composition, not by itself a universal scaling law or scale invariance.
