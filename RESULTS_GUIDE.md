# Accessing and reconstructing the results

Nothing in this repository is encrypted or password protected. The numerical
records are ordinary UTF-8 JSON files that GitHub can display directly in a
browser and any text editor can open after download.

For a claim-first audit, start with
`reproducibility/CLAIM_EVIDENCE_MAP.md`. It distinguishes proved statements,
finite-campaign results, falsified predictions and claims that the article does
not make.

## Direct result files

After the release is assembled, the principal records are:

- `reproducibility/results/abm_main_v2.json` — five-seed, 1000-unit primary agent-based campaign.
- `reproducibility/results/abm_mild_v2.json` — five-seed mild-offset control.
- `reproducibility/results/abm_pilot_v2.json` and `abm_scan_*_v2.json` — the
  method-freezing pilot and sensitivity controls reported in the final article.
- `reproducibility/results/cnn_*.json` — every CIFAR-10 configuration, seed, and regularisation-strength record used in the paper.
- `reproducibility/results/diagnostic_controls.json` — positive and negative graph/covariance controls.
- `reproducibility/results/reduction_dynamics.json` — static Schur-complement and transient-memory checks.
- `reproducibility/results/abm_hierarchy_main.json` — held-out five-seed hierarchical-scale campaign, including every tested k, matched non-gauge, random-link and oracle controls, and the label-free selected scale for each seed.
- `reproducibility/results/abm_hierarchy_pilot2.json` — the method-freezing
  scale-selection record described in the final article; it is not pooled with
  the held-out result.
- `reproducibility/results/hierarchy_controls.json` — deterministic covariance and support-invariance checks for promotion to the next scale.
- `reproducibility/results/phase_routing_controls.json` — deterministic positive and negative controls for the routing layer's representation symmetry.
- `reproducibility/results/cnn_phase_lambda*_g0.01_s*.json` — paired-seed interpolation of the phase read-out from a non-invariant real-axis magnitude at lambda 0 through lambda 0.5 to the invariant modulus at lambda 1. A fixed factor equalises the expected squared output across lambda. Initial-state and split fingerprints verify the matched comparison.

Each experiment JSON contains its parameters, per-seed or per-run measurements,
aggregate statistics, and the producing-script hash. The JSON records are the
authoritative numerical source; the adjacent text logs are execution transcripts.
The matching `manuscript/tables/*.csv` files provide the final aggregate tables
in a spreadsheet-friendly form generated from those same JSON records.

## Read a record without special software

Open a JSON file directly on GitHub, or download the repository and use a text
editor. Python can also pretty-print any record:

```text
python3 -m json.tool reproducibility/results/abm_main_v2.json
```

## Rebuild the submitted tables and figures

From the `reproducibility` directory:

```text
python3 check_revision_v2.py
python3 check_reduction.py
python3 check_gauge_rule.py
python3 check_hierarchy.py
python3 check_phase_routing.py
python3 make_tables.py --abm_tag main_v2 --abm_mild_tag mild_v2
python3 make_figures.py --abm_tag main_v2
python3 make_hierarchy_tables.py --tag hierarchy_main
python3 make_hierarchy_figure.py --tag hierarchy_main
python3 make_phase_ablation_table.py
```

All five checks must print `OVERALL PASS`. The generators read the archived
JSON records and recreate the manuscript tables, numerical macros, and figures;
retraining is unnecessary for this reconstruction.

## Re-run the campaigns

The detailed environment and exact commands are in
`reproducibility/README.md` and `reproducibility/ENVIRONMENT.md`. CIFAR-10 is
downloaded from the public University of Toronto source recorded in the script,
and `cifar10_input_manifest.json` records the checksums of the archive and every
batch file.

## What the checksums mean

`MANIFEST.sha256` contains integrity fingerprints. It lets a reader verify that
a downloaded file is byte-for-byte identical to the released file. It is not
an encryption key, password, or access-control mechanism.
