# Composite-neuron clustering with gauge-inspired adaptive connectivity

This repository is the reproducibility record for:

**Andrei T. Pătrașcu, “Emergent Composite Clustering and Gauge-Inspired Adaptive Connectivity in Agent-Based Neural Networks.”**

It contains the final source code, final reported numerical records, execution
receipts, and scripts needed to verify or reconstruct every numerical table and
figure.  The current author manuscript is available as [manuscript.pdf](manuscript.pdf). Private working records, journal correspondence, manuscript source, and superseded code are not part of this release.

## Scientific scope

The paper studies complex oscillator units connected by fixed transmission offsets and learnable phase transporters. It separates four questions:

1. static local phase covariance of the representation;
2. complete cycle consistency of candidate composite clusters;
3. static and transient elimination of a collective cluster;
4. empirical recovery after physical perturbations.

The repository also contains a bounded CIFAR-10 counterpart based on soft weight sharing and two routing layers. Its negative adversarial-robustness result is retained.

## Repository layout

- `manuscript/tables/` and `manuscript/figures/` — final generated outputs.  The
  directory name is retained because the reconstruction scripts write to these
  relative paths; no manuscript source or journal correspondence is included.
- `reproducibility/` — experiment code, deterministic checks, raw JSON records, logs, requirements and run instructions.
- `RESULTS_GUIDE.md` — a plain-language map from each reported result to its directly readable JSON file and regeneration command.
- `reproducibility/CLAIM_EVIDENCE_MAP.md` — a claim-by-claim map of proofs, raw records, negative results and scope limits.
- `MANIFEST.sha256` — SHA-256 checksum of every released file.

The original v1.0.0 reproducibility release begins with one clean release commit. This update adds the current author manuscript without importing private development history or abandoned versions.  Records named `pilot` are retained only
where the final article explicitly reports them as method-freezing or
sensitivity controls; they are evidence used by the submitted analysis, not
discarded versions.

All released files and ZIP archives are unencrypted and have no password.  The
SHA-256 values are integrity fingerprints: they let a reader confirm that a
download is byte-for-byte identical to the released file and do not restrict
access to its contents.

The reproducibility folder has its own detailed `README.md`. The shortest verification path is:

```text
cd reproducibility
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

All five checks must print `OVERALL PASS`. The table generators also write plain CSV copies. The table and figure generators consume the archived JSON files, so rerunning the simulations or retraining is not required to reconstruct the submitted artifacts.

## Repeating the numerical campaigns

The exact campaign commands are:

```text
bash run_abm_validation_v2.sh
bash run_abm_production_v2.sh
bash run_cnn_batch.sh
bash run_cnn_provenance_v2.sh
python3 abm_hierarchy.py --tag hierarchy_main --N 1000 --seeds 5 --seed_start 10
WORKERS=2 THREADS=1 EPOCHS=8 bash run_phase_ablation.sh
```

The production agent-based campaign is CPU-only. Repeating all CNN training is substantially slower; the archived JSON records include the metrics and producing-script hashes used by the paper.

## Interpretation boundaries

- The proved symmetry is a static local rephasing.
- An offset redraw is a physical perturbation; the exact compensator is a separate algebraic control.
- Only the transporter rule is derived as a mismatch-energy gradient.
- A strong-coupling component is a candidate; complete cycle tests decide certification.
- Cluster elimination is exact for fixed points and static gains, while transients retain memory.
- The paired phase-readout interpolation tests whether clean accuracy follows the strength of the read-out invariance; a fixed normalisation holds the expected squared activation constant across the interpolation. Its result is reported whichever way it comes out and does not imply adversarial robustness.
- Scale information proposes inter-composite support; gauge transport tests whether those relations glue consistently. The hierarchy does not claim that gauge symmetry manufactures missing information.
- The two tested levels establish hierarchical composition only. They do not establish a universal scaling law.

## Data source

CIFAR-10 is the public data set described in the manuscript. If it is not present locally, the training script downloads the Python-format archive from the University of Toronto URL recorded in the source. The release also supplies the SHA-256 manifest of the archive and every batch file used by the reported runs.

## Citation

Please cite the article and use `CITATION.cff` for this repository snapshot.

## Rights

The software in `reproducibility/` is released under the BSD 3-Clause License
in `LICENSE-CODE`. The numerical records, generated tables, generated figures,
execution logs and repository documentation are released under CC BY 4.0 as
specified in `LICENSE-DATA.md`.

The author manuscript in `paper/` is supplied for reading and citation; its copyright is retained by the author and it is outside the code/data licenses. Response letters, cover letters, private working notes and journal correspondence are not included.

## Manuscript version

Author manuscript; revision submitted to Neurocomputing on 26 September 2026. Not an accepted or publisher-formatted article.
