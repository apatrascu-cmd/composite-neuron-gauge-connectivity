# Recorded execution environments

The corrected agent-based simulations, table generation and figure generation are compatible with Python 3.8 or later and the versions in `requirements.txt`.

The archived CIFAR-10 records were produced on macOS with:

- Python 3.8.5
- NumPy 1.22.4
- Matplotlib 3.7.5
- PyTorch 2.2.2 (CPU)

The symbolic and deterministic analytical checks were also run with:

- Python 3.8.12
- NumPy 1.24.4
- SymPy 1.13.3
- Matplotlib 3.7.5

The LaTeX source was built with pdfLaTeX from TeX Live 2019. The scripts set no machine-specific paths. CIFAR-10 is read from `data/cifar-10-batches-py` when bundled locally or is downloaded from the University of Toronto data URL recorded in `cnn_cifar10_composite.py`; `cifar10_input_manifest.json` records the archive and batch-file checksums used in this work.
