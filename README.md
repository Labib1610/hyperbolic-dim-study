# hyperbolic-dim-study

A controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean
embeddings of linguistic hierarchies (WordNet), as a function of hierarchy
structure (δ-hyperbolicity).

## Environment setup

**GPU note (Blackwell / RTX 5060 Ti / sm_120):** a plain `pip install torch`
installs wheels **without sm_120 kernels**, which triggers a
`sm_120 is not compatible` warning and silent CPU fallback at runtime. You MUST
install torch from the **cu128** (CUDA 12.8) channel.

```bash
cd hyperbolic-dim-study
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# PyTorch for Blackwell/sm_120 — try STABLE cu128 first:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then the rest of the dependencies:
pip install -r requirements.txt

# Editable install of the local package:
pip install -e .
```

**Fallback** if the stable cu128 wheel does not exist for your Python version,
or if the smoke test reports `sm_120` NOT supported — install the nightly
cu128 build instead:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

If neither yields sm_120 support, stop and report the exact torch version,
Python version, and `torch.cuda.get_arch_list()` output. Do not silently
proceed on CPU.

## Run the smoke test

```bash
python -m src.smoke
```

A successful run ends with `SMOKE TEST PASSED`. It verifies Python/PyTorch
versions, CUDA availability, the compiled arch list (must contain `sm_120`),
the GPU name + compute capability `(12, 0)`, a real GPU tensor op, and a
`geoopt` Poincaré-ball distance computation.

## Run tests

```bash
pytest -q
```

The pytest suite is CPU-safe and does not require a GPU; GPU verification lives
in `src/smoke.py`.
