# Stage 0 — Environment & Repo Skeleton

## Context (read first)
This is the first stage of a research project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet), as a function of hierarchy structure (δ-hyperbolicity).**

This stage builds ONLY the foundation: a clean repo, a working Python environment (with correct PyTorch for an RTX 5060 Ti / Blackwell / sm_120 GPU), and a trivial smoke test. **Do not implement any geometry, data, or training logic in this stage.** Those come in later stages. Keep everything minimal and correct.

## Target machine (already confirmed)
- OS: Linux
- Env manager: **venv + pip** (do NOT use conda)
- GPU: NVIDIA RTX 5060 Ti 16GB — **Blackwell, compute capability sm_120**
- CUDA runtime/driver: already installed and working

## Critical GPU note (must handle)
The RTX 5060 Ti is sm_120. A plain `pip install torch` installs wheels **without sm_120 kernels**, causing this at runtime:
`NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.`
followed by silent CPU fallback. To get real GPU support you MUST install the **cu128** (CUDA 12.8) build channel. Handle this explicitly and verify it (see Task 3 + Task 6).

---

## Objective / Definition of Done
A reproducible repo where running `python -m src.smoke` prints:
1. Python version
2. PyTorch version + whether it was built with CUDA
3. Whether `torch.cuda.is_available()` is True
4. The detected GPU name and compute capability
5. Result of a tiny tensor op run **on the GPU** (not CPU)
6. `geoopt` version and a tiny Poincaré-ball manifold sanity op
7. A final line: `SMOKE TEST PASSED` (only if GPU op succeeded) or a clear diagnostic if GPU is unavailable.

All checklist boxes at the bottom must pass.

---

## Task 1 — Create repo structure
Create exactly this layout (empty `__init__.py` where noted; placeholder modules just contain a docstring + `pass` for now):

```
hyperbolic-dim-study/
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml            # minimal, for editable install of `src`
├── configs/
│   └── default.yaml          # global config (seeds, paths, device)
├── data/                     # gitignored; datasets land here later
│   └── .gitkeep
├── outputs/                  # gitignored; runs/checkpoints/figures later
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── smoke.py              # the smoke test (implement fully this stage)
│   ├── config.py             # load YAML config + set seeds + pick device
│   ├── geometry.py           # docstring + pass (Stage 2)
│   ├── data.py               # docstring + pass (Stage 1)
│   ├── trainer.py            # docstring + pass (Stage 3)
│   └── metrics.py            # docstring + pass (Stage 4)
└── tests/
    ├── __init__.py
    └── test_smoke.py         # pytest that imports src and checks config loads
```

## Task 2 — .gitignore
Include at least: `venv/`, `__pycache__/`, `*.pyc`, `data/`, `outputs/`, `.pytest_cache/`, `*.egg-info/`, `.DS_Store`.

## Task 3 — requirements.txt + install instructions
Create `requirements.txt` WITHOUT torch pinned to a plain version (torch is installed separately from the cu128 channel). Include:
```
geoopt>=0.5.0
nltk>=3.8
networkx>=3.0
numpy>=1.24
scipy>=1.10
pyyaml>=6.0
matplotlib>=3.7
pandas>=2.0
tqdm>=4.65
pytest>=7.0
```
Put the torch install as an explicit documented command in README (Task 5), NOT in requirements.txt, because it needs the special index URL.

## Task 4 — Set up the environment (run these; capture output)
```bash
cd hyperbolic-dim-study
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# PyTorch for Blackwell/sm_120 — try STABLE cu128 first:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then the rest:
pip install -r requirements.txt

# editable install of our package
pip install -e .
```
**Fallback if the stable cu128 wheel does not exist for this Python version, or if the smoke test in Task 6 reports sm_120 NOT supported:** uninstall torch and install the nightly cu128 build:
```bash
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
If neither yields sm_120 support, STOP and report the exact torch version, python version, and the `torch.cuda.get_arch_list()` output so the human can decide. Do not silently proceed on CPU.

## Task 5 — README.md
Document: project one-liner, the exact env setup commands above (including the Blackwell note and the nightly fallback), how to run the smoke test (`python -m src.smoke`), and how to run tests (`pytest -q`).

## Task 6 — Implement src/config.py
- Function `load_config(path="configs/default.yaml") -> dict`.
- Function `set_seed(seed: int)` that seeds `random`, `numpy`, and `torch` (+ `torch.cuda`), and sets `torch.backends.cudnn.deterministic = True`, `benchmark = False`.
- Function `get_device(prefer_cuda=True) -> torch.device` returning `cuda` if available else `cpu`.
- `configs/default.yaml` contains at least:
  ```yaml
  seed: 42
  device: cuda
  paths:
    data_dir: data
    output_dir: outputs
  ```

## Task 7 — Implement src/smoke.py (this is the deliverable)
It must, in order:
1. Print Python version.
2. Print `torch.__version__` and `torch.version.cuda`.
3. Print `torch.cuda.is_available()`.
4. Print `torch.cuda.get_arch_list()` (the compiled arch list — MUST contain `sm_120` for our GPU).
5. If CUDA available: print `torch.cuda.get_device_name(0)` and `torch.cuda.get_device_capability(0)` (expected `(12, 0)`).
6. Create two random tensors, move them to the device from `get_device()`, multiply them, and `.sum()` — confirming a real GPU op runs without the sm_120 warning/error.
7. Import `geoopt`; print its version; create a `geoopt.PoincareBall()` manifold; take two points, project them into the ball, and compute their distance with the manifold's `.dist()` — print the value.
8. Determine pass/fail:
   - If CUDA available AND `sm_120` in arch list AND the GPU tensor op succeeded AND geoopt dist computed → print `SMOKE TEST PASSED`.
   - Else → print `SMOKE TEST FAILED` plus a one-line reason and the specific remediation (e.g. "install nightly cu128").

Wrap the GPU section so that a raw sm_120 incompatibility is caught and reported as FAILED with the remediation, not as an uncaught traceback.

## Task 8 — tests/test_smoke.py
A minimal pytest that:
- imports `src.config`, calls `load_config()`, asserts `seed` and `paths` keys exist.
- calls `set_seed(0)` and `get_device()` without error.
(Do not require a GPU in the pytest itself, so CI/CPU won't fail — GPU verification lives in `smoke.py`.)

---

## Definition of Done — checklist (all must be true)
- [ ] Repo structure created exactly as specified.
- [ ] `venv` created; torch installed from a cu128 channel; requirements installed; `pip install -e .` succeeds.
- [ ] `python -m src.smoke` runs and prints all 8 items.
- [ ] `torch.cuda.is_available()` is `True`.
- [ ] `torch.cuda.get_arch_list()` contains `sm_120`.
- [ ] `torch.cuda.get_device_capability(0)` returns `(12, 0)`.
- [ ] The GPU tensor multiply runs with NO sm_120 incompatibility warning.
- [ ] geoopt imports and computes a Poincaré distance.
- [ ] Final line is `SMOKE TEST PASSED`.
- [ ] `pytest -q` passes.

## Report back to the human
After completing, report: the installed torch version + cuda tag, the printed arch list, the GPU name + capability, the geoopt distance value printed, and confirmation that `SMOKE TEST PASSED`. If anything failed, report the exact error and which fallback (if any) was attempted.
```
