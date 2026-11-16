# Python companion to Chapter 1, "A Review of Linear Mixed Models"

An unofficial Python companion to the March 2025 draft of Chapter 1 of the second edition of
*Genetic Data Analysis for Plant and Animal Breeding* by **Fikret Isik, James B. Holland and Christian Maltecca**
(North Carolina State University; first edition Springer, 2017). All concepts, examples and data are the authors'.
The authors share the draft chapters for personal research use and welcome feedback.

The companion re-implements every SAS, R and ASReml-R analysis in the chapter in Python. It checks each reproducible
number against the draft and documents the corrections it found.

| Deliverable | Where |
|---|---|
| Companion PDF (narrative, imports per section, original and Python code, outputs, margin corrections, appendices) | `GDAB2e-py-01_LinearMixedModels_PythonCompanion.pdf` |
| Annotated, executed notebooks | `notebooks/` |
| Custom package (sparse REML, SAS-style ANOVA/EMS, MME) | `src/lmmtools/` |
| Unit tests based on the chapter's examples | `tests/` |
| Benchmarks (statsmodels vs pyreml vs custom) | `benchmarks/` |
| Data download and verification | `scripts/get_data.py`, `src/lmmtools/data.py` |
| Corrections register | PDF Appendix D, `build/corrections.py` |

## Contents

| Notebook | Chapter pages | Topic |
|---|---|---|
| `00_data_sources.ipynb` | — | Appendix A: data sources, verification, items to request from the authors |
| `01_anova_vs_mixed_models.ipynb` | 3–12 | Maize RIL data; ANOVA (SAS GLM, R `lm`, ASReml); EMS and random-model F-tests; moments; LS-means |
| `02_mixed_models_balanced.ipynb` | 12–23 | REML (SAS MIXED, lme4, ASReml); SEs; profile CIs; LRT; AIC/BIC; BLUE vs BLUP; shrinkage |
| `03_unbalanced_data.ipynb` | 23–28 | Unbalanced data: Type I/III, EMS, synthesised tests, non-estimable LS-means, REML, BLUPs |
| `04_theory_lactation.ipynb` | 28–45 | Model theory, fixed vs random, lactation example, MME, estimability, PEV/reliability, REML |
| `05_matrix_algebra.ipynb` | 45–59 | Matrix appendix in NumPy/SciPy |
| `06_appendix_B_alternatives.ipynb` | — | Appendix B: statsmodels, pyreml, lmmtools, pymer4, Bambi, scikit-learn, and when to use each |
| `07_appendix_C_tests_benchmarks.ipynb` | — | Appendix C: custom-code inventory, unit-test results, benchmarks |

## Setup with uv

[uv](https://docs.astral.sh/uv/) manages the Python version and the virtual environment. The same steps work on every
platform; only the uv installer and the venv activation command differ.

### 1. Install uv

**Linux and macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv   (macOS/Homebrew)     or: pipx install uv
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# or: winget install --id=astral-sh.uv -e
```
Open a new terminal afterwards so `uv` is on your PATH, then check with `uv --version`.

### 2. Create the environment and install the requirements

Run these from the project folder (the one that holds this README).

**Linux**
```bash
uv venv --python 3.12
# Optional but recommended: CPU-only PyTorch (pyreml's backend) instead of the ~5 GB CUDA build.
# Skip this line if you have an NVIDIA GPU and want pyreml on CUDA.
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
source .venv/bin/activate
```

**macOS** (Apple silicon or Intel; PyPI's PyTorch wheels are already CPU/MPS builds)
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
uv venv --python 3.12
uv pip install -r requirements.txt          # PyPI's Windows PyTorch wheel is CPU-only
.venv\Scripts\activate
```
If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
You can also skip activation and prefix commands with `uv run --no-project`, for example
`uv run --no-project python scripts/get_data.py`.

`requirements.txt` ends with `-e .`, which installs the custom `lmmtools` package (from `src/`) in editable mode.

### 3. Get the data

```bash
python scripts/get_data.py
```
This downloads `MaizeRILs.csv` and `MaizeRIL_miss.csv` into `./data` (or `$LMM_DATA_DIR`) and verifies their SHA-256
digests. See **Data** below.

### 4. Register the Jupyter kernel and open the notebooks

```bash
python -m ipykernel install --user --name lmm-companion --display-name "LMM companion (uv)"
jupyter lab notebooks/
```
The notebooks expect to run from the project folder: Jupyter Lab started there does this automatically.

### 5. Run the tests and the benchmark

```bash
pytest                       # all tests (about 1-2 minutes); data-dependent tests skip if data are missing
pytest -m "not slow"         # skip the statsmodels/pyreml cross-checks
python benchmarks/bench_reml.py   # REML engines on growing data sets; can take 1-2 hours
```

### 6. Rebuild the notebooks and the PDF (optional)

```bash
uv pip install -r requirements-build.txt
python -m playwright install chromium
python build/build_nb.py     # executes build/src/*.txt into notebooks/*.ipynb
python build/build_pdf.py    # renders GDAB2e-py-01_LinearMixedModels_PythonCompanion.pdf
```
On Linux, `playwright install --with-deps chromium` also installs the system libraries Chromium needs.

## Data

| File | Status | Source |
|---|---|---|
| `MaizeRILs.csv` (balanced, 496 plots) | Openly available | The authors' sample-data folder, linked from F. Isik's book page at NC State (*faculty.cnr.ncsu.edu/fikretisik*). Public mirror used by `get_data.py`: `github.com/malachycampbell/StatisticalMethodsforOmicsAssistedBreeding` (`OLS.MM/MaizeRILs.csv`). |
| `MaizeRIL_miss.csv` (unbalanced, 474 plots) | Openly available | Authors' folder (as above). Public mirror: `github.com/Robbins-Lab/PLSCI-7201` (`Modules/Module8/ProblemSets/MaizeRILs2.csv`). |
| Lactation example, matrix examples | Printed in the chapter | Typed into the notebooks and tests. |

Both mirrors reproduce every statistic the chapter prints for these data: totals, Type I/III sums of squares and REML
estimates. They are pinned by SHA-256. The data belong to the authors and are not redistributed here.

**To request from the authors:** confirmation that the mirrored files match the originals; the `fs` data frame and
the variables `x1`, `x2` used in the ML examples (pp. 18–19); the cited code files (`example2a.r`, `Example2B.sas`,
`Code02-1_ANOVA with SAS GLM procedure.sas`); and the code for the `NoRIL`/`NoGxE`/`Full` fits behind `lrt.asreml()`
(p. 20). Details are in Appendix A and `notebooks/00_data_sources.ipynb`.

## Custom code

Only `src/lmmtools` is custom. Everything else uses NumPy, SciPy, pandas, statsmodels, scikit-learn and pyreml.

* `reml.SparseREML`: REML/ML via the sparse mixed model equations. SEs come from observed information (SAS COVTEST) or
  average information (ASReml); it also gives profile-likelihood CIs. In the benchmark it fitted 38 000 records in about 5.5 minutes on 2 CPU cores; 95 000 records exceeded the 15-minute limit.
* `anova.FactorialANOVA`: SAS PROC GLM conventions. It gives Type I/III SS with rank-based df, Type III EMS (Hartley),
  synthesised F-tests (Satterthwaite), moment estimates and LS-means with estimability checks.
* `mme.solve_mme`: Henderson's MME with known variances, giving BLUE, BLUP, PEV, SEP and reliability.

Design notes, unit-test results and benchmarks are in PDF Appendix C and `notebooks/07_appendix_C_tests_benchmarks.ipynb`.

## Corrections to the draft

The companion found 23 corrections (C1–C23) and 4 software notes (N1–N4). They are highlighted in the PDF margin and
listed in Appendix D. The numerical ones are verified by tests, for example C6 (the unbalanced moment estimate of σ²(RIL)
is 309.04, not 304.84) and C19 (the eigenvector example).

## Licence

The companion's code is provided for study alongside the chapter. pyreml is GPL-3.0; statsmodels, scikit-learn,
NumPy, SciPy and pandas are BSD-licensed.
