# `GDAB2e-py`: Python Companion to *Genetic Data Analysis for Plant and Animal Breeding* (2nd Edition)

An unofficial Python companion to the second edition of *Genetic Data Analysis for Plant and Animal Breeding*
by **Fikret Isik, James B. Holland and Christian Maltecca** (North Carolina State University). The second edition is
in preparation, and the authors share draft chapters for personal research use on
[the second-edition page](https://faculty.cnr.ncsu.edu/fikretisik/second-edition-genetic-data-analysis-for-animal-and-plant-breeding/).
The first edition was published by Springer in 2017.

The book's analyses use **SAS**, **R** and **ASReml/ASReml-R**. This repository re-implements them, chapter by
chapter, in open-source Python (NumPy, SciPy, pandas, statsmodels, scikit-learn, pyreml). It checks every reproducible
number against the draft text and documents the corrections it finds.

> **Not an official product.** All concepts, examples, data and interpretations belong to the book's authors.
> This companion has not been reviewed or endorsed by them. Read it alongside the draft chapters, not instead of them.

## Chapters

| # | Draft chapter | Folder | Status | Python companion PDF |
|---|---|---|---|---|
| 1 | A Review of Linear Mixed Models (draft, March 2025) | [`01_LinearMixedModels/`](01_LinearMixedModels/) | Complete | [PDF, 57 pp.](01_LinearMixedModels/GDAB2e-py-01_LinearMixedModels_PythonCompanion.pdf) |

Later chapters will be added as the authors release their drafts.

### Chapter 1 at a glance

* **Topics.** ANOVA versus mixed models, expected mean squares and corrected F-tests, REML, likelihood-ratio
  tests, BLUE and BLUP, unbalanced data, the mixed model equations, estimability, prediction error variance, and the
  chapter's matrix-algebra appendix.
* **Verification.** Eight executed notebooks make 169 checks against values printed in the chapter. There are 33 unit
  tests, and the results agree with the chapter's SAS PROC GLM/MIXED, lme4 and ASReml-R output.
* **Corrections.** The PDF margin marks 23 corrections to the draft and 4 software notes, collected in
  Appendix D. Examples: the unbalanced moment estimate of σ²(RIL) is 309.04, not 304.84, and the eigenvector example in
  the matrix appendix is corrected.
* **Custom code.** `lmmtools` adds features Python lacks: SAS-style Type III ANOVA with expected mean squares and
  synthesised F-tests, sparse REML with SAS/ASReml-style standard errors and profile confidence intervals, and a mixed
  model equations solver. It is documented, unit-tested and benchmarked against statsmodels and pyreml (Appendix C).
* **Alternatives.** Appendix B shows when to use statsmodels, pyreml, `lmmtools`, pymer4, Bambi or scikit-learn,
  with alternative code for small and large data sets.

## Repository layout

Each chapter is a self-contained project with its own environment, data script, tests and PDF:

```
GDAB2e-py/
├── README.md                          ← this file
└── 01_LinearMixedModels/
    ├── README.md                      ← chapter setup and details
    ├── GDAB2e-py-01_..._PythonCompanion.pdf
    ├── notebooks/                     ← annotated, executed Jupyter notebooks (00-07)
    ├── src/lmmtools/                  ← custom package (installed in editable mode)
    ├── tests/                         ← pytest suite based on the chapter's examples
    ├── benchmarks/                    ← REML engine benchmark script and results
    ├── scripts/get_data.py            ← downloads and SHA-256-verifies the chapter's data
    ├── build/                         ← sources that regenerate the notebooks and the PDF
    ├── requirements.txt               ← runtime + notebook + test dependencies
    ├── requirements-build.txt         ← extra dependencies to rebuild the PDF
    └── pyproject.toml
```

## Quick start

Each chapter uses [uv](https://docs.astral.sh/uv/) and Python 3.12. Install uv first:

* **Linux / macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`)
* **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Then, from the repository root:

```bash
cd 01_LinearMixedModels
uv venv --python 3.12
uv pip install -r requirements.txt               # Linux: see the chapter README for CPU-only PyTorch first
uv run --no-project python scripts/get_data.py   # fetch and verify the data
uv run --no-project pytest                       # run the tests
uv run --no-project jupyter lab notebooks/       # open the notebooks
```

The chapter README covers platform-specific details (venv activation on Windows, the CPU-only PyTorch
wheel for pyreml on Linux), registering the Jupyter kernel, running the benchmark and rebuilding the PDF.

## Data

The data belong to the book's authors and are **not** stored in this repository. Each chapter's
`scripts/get_data.py` downloads verified copies and pins their SHA-256 digests. Each chapter's Appendix A (and
`notebooks/00_data_sources.ipynb`) lists the sources and the items to request from the authors. Please obtain the
data from the authors' book page for any use beyond personal study.

## Conventions

* **Narrative.** Text is a condensed restatement in each chapter's order, never a verbatim copy. The original SAS,
  R and ASReml code is quoted only where it is translated.
* **Imports.** Each section opens with the Python modules it uses.
* **Checks.** `check(label, python_value, chapter_value, tol)` asserts agreement with a printed result. A notebook
  that finishes has passed all of its checks.
* **Corrections.** Errors in a draft are marked in the text, explained in the PDF margin and listed in a
  register (`build/corrections.py`), with the test or check that verifies each one.
* **Custom code** is limited to each chapter's package. It is labelled `CUSTOM` where imported and is unit-tested
  and benchmarked.

## Feedback to the authors

The authors invite feedback on the drafts. Each chapter's corrections register (PDF Appendix D) is written so it
can be sent to them as it stands. Issues found in this companion itself are welcome as GitHub issues.

## Citation

Please cite the book for any concept, example or result:

> Isik, F., Holland, J. B. and Maltecca, C. *Genetic Data Analysis for Plant and Animal Breeding*, 2nd edition
> (in preparation; draft chapters 2025–). First edition: Springer, Cham, 2017.

If the Python code is useful, cite this repository as well:
[github.com/NearIdentity/GDAB2e-py](https://github.com/NearIdentity/GDAB2e-py).

## Licensing

The book text, examples and data are © their authors. The companion code is provided for study alongside the
book. The dependencies carry their own licences: pyreml is GPL-3.0, while NumPy, SciPy, pandas, statsmodels and
scikit-learn are BSD-licensed.
