"""lmmtools: custom code for the Python companion to Chapter 1, "A Review of Linear Mixed Models"
(Isik, Holland & Maltecca, draft 2025). See Appendix C of the companion PDF.

Modules: reml (sparse REML/ML, SEs, LRT, profile CIs), anova (SAS-style ANOVA, EMS, random-model tests,
LS-means), mme (mixed model equations with known variances), design (incidence matrices),
data (download and verification of the chapter's data), report (checks against the chapter).
"""
__version__ = "0.1.0"

from .anova import FactorialANOVA, sweep_ginv
from .mme import solve_mme
from .reml import REMLFit, SparseREML, boundary_lrt

__all__ = ["FactorialANOVA", "sweep_ginv", "solve_mme", "SparseREML", "REMLFit", "boundary_lrt", "__version__"]
