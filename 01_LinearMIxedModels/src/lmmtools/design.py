"""Design-matrix helpers: indicator (incidence) matrices for factors and their interactions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def factor_key(df: pd.DataFrame, factors: str | list[str]) -> pd.Series:
    """Combine one or more columns into a single label, e.g. ['location', 'rep'] -> 'ARC:1'."""
    if isinstance(factors, str):
        factors = [factors]
    return df[factors].astype(str).agg(":".join, axis=1)


def incidence(df: pd.DataFrame, factors: str | list[str], sparse: bool = True):
    """Incidence matrix with one column per *observed* level combination.

    Returns (Z, levels). Z is scipy.sparse CSC if sparse=True, else a dense ndarray.
    """
    key = factor_key(df, factors)
    codes, levels = pd.factorize(key, sort=True)
    n, q = len(df), len(levels)
    Z = sp.csc_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, q))
    return (Z if sparse else Z.toarray()), list(levels)
