"""Least-squares ANOVA for (possibly unbalanced) factorial models, reproducing SAS PROC GLM.

CUSTOM CODE (companion Appendix C). statsmodels' anova_lm gives Type I/II/III sums of squares, but
(i) its Type III with empty cells is not SAS's Type III, (ii) degrees of freedom are column counts
rather than ranks when cells are empty, and (iii) it has no expected-mean-square (EMS) machinery for
random effects. This module follows SAS's computing method (Goodnight 1978):

* indicator (non-full-rank) coding of every effect, in model order;
* the sweep "g2" generalized inverse G of X'X and the Hermite form H = G X'X, whose rows are the
  general form of estimable functions;
* Type I SS: sequential projections; Type III SS: for each effect, the rows of H for that effect made
  orthogonal (Euclidean, on their coefficients) to the rows of every effect that contains it;
* Type III EMS coefficients by Hartley's synthesis: for SS = y'Qy, coef(s2_k) = tr(Z_k'QZ_k) / df;
* synthesised error terms with Satterthwaite df ("RANDOM ... / TEST");
* method-of-moments variance components; LS-means with an estimability check ("Non-est").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def sweep_ginv(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """SAS-style g2 generalized inverse of a symmetric matrix by sweeping pivots in order; columns
    that are linearly dependent on earlier ones get zero rows/columns."""
    A = np.array(A, dtype=float)
    k = A.shape[0]
    d0 = np.diag(A).copy()
    keep = np.zeros(k, bool)
    for j in range(k):
        if abs(A[j, j]) <= tol * max(d0[j], 1.0):
            continue
        piv = A[j, j]
        r, c = A[j].copy(), A[:, j].copy()
        A -= np.outer(c, r) / piv
        A[j], A[:, j] = r / piv, c / piv
        A[j, j] = -1.0 / piv
        keep[j] = True
    G = np.zeros_like(A)
    idx = np.where(keep)[0]
    G[np.ix_(idx, idx)] = -A[np.ix_(idx, idx)]
    return G


def _rank_rows(L: np.ndarray, tol=1e-8) -> np.ndarray:
    if L.size == 0:
        return L
    _, s, Vt = np.linalg.svd(L, full_matrices=False)
    return Vt[: int(np.sum(s > tol * s.max()))]


def _indicator(df: pd.DataFrame, factors: tuple[str, ...], levels=None):
    key = df[list(factors)].astype(str).agg(":".join, axis=1)
    if levels is None:
        levels = sorted(key.unique())
    M = (key.to_numpy()[:, None] == np.asarray(levels)[None, :]).astype(float)
    return M, list(levels)


@dataclass
class Term:
    name: str
    factors: tuple[str, ...]       # all factors in the term, e.g. ('location', 'rep') for rep(location)


class FactorialANOVA:
    """Fixed-effects ANOVA y ~ terms (indicator coding, SAS PROC GLM conventions).

    terms: ordered dict  name -> tuple of factor names, e.g.
        {"location": ("location",), "rep(location)": ("location", "rep"),
         "RIL": ("RIL",), "location*RIL": ("location", "RIL")}
    A term contains another if its factor set is a strict superset (rep(location) contains location).
    """

    def __init__(self, df: pd.DataFrame, response: str, terms: dict):
        self.df = df.reset_index(drop=True)
        self.y = self.df[response].to_numpy(float)
        self.n = len(self.y)
        self.terms = [Term(k, tuple(v)) for k, v in terms.items()]
        blocks, self.levels, self.cols, c0 = [np.ones((self.n, 1))], {}, {}, 1
        for t in self.terms:
            M, lev = _indicator(self.df, t.factors)
            blocks.append(M); self.levels[t.name] = lev
            self.cols[t.name] = np.arange(c0, c0 + M.shape[1]); c0 += M.shape[1]
        self.X = np.column_stack(blocks)
        self.XtX = self.X.T @ self.X
        self.G = sweep_ginv(self.XtX)
        self.H = self.G @ self.XtX                     # Hermite form: general form of estimable functions
        self.b = self.G @ self.X.T @ self.y
        self.rank = int(round(np.trace(self.H)))
        resid = self.y - self.X @ self.b
        self.dfe = self.n - self.rank
        self.sse = float(resid @ resid)
        self.mse = self.sse / self.dfe
        self._t3 = None

    def contains(self, outer: Term, inner: Term) -> bool:
        return set(inner.factors) < set(outer.factors)

    # ---------------------------------------------------------------- sums of squares
    def _table(self, ss: dict, dfs: dict) -> pd.DataFrame:
        rows = []
        for t in self.terms:
            ms = ss[t.name] / dfs[t.name]
            F = ms / self.mse
            rows.append((t.name, dfs[t.name], ss[t.name], ms, F, stats.f.sf(F, dfs[t.name], self.dfe)))
        rows.append(("Error", self.dfe, self.sse, self.mse, np.nan, np.nan))
        return pd.DataFrame(rows, columns=["Source", "DF", "SS", "MS", "F", "Pr > F"]).set_index("Source")

    def type1(self) -> pd.DataFrame:
        ss, dfs = {}, {}
        prev_cols = [0]
        def fit_ss(cols):
            Xs = self.X[:, cols]
            Gs = sweep_ginv(Xs.T @ Xs)
            fitted = Xs @ (Gs @ Xs.T @ self.y)
            return float(fitted @ fitted), int(round(np.trace(Gs @ Xs.T @ Xs)))
        r_prev = fit_ss(prev_cols)
        for t in self.terms:
            cur = prev_cols + list(self.cols[t.name])
            r_cur = fit_ss(cur)
            ss[t.name], dfs[t.name] = r_cur[0] - r_prev[0], r_cur[1] - r_prev[1]
            prev_cols, r_prev = cur, r_cur
        return self._table(ss, dfs)

    def type3_L(self) -> dict:
        """Type III hypothesis matrices L (rows = estimable functions) for each term."""
        if self._t3 is None:
            L3 = {}
            for t in self.terms:
                L = _rank_rows(self.H[self.cols[t.name]])
                containing = [u for u in self.terms if self.contains(u, t)]
                if containing:
                    L2 = _rank_rows(np.vstack([self.H[self.cols[u.name]] for u in containing]))
                    L = L - (L @ L2.T) @ np.linalg.pinv(L2 @ L2.T) @ L2
                L3[t.name] = _rank_rows(L)
            self._t3 = L3
        return self._t3

    def _Q(self, L):
        """Quadratic-form matrix of the hypothesis SS: SS = y'Qy, Q = X G L'(L G L')^- L G X'."""
        A = self.X @ self.G @ L.T
        return A @ np.linalg.pinv(L @ self.G @ L.T) @ A.T

    def type3(self) -> pd.DataFrame:
        ss, dfs = {}, {}
        for t in self.terms:
            L = self.type3_L()[t.name]
            Lb = L @ self.b
            ss[t.name] = float(Lb @ np.linalg.pinv(L @ self.G @ L.T) @ Lb)
            dfs[t.name] = L.shape[0]
        return self._table(ss, dfs)

    # ---------------------------------------------------------------- random-model machinery
    def ems_type3(self, random: list[str] | None = None) -> pd.DataFrame:
        """Type III expected-mean-square coefficients for the random terms (default: all terms)."""
        random = random or [t.name for t in self.terms]
        Z = {t.name: self.X[:, self.cols[t.name]] for t in self.terms if t.name in random}
        rows = {}
        for t in self.terms:
            L = self.type3_L()[t.name]
            Q = self._Q(L)
            rows[t.name] = {k: float(np.sum(Zk * (Q @ Zk))) / L.shape[0] for k, Zk in Z.items()}
        ems = pd.DataFrame(rows).T
        ems["Error"] = 1.0
        return ems

    def random_tests(self, random: list[str] | None = None) -> pd.DataFrame:
        """PROC GLM 'RANDOM ... / TEST': synthesised denominators with Satterthwaite df."""
        t3, ems = self.type3(), self.ems_type3(random)
        comps = list(ems.columns)
        rows = []
        for t in self.terms:
            nm = t.name
            target = ems.loc[nm, comps].copy()
            if nm in comps:
                target[nm] = 0.0
            cand = [u.name for u in self.terms if u.name != nm and (nm not in comps or abs(ems.loc[u.name, nm]) < 1e-10)]
            A = np.column_stack([ems.loc[u, comps].to_numpy() for u in cand] + [np.eye(len(comps))[-1]])
            c = np.linalg.lstsq(A, target.to_numpy(), rcond=None)[0]
            c[np.abs(c) < 1e-10] = 0.0
            names = cand + ["Error"]
            ms = np.array([t3.loc[u, "MS"] for u in names]); dfs = np.array([t3.loc[u, "DF"] for u in names])
            ms_syn = float(c @ ms)
            df_syn = ms_syn**2 / np.sum((c * ms) ** 2 / dfs)
            F = t3.loc[nm, "MS"] / ms_syn
            desc = " + ".join(f"{ci:.4f}*MS({u})" for ci, u in zip(c, names) if ci != 0).replace("+ -", "- ")
            rows.append((nm, t3.loc[nm, "DF"], t3.loc[nm, "MS"], df_syn, ms_syn, F,
                         stats.f.sf(F, t3.loc[nm, "DF"], df_syn), desc))
        return pd.DataFrame(rows, columns=["Source", "DF", "MS", "Error DF", "Error MS", "F", "Pr > F",
                                           "Error term"]).set_index("Source")

    def moments(self, random: list[str] | None = None) -> pd.Series:
        """Method-of-moments variance components from Type III mean squares (E[MS] = MS)."""
        t3, ems = self.type3(), self.ems_type3(random)
        comps = list(ems.columns)
        A = np.vstack([ems.loc[t.name, comps].to_numpy() for t in self.terms] + [np.eye(len(comps))[-1]])
        rhs = np.r_[[t3.loc[t.name, "MS"] for t in self.terms], self.mse]
        return pd.Series(np.linalg.lstsq(A, rhs, rcond=None)[0], index=comps)

    # ---------------------------------------------------------------- LS-means
    def lsmeans(self, term: str) -> pd.DataFrame:
        """LS-means for a main-effect term: intercept + that level + averages over the levels of every
        other effect (containing effects averaged within the level), as SAS LSMEANS. Non-estimable
        LS-means (L H != L) are returned as NaN, which SAS prints as 'Non-est'."""
        t = next(u for u in self.terms if u.name == term)
        out = []
        for j, lev in enumerate(self.levels[term]):
            L = np.zeros(self.X.shape[1]); L[0] = 1.0
            for u in self.terms:
                cols = self.cols[u.name]
                if u.name == term:
                    L[cols[j]] = 1.0
                elif self.contains(u, t):
                    # average over the full cross of the other factors (including empty cells -> Non-est)
                    pos = u.factors.index(t.factors[0])
                    other = [f for f in u.factors if f != t.factors[0]]
                    n_other = int(np.prod([self.df[f].nunique() for f in other]))
                    for c, lv in zip(cols, self.levels[u.name]):
                        if lv.split(":")[pos] == lev:
                            L[c] = 1.0 / n_other
                    present = sum(1 for lv in self.levels[u.name] if lv.split(":")[pos] == lev)
                    if present < n_other:
                        L[0] = np.nan                       # a needed cell is empty
                else:
                    L[cols] = 1.0 / len(cols)
            est = np.isfinite(L).all() and np.allclose(L @ self.H, L, atol=1e-8)
            se = np.sqrt(L @ self.G @ L * self.mse) if est else np.nan
            out.append((lev, float(L @ self.b) if est else np.nan, se, est))
        return pd.DataFrame(out, columns=[term, "LSMEAN", "SE", "estimable"]).set_index(term)
