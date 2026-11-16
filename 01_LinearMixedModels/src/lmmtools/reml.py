"""Sparse REML / ML for linear mixed models with independent (iid) random factors.

CUSTOM CODE (companion Appendix C).  Model:

    y = X b + sum_k Z_k u_k + e,   u_k ~ N(0, I s2_k),   e ~ N(0, I s2_e).

The likelihood is evaluated through Henderson's mixed model equations (MME) on the scaled design
Z*_k = Z_k sqrt(g_k), g_k = s2_k / s2_e, so that zero variances are allowed:

    C* = [[X'X, X'Z*], [Z*'X, I + Z*'Z*]],   rhs = [X'y; Z*'y],   Q = y'y - sol'rhs,

    REML:  -2 logL_R = (n - p) log(2 pi s2_e) + log|C*| + Q / s2_e
    ML:    -2 logL   =  n      log(2 pi s2_e) + log|C*| - log|X'H^-1 X| + Q / s2_e,  H = I + Z*Z*'

Each evaluation needs one sparse LU factorisation of C* (never the dense n x n matrix V). The
residual variance is profiled out analytically; the variance ratios g_k = t_k^2 are optimised
without bounds by BFGS (central-difference gradients) and polished by Newton steps.

Standard errors of variance components:
    se="observed"  exact observed information (dense traces; SAS PROC MIXED COVTEST), n <~ 5000
    se="ai"        average information (ASReml's algorithm); sparse solves only, any n
    se=None        skip
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import optimize, stats

LOG2PI = np.log(2 * np.pi)


@dataclass
class REMLFit:
    names: list[str]
    sigma2: np.ndarray            # random-factor variances (order of names)
    sigma2_e: float
    neg2_loglik: float            # -2 log (restricted) likelihood; SAS "-2 Res Log Likelihood"
    beta: np.ndarray
    beta_se: np.ndarray
    method: str = "reml"
    blups: dict = field(default_factory=dict)
    blup_se: dict = field(default_factory=dict)   # SE of prediction, sqrt(PEV)
    se: np.ndarray | None = None  # SEs of (sigma2..., sigma2_e)
    se_method: str | None = None
    n: int = 0
    p: int = 0
    n_evals: int = 0

    @property
    def theta(self):
        return np.r_[self.sigma2, self.sigma2_e]

    def table(self):
        import pandas as pd
        out = pd.DataFrame({"Estimate": self.theta}, index=self.names + ["Residual"])
        if self.se is not None:
            out["Std Error"] = self.se
            out["Z value"] = self.theta / self.se
            out["Pr Z"] = stats.norm.sf(out["Z value"])          # one-sided, as SAS COVTEST
        return out

    @property
    def n_cov(self):
        return len(self.sigma2) + 1

    @property
    def aic(self):
        """SAS PROC MIXED convention: -2 logL + 2 * (number of covariance parameters) for REML;
        fixed effects are counted as well for ML."""
        k = self.n_cov + (self.p if self.method == "ml" else 0)
        return self.neg2_loglik + 2 * k

    def bic(self, nu=None):
        """BIC = -2 logL + t log(nu); ASReml-R uses nu = residual df (n - p) for REML (eq. 1.6)."""
        k = self.n_cov + (self.p if self.method == "ml" else 0)
        nu = (self.n - self.p if self.method == "reml" else self.n) if nu is None else nu
        return self.neg2_loglik + k * np.log(nu)


class SparseREML:
    """Fit y ~ X b + sum Z_k u_k + e by REML (default) or ML using sparse MME."""

    def __init__(self, y, X, Z_list, names=None):
        self.y = np.asarray(y, float)
        X = np.asarray(X, float)
        _, r = np.linalg.qr(X)                 # keep a full-column-rank X (drop aliased columns)
        d = np.abs(np.diag(r))
        self.X = X[:, d > 1e-8 * d.max()]
        self.Z = [sp.csc_matrix(Z, dtype=float) for Z in Z_list]
        self.names = list(names) if names is not None else [f"u{k}" for k in range(len(self.Z))]
        self.n, self.p = self.X.shape
        self.sizes = [Z.shape[1] for Z in self.Z]
        self.Zall = sp.hstack(self.Z, format="csc")
        Xs = sp.csc_matrix(self.X)
        self.XtX = self.X.T @ self.X
        self.XtZ = (Xs.T @ self.Zall).tocsc()
        self.ZtZ = (self.Zall.T @ self.Zall).tocsc()
        self.Xty = self.X.T @ self.y
        self.Zty = self.Zall.T @ self.y
        self.yty = float(self.y @ self.y)
        self.n_evals = 0

    # ------------------------------------------------------------------ MME factorisation
    def _mme(self, g):
        g = np.maximum(np.asarray(g, float), 0.0)
        s = np.repeat(np.sqrt(g), self.sizes)
        S = sp.diags(s)
        q = self.Zall.shape[1]
        C = sp.bmat([[sp.csc_matrix(self.XtX), self.XtZ @ S],
                     [S @ self.XtZ.T, sp.identity(q) + S @ self.ZtZ @ S]], format="csc")
        rhs = np.r_[self.Xty, s * self.Zty]
        lu = spla.splu(C, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options={"SymmetricMode": True})
        sol = lu.solve(rhs)
        logdet = np.log(np.abs(lu.U.diagonal())).sum() + np.log(np.abs(lu.L.diagonal())).sum()
        self.n_evals += 1
        return logdet, self.yty - sol @ rhs, sol, lu, s

    def _cinv_bb(self, lu):
        p, dim = self.p, lu.shape[0]
        E = np.zeros((dim, p)); E[:p, :p] = np.eye(p)
        return lu.solve(E)[:p, :]

    def _dfe(self, method):
        return self.n - self.p if method == "reml" else self.n

    def neg2ll_profiled(self, g, method="reml"):
        logdet, Q, sol, lu, s = self._mme(g)
        dfe = self._dfe(method)
        out = dfe * (LOG2PI + np.log(Q / dfe) + 1.0) + logdet
        if method == "ml":
            out += np.linalg.slogdet(self._cinv_bb(lu))[1]       # - log|X'H^-1 X|
        return out

    def neg2ll(self, sigma2, sigma2_e, method="reml"):
        """-2 log (restricted) likelihood at given variances."""
        logdet, Q, sol, lu, s = self._mme(np.asarray(sigma2, float) / sigma2_e)
        out = self._dfe(method) * (LOG2PI + np.log(sigma2_e)) + logdet + Q / sigma2_e
        if method == "ml":
            out += np.linalg.slogdet(self._cinv_bb(lu))[1]
        return out

    # ------------------------------------------------------------------ fitting
    def fit(self, method="reml", start=None, se="observed", gtol=1e-9):
        k = len(self.Z)
        f = lambda tt: self.neg2ll_profiled(tt**2, method)
        t = np.ones(k) if start is None else np.sqrt(np.asarray(start, float))
        t = optimize.minimize(f, t, jac=lambda tt: _grad(f, tt), method="BFGS",
                              options={"gtol": gtol, "maxiter": 10_000}).x
        for _ in range(20):                                       # Newton polishing
            step = np.linalg.lstsq(_hessian(f, t), _grad(f, t), rcond=None)[0]
            if not np.all(np.isfinite(step)) or np.max(np.abs(step)) < 1e-10 or f(t - step) > f(t) + 1e-12:
                break
            t = t - step
        g = t**2
        logdet, Q, sol, lu, s = self._mme(g)
        s2e = Q / self._dfe(method)
        sigma2 = g * s2e
        p = self.p
        beta = sol[:p]
        beta_se = np.sqrt(np.diag(self._cinv_bb(lu)) * s2e)
        u = s * sol[p:]
        cuts = np.cumsum([0] + self.sizes)
        fit = REMLFit(self.names, sigma2, s2e, self.neg2ll(sigma2, s2e, method), beta, beta_se, method,
                      n=self.n, p=p, n_evals=self.n_evals)
        fit.blups = {nm: u[a:b] for nm, a, b in zip(self.names, cuts[:-1], cuts[1:])}
        if se is not None:
            fit.se, fit.se_method = self.variance_se(sigma2, s2e, se), se
        return fit

    def blup_pev(self, fit: REMLFit, which: str):
        """Prediction error variances of one random factor's BLUPs: s2e * g_k * [C*^-1]_uu (diagonal)."""
        k = self.names.index(which)
        g = fit.sigma2 / fit.sigma2_e
        *_, lu, s = self._mme(g)
        a = self.p + int(np.sum(self.sizes[:k]))
        idx = np.arange(a, a + self.sizes[k])
        E = np.zeros((lu.shape[0], len(idx))); E[idx, np.arange(len(idx))] = 1
        diag = np.einsum("ij,ij->j", E, lu.solve(E))
        return fit.sigma2_e * g[k] * diag

    # ------------------------------------------------------------------ standard errors
    def _P(self, g, s2e):
        """Return a function w -> P w, with P = V^-1 - V^-1 X (X'V^-1X)^-1 X'V^-1, via the MME."""
        *_, lu, s = self._mme(g)
        W = sp.hstack([sp.csc_matrix(self.X), self.Zall @ sp.diags(s)], format="csc")
        def P(w):
            return (w - W @ lu.solve(W.T @ w)) / s2e
        return P

    def variance_se(self, sigma2, s2e, method="observed"):
        theta = np.r_[sigma2, s2e]
        P = self._P(np.asarray(sigma2) / s2e, s2e)
        Py = P(self.y)
        dV = [(Z, None) for Z in self.Z] + [(None, None)]
        # V_k Py for each parameter (V_k = Z_k Z_k' or I)
        VkPy = [Z @ (Z.T @ Py) for Z in self.Z] + [Py]
        if method == "ai":
            PVPy = np.column_stack([P(v) for v in VkPy])
            info = 0.5 * np.column_stack(VkPy).T @ PVPy
        elif method == "observed":
            n = self.n
            Pd = np.column_stack([P(col) for col in np.eye(n)])       # dense P (n x n); small n only
            PV = [Pd @ (Z @ Z.T).toarray() for Z in self.Z] + [Pd]
            PVPy = np.column_stack([P(v) for v in VkPy])
            Vk = np.column_stack(VkPy)
            info = np.array([[-0.5 * np.sum(PV[i] * PV[j].T) + Vk[:, i] @ PVPy[:, j]
                              for j in range(len(theta))] for i in range(len(theta))])
        else:
            raise ValueError("method must be 'observed' or 'ai'")
        cov = np.linalg.pinv(info)
        return np.sqrt(np.clip(np.diag(cov), 0, None))

    # ------------------------------------------------------------------ profile likelihood CIs
    def profile_ci(self, fit: REMLFit, which: int, level=0.95, method=None, scale="sd"):
        """Profile-likelihood CI for parameter `which` (index into names; len(names) = residual).

        method defaults to fit.method. lme4's confint() profiles the ML deviance even for REML fits,
        so method="ml" (with an ML fit) reproduces lme4; method="reml" profiles the REML criterion.
        Returns (lower, upper) on the SD scale (as lme4) or the variance scale.
        """
        method = method or fit.method
        crit = stats.chi2.ppf(level, 1)
        theta_hat, best = fit.theta, fit.neg2_loglik
        k = len(theta_hat)
        others = [i for i in range(k) if i != which]
        state = {"x": np.sqrt(theta_hat[others])}

        def prof(v):
            def f(x):
                tt = np.empty(k); tt[which] = v; tt[others] = x**2
                return self.neg2ll(tt[:-1], tt[-1], method)
            r = optimize.minimize(f, state["x"], jac=lambda x: _grad(f, x), method="BFGS",
                                  options={"gtol": 1e-7, "maxiter": 2000})
            state["x"] = r.x
            return r.fun - best - crit

        v_hat = theta_hat[which]
        lo_end = 1e-10 if which < k - 1 else v_hat * 1e-3
        lower = lo_end if prof(lo_end) < 0 else optimize.brentq(prof, lo_end, v_hat, xtol=1e-7 * v_hat)
        state["x"] = np.sqrt(theta_hat[others])
        hi = 2 * v_hat
        while prof(hi) < 0:
            hi *= 2
        upper = optimize.brentq(prof, v_hat, hi, xtol=1e-7 * v_hat)
        return (np.sqrt(lower), np.sqrt(upper)) if scale == "sd" else (lower, upper)


def boundary_lrt(neg2ll_reduced, neg2ll_full, df=1):
    """LRT for one variance component tested against zero (Self & Liang 1987):
    D = -2logL_reduced - (-2logL_full);  p_adjusted = 0.5 * P(chi2_df > D)."""
    D = neg2ll_reduced - neg2ll_full
    p_raw = stats.chi2.sf(D, df)
    return D, p_raw, 0.5 * p_raw


def _grad(f, x, h=1e-6):
    x = np.asarray(x, float); g = np.empty_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h * max(abs(x[i]), 1e-2)
        g[i] = (f(x + e) - f(x - e)) / (2 * e[i])
    return g


def _hessian(f, x, rel=1e-4):
    x = np.asarray(x, float); k = len(x); H = np.empty((k, k))
    h = rel * np.maximum(np.abs(x), 1e-2)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h[i]; ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H
