"""Henderson's mixed model equations (MME) with known variance components.

Solves  [X'R^-1X  X'R^-1Z; Z'R^-1X  Z'R^-1Z + G^-1] [b; u] = [X'R^-1y; Z'R^-1y]
for R = I*sigma2_e and G = blockdiag(I*sigma2_k), returning BLUEs, BLUPs, prediction error
variances (PEV), standard errors of prediction (SEP) and reliabilities, as in Chapter 1
(equations 1.24-1.33).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class MMESolution:
    blue: np.ndarray            # fixed-effect solutions b
    blup: list[np.ndarray]      # one array of random-effect predictions per random factor
    pev: list[np.ndarray]       # prediction error variances  V(u_hat - u)
    sep: list[np.ndarray]       # standard errors of prediction sqrt(PEV)
    reliability: list[np.ndarray]  # 1 - PEV / sigma2_k
    lhs: np.ndarray             # coefficient matrix C (dense, for teaching / small problems)
    lhs_inv: np.ndarray         # C^-1 (scaled by sigma2_e gives the sampling (co)variances)


def solve_mme(X, Z_list, y, sigma2_u, sigma2_e) -> MMESolution:
    """Solve the MME for known variances. X: n x p (full column rank), Z_list: list of n x q_k."""
    X = np.asarray(X, float)
    Zs = [np.asarray(Z.toarray() if sp.issparse(Z) else Z, float) for Z in Z_list]
    sigma2_u = np.atleast_1d(np.asarray(sigma2_u, float))
    y = np.asarray(y, float)
    Z = np.hstack(Zs)
    sizes = [Zk.shape[1] for Zk in Zs]
    # With R = I*sigma2_e factored out, G^-1 * sigma2_e = diag(alpha_k), alpha_k = sigma2_e / sigma2_k  (eq. 1.25)
    alpha = np.repeat(sigma2_e / sigma2_u, sizes)
    C = np.block([[X.T @ X, X.T @ Z], [Z.T @ X, Z.T @ Z + np.diag(alpha)]])
    rhs = np.concatenate([X.T @ y, Z.T @ y])
    Cinv = np.linalg.inv(C)
    sol = Cinv @ rhs
    p = X.shape[1]
    blue, u = sol[:p], sol[p:]
    pev_all = np.diag(Cinv)[p:] * sigma2_e                    # eq. 1.32: PEV = C^22 * sigma2_e
    cuts = np.cumsum([0] + sizes)
    blups = [u[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    pevs = [pev_all[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    rel = [1 - pv / s2 for pv, s2 in zip(pevs, sigma2_u)]       # eq. 1.33
    return MMESolution(blue, blups, pevs, [np.sqrt(pv) for pv in pevs], rel, C, Cinv)
