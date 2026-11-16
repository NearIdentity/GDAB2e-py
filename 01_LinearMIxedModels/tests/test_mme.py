"""Mixed model equations with known variances: the daughters' lactation yield example (eqs. 1.22-1.33)."""
import numpy as np
from numpy.testing import assert_allclose

from lmmtools.mme import solve_mme


def test_gls_blue_blup_direct(lactation):
    """Eqs. 1.22-1.23 via V^-1: b = (105.64, 104.28, 105.46), u = (0.40, 0.52, 0.76, -1.67)."""
    y, X, Z = lactation
    G, R = np.eye(4) * 0.1, np.eye(9) * 1.0
    V = Z @ G @ Z.T + R
    Vi = np.linalg.inv(V)
    b = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y)
    u = G @ Z.T @ Vi @ (y - X @ b)
    assert_allclose(b, [105.64, 104.28, 105.46], atol=5e-3)
    assert_allclose(u, [0.40, 0.52, 0.76, -1.67], atol=5e-3)


def test_mme_matches_gls(lactation):
    y, X, Z = lactation
    sol = solve_mme(X, [Z], y, sigma2_u=0.1, sigma2_e=1.0)
    assert_allclose(sol.blue, [105.64, 104.28, 105.46], atol=5e-3)
    assert_allclose(sol.blup[0], [0.40, 0.52, 0.76, -1.67], atol=5e-3)


def test_lhs_printed_in_text(lactation):
    """The 7 x 7 LHS printed in the chapter (alpha = 10 added to the sire diagonal)."""
    y, X, Z = lactation
    sol = solve_mme(X, [Z], y, 0.1, 1.0)
    expected = np.array([[2, 0, 0, 1, 0, 0, 1], [0, 3, 0, 0, 1, 0, 2], [0, 0, 4, 0, 0, 2, 2],
                         [1, 0, 0, 11, 0, 0, 0], [0, 1, 0, 0, 11, 0, 0], [0, 0, 2, 0, 0, 12, 0],
                         [1, 2, 2, 0, 0, 0, 15]])
    assert_allclose(sol.lhs, expected)


def test_pev_sep_reliability(lactation):
    """PEV = (0.095, 0.094, 0.092, 0.083), SEP and REL table."""
    y, X, Z = lactation
    sol = solve_mme(X, [Z], y, 0.1, 1.0)
    assert_allclose(sol.pev[0], [0.095, 0.094, 0.092, 0.083], atol=5e-4)
    assert_allclose(sol.sep[0], [0.308, 0.306, 0.303, 0.288], atol=1e-3)
    assert_allclose(sol.reliability[0], [0.05, 0.06, 0.08, 0.17], atol=5e-3)


def test_c22_row_sums_equal_variance_ratio(lactation):
    """Row sums of C^22 equal sigma2_u/sigma2_e = 0.1 exactly (the text shows 0.099 after rounding)."""
    y, X, Z = lactation
    sol = solve_mme(X, [Z], y, 0.1, 1.0)
    C22 = sol.lhs_inv[3:, 3:]
    assert_allclose(C22.sum(axis=1), 0.1, atol=1e-12)


def test_fixed_sire_estimable_functions(lactation):
    """Herd + sire both fixed: least-squares differences from sire AD are +10 (estimable functions)."""
    y, X, Z = lactation
    W = np.column_stack([X, Z])
    assert np.linalg.matrix_rank(W.T @ W) == 6                 # 7 x 7 but rank 6
    b = np.linalg.pinv(W) @ y
    sires = b[3:]
    assert_allclose(sires[:3] - sires[3], [10, 10, 10], atol=1e-10)
