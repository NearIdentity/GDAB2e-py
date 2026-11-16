"""Matrix-algebra appendix examples (with the corrections C8-C14 applied)."""
import numpy as np
from numpy.testing import assert_allclose


def test_inner_outer_products():
    a, b = np.array([1, 0, 4]), np.array([2, 1, 5])
    assert a @ b == 22
    assert_allclose(np.outer(a, b), [[2, 1, 5], [0, 0, 0], [8, 4, 20]])


def test_transpose_of_product_reverses_order():
    """Correction C8: (AB)' = B'A' (not A'B')."""
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(2, 3)), rng.normal(size=(3, 4))
    assert_allclose((A @ B).T, B.T @ A.T)


def test_solve_linear_system():
    """Correction C9: x = A^-1 b. The appendix system 6x-3y+2z=7, x+2y+5z=0, 2x-8y-z=-2."""
    A = np.array([[6, -3, 2], [1, 2, 5], [2, -8, -1]], float)
    b = np.array([7, 0, -2], float)
    x = np.linalg.solve(A, b)
    assert_allclose(A @ x, b, atol=1e-12)
    assert_allclose(x, np.linalg.inv(A) @ b)


def test_two_by_two_inverse():
    """Correction C10: [[4,7],[2,6]]^-1 = 1/10 [[6,-7],[-2,4]] = [[0.6,-0.7],[-0.2,0.4]]."""
    assert_allclose(np.linalg.inv([[4, 7], [2, 6]]), [[0.6, -0.7], [-0.2, 0.4]])


def test_three_by_three_inverse_example():
    A = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 4]], float)
    Ainv = np.array([[1, 1, -1], [1, -3, 2], [-1, 2, -1]], float)
    assert_allclose(A @ Ainv, np.eye(3))


def test_direct_sum_and_kronecker():
    from scipy.linalg import block_diag
    D, F = np.array([[4, 3], [6, 8]]), np.array([[1, 2], [3, 5]])
    assert_allclose(block_diag(D, F), [[4, 3, 0, 0], [6, 8, 0, 0], [0, 0, 1, 2], [0, 0, 3, 5]])
    assert_allclose(np.kron(D, F), [[4, 8, 3, 6], [12, 20, 9, 15], [6, 12, 8, 16], [18, 30, 24, 40]])
    G = np.array([[5.55, 0.44, 0.43], [0.44, 8.88, 0.42], [0.43, 0.42, 3.33]])
    assert not np.allclose(np.kron(G, np.eye(2)), np.kron(np.eye(2), G))


def test_eigen_example_and_properties():
    """Correction C15: for A = [[3,2],[3,-2]] the eigenvector with eigenvalue 4 is (2,1), not (1,2)."""
    A = np.array([[3, 2], [3, -2]], float)
    assert_allclose(A @ [2, 1], 4 * np.array([2, 1]))
    assert not np.allclose(A @ [1, 2], 4 * np.array([1, 2]))            # the draft's x = (1,2) gives (7,-1)
    assert_allclose(sorted(np.linalg.eigvals(A).real), [-3, 4])
    S = np.array([[13, -4, 2], [-4, 11, -2], [2, -2, 8]], float)
    lam = np.linalg.eigvalsh(S)
    assert_allclose([np.trace(S), np.linalg.det(S), np.linalg.matrix_rank(S)], [32, 952, 3])
    assert_allclose([lam.sum(), lam.prod()], [32, 952])
    assert_allclose(np.sum(S**2), np.sum(lam**2))                       # C14: holds for symmetric matrices


def test_generalized_inverse_definition():
    """Correction C12: G is a g-inverse of A if A G A = A (Moore-Penrose via pinv)."""
    X = np.column_stack([np.ones(4), [1, 1, 0, 0], [0, 0, 1, 1]])
    A = X.T @ X                                                         # singular (rank 2)
    G = np.linalg.pinv(A)
    assert np.linalg.matrix_rank(A) == 2
    assert_allclose(A @ G @ A, A, atol=1e-12)


def test_random_matrix_example_from_r():
    """R: set.seed(1); round(matrix(rnorm(16),4)*5). Values copied from the text (R's RNG differs)."""
    A = np.array([[-3, 2, 3, -3], [1, -4, -2, -11], [-4, 2, 8, 6], [8, 4, 2, 0]], float)
    assert np.trace(A) == 1
    assert_allclose(np.round(np.linalg.inv(A), 2)[0], [-0.12, 0.05, 0.03, 0.09])
