"""REML variance components, SEs, LRTs, BLUPs and profile CIs vs. SAS PROC MIXED / ASReml / lme4."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from lmmtools.reml import SparseREML, boundary_lrt

from conftest import needs_data

pytestmark = needs_data


@pytest.fixture(scope="module")
def bal_fit(balanced):
    d, X, Zs, names = balanced
    m = SparseREML(d.height, X, Zs, names)
    return m, m.fit()


@pytest.fixture(scope="module")
def unbal_fit(unbalanced):
    d, X, Zs, names = unbalanced
    m = SparseREML(d.height, X, Zs, names)
    return m, m.fit()


def test_balanced_components_and_se(bal_fit):
    """SAS PROC MIXED covtest (balanced)."""
    _, f = bal_fit
    assert_allclose(f.theta, [220.66, 13.4463, 303.15, 24.9323, 64.8862], rtol=2e-4)
    assert_allclose(f.se, [186.48, 10.2484, 57.5088, 6.6787, 5.8745], rtol=2e-4)
    assert_allclose(f.neg2_loglik, 3833.2, atol=0.05)
    assert_allclose(f.aic, 3843.2, atol=0.05)
    assert_allclose([f.beta[0], f.beta_se[0]], [182.12, 7.8719], atol=5e-3)


def test_balanced_ai_se_close_to_asreml(bal_fit):
    m, f = bal_fit
    se_ai = m.variance_se(f.sigma2, f.sigma2_e, "ai")
    assert_allclose(se_ai, [186.453, 10.248, 57.507, 6.679, 5.875], rtol=5e-4)


def test_balanced_moments_equal_reml(bal_fit):
    """REML = method of moments for balanced data with positive estimates."""
    _, f = bal_fit
    assert_allclose(f.theta[[2, 3, 4]], [303.15, 24.93, 64.89], atol=6e-3)


def test_lrt_reduced_models(balanced, bal_fit):
    """-2RLL: full 3833.2; no location 3841.3; no RIL 4092.7; no location*RIL 3850.4. Location LRT p = 0.002."""
    d, X, Zs, names = balanced
    _, full = bal_fit
    expected = {0: 3841.3, 2: 4092.7, 3: 3850.4}
    for drop, val in expected.items():
        red = SparseREML(d.height, X, [Z for i, Z in enumerate(Zs) if i != drop]).fit(se=None)
        assert_allclose(red.neg2_loglik, val, atol=0.06)
    red = SparseREML(d.height, X, Zs[1:]).fit(se=None)
    D, p_raw, p_adj = boundary_lrt(red.neg2_loglik, full.neg2_loglik)
    assert_allclose(D, 8.145, atol=1e-3)
    assert_allclose([p_raw, p_adj], [0.0043, 0.002159], atol=5e-5)


def test_balanced_blups(bal_fit, balanced):
    """RIL BLUPs and marginal BLUPs (mu + u): table 'Random effect predictor / BLUP'."""
    d, *_ = balanced
    _, f = bal_fit
    levels = sorted(d.RIL.unique())
    u = dict(zip(levels, f.blups["RIL"]))
    want = {"RIL-1": -0.02, "RIL-11": 0.72, "RIL-12": 2.94, "RIL-14": 11.58, "RIL-15": 13.04,
            "RIL-16": -8.87, "RIL-20": 26.38, "RIL-21": -15.53}
    assert_allclose([u[k] for k in want], list(want.values()), atol=6e-3)
    assert_allclose(f.beta[0] + u["RIL-21"], 166.59, atol=6e-3)          # correction C5 (text says 166.50)


def test_shrinkage_equals_h2_times_deviation(bal_fit, balanced):
    """Balanced data: BLUP(RIL) = h2 * (LSmean - grand mean), h2 = 0.954."""
    d, *_ = balanced
    _, f = bal_fit
    s2g, s2gl, s2e = f.sigma2[2], f.sigma2[3], f.sigma2_e
    h2 = s2g / (s2g + s2gl / 4 + s2e / 8)
    dev = d.groupby("RIL").height.mean().sort_index() - d.height.mean()
    assert_allclose(f.blups["RIL"], h2 * dev.to_numpy(), atol=1e-6)
    assert_allclose(h2, 0.9548, atol=5e-5)


def test_unbalanced_components_se_blups(unbal_fit, unbalanced):
    d, *_ = unbalanced
    _, f = unbal_fit
    assert_allclose(f.theta, [217.69, 15.0591, 309.19, 17.1466, 67.9184], rtol=2e-4)
    assert_allclose(f.se, [184.69, 11.4593, 58.5756, 6.4757, 6.4110], rtol=5e-4)
    levels = sorted(d.RIL.unique())
    u = dict(zip(levels, f.blups["RIL"]))
    assert_allclose(f.beta[0], 181.70, atol=5e-3)
    assert_allclose([u["RIL-5"], u["RIL-51"], u["RIL-49"]], [-28.74, 0.53, -4.53], atol=6e-3)


def test_profile_ci_lme4(balanced):
    """lme4 confint(): RIL SD interval (14.574, 21.155) -> variance (212.4, 447.5). lme4 profiles the ML
    deviance, so an ML fit reproduces it."""
    d, X, Zs, names = balanced
    m = SparseREML(d.height, X, Zs, names)
    ml = m.fit(method="ml", se=None)
    lo, hi = m.profile_ci(ml, 2)
    assert_allclose([lo, hi], [14.574, 21.155], atol=0.02)
    lo, hi = m.profile_ci(ml, 4)
    assert_allclose([lo, hi], [7.390, 8.826], atol=0.01)


@pytest.mark.slow
def test_statsmodels_and_pyreml_agree(balanced):
    import pandas as pd
    import statsmodels.formula.api as smf
    from pyreml import MixedModel, Random, Residual
    d, *_ = balanced
    d = d.assign(one=1, loc_rep=d.location + ":" + d.rep, loc_ril=d.location + ":" + d.RIL)
    vc = {"location": "0 + C(location)", "rep_loc": "0 + C(loc_rep)", "RIL": "0 + C(RIL)", "RIL_loc": "0 + C(loc_ril)"}
    sm = smf.mixedlm("height ~ 1", d, groups="one", vc_formula=vc).fit(reml=True, method=["bfgs"], gtol=1e-10, maxiter=5000)
    vcomp = dict(zip(sm.model.exog_vc.names, sm.vcomp))
    assert_allclose([vcomp["location"], vcomp["rep_loc"], vcomp["RIL"], vcomp["RIL_loc"], sm.scale],
                    [220.66, 13.4463, 303.15, 24.9323, 64.8862], rtol=5e-4)
    pr = MixedModel.from_dataframe(data=d, response="height", fixed="1",
                                   random=[Random(unit="location"), Random(unit="loc_rep"), Random(unit="RIL"), Random(unit="loc_ril")],
                                   residual=Residual())
    pr.fit(dtype="double", verbose=False)
    assert_allclose([r.variance["sigma"] for r in pr.random] + [pr.residual.variance["sigma"]],
                    [220.66, 13.4463, 303.15, 24.9323, 64.8862], rtol=5e-4)
