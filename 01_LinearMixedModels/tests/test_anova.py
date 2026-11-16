"""Least-squares ANOVA and random-model tests vs. SAS PROC GLM output printed in Chapter 1."""
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from lmmtools.anova import FactorialANOVA
from lmmtools.data import load_maize

from conftest import TERMS, needs_data

pytestmark = needs_data


def _anova(balanced):
    return FactorialANOVA(load_maize(balanced), "height", TERMS)


def test_balanced_type1_type3():
    a = _anova(True)
    for tab in (a.type1(), a.type3()):
        assert_allclose(tab["SS"], [84931.3312, 3594.2244, 154937.5322, 20999.4038, 15832.2400], atol=1e-3)
        assert list(tab["DF"]) == [3, 4, 61, 183, 244]
        assert_allclose(tab["F"].iloc[:4], [436.31, 13.85, 39.14, 1.77], atol=5e-3)
    assert_allclose(a.mse, 64.8862, atol=1e-4)


def test_balanced_random_tests():
    """SAS 'Tests of Hypotheses for Random Model Analysis of Variance' (balanced)."""
    rt = _anova(True).random_tests()
    assert_allclose(rt.loc["location", ["Error DF", "Error MS", "F", "Pr > F"]], [4.4543, 948.420722, 29.85, 0.0022], atol=5e-3)
    assert_allclose(rt.loc["RIL", "F"], 22.13, atol=5e-3)                 # = MS(RIL)/MS(location*RIL)
    assert rt.loc["RIL", "Error DF"] == 183


def test_balanced_moments_and_h2():
    m = _anova(True).moments()
    assert_allclose(m[["Error", "location*RIL", "RIL"]], [64.89, 24.93, 303.15], atol=6e-3)
    h2 = m["RIL"] / (m["RIL"] + m["location*RIL"] / 4 + m["Error"] / 8)
    assert_allclose(h2, 0.9548, atol=5e-5)                             # the text prints 0.954 (truncated)


def test_balanced_lsmeans():
    ls = _anova(True).lsmeans("RIL")
    assert_allclose(ls.loc[["RIL-1", "RIL-11", "RIL-12", "RIL-14"], "LSMEAN"], [182.1, 182.875, 185.2, 194.25], atol=1e-6)
    assert_allclose(ls["SE"].iloc[0], 2.85, atol=5e-3)                    # R lsmeans(): SE 2.85, df 244


def test_unbalanced_type1_type3():
    a = _anova(False)
    t1, t3 = a.type1(), a.type3()
    assert_allclose(t1["SS"], [79345.6274, 3693.2364, 150287.3376, 18290.6367, 15157.9644], atol=1e-3)
    assert_allclose(t3["SS"], [77171.8286, 3677.7950, 149644.4648, 18290.6367, 15157.9644], atol=1e-3)
    assert list(t3["DF"]) == [3, 4, 61, 180, 225]
    assert_allclose(t3["SS"].iloc[:4].sum(), 248784.7251, atol=1e-3)


def test_unbalanced_ems_coefficients():
    e = _anova(False).ems_type3()
    assert_allclose(e.loc["location", ["location*RIL", "rep(location)", "location"]], [1.8769, 57.243, 114.49], atol=6e-3)
    assert_allclose(e.loc["rep(location)", "rep(location)"], 57.25, atol=5e-3)
    assert_allclose(e.loc["RIL", ["location*RIL", "RIL"]], [1.9149, 7.6105], atol=5e-5)
    assert_allclose(e.loc["location*RIL", "location*RIL"], 1.9348, atol=5e-5)


def test_unbalanced_random_tests():
    rt = _anova(False).random_tests()
    assert_allclose(rt.loc["location", ["Error DF", "Error MS", "F", "Pr > F"]], [4.2929, 952.566239, 27.00, 0.0031], atol=5e-3)
    assert_allclose(rt.loc["RIL", ["Error DF", "Error MS", "F"]], [182.48, 101.262345, 24.23], atol=5e-3)
    assert "0.9897*MS(location*RIL) + 0.0103*MS(Error)" in rt.loc["RIL", "Error term"]


def test_unbalanced_moments_correction_C6():
    """Correction C6: with the unbalanced MS(RIL) = 2453.19 and coefficient 7.6105 the moment estimate
    is 309.04 (the draft's 304.84 used the balanced MS(RIL) = 2539.96 and divisor 8)."""
    m = _anova(False).moments()
    assert_allclose(m["location*RIL"], 17.70, atol=5e-3)
    assert_allclose(m["RIL"], 309.04, atol=5e-3)
    draft = (2539.96 - 1.9149 * 17.7 - 67.37) / 8
    assert_allclose(draft, 304.84, atol=5e-3)


def test_unbalanced_lsmeans_non_estimable():
    ls = _anova(False).lsmeans("RIL")
    assert not ls.loc["RIL-5", "estimable"] and not ls.loc["RIL-51", "estimable"]
    assert ls["estimable"].sum() == 60
    assert_allclose(ls.loc[["RIL-1", "RIL-49", "RIL-50", "RIL-53"], "LSMEAN"], [182.1, 176.975, 200.275, 174.425], atol=1e-6)
