import numpy as np
import pytest

from lmmtools.data import available, load_maize
from lmmtools.design import incidence

needs_data = pytest.mark.skipif(not available(), reason="run `python scripts/get_data.py` to download the maize data")

TERMS = {"location": ("location",), "rep(location)": ("location", "rep"),
         "RIL": ("RIL",), "location*RIL": ("location", "RIL")}


def maize_design(balanced=True):
    d = load_maize(balanced)
    Zs = [incidence(d, list(f) if len(f) > 1 else f[0])[0] for f in TERMS.values()]
    return d, np.ones((len(d), 1)), Zs, list(TERMS)


@pytest.fixture(scope="session")
def balanced():
    return maize_design(True)


@pytest.fixture(scope="session")
def unbalanced():
    return maize_design(False)


@pytest.fixture(scope="session")
def lactation():
    """Robinson (1991) daughters' lactation yield example (Chapter 1, 'A trivial example')."""
    y = np.array([110, 100, 110, 100, 100, 110, 110, 100, 100], float)
    herd = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    sire = [0, 3, 1, 3, 3, 2, 2, 3, 3]                     # columns ordered ZA, BB, CC, AD as in the text
    return y, np.eye(3)[herd], np.eye(4)[sire]
