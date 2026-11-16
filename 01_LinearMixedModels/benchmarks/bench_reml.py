"""Benchmark REML engines on the maize model  height ~ 1 + (location) + (rep:location) + (RIL) + (RIL:location).

Engines: statsmodels MixedLM (crossed effects as variance components in one group), pyreml (PyTorch),
and the custom lmmtools.SparseREML. Data: the chapter's maize sets plus simulated multi-environment
trials of increasing size (variances set to the chapter's REML estimates, 5% plots missing).

    uv run python benchmarks/bench_reml.py            # writes benchmarks/results.csv
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("results.csv")
TRUE = dict(location=220.66, rep=13.4463, RIL=303.15, RIL_loc=24.9323, resid=64.8862)
SIZES = [(62, 4), (200, 8), (500, 10), (1000, 20), (2000, 25)]     # (lines, locations), 2 reps each
TIMEOUT = 900


def simulate(n_ril, n_loc, seed=1, miss=0.05):
    rng = np.random.default_rng(seed)
    L, R = np.arange(n_loc), np.arange(n_ril)
    g = rng.normal(0, np.sqrt(TRUE["RIL"]), n_ril)
    l = rng.normal(0, np.sqrt(TRUE["location"]), n_loc)
    rl = rng.normal(0, np.sqrt(TRUE["rep"]), (n_loc, 2))
    gl = rng.normal(0, np.sqrt(TRUE["RIL_loc"]), (n_loc, n_ril))
    rows = [(f"L{i}", str(r + 1), f"RIL-{k}", 180 + l[i] + rl[i, r] + g[k] + gl[i, k] + rng.normal(0, np.sqrt(TRUE["resid"])))
            for i in L for r in range(2) for k in R]
    d = pd.DataFrame(rows, columns=["location", "rep", "RIL", "height"])
    return d.drop(rng.choice(len(d), int(miss * len(d)), replace=False)).reset_index(drop=True)


def run_custom(d):
    from lmmtools.design import incidence
    from lmmtools.reml import SparseREML
    Zs = [incidence(d, f)[0] for f in ["location", ["location", "rep"], "RIL", ["location", "RIL"]]]
    fit = SparseREML(d.height, np.ones((len(d), 1)), Zs).fit(se="ai")
    return list(fit.theta)


def run_statsmodels(d):
    import statsmodels.formula.api as smf
    warnings.simplefilter("ignore")
    d = d.assign(one=1, lr=d.location + ":" + d.rep, gl=d.location + ":" + d.RIL)
    vc = {"a": "0 + C(location)", "b": "0 + C(lr)", "c": "0 + C(RIL)", "d": "0 + C(gl)"}
    m = smf.mixedlm("height ~ 1", d, groups="one", vc_formula=vc).fit(reml=True, method=["bfgs"], gtol=1e-8, maxiter=5000)
    v = dict(zip(m.model.exog_vc.names, m.vcomp))
    return [v["a"], v["b"], v["c"], v["d"], m.scale]


def run_pyreml(d):
    from pyreml import MixedModel, Random, Residual
    d = d.assign(lr=d.location + ":" + d.rep, gl=d.location + ":" + d.RIL)
    m = MixedModel.from_dataframe(data=d, response="height", fixed="1",
                                  random=[Random(unit="location"), Random(unit="lr"), Random(unit="RIL"), Random(unit="gl")],
                                  residual=Residual())
    m.fit(dtype="double", verbose=False)
    return [r.variance["sigma"] for r in m.random] + [m.residual.variance["sigma"]]


ENGINES = {"lmmtools.SparseREML (custom)": run_custom, "statsmodels MixedLM": run_statsmodels, "pyreml": run_pyreml}


def _worker(engine, data_path, q):
    try:
        d = pd.read_pickle(data_path)
        t = time.perf_counter()
        est = ENGINES[engine](d)
        q.put(("ok", time.perf_counter() - t, est))
    except Exception as e:                                   # noqa: BLE001
        q.put(("error", np.nan, repr(e)[:200]))


def timed(engine, d, tmp):
    path = tmp / "d.pkl"; d.to_pickle(path)
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(engine, path, q)); p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join()
        return "timeout", np.nan, None
    return q.get() if not q.empty() else ("crashed (memory?)", np.nan, None)


def main():
    from lmmtools.data import load_maize
    tmp = Path(__file__).parent / ".tmp"; tmp.mkdir(exist_ok=True)
    datasets = [("MaizeRILs.csv (balanced)", load_maize(True)), ("MaizeRIL_miss.csv (unbalanced)", load_maize(False))]
    datasets += [(f"simulated {g} lines x {l} locations", simulate(g, l)) for g, l in SIZES]
    rows, skip = [], set()
    for label, d in datasets:
        n_q = d.location.nunique() * 3 + d.RIL.nunique() * (1 + d.location.nunique())
        ref = None
        for eng in ENGINES:
            if eng in skip:
                rows.append((label, len(d), n_q, eng, "skipped (slower size failed)", np.nan, np.nan)); continue
            status, secs, est = timed(eng, d, tmp)
            if status == "ok" and eng.startswith("lmmtools"):
                ref = np.array(est)
            dev = float(np.max(np.abs(np.array(est) / ref - 1))) if status == "ok" and ref is not None else np.nan
            rows.append((label, len(d), n_q, eng, status, secs, dev))
            print(f"{label:40s} n={len(d):7d}  {eng:30s} {status:10s} {secs:8.2f}s  max rel diff vs custom {dev:.2e}", flush=True)
            if status != "ok":
                skip.add(eng)
        pd.DataFrame(rows, columns=["dataset", "n", "random effects", "engine", "status", "seconds",
                                    "max rel. diff vs custom"]).to_csv(OUT, index=False)
    print("written", OUT)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.exit(main())
