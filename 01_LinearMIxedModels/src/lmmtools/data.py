"""Locate, download and verify the Chapter 1 data files.

The chapter uses MaizeRILs.csv (balanced, 496 plots) and MaizeRIL_miss.csv (unbalanced, 474 plots).
The authors distribute them through the book's Google Drive folder (see README / Appendix A). Public
copies identical to the chapter's analyses are mirrored in two teaching repositories on GitHub; this
module downloads those and checks SHA-256 digests, so a changed file is detected.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd

FILES = {
    "MaizeRILs.csv": {
        "url": "https://raw.githubusercontent.com/malachycampbell/StatisticalMethodsforOmicsAssistedBreeding/master/OLS.MM/MaizeRILs.csv",
        "sha256": "1ec465ef3709877b569100a81fc71d9293253c2152bcc05470621564ee98bec6",
        "rows": 496,
    },
    "MaizeRIL_miss.csv": {
        "url": "https://raw.githubusercontent.com/Robbins-Lab/PLSCI-7201/refs/heads/main/Modules/Module8/ProblemSets/MaizeRILs2.csv",
        "sha256": "0004658a110bb44b98b9a3a36ab62e7ad0a59e4cd9b16094bcf02df7964e830b",
        "rows": 474,
    },
}


def data_dir() -> Path:
    """$LMM_DATA_DIR if set, else <project>/data."""
    env = os.environ.get("LMM_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(force: bool = False, verbose: bool = True) -> dict[str, Path]:
    d = data_dir(); d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, meta in FILES.items():
        path = d / name
        if force or not path.exists():
            if verbose:
                print(f"downloading {name} <- {meta['url']}")
            urllib.request.urlretrieve(meta["url"], path)
        digest = sha256(path)
        if digest != meta["sha256"]:
            raise ValueError(f"{name}: SHA-256 {digest} does not match the verified copy {meta['sha256']}")
        out[name] = path
    if verbose:
        print("all files present and verified in", d)
    return out


def load_maize(balanced: bool = True) -> pd.DataFrame:
    """Load the maize RIL data with factors as strings (location, rep, block, plot, RIL)."""
    name = "MaizeRILs.csv" if balanced else "MaizeRIL_miss.csv"
    path = data_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python scripts/get_data.py` first")
    df = pd.read_csv(path)
    for c in ("location", "rep", "block", "plot", "RIL"):
        df[c] = df[c].astype(str)
    return df


def available() -> bool:
    return all((data_dir() / n).exists() for n in FILES)
