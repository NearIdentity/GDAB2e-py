"""Download and verify the Chapter 1 data files into ./data (or $LMM_DATA_DIR).

    uv run python scripts/get_data.py
"""
from lmmtools.data import fetch

if __name__ == "__main__":
    fetch()
