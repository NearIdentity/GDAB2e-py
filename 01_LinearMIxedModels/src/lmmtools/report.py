"""Tiny reporting helper used by the notebooks: compare a Python result with the chapter's value."""


def check(label, python_value, original_value, tol):
    """Print PASS/FAIL and raise AssertionError on mismatch (|python - original| > tol)."""
    ok = abs(float(python_value) - float(original_value)) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {label}: Python = {float(python_value):.6g}   chapter = {float(original_value):.6g}")
    assert ok, f"{label} does not match the chapter"
