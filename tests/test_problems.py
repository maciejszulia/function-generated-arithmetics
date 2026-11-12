"""The twenty problems of the second paper as unit tests.

Each test evaluates the classical route and the route of the generator calculus
and compares the two numbers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gcalc import _selfcheck  # noqa: E402
from src.problems import PROBLEMS, run, tex_len  # noqa: E402


def test_gcalc_selfcheck():
    _selfcheck()


@pytest.mark.parametrize("p", PROBLEMS, ids=[p["id"] for p in PROBLEMS])
def test_two_notations_agree(p):
    r = run(p)
    assert r["resid"] <= r["tol"], r


def test_symbol_count_rules():
    assert tex_len(r"a \oplus b") == 3
    assert tex_len(r"\frac{a_{1}}{b^2}\,\mathbf{c}") == 7     # frac a 1 b ^ 2 c; braces and _ are free
    assert tex_len(r"x \\ y") == 2
