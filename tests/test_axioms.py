"""The algebraic laws of the paper, as unit tests.

Every test here corresponds to a row of Table 2 or to a statement of
Theorem 2.1 / Proposition 3.1 / Proposition 6.1.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fga import (  # noqa: E402
    CubicArithmetic,
    GeometricArithmetic,
    HarmonicArithmetic,
    KeplerArithmetic,
    LambertArithmetic,
    LogisticArithmetic,
    PowerArithmetic,
    RelativisticArithmetic,
    SoftplusArithmetic,
    SynergyArithmetic,
    TsallisArithmetic,
)

GROUP_MODELS = [
    (CubicArithmetic(), -8.0, 8.0),
    (GeometricArithmetic(), 0.05, 20.0),
    (LogisticArithmetic(), 0.02, 0.98),
    (SoftplusArithmetic(), 0.05, 20.0),
    (LambertArithmetic(), 0.02, 20.0),
    (RelativisticArithmetic(1.0), -0.99, 0.99),
    (KeplerArithmetic(0.85), -6.0, 6.0),
    (TsallisArithmetic(0.5), -1.5, 20.0),
]


def rel(x, y):
    return abs(x - y) / max(1.0, abs(x), abs(y))


@pytest.mark.parametrize("model,lo,hi", GROUP_MODELS, ids=[m.name for m, _, _ in GROUP_MODELS])
def test_group_axioms(model, lo, hi):
    rng = np.random.default_rng(3)
    a = rng.uniform(lo, hi, 120)
    b = rng.uniform(lo, hi, 120)
    c = rng.uniform(lo, hi, 120)
    z0 = model.zero()

    assert max(rel(model.add(x, y), model.add(y, x)) for x, y in zip(a, b)) < 1e-9
    assert max(rel(model.add(model.add(x, y), z), model.add(x, model.add(y, z)))
               for x, y, z in zip(a, b, c)) < 1e-9
    assert max(rel(model.add(x, z0), x) for x in a) < 1e-9
    # the inverse law is checked in the chart, where it is exact addition;
    # in the original coordinates the cube root amplifies cancellation
    assert max(abs(model.g(model.add(x, model.neg(x))) - model.g(z0))
               / max(1.0, abs(model.g(x))) for x in a) < 1e-12
    assert max(rel(model.mul(x, model.add(y, z)),
                   model.add(model.mul(x, y), model.mul(x, z)))
               for x, y, z in zip(a, b, c)) < 1e-9


def test_power_arithmetic_is_a_monoid_without_inverses():
    m = PowerArithmetic(2.0)
    assert rel(m.add(3.0, 4.0), 5.0) < 1e-15          # 3-4-5
    assert m.zero() == 0.0
    # g(D) = [0, inf), so -g(a) is outside the image and the additive inverse
    # produced by g^-1 collapses to 0. The group law must therefore fail.
    assert rel(m.add(3.0, m.neg(3.0)), m.zero()) > 1e-3


def test_harmonic_arithmetic_is_a_semigroup():
    m = HarmonicArithmetic()
    assert rel(m.add(6.0, 3.0), 2.0) < 1e-15
    with pytest.raises(ArithmeticError):
        m.zero()


def test_lambert_constants_and_regimes():
    m = LambertArithmetic()
    assert abs(m.zero() - 0.5671432904097838) < 1e-15   # Omega
    assert abs(m.one() - 1.0) < 1e-15
    # small arguments behave like multiplication, with relative error ~ a + b
    dev = abs(m.add(1e-3, 1e-3) - 1e-6) / 1e-6
    assert 1e-3 < dev < 5e-3
    # large arguments behave like addition
    assert rel(m.add(0.8, 1e6), 0.8 + 1e6) < 1e-2


def test_lambert_inverse_survives_large_arguments():
    m = LambertArithmetic()
    for y in (10.0, 500.0, 1e4, 1e6):
        x = m.g_inv(y)
        assert math.isfinite(x)
        assert abs((x + math.log(x)) - y) <= 1e-9 * max(1.0, y)


def test_kepler_one_is_not_one():
    m = KeplerArithmetic(0.85)
    assert m.zero() == 0.0
    assert abs(m.one() - 0.553315638360866) < 1e-12
    assert abs(m.g(1.0) - 1.715250337086712) < 1e-12


def test_relativistic_composition_never_exceeds_c():
    m = RelativisticArithmetic(1.0)
    assert rel(m.add(0.5, 0.5), 0.8) < 1e-14
    assert m.add(0.9, 0.9) < 1.0
    assert rel(m.add(0.99, 0.99), 2 * 0.99 / (1 + 0.99 ** 2)) < 1e-12


def test_logistic_pooling_of_evidence():
    m = LogisticArithmetic()
    assert rel(m.add(0.8, 0.7), 0.9032258064516129) < 1e-12
    assert 0.0 < m.add(0.99, 0.99) < 1.0


def test_synergy_is_not_an_arithmetic():
    base = LambertArithmetic()
    m = SynergyArithmetic(base, s=0.5)
    a, b, c = 1.3, 1.7, 2.3
    left, right = m.add(m.add(a, b), c), m.add(a, m.add(b, c))
    assert rel(left, right) > 1e-2                 # not associative
    assert abs(m.add(a, base.zero()) - a) > 0.1    # 0_g is not neutral
    # for s = 0 the operator degenerates to the arithmetic
    m0 = SynergyArithmetic(base, s=0.0)
    assert rel(m0.add(a, b), base.add(a, b)) < 1e-14
