"""Reference implementation of function-generated arithmetics.

Every arithmetic in this module is a push-forward of (R, +) along a bijection g.
For a bijection g: D -> R the induced operation is

    a (+) b = g^{-1}( g(a) + g(b) )

Because the transported structure is a group, associativity, commutativity,
neutral element, inverses and distributivity of the induced multiplication all
hold identically. The module exists to make that claim testable, and to record
the exact numerical cost of each generator.

Only numpy is required. The inverse of the Lambert generator is evaluated by
solving x + ln x = y in the log domain; the textbook closed form W(exp(y))
overflows for y > 709.78 in IEEE-754 double precision, which is precisely the
range in which the arithmetic is used for large arguments.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "Arithmetic",
    "PowerArithmetic",
    "CubicArithmetic",
    "HarmonicArithmetic",
    "GeometricArithmetic",
    "LogisticArithmetic",
    "SoftplusArithmetic",
    "LambertArithmetic",
    "RelativisticArithmetic",
    "KeplerArithmetic",
    "TsallisArithmetic",
    "SynergyArithmetic",
    "lambert_w",
    "solve_monotone",
]


# --------------------------------------------------------------------------
# numerical helpers
# --------------------------------------------------------------------------

def solve_monotone(func, target, x0, deriv=None, tol=1e-15, maxiter=100):
    """Newton iteration with step halving, used for generators without a
    closed-form inverse. Returns a float."""
    x = float(x0)
    for _ in range(maxiter):
        f = func(x) - target
        if f == 0.0:
            return x
        d = deriv(x) if deriv is not None else (func(x + 1e-7) - func(x - 1e-7)) / 2e-7
        if d == 0.0 or not math.isfinite(d):
            break
        step = f / d
        x_new = x - step
        # keep the iterate finite; halve the step if the residual grows
        for _ in range(40):
            if math.isfinite(x_new) and abs(func(x_new) - target) <= abs(f):
                break
            step *= 0.5
            x_new = x - step
        if abs(x_new - x) <= tol * max(1.0, abs(x_new)):
            return x_new
        x = x_new
    return x


def lambert_w(z: float) -> float:
    """Principal real branch of the Lambert W function for z > 0.

    Solves w + ln w = ln z in the log domain, so it does not overflow when z is
    replaced by exp(y) with y large.
    """
    if z <= 0.0:
        raise ValueError("real principal branch requires z > 0")
    lz = math.log(z)
    if lz > 1.0:
        w = lz - math.log(lz)
    else:
        w = z
    return solve_monotone(lambda w: w + math.log(w), lz, w, lambda w: 1.0 + 1.0 / w)


# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------

class Arithmetic:
    """A pair of mutually inverse maps g, g_inv plus the induced operations."""

    name = "generic"
    domain = "R"
    has_closed_form_inverse = True

    # -- to be provided by subclasses -------------------------------------
    def g(self, x):
        raise NotImplementedError

    def g_inv(self, y):
        raise NotImplementedError

    # -- induced operations ------------------------------------------------
    def add(self, a, b):
        return self.g_inv(self.g(a) + self.g(b))

    def sub(self, a, b):
        return self.g_inv(self.g(a) - self.g(b))

    def neg(self, a):
        return self.g_inv(-self.g(a))

    def scale(self, n, a):
        """n (*) a, where (*) is multiplication by a real scalar."""
        return self.g_inv(n * self.g(a))

    def div_scale(self, a, n):
        return self.g_inv(self.g(a) / n)

    def mul(self, a, b):
        """Full multiplication a (x) b = g^{-1}(g(a) * g(b))."""
        return self.g_inv(self.g(a) * self.g(b))

    def div(self, a, b):
        return self.g_inv(self.g(a) / self.g(b))

    def power(self, a, n):
        return self.g_inv(self.g(a) ** n)

    def mean(self, values):
        values = list(values)
        return self.g_inv(sum(self.g(v) for v in values) / len(values))

    # -- constants ---------------------------------------------------------
    def zero(self):
        return self.g_inv(0.0)

    def one(self):
        return self.g_inv(1.0)


# --------------------------------------------------------------------------
# generators with a closed-form inverse
# --------------------------------------------------------------------------

class PowerArithmetic(Arithmetic):
    """g(x) = x**p on [0, inf); p > 0. p = 2 gives the Pythagorean arithmetic."""

    domain = "[0, inf)"

    def __init__(self, p: float = 2.0):
        if p <= 0:
            raise ValueError("p must be positive")
        self.p = float(p)
        self.name = f"power p={self.p:g}"

    def g(self, x):
        return np.power(x, self.p)

    def g_inv(self, y):
        return np.power(np.maximum(y, 0.0), 1.0 / self.p)


class CubicArithmetic(Arithmetic):
    """g(x) = x**3 on R; the only odd power with a real inverse on all of R."""

    domain = "R"

    def __init__(self):
        self.name = "cubic p=3"

    def g(self, x):
        return x ** 3

    def g_inv(self, y):
        return np.cbrt(y)


class HarmonicArithmetic(Arithmetic):
    """g(x) = 1/x on (0, inf). Models parallel resistances and lens focal length.

    Warning: (0, inf) contains no neutral element for this operation. The
    identity would be g_inv(0) = +inf, so the structure is a semigroup, not a
    group, on the stated domain.
    """

    domain = "(0, inf)"
    has_identity_in_domain = False

    def __init__(self):
        self.name = "harmonic"

    def g(self, x):
        return 1.0 / x

    def g_inv(self, y):
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore"):
            out = 1.0 / y
        return float(out) if out.ndim == 0 else out

    def zero(self):
        raise ArithmeticError(
            "the harmonic operation has no neutral element in (0, inf); "
            "g^-1(0) = +inf lies outside the domain")


class GeometricArithmetic(Arithmetic):
    """g(x) = ln x on (0, inf). Addition becomes multiplication."""

    domain = "(0, inf)"

    def __init__(self):
        self.name = "geometric (log)"

    def g(self, x):
        return np.log(x)

    def g_inv(self, y):
        return np.exp(y)


class LogisticArithmetic(Arithmetic):
    """g(x) = ln(x/(1-x)) on (0, 1); the inverse is the logistic function.

    Addition combines log-odds, which is the Bayes update for independent
    evidence in favour of an event.
    """

    domain = "(0, 1)"

    def __init__(self):
        self.name = "logistic (logit)"

    def g(self, x):
        return np.log(x / (1.0 - x))

    def g_inv(self, y):
        return 1.0 / (1.0 + np.exp(-np.clip(y, -500.0, 500.0)))


class SoftplusArithmetic(Arithmetic):
    """g(x) = ln(e**x - 1) on (0, inf); the inverse is ln(e**y + 1)."""

    domain = "(0, inf)"

    def __init__(self):
        self.name = "softplus"

    def g(self, x):
        return np.log(np.expm1(x))

    def g_inv(self, y):
        return np.log1p(np.exp(-np.abs(y))) + np.maximum(y, 0.0)


class LambertArithmetic(Arithmetic):
    """g(x) = x + ln x on (0, inf); the inverse solves x + ln x = y.

    For x -> 0 the logarithm dominates, so addition behaves like multiplication.
    For x -> inf the identity dominates, so addition behaves like addition.
    The neutral element is g_inv(0) = W(1) = Omega = 0.5671432904097838.
    """

    domain = "(0, inf)"

    def __init__(self):
        self.name = "Lambert (x + ln x)"

    def g(self, x):
        return x + np.log(x)

    def g_inv(self, y):
        scalar = np.isscalar(y)
        arr = np.atleast_1d(np.asarray(y, dtype=float))
        out = np.empty_like(arr)
        for i, v in enumerate(arr):
            if v < -700.0:
                out[i] = 0.0  # true root underflows the double range
                continue
            x0 = max(v - math.log(v), 1e-12) if v > 1.0 else math.exp(v)
            out[i] = solve_monotone(
                lambda x: x + math.log(max(x, 1e-300)), v, x0,
                lambda x: 1.0 + 1.0 / max(x, 1e-300))
        return float(out[0]) if scalar else out


class RelativisticArithmetic(Arithmetic):
    """g(x) = artanh(x/c) on (-c, c); composition of collinear velocities."""

    domain = "(-c, c)"

    def __init__(self, c: float = 1.0):
        self.c = float(c)
        self.name = f"relativistic c={self.c:g}"

    def g(self, x):
        r = np.clip(x / self.c, -1.0 + 1e-16, 1.0 - 1e-16)
        return np.arctanh(r)

    def g_inv(self, y):
        return self.c * np.tanh(y)


class TsallisArithmetic(Arithmetic):
    """g(x) = ln(1 + eps*x) with eps = 1 - q, on (-1/eps, inf).

    The induced addition is a (+) b = a + b + eps*a*b, the q-addition of
    Tsallis statistics. Unlike the synergy operator below, it is associative.
    """

    def __init__(self, q: float = 0.5):
        self.q = float(q)
        self.eps = 1.0 - self.q
        if self.eps == 0.0:
            raise ValueError("q = 1 degenerates to ordinary addition")
        self.name = f"Tsallis q={self.q:g}"

    @property
    def domain(self):
        return f"(-{1.0/self.eps:g}, inf)"

    def g(self, x):
        return np.log1p(self.eps * x)

    def g_inv(self, y):
        return np.expm1(y) / self.eps


class KeplerArithmetic(Arithmetic):
    """g(x) = x + k*sin(x) on R, 0 < k < 1. No closed-form inverse."""

    domain = "R"
    has_closed_form_inverse = False

    def __init__(self, k: float = 0.85):
        if not 0.0 < k < 1.0:
            raise ValueError("k must lie in (0, 1) for g to be a bijection")
        self.k = float(k)
        self.name = f"Kepler k={self.k:g}"

    def g(self, x):
        return x + self.k * np.sin(x)

    def g_inv(self, y):
        scalar = np.isscalar(y)
        arr = np.atleast_1d(np.asarray(y, dtype=float))
        out = np.array([
            solve_monotone(lambda x, v=v: x + self.k * math.sin(x), v, v,
                           lambda x: 1.0 + self.k * math.cos(x))
            for v in arr
        ])
        return float(out[0]) if scalar else out


# --------------------------------------------------------------------------
# the non-associative variant
# --------------------------------------------------------------------------

class SynergyArithmetic(Arithmetic):
    """a (+) b = g^{-1}(g(a) + g(b) + 2*sqrt(s*(g(a) + g(b)))), s >= 0.

    This is the operator proposed in the source material as an "open system"
    arithmetic. It is commutative but neither associative nor unital for s > 0,
    so it is not a group and not an isomorphism of (R, +). It is included here
    so that the failure can be measured rather than asserted.
    """

    def __init__(self, base: Arithmetic, s: float = 0.5):
        if s < 0:
            raise ValueError("s must be non-negative")
        self.base = base
        self.s = float(s)
        self.name = f"synergy s={self.s:g} on {base.name}"
        self.domain = base.domain

    def add(self, a, b):
        x = self.base.g(a) + self.base.g(b)
        boost = 2.0 * math.sqrt(self.s * max(x, 0.0))
        return self.base.g_inv(x + boost)

    def sub(self, a, b):
        """Recover a from (a (+) b) and b."""
        y = self.base.g(a)  # placeholder to keep symmetry of the API
        raise NotImplementedError(
            "sub is not the inverse of add for s > 0; see paper section 6"
        )
