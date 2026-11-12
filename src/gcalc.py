"""Generator calculus: the two axes that extend src/fga.py.

Axis "scale"        src.fga.Arithmetic    a (+) b = g^-1(g(a) + g(b))
Axis "unit"         Algebra, MV           units e with e*e = +1, -1, 0 (or any real kappa)
                    Jet                   eps**(n+1) = 0, derivatives as arithmetic
Axis "temperature"  oplus_h, minplus      g_h(x) = exp(x/h); h -> 0 gives (max, +) and (min, +)

Only numpy is required. Self-check:  python -m src.gcalc
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Algebra", "MV", "Jet", "jexp", "jlog", "jsin", "jcos", "deriv",
           "unit_exp", "oplus_h", "minplus", "E3", "I3", "rotor3", "rotate", "rk4"]


# --------------------------------------------------------------------------
# axis "unit": Clifford algebra with prescribed squares of the units
# --------------------------------------------------------------------------

class Algebra:
    """Units e_1..e_n with e_i*e_i = squares[i] and e_i*e_j = -e_j*e_i.

    Algebra(3) is Cl(3,0). Algebra(1, 3) is Cl(1,3). Algebra(0, 0, 2) is the
    Grassmann algebra on two generators. Algebra(squares=[k]) is the planar
    number system with e*e = k: complex (k<0), dual (k=0), double (k>0).
    """

    def __init__(self, p=0, q=0, r=0, squares=None):
        self.sq = list(squares) if squares is not None else [1.0] * p + [-1.0] * q + [0.0] * r
        self.n = len(self.sq)

    def scalar(self, x):
        return MV(self, {0: x})

    def vector(self, comps):
        return MV(self, {1 << i: c for i, c in enumerate(comps)})

    def blade(self, *idx):
        """blade(1, 2) is the product e_1 e_2 (indices start at 1)."""
        out = MV(self, {0: 1.0})
        for i in idx:
            out = out * MV(self, {1 << (i - 1): 1.0})
        return out

    def _mul(self, a, b):
        """Product of two basis blades given as bit masks: (sign, mask)."""
        s, t = 1.0, a >> 1
        while t:
            if bin(t & b).count("1") & 1:
                s = -s
            t >>= 1
        c, i = a & b, 0
        while c:
            if c & 1:
                s *= self.sq[i]
            c >>= 1
            i += 1
        return s, a ^ b


class MV:
    """Multivector: a dict {blade mask: coefficient}. Coefficients are float or complex."""

    def __init__(self, alg, c):
        self.alg, self.c = alg, {k: v for k, v in c.items() if v != 0}

    def _lift(self, x):
        return x if isinstance(x, MV) else MV(self.alg, {0: x})

    def __add__(self, o):
        d = dict(self.c)
        for k, v in self._lift(o).c.items():
            d[k] = d.get(k, 0) + v
        return MV(self.alg, d)

    __radd__ = __add__

    def __neg__(self):
        return MV(self.alg, {k: -v for k, v in self.c.items()})

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        if not isinstance(o, MV):
            return MV(self.alg, {k: v * o for k, v in self.c.items()})
        d = {}
        for a, x in self.c.items():
            for b, y in o.c.items():
                s, m = self.alg._mul(a, b)
                if s:
                    d[m] = d.get(m, 0) + s * x * y
        return MV(self.alg, d)

    def __rmul__(self, o):          # a scalar on the left commutes
        return self * o

    def __invert__(self):
        """Reversion: the sign of grade k is (-1)**(k(k-1)/2)."""
        return MV(self.alg, {k: (-v if (bin(k).count("1") // 2) & 1 else v)
                             for k, v in self.c.items()})

    def grade(self, g):
        return MV(self.alg, {k: v for k, v in self.c.items() if bin(k).count("1") == g})

    def s(self):
        """Scalar part."""
        return self.c.get(0, 0.0)

    def coef(self, *idx):
        """Coefficient of e_i e_j ... (ascending indices starting at 1)."""
        return self.c.get(sum(1 << (i - 1) for i in idx), 0.0)

    def vec(self):
        return np.array([self.c.get(1 << i, 0.0) for i in range(self.alg.n)])

    def norm1(self):
        return sum(abs(v) for v in self.c.values())

    def exp(self, terms=30):
        """Exponential by Taylor series with scaling and squaring; valid for every signature."""
        n1 = self.norm1()
        k = max(0, math.ceil(math.log2(n1)) + 1) if n1 > 0.5 else 0
        x = self * (0.5 ** k)
        out = term = MV(self.alg, {0: 1.0})
        for n in range(1, terms):
            term = term * x * (1.0 / n)
            out = out + term
        for _ in range(k):
            out = out * out
        return out


E3 = Algebra(3)                 # space: three units with square +1
I3 = E3.blade(1, 2, 3)          # unit pseudoscalar, I3*I3 = -1


def rotor3(axis, angle):
    """Rotor exp(-I n angle/2): rotation by `angle` about the unit vector `axis`."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    return (I3 * E3.vector(n) * (-0.5 * angle)).exp()


def rotate(R, v):
    """R v R~ for a rotor R and a 3-vector v given as an array."""
    return (R * E3.vector(v) * ~R).vec()


def unit_exp(kappa, t):
    """(C, S) with exp(e t) = C + e S for a unit e with e*e = kappa."""
    t = np.asarray(t, dtype=float)
    if kappa < 0:
        w = math.sqrt(-kappa)
        return np.cos(w * t), np.sin(w * t) / w
    if kappa > 0:
        w = math.sqrt(kappa)
        return np.cosh(w * t), np.sinh(w * t) / w
    return np.ones_like(t), t


# --------------------------------------------------------------------------
# axis "unit", nilpotent case: jets
# --------------------------------------------------------------------------

class Jet:
    """c[0] + c[1] eps + ... + c[n] eps**n with eps**(n+1) = 0.

    For the jet f(a + eps) the coefficient c[k] equals f^(k)(a) / k!.
    """

    def __init__(self, c):
        self.c = list(c)

    @classmethod
    def var(cls, a, order=1):
        return cls([a, 1.0] + [0.0] * (order - 1))

    def _lift(self, o):
        return o if isinstance(o, Jet) else Jet([o] + [0.0] * (len(self.c) - 1))

    def __add__(self, o):
        return Jet([x + y for x, y in zip(self.c, self._lift(o).c)])

    __radd__ = __add__

    def __neg__(self):
        return Jet([-x for x in self.c])

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        o = self._lift(o)
        return Jet([sum(self.c[j] * o.c[k - j] for j in range(k + 1)) for k in range(len(self.c))])

    __rmul__ = __mul__

    def __truediv__(self, o):
        o, q = self._lift(o), []
        for k in range(len(self.c)):
            q.append((self.c[k] - sum(o.c[j] * q[k - j] for j in range(1, k + 1))) / o.c[0])
        return Jet(q)

    def __rtruediv__(self, o):
        return self._lift(o) / self

    def __pow__(self, p):
        if isinstance(p, int) and p >= 0:
            out = self._lift(1.0)
            for _ in range(p):
                out = out * self
            return out
        return jexp(p * jlog(self))     # real exponent, needs c[0] > 0


def jexp(x):
    if not isinstance(x, Jet):
        return math.exp(x)
    e = [math.exp(x.c[0])]
    for k in range(1, len(x.c)):
        e.append(sum(j * x.c[j] * e[k - j] for j in range(1, k + 1)) / k)
    return Jet(e)


def jlog(x):
    if not isinstance(x, Jet):
        return math.log(x)
    out = [math.log(x.c[0])]
    for k in range(1, len(x.c)):
        out.append((x.c[k] - sum(j * out[j] * x.c[k - j] for j in range(1, k)) / k) / x.c[0])
    return Jet(out)


def _jsincos(x):
    if not isinstance(x, Jet):
        return math.sin(x), math.cos(x)
    s, c = [math.sin(x.c[0])], [math.cos(x.c[0])]
    for k in range(1, len(x.c)):
        s.append(sum(j * x.c[j] * c[k - j] for j in range(1, k + 1)) / k)
        c.append(-sum(j * x.c[j] * s[k - j] for j in range(1, k + 1)) / k)
    return Jet(s), Jet(c)


def jsin(x):
    return _jsincos(x)[0]


def jcos(x):
    return _jsincos(x)[1]


def deriv(f, a, order=1):
    """Derivatives f(a), f'(a), ..., f^(order)(a) from one evaluation on a jet."""
    y = f(Jet.var(a, order))
    c = y.c if isinstance(y, Jet) else [y] + [0.0] * order
    return [c[k] * math.factorial(k) for k in range(order + 1)]


# --------------------------------------------------------------------------
# axis "temperature"
# --------------------------------------------------------------------------

def oplus_h(values, h, axis=0):
    """Thermal sum h*ln(sum(exp(x/h))). h -> 0+ gives max, h -> 0- gives min."""
    x = np.asarray(values, dtype=float) / h
    m = x.max(axis=axis, keepdims=True)
    return h * (np.squeeze(m, axis=axis) + np.log(np.exp(x - m).sum(axis=axis)))


def minplus(A, B):
    """Tropical matrix product: C[i, j] = min_k (A[i, k] + B[k, j])."""
    return (A[:, :, None] + B[None, :, :]).min(axis=1)


# --------------------------------------------------------------------------
# shared numerics
# --------------------------------------------------------------------------

def rk4(f, y, t0, t1, n):
    """Classical Runge-Kutta, n steps from t0 to t1; returns the final state."""
    y, h, t = np.array(y, dtype=complex if np.iscomplexobj(y) else float), (t1 - t0) / n, t0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y, t = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + h
    return y


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _selfcheck():
    e1, e2, e3 = (E3.blade(i) for i in (1, 2, 3))
    assert (e1 * e2 + e2 * e1).norm1() == 0 and (e1 * e1).s() == 1.0
    assert (I3 * I3).s() == -1.0
    i, j = e2 * e3, e1 * e3                       # quaternion units inside Cl(3,0)
    assert (i * i).s() == (j * j).s() == -1.0 and ((i * j) * (i * j)).s() == -1.0
    assert np.allclose(rotate(rotor3([0, 0, 1], math.pi / 2), [1, 0, 0]), [0, 1, 0])
    for kappa in (-2.0, 0.0, 3.0):                # one exponential, three unit kinds
        z = (Algebra(squares=[kappa]).blade(1) * 0.7).exp()
        C, S = unit_exp(kappa, 0.7)
        assert abs(z.s() - C) < 1e-13 and abs(z.coef(1) - S) < 1e-13
    g = Algebra(0, 0, 2)                          # Grassmann: squares vanish, units anticommute
    assert (g.blade(1) * g.blade(1)).norm1() == 0 and (g.blade(1, 2) + g.blade(2, 1)).norm1() == 0
    d = deriv(lambda x: jsin(x * x) / jexp(x), 0.3, 2)
    f = lambda x: math.sin(x * x) / math.exp(x)
    assert abs(d[1] - (f(0.3 + 1e-6) - f(0.3 - 1e-6)) / 2e-6) < 1e-8
    assert abs(d[2] - (f(0.3 + 1e-4) - 2 * f(0.3) + f(0.3 - 1e-4)) / 1e-8) < 1e-6
    assert abs(deriv(lambda x: x ** 0.5, 4.0, 2)[2] + 1 / 32) < 1e-14
    assert abs(oplus_h([1.0, 3.0], 1e-3) - 3.0) < 1e-9 and abs(oplus_h([1.0, 3.0], -1e-3) - 1.0) < 1e-9
    assert abs(oplus_h([2.0, 2.0], 1.0) - (2.0 + math.log(2.0))) < 1e-14
    A = np.array([[0.0, 4.0], [1.0, 0.0]])
    assert np.array_equal(minplus(A, A), [[0.0, 4.0], [1.0, 0.0]])
    assert abs(rk4(lambda t, y: y, [1.0], 0.0, 1.0, 100)[0] - math.e) < 1e-9
    print("gcalc self-check: OK")


if __name__ == "__main__":
    _selfcheck()
