"""Twenty physics problems, two notations each, and a numerical test of agreement.

Every problem carries the classical notation and the notation of the generator
calculus as LaTeX, a symbol count of both, and a check that evaluates both
routes numerically. Nothing in the second paper is typed by hand: this module
writes results_problems.json and paper/data/problems_table.tex.

Usage:  python -m src.problems
"""

from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from src.fga import Arithmetic, HarmonicArithmetic, PowerArithmetic
from src.gcalc import (E3, I3, Algebra, Jet, deriv, jcos, jexp, jlog, jsin, minplus,
                       oplus_h, rk4, rotate, rotor3, unit_exp)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HARM, POW2 = HarmonicArithmetic(), PowerArithmetic(2.0)


def _bisect(f, lo, hi, it=200):
    """Root of a decreasing function on [lo, hi]."""
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def _d(y):
    """First derivative carried by a jet; zero for a constant."""
    return y.c[1] if isinstance(y, Jet) else 0.0


# ==========================================================================
# 1. classical mechanics
# ==========================================================================

def top_rotor(t, I1=1.0, I3m=0.5, w=(0.6, 0.0, 2.0)):
    """Free symmetric top, closed form: R(t) = exp(-t I L/(2 I1)) exp(-t Omega I e3/2), R(0) = 1."""
    L = np.array([I1 * w[0], I1 * w[1], I3m * w[2]])
    return rotor3(L, np.linalg.norm(L) * t / I1) * rotor3([0, 0, 1], (I1 - I3m) / I1 * w[2] * t)


def top_matrix(R):
    return np.column_stack([rotate(R, e) for e in np.eye(3)])


def top_rk4(t, n, I1=1.0, I3m=0.5, w=(0.6, 0.0, 2.0)):
    """Euler's equations together with dR/dt = R [w]x, integrated numerically."""
    def f(_, y):
        R, wb = y[:9].reshape(3, 3), y[9:]
        W = np.array([[0, -wb[2], wb[1]], [wb[2], 0, -wb[0]], [-wb[1], wb[0], 0]])
        k = (I1 - I3m) / I1 * wb[2]
        return np.concatenate([(R @ W).ravel(), [k * wb[1], -k * wb[0], 0.0]])
    return rk4(f, np.concatenate([np.eye(3).ravel(), w]), 0.0, t, n)[:9].reshape(3, 3)


def check_top():
    resid = abs(top_matrix(top_rotor(10.0)) - top_rk4(10.0, 10000)).max()
    return dict(resid=resid, tol=1e-9)


def lj(r):
    """Lennard-Jones potential in reduced units; accepts a float or a jet."""
    x = (1.0 / r) ** 6
    return 4.0 * (x * x - x)


LJ_R0 = 2.0 ** (1.0 / 6.0)
LJ_K = 72.0 / 2.0 ** (1.0 / 3.0)        # V''(r0), derived by hand


def check_lj():
    _, c1, k = deriv(lj, LJ_R0, 2)
    h = np.logspace(-7, -1, 60)                     # central differences, for comparison
    fd = np.abs((lj(LJ_R0 + h) - 2 * lj(LJ_R0) + lj(LJ_R0 - h)) / h ** 2 - LJ_K) / LJ_K
    return dict(resid=max(abs(c1), abs(k - LJ_K) / LJ_K), tol=1e-12,
                facts={"LJk": (k, 3), "LJomega": (math.sqrt(k), 3), "LJfd": (float(fd.min()), None)})


# ==========================================================================
# 2. gravitation and celestial mechanics
# ==========================================================================

def kepler_spinor(s, a=1.0, e=0.6, mu=1.0):
    """Levi-Civita spinor: z = w**2 and w'' = -omega**2 w. Returns (z, t)."""
    om = 0.5 * math.sqrt(mu / a)
    w = math.sqrt(a * (1 - e)) * np.cos(om * s) + 1j * math.sqrt(a * (1 + e)) * np.sin(om * s)
    return w * w, a * s - a * e / (2 * om) * np.sin(2 * om * s)


def _newton(_, y, mu=1.0):
    return np.array([y[1], -mu * y[0] / abs(y[0]) ** 3])


def kepler_newton(t, n, a=1.0, e=0.6, mu=1.0):
    """Newton's equation in physical time from the pericentre. Returns z."""
    y0 = np.array([a * (1 - e), 1j * math.sqrt(mu * (1 + e) / (a * (1 - e)))], dtype=complex)
    return rk4(_newton, y0, 0.0, t, n)[0]


def kepler_period_error(n, e=0.9, regular=True, a=1.0, mu=1.0):
    """Position error after one period with n RK4 steps: in fictitious time s or in time t."""
    z0, om = a * (1 - e), 0.5 * math.sqrt(mu / a)
    if not regular:
        return abs(kepler_newton(2 * math.pi * math.sqrt(a ** 3 / mu), n, a, e, mu) - z0)
    y0 = np.array([math.sqrt(z0), 1j * om * math.sqrt(a * (1 + e))], dtype=complex)
    w = rk4(lambda _, y: np.array([y[1], -om ** 2 * y[0]]), y0, 0.0, math.pi / om, n)[0]
    return abs(w * w - z0)


def check_kepler():
    z, t = kepler_spinor(2.0)                       # eccentric anomaly of 1 rad
    orders = math.log10(kepler_period_error(1000, regular=False) / kepler_period_error(1000, regular=True))
    return dict(resid=abs(z - kepler_newton(float(t), 20000)), tol=1e-9, facts={"KeplerOrders": (round(orders), 0)})


def v_escape(r, mu=1.0):
    return math.sqrt(2 * mu / r)


def check_excess():
    v0 = 1.2 * v_escape(1.0)
    z, v = rk4(_newton, np.array([1.0, 1j * v0], dtype=complex), 0.0, 30.0, 20000)
    classic = math.sqrt(v0 ** 2 - 2.0)              # energy conservation, written out
    new = float(POW2.sub(abs(v), v_escape(abs(z))))  # subtraction in the scale x**2, far away
    oberth = float(POW2.sub(1.1 * v_escape(1.0), v_escape(1.0))) / v_escape(1.0)
    return dict(resid=abs(classic - new), tol=1e-9, facts={"Oberth": (oberth, 3)})


# ==========================================================================
# 3. oscillations, waves, acoustics
# ==========================================================================

def damped(t, gamma, w0=1.0, x0=1.0, v0=0.0):
    """One formula for three regimes: the unit e has e*e = gamma**2 - w0**2."""
    C, S = unit_exp(gamma ** 2 - w0 ** 2, t)
    return np.exp(-gamma * t) * (x0 * C + (v0 + gamma * x0) * S)


def check_damped():
    resid = 0.0
    for gamma in (0.2, 1.0, 2.5):
        num = rk4(lambda _, y: np.array([y[1], -2 * gamma * y[1] - y[0]]), [1.0, 0.0], 0.0, 6.0, 6000)[0]
        resid = max(resid, abs(num - float(damped(6.0, gamma))))
    return dict(resid=resid, tol=1e-9)


class Decibel(Arithmetic):
    """Sound level: the generator maps a level to an intensity."""
    name, domain = "decibel", "R"

    def g(self, x):
        return 10.0 ** (np.asarray(x, dtype=float) / 10.0)

    def g_inv(self, y):
        return 10.0 * np.log10(y)


H_DB = 10.0 / math.log(10.0)     # the "temperature" of the decibel scale


def db_correction(d):
    """Level added to the louder source when the other one is d decibels weaker."""
    return oplus_h(np.array([0.0 * np.asarray(d, dtype=float), -np.asarray(d, dtype=float)]), H_DB)


def check_decibel():
    classic = 10.0 * math.log10(10 ** 8.0 + 10 ** 7.4)
    resid = max(abs(classic - float(oplus_h([80.0, 74.0], H_DB))),
                abs(classic - float(Decibel().add(80.0, 74.0))))
    return dict(resid=resid, tol=1e-12, facts={
        "DbSame": (float(oplus_h([80.0, 80.0], H_DB)), 2),
        "DbTen": (float(db_correction(10.0)), 2), "DbH": (H_DB, 2)})


# ==========================================================================
# 4. thermodynamics and statistical physics
# ==========================================================================

def f_two_phase(T, N):
    """Free energy per particle for two phases f1 = 0 and f2 = 1 - T; accepts a float or a jet."""
    return -(T / N) * jlog(1.0 + jexp(-N * (1.0 - T) / T))


def schottky(T):
    """Heat capacity of a two-level system (gap 1, k_B = 1): -T F'' with F from the thermal sum."""
    return -T * deriv(lambda x: -x * jlog(1.0 + jexp(-1.0 / x)), T, 2)[2]


def check_free_energy():
    resid = 0.0
    for T in (0.5, 0.9, 1.0, 1.3):
        for N in (1, 10, 200):
            direct = -(T / N) * math.log(math.exp(0.0) + math.exp(-N * (1 - T) / T))
            new = float(oplus_h([0.0, 1.0 - T], -T / N))
            assert 0.0 <= min(0.0, 1.0 - T) - new <= T / N * math.log(2) + 1e-15
            resid = max(resid, abs(direct - new), abs(direct - f_two_phase(T, N)))
        exact = (1 / T) ** 2 * math.exp(1 / T) / (1 + math.exp(1 / T)) ** 2
        resid = max(resid, abs(schottky(T) - exact))
    Ts = np.linspace(0.2, 1.0, 8001)
    C = np.array([schottky(T) for T in Ts])
    return dict(resid=resid, tol=1e-12, facts={
        "SchottkyT": (float(Ts[C.argmax()]), 3), "SchottkyC": (float(C.max()), 3)})


def two_form_defect(T, S, p, V, x, y):
    """Coefficient of dT^dS - dp^dV in the coordinates (x, y), in the Grassmann algebra."""
    G = Algebra(0, 0, 2)
    d = lambda f: G.vector([_d(f(Jet.var(x), y)), _d(f(x, Jet.var(y)))])
    return (d(T) * d(S) - d(p) * d(V)).coef(1, 2)


def carnot_cycle(n=2000, Th=2.0, Tc=1.0, V1=1.0, V2=2.0, cv=1.5):
    """Carnot cycle of an ideal gas (R = 1) as closed polylines (V, p, S, T)."""
    k = (Th / Tc) ** cv
    u = np.linspace(0.0, 1.0, n)
    V = np.concatenate([V1 + (V2 - V1) * u, V2 * k ** u, V2 * k + (V1 * k - V2 * k) * u, V1 * k ** (1 - u)])
    T = np.concatenate([Th + 0 * u, Th * (V2 / (V2 * k ** u)) ** (1 / cv), Tc + 0 * u,
                        Tc * (V1 * k / (V1 * k ** (1 - u))) ** (1 / cv)])
    return V, T / V, cv * np.log(T) + np.log(V), T


def loop_area(x, y):
    """Signed area enclosed by a closed polyline (integral of y dx)."""
    xs, ys = np.append(x, x[0]), np.append(y, y[0])
    return float(((ys[1:] + ys[:-1]) * np.diff(xs)).sum() / 2)


def check_maxwell_relations():
    R, cv, a, b = 1.0, 1.5, 0.4, 0.1
    ident = (lambda x, y: x), (lambda x, y: y)
    T_sv = lambda S, V: jexp((S - R * jlog(V - b)) / cv)
    p_sv = lambda S, V: R * T_sv(S, V) / (V - b) - a / (V * V)
    S_tv = lambda T, V: cv * jlog(T) + R * jlog(V - b)
    p_tv = lambda T, V: R * T / (V - b) - a / (V * V)
    T_sp = lambda S, p: jexp((S + R * jlog(p)) / (cv + R))      # ideal gas
    V_sp = lambda S, p: R * T_sp(S, p) / p
    S_tp = lambda T, p: (cv + R) * jlog(T) - R * jlog(p)
    V_tp = lambda T, p: R * T / p
    defects = [two_form_defect(T_sv, ident[0], p_sv, ident[1], 0.7, 1.9),      # van der Waals, (S, V)
               two_form_defect(ident[0], S_tv, p_tv, ident[1], 1.4, 1.9),      # van der Waals, (T, V)
               two_form_defect(T_sp, ident[0], ident[1], V_sp, 0.7, 1.3),      # ideal gas, (S, p)
               two_form_defect(ident[0], S_tp, ident[1], V_tp, 1.4, 1.3)]      # ideal gas, (T, p)
    V, p, S, T = carnot_cycle()
    work, heat = loop_area(V, p), loop_area(S, T)
    return dict(resid=max(max(abs(x) for x in defects), abs(work - heat) / abs(work)), tol=1e-6,
                facts={"CarnotArea": (abs(work), 4)})


# ==========================================================================
# 5. electricity and magnetism
# ==========================================================================

def field_partials(E, B, pt):
    """dE[mu][i], dB[mu][i] for mu in (t, x, y, z): partial derivatives by jets."""
    dE, dB = np.zeros((4, 3)), np.zeros((4, 3))
    for mu in range(4):
        q = [Jet.var(v) if k == mu else v for k, v in enumerate(pt)]
        dE[mu], dB[mu] = [_d(c) for c in E(*q)], [_d(c) for c in B(*q)]
    return dE, dB


def maxwell_classic(dE, dB, c=1.0):
    """Left sides of the four vacuum equations, as 8 numbers in the order of the grades."""
    curl = lambda d: np.array([d[2][2] - d[3][1], d[3][0] - d[1][2], d[1][1] - d[2][0]])
    div = lambda d: d[1][0] + d[2][1] + d[3][2]
    return np.concatenate([[div(dE)], dE[0] / c - c * curl(dB), curl(dE) + dB[0], [c * div(dB)]])


def maxwell_one(dE, dB, c=1.0):
    """Left side of the single equation (d_t/c + nabla) F, F = E + I c B, as 8 numbers."""
    F = lambda mu: E3.vector(dE[mu]) + I3 * E3.vector(dB[mu]) * c
    M = F(0) * (1.0 / c)
    for k in (1, 2, 3):
        M = M + E3.blade(k) * F(k)
    return np.array([M.s(), *M.grade(1).vec(), M.coef(2, 3), -M.coef(1, 3), M.coef(1, 2), M.coef(1, 2, 3)])


def test_field():
    """A smooth field that does not solve the equations, so that no component vanishes."""
    E = lambda t, x, y, z: (jsin(y - 0.7 * t) + 0.3 * x * z, jcos(x + 2.0 * t) * z, x * y + 0.5 * t * t)
    B = lambda t, x, y, z: (0.2 * y * z + jsin(t), jcos(z - t) + x * x, jsin(x * y) - 0.4 * t * z)
    return E, B


def maxwell_samples(n=40, seed=5):
    """Pairs (four equations, one equation) at random points: two arrays of shape (n, 8)."""
    E, B = test_field()
    pts = np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 4))
    parts = [field_partials(E, B, list(pt)) for pt in pts]
    return np.array([maxwell_classic(*d) for d in parts]), np.array([maxwell_one(*d) for d in parts])


def check_maxwell():
    four, one = maxwell_samples()
    wave_E = lambda t, x, y, z: (jcos(z - t), jsin(z - t), 0.0)      # circular plane wave, c = 1
    wave_B = lambda t, x, y, z: (-jsin(z - t), jcos(z - t), 0.0)
    vacuum = abs(maxwell_one(*field_partials(wave_E, wave_B, [0.3, 0.1, -0.4, 0.8]))).max()
    return dict(resid=max(abs(four - one).max(), vacuum), tol=1e-12)


def zin_classic(zL, bl):
    return (zL + 1j * np.tan(bl)) / (1 + 1j * zL * np.tan(bl))


def zin_new(zL, bl):
    """Addition of the electrical length in the scale artanh."""
    return np.tanh(np.arctanh(np.asarray(zL, dtype=complex)) + 1j * np.asarray(bl))


def check_line():
    rng = np.random.default_rng(2)
    zL = rng.uniform(0.2, 3.0, 200) + 1j * rng.uniform(-2.0, 2.0, 200)
    bl = rng.uniform(0.05, 1.45, 200)
    quarter = abs(zin_new(zL, math.pi / 2) - 1 / zL).max()
    return dict(resid=max(abs(zin_classic(zL, bl) - zin_new(zL, bl)).max(), quarter), tol=1e-11)


# ==========================================================================
# 6. optics
# ==========================================================================

def snell_tropical(a=1.0, b=1.0, d=1.5, n1=1.0, n2=1.5, k=200001):
    """Arrival time as a (min, +) product over the interface points. Returns (T, x)."""
    x = np.linspace(0.0, d, k)
    t = n1 * np.hypot(a, x) + n2 * np.hypot(b, d - x)
    return float(t.min()), float(x[t.argmin()])


def snell_classic(a=1.0, b=1.0, d=1.5, n1=1.0, n2=1.5):
    """Snell's law solved for the crossing point by bisection. Returns (T, x)."""
    x = _bisect(lambda x: n2 * (d - x) / math.hypot(b, d - x) - n1 * x / math.hypot(a, x), 0.0, d)
    return n1 * math.hypot(a, x) + n2 * math.hypot(b, d - x), x


def check_snell():
    (Tt, xt), (Tc, xc) = snell_tropical(), snell_classic()
    x = np.linspace(0.0, 1.5, 2001)                  # the same sum as a tropical matrix product
    prod = minplus((1.0 * np.hypot(1.0, x))[None, :], (1.5 * np.hypot(1.0, 1.5 - x))[:, None])[0, 0]
    assert abs(prod - Tc) < 1e-6
    defect = abs(1.0 * xt / math.hypot(1.0, xt) - 1.5 * (1.5 - xt) / math.hypot(1.0, 1.5 - xt))
    return dict(resid=abs(Tt - Tc), tol=1e-9, facts={"SnellDefect": (defect, None)})


def slits_classic(delta, N):
    k = np.arange(N)[:, None]
    return np.cos(k * delta).sum(0) ** 2 + np.sin(k * delta).sum(0) ** 2


def slits_new(delta, N):
    q = np.exp(1j * np.asarray(delta))
    return np.abs((1 - q ** N) / (1 - q)) ** 2


def check_slits():
    delta = np.random.default_rng(3).uniform(0.1, 6.1, 300)
    resid = max(abs(slits_classic(delta, N) - slits_new(delta, N)).max() / N ** 2 for N in (2, 5, 10))
    return dict(resid=resid, tol=1e-12)


# ==========================================================================
# 7. relativity
# ==========================================================================

def boost_matrix(eta, n):
    n, B = np.asarray(n, dtype=float), np.eye(4)
    B[0, 0] = math.cosh(eta)
    B[0, 1:] = B[1:, 0] = math.sinh(eta) * n
    B[1:, 1:] += (math.cosh(eta) - 1) * np.outer(n, n)
    return B


def wigner_matrix(eta1, eta2, alpha):
    """Rotation angle in B2 B1 = R B from 4x4 matrices and a polar decomposition."""
    L = boost_matrix(eta2, [math.cos(alpha), math.sin(alpha), 0]) @ boost_matrix(eta1, [1, 0, 0])
    w, V = np.linalg.eigh(L.T @ L)
    R = L @ V @ np.diag(w ** -0.5) @ V.T
    return math.acos(max(-1.0, min(1.0, (np.trace(R[1:, 1:]) - 1) / 2)))


def wigner_rotor(eta1, eta2, alpha):
    """The same angle from the product of two exponentials in Cl(3,0)."""
    boost = lambda eta, n: (E3.vector(n) * (eta / 2)).exp()
    L = boost(eta2, [math.cos(alpha), math.sin(alpha), 0]) * boost(eta1, [1, 0, 0])
    B2 = ~L * L                                       # the square of the boost part: scalar + vector
    eta = math.acosh(B2.s())
    R = L * boost(-eta, B2.grade(1).vec() / math.sinh(eta))
    return 2 * math.atan2(math.sqrt(sum(v * v for v in R.grade(2).c.values())), R.s())


def wigner_formula(eta1, eta2, alpha):
    return 2 * math.atan2(math.sin(alpha), 1 / (math.tanh(eta1 / 2) * math.tanh(eta2 / 2)) + math.cos(alpha))


def check_wigner():
    resid = 0.0
    for e1, e2, al in ((0.3, 0.5, 1.0), (1.2, 0.8, math.pi / 2), (2.0, 2.0, 2.5)):
        m, r, f = wigner_matrix(e1, e2, al), wigner_rotor(e1, e2, al), wigner_formula(e1, e2, al)
        resid = max(resid, abs(m - r), abs(f - r))
    eta = math.atanh(0.9)
    small = wigner_rotor(0.1, 0.1, math.pi / 2) / (0.5 * 0.1 * 0.1)      # commutator estimate
    return dict(resid=resid, tol=1e-10, facts={
        "WignerDeg": (math.degrees(wigner_rotor(eta, eta, math.pi / 2)), 1),
        "WignerSmall": (small, 3)})


# Sources: GM of the Sun, IAU 2015 Resolution B3; orbit of Mercury, NASA Mercury Fact Sheet.
MERCURY = dict(GM=1.3271244e20, c=299792458.0, a=57.909e9, e=0.2056, period_days=87.969)


def precession_jet():
    """Advance of the perihelion per orbit divided by lambda: arithmetic with lambda**2 = 0."""
    lam, u0 = Jet([0.0, 1.0]), 1.0
    for _ in range(3):
        u0 = 1.0 + lam * u0 * u0                      # circular orbit as a fixed point
    Om = (1.0 - 2.0 * lam * u0) ** 0.5                # frequency of the radial oscillation
    return (2 * math.pi * (1.0 / Om - 1.0)).c[1]


def precession_numeric(lam, e=0.2, h=1e-3):
    """Angle between two successive perihelia of u'' + u = 1 + lam u**2, minus 2 pi."""
    f = lambda _, y: np.array([y[1], 1.0 + lam * y[0] ** 2 - y[0]])
    y, phi, past = np.array([1.0 + e, 0.0]), 0.0, False
    while True:
        y1 = rk4(f, y, 0.0, h, 1)
        past = past or y1[1] > 0
        if past and y1[1] <= 0:
            break
        y, phi = y1, phi + h
    return phi + _bisect(lambda s: rk4(f, y, 0.0, s, 1)[1], 0.0, h, 60) - 2 * math.pi


def mercury_lambda():
    m = MERCURY
    return 3 * m["GM"] / (m["c"] ** 2 * m["a"] * (1 - m["e"] ** 2))


def check_mercury():
    lam = 1e-5
    num, jet = precession_numeric(lam), precession_jet() * lam
    arcsec = 2 * math.pi * mercury_lambda() * 36525.0 / MERCURY["period_days"] * math.degrees(1) * 3600
    return dict(resid=abs(num - jet) / jet, tol=1e-3, facts={
        "MercuryArcsec": (arcsec, 2), "MercuryLambda": (mercury_lambda() * 1e8, 2)})


# ==========================================================================
# 8. quantum mechanics
# ==========================================================================

def rabi_rotor(t, Om=1.0, De=0.0):
    """Flip probability (1 - s.e3)/2 with s = R e3 R~, R = exp(-t/2 I (Om e1 + De e3))."""
    R = (I3 * E3.vector([Om, 0.0, De]) * (-t / 2)).exp()
    return (1 - rotate(R, [0, 0, 1])[2]) / 2


def rabi_schrodinger(t, Om=1.0, De=0.0, n=2000):
    H = 0.5 * np.array([[De, Om], [Om, -De]], dtype=complex)
    psi = rk4(lambda _, y: -1j * H @ y, np.array([1, 0], dtype=complex), 0.0, t, n)
    return abs(psi[1]) ** 2


def rabi_formula(t, Om=1.0, De=0.0):
    W = math.hypot(Om, De)
    return Om ** 2 / W ** 2 * np.sin(W * np.asarray(t) / 2) ** 2


def check_rabi():
    resid = 0.0
    for t, De in ((1.0, 0.0), (4.0, 1.0), (7.5, 2.0)):
        r = rabi_rotor(t, 1.0, De)
        resid = max(resid, abs(r - rabi_schrodinger(t, 1.0, De)), abs(r - float(rabi_formula(t, 1.0, De))))
    return dict(resid=resid, tol=1e-10)


PATH = dict(m=1.0, X=2.0, T=1.0)                    # free particle from (0, 0) to (X, T)
S_CL = PATH["m"] * PATH["X"] ** 2 / (2 * PATH["T"])  # action of the straight path


def path_action(x):
    """Action of the broken path through the point x at half time."""
    return PATH["m"] / PATH["T"] * (x ** 2 + (PATH["X"] - x) ** 2)


def path_sum(hbar, L=12.0, dx=1e-4):
    """Sum of exp(i S(x)/hbar) over the midpoints x, with a smooth window of half-width L."""
    x = np.arange(PATH["X"] / 2 - L, PATH["X"] / 2 + L, dx)
    win = np.cos(np.pi * (x - PATH["X"] / 2) / (2 * L)) ** 2
    return (np.exp(1j * path_action(x) / hbar) * win).sum() * dx


def check_path_sum():
    resid, a = 0.0, 2 * PATH["m"] / PATH["T"]
    for hbar in (0.1, 0.05, 0.02):
        I = path_sum(hbar) * np.exp(-1j * S_CL / hbar)
        # h ln(sum) with h = -i hbar equals S_cl + hbar pi/4 - (i hbar/2) ln(pi hbar/a)
        resid = max(resid, abs(np.angle(I) - math.pi / 4), abs(abs(I) / math.sqrt(math.pi * hbar / a) - 1))
    return dict(resid=resid, tol=1e-3)                # the window adds a term of order hbar/L**2


# ==========================================================================
# 9. atomic, nuclear and particle physics
# ==========================================================================

HALF = dict(T1=8.0, T2=80.0)                        # example values, in days


def check_half_life():
    T = float(HARM.add(HALF["T1"], HALF["T2"]))
    lam = math.log(2) / HALF["T1"] + math.log(2) / HALF["T2"]
    N = rk4(lambda _, y: -lam * y, [1.0], 0.0, T, 2000)[0]      # two channels, integrated
    return dict(resid=abs(N - 0.5), tol=1e-12, facts={"HalfEff": (T, 2)})


def compton_classic(E, th):
    """Energy and two momentum components, solved numerically (m = c = 1)."""
    def f(E2):
        p2 = (E - E2 * math.cos(th)) ** 2 + (E2 * math.sin(th)) ** 2
        return E + 1.0 - E2 - math.sqrt(p2 + 1.0)
    return _bisect(f, 0.0, E)


def compton_new(E, th):
    """Harmonic sum: 1/E' = 1/E + (1 - cos th)."""
    return HARM.add(E, 1.0 / (1.0 - np.cos(th)))


def check_compton():
    ST, resid = Algebra(1, 3), 0.0
    for E, th in ((0.1, 0.5), (1.0, 1.5), (10.0, 2.8)):
        E2 = float(compton_new(E, th))
        p2 = ST.vector([1.0, 0, 0, 0]) + ST.vector([E, E, 0, 0]) - ST.vector(
            [E2, E2 * math.cos(th), E2 * math.sin(th), 0])
        resid = max(resid, abs(E2 - compton_classic(E, th)) / E2, abs((p2 * p2).s() - 1.0))
    return dict(resid=resid, tol=1e-12)


# ==========================================================================
# 10. condensed matter and fluids
# ==========================================================================

def phonon_matrix(N):
    """Squared frequencies of a ring of N masses (K = m = 1) from the dynamical matrix."""
    D = 2 * np.eye(N) - np.roll(np.eye(N), 1, 0) - np.roll(np.eye(N), -1, 0)
    return np.sort(np.linalg.eigvalsh(D))


def phonon_cyclic(N):
    """The same numbers: the cyclic number 2 - s - 1/s evaluated at the N-th roots of unity."""
    s = np.exp(2j * np.pi * np.arange(N) / N)
    return np.sort((2 - s - 1 / s).real)


def check_phonons():
    resid = max(abs(phonon_matrix(N) - phonon_cyclic(N)).max() for N in (5, 12, 64))
    return dict(resid=resid, tol=1e-12)


def cylinder_dw(z, U=1.0, Gam=2.0, a=1.0):
    """Complex velocity u - i v of the flow past a cylinder with circulation."""
    return U * (1 - a ** 2 / z ** 2) + 1j * Gam / (2 * np.pi * z)


def cylinder_force(U=1.0, Gam=2.0, a=1.0, rho=1.0, n=720):
    """Pressure integrated over the surface. Returns (X, Y)."""
    th = np.arange(n) * 2 * np.pi / n
    p = 0.5 * rho * (U ** 2 - np.abs(cylinder_dw(a * np.exp(1j * th), U, Gam, a)) ** 2)
    return -(p * np.cos(th)).sum() * a * 2 * np.pi / n, -(p * np.sin(th)).sum() * a * 2 * np.pi / n


def check_cylinder():
    U, Gam, rho, n = 1.0, 2.0, 1.0, 720
    X, Y = cylinder_force(U, Gam)
    z = 2.0 * np.exp(1j * np.arange(n) * 2 * np.pi / n)             # Blasius integral on |z| = 2
    blasius = 0.5j * rho * (cylinder_dw(z, U, Gam) ** 2 * 1j * z).sum() * 2 * np.pi / n
    return dict(resid=max(abs(X), abs(Y - rho * U * Gam), abs(blasius + 1j * rho * U * Gam)), tol=1e-12)


# ==========================================================================
# the catalogue
# ==========================================================================

BRANCHES = ["Mechanika klasyczna", "Grawitacja i mechanika nieba", "Drgania, fale i akustyka",
            "Termodynamika i fizyka statystyczna", "Elektryczność i magnetyzm", "Optyka",
            "Teoria względności", "Mechanika kwantowa", "Fizyka atomowa, jądrowa i cząstek",
            "Materia skondensowana i płyny"]

PROBLEMS = [
    dict(id="A", title="Swobodny bąk symetryczny", axes="jednostka", check=check_top,
         classic=r"\theta = \arccos\frac{I_3\omega_3}{L} \\ \varphi(t) = \frac{L}{I_1}\,t \\ "
                 r"\psi(t) = \frac{I_1 - I_3}{I_1}\,\omega_3 t \\ "
                 r"\mathsf{R}(t) = \mathsf{R}_z(\varphi)\,\mathsf{R}_x(\theta)\,\mathsf{R}_z(\psi)",
         new=r"R(t) = \exp\left(-\frac{t}{2I_1}\,\mathcal{I}\mathbf{L}\right) R_0\, "
             r"\exp\left(-\frac{t}{2}\,\Omega\,\mathcal{I}e_3\right) \\ \Omega = \frac{I_1 - I_3}{I_1}\,\omega_3"),
    dict(id="B", title="Częstość drgań wokół minimum", axes="jednostka", check=check_lj,
         classic=r"V'(r) = 4\epsilon\left(-\frac{12\sigma^{12}}{r^{13}} + \frac{6\sigma^{6}}{r^{7}}\right) \\ "
                 r"V''(r) = 4\epsilon\left(\frac{156\sigma^{12}}{r^{14}} - \frac{42\sigma^{6}}{r^{8}}\right) \\ "
                 r"V'(r_0) = 0, \quad \omega = \sqrt{\frac{V''(r_0)}{m}}",
         new=r"V(r_0 + \varepsilon) = V_0 + c_1\varepsilon + c_2\varepsilon^2, \quad \varepsilon^3 = 0 \\ "
             r"c_1 = 0, \quad \omega = \sqrt{\frac{2c_2}{m}}"),
    dict(id="C", title="Orbita Keplera", axes="jednostka", check=check_kepler,
         classic=r"M = \sqrt{\frac{\mu}{a^3}}\,t \\ M = E - e\sin E \\ x = a(\cos E - e) \\ "
                 r"y = a\sqrt{1 - e^2}\,\sin E",
         new=r"z = w^2, \quad \frac{d^2w}{ds^2} = -\omega^2 w, \quad \omega = \frac{1}{2}\sqrt{\frac{\mu}{a}} \\ "
             r"w = \sqrt{a(1-e)}\cos\omega s + i\sqrt{a(1+e)}\sin\omega s \\ "
             r"t = as - \frac{ae}{2\omega}\sin 2\omega s"),
    dict(id="D", title="Nadwyżka hiperboliczna", axes="skala", check=check_excess,
         classic=r"v_\infty = \sqrt{(v_p + \Delta v)^2 - \frac{2\mu}{r_p}}",
         new=r"v_\infty = (v_p + \Delta v) \ominus v_e, \quad v_e = \sqrt{\frac{2\mu}{r_p}}, \quad g(v) = v^2"),
    dict(id="E", title="Oscylator tłumiony", axes="jednostka", check=check_damped,
         classic=r"x = e^{-\gamma t}\left(x_0\cos\Omega t + \frac{v_0 + \gamma x_0}{\Omega}\sin\Omega t\right), "
                 r"\quad \gamma < \omega_0 \\ "
                 r"x = e^{-\gamma t}\left(x_0 + (v_0 + \gamma x_0)\,t\right), \quad \gamma = \omega_0 \\ "
                 r"x = e^{-\gamma t}\left(x_0\cosh\Omega t + \frac{v_0 + \gamma x_0}{\Omega}\sinh\Omega t\right), "
                 r"\quad \gamma > \omega_0 \\ \Omega = \sqrt{|\omega_0^2 - \gamma^2|}",
         new=r"x = e^{-\gamma t}\left(x_0 C + (v_0 + \gamma x_0) S\right) \\ "
             r"C + \mathbf{e}S = \exp(\mathbf{e}t), \quad \mathbf{e}^2 = \gamma^2 - \omega_0^2"),
    dict(id="F", title="Suma poziomów dźwięku", axes="temperatura", check=check_decibel,
         classic=r"L = 10\log_{10}\left(10^{L_1/10} + 10^{L_2/10}\right)",
         new=r"L = L_1 \oplus_h L_2, \quad h = \frac{10}{\ln 10}"),
    dict(id="G", title="Energia swobodna dwóch faz", axes="temperatura", check=check_free_energy,
         classic=r"Z = e^{-Nf_1/T} + e^{-Nf_2/T} \\ f = -\frac{T}{N}\ln Z \\ "
                 r"\lim_{N\to\infty} f = \min(f_1, f_2)",
         new=r"f = f_1 \oplus_h f_2, \quad h = -\frac{T}{N} \\ \lim_{h\to 0^-} f = \min(f_1, f_2)"),
    dict(id="H", title="Relacje Maxwella", axes="jednostka", check=check_maxwell_relations,
         classic=r"\left(\frac{\partial T}{\partial V}\right)_S = -\left(\frac{\partial p}{\partial S}\right)_V \\ "
                 r"\left(\frac{\partial T}{\partial p}\right)_S = \left(\frac{\partial V}{\partial S}\right)_p \\ "
                 r"\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial p}{\partial T}\right)_V \\ "
                 r"\left(\frac{\partial S}{\partial p}\right)_T = -\left(\frac{\partial V}{\partial T}\right)_p",
         new=r"dT \wedge dS = dp \wedge dV"),
    dict(id="I", title="Równania Maxwella", axes="jednostka", check=check_maxwell,
         classic=r"\nabla\cdot\mathbf{E} = \frac{\rho}{\varepsilon_0} \\ \nabla\cdot\mathbf{B} = 0 \\ "
                 r"\nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t} \\ "
                 r"\nabla\times\mathbf{B} = \mu_0\mathbf{J} + \frac{1}{c^2}\frac{\partial\mathbf{E}}{\partial t}",
         new=r"\left(\frac{1}{c}\frac{\partial}{\partial t} + \nabla\right)F = "
             r"\frac{\rho}{\varepsilon_0} - c\mu_0\mathbf{J}, \quad F = \mathbf{E} + \mathcal{I}c\mathbf{B}"),
    dict(id="J", title="Impedancja wejściowa linii", axes="skala", check=check_line,
         classic=r"Z_i = Z_0\,\frac{Z_L + jZ_0\tan\beta l}{Z_0 + jZ_L\tan\beta l}",
         new=r"g(z_i) = g(z_L) + j\beta l, \quad g = \artanh, \quad z = \frac{Z}{Z_0}"),
    dict(id="K", title="Załamanie światła", axes="temperatura", check=check_snell,
         classic=r"n_1\sin\theta_1 = n_2\sin\theta_2 \\ \sin\theta_1 = \frac{x}{\sqrt{a^2 + x^2}}, \quad "
                 r"\sin\theta_2 = \frac{d - x}{\sqrt{b^2 + (d - x)^2}} \\ "
                 r"T = \frac{n_1}{c}\sqrt{a^2 + x^2} + \frac{n_2}{c}\sqrt{b^2 + (d - x)^2}",
         new=r"T = \bigoplus_x \left(T_1(x) + T_2(x)\right), \quad \oplus = \min \\ "
             r"T_1 = \frac{n_1}{c}\sqrt{a^2 + x^2}, \quad T_2 = \frac{n_2}{c}\sqrt{b^2 + (d - x)^2}"),
    dict(id="L", title="Dyfrakcja na N szczelinach", axes="jednostka", check=check_slits,
         classic=r"I = I_0\left(\sum_{k=0}^{N-1}\cos k\delta\right)^2 + I_0\left(\sum_{k=0}^{N-1}\sin k\delta\right)^2 \\ "
                 r"\sum_{k=0}^{N-1}\cos k\delta = \frac{\sin\frac{N\delta}{2}}{\sin\frac{\delta}{2}}\cos\frac{(N-1)\delta}{2} \\ "
                 r"\sum_{k=0}^{N-1}\sin k\delta = \frac{\sin\frac{N\delta}{2}}{\sin\frac{\delta}{2}}\sin\frac{(N-1)\delta}{2}",
         new=r"I = I_0\left|\sum_{k=0}^{N-1} q^k\right|^2 = I_0\left|\frac{1 - q^N}{1 - q}\right|^2, "
             r"\quad q = \exp(i\delta)"),
    dict(id="M", title="Obrót Thomasa--Wignera", axes="komutator", check=check_wigner,
         classic=r"\tan\frac{\theta}{2} = \frac{\sin\alpha}{D + \cos\alpha}, \quad "
                 r"D = \sqrt{\frac{(\gamma_1 + 1)(\gamma_2 + 1)}{(\gamma_1 - 1)(\gamma_2 - 1)}}",
         new=r"R\,B = \exp\left(\frac{\eta_2}{2}\mathbf{n}_2\right)\exp\left(\frac{\eta_1}{2}\mathbf{n}_1\right) \\ "
             r"\theta \approx \frac{1}{2}\eta_1\eta_2\sin\alpha"),
    dict(id="N", title="Peryhelium Merkurego", axes="jednostka", check=check_mercury,
         classic=r"\frac{d^2u}{d\varphi^2} + u = \frac{\mu}{h^2} + \frac{3\mu}{c^2}u^2 \\ "
                 r"u \approx \frac{\mu}{h^2}\left(1 + e\cos\varphi + \lambda e\varphi\sin\varphi\right) \\ "
                 r"u \approx \frac{\mu}{h^2}\left(1 + e\cos((1 - \lambda)\varphi)\right) \\ "
                 r"\Delta\varphi = 2\pi\lambda, \quad \lambda = \frac{3\mu^2}{c^2h^2}",
         new=r"u_0 = 1 + \lambda u_0^2, \quad \Omega^2 = 1 - 2\lambda u_0, \quad \lambda^2 = 0 \\ "
             r"\Delta\varphi = 2\pi\left(\frac{1}{\Omega} - 1\right) = 2\pi\lambda, "
             r"\quad \lambda = \frac{3\mu^2}{c^2h^2}"),
    dict(id="O", title="Oscylacje Rabiego spinu", axes="jednostka", check=check_rabi,
         classic=r"i\hbar\frac{d\psi}{dt} = \frac{\hbar}{2}\begin{pmatrix}\Delta & \Omega_R \\ "
                 r"\Omega_R & -\Delta\end{pmatrix}\psi \\ "
                 r"P = \frac{\Omega_R^2}{\Omega_R^2 + \Delta^2}\sin^2\frac{\sqrt{\Omega_R^2 + \Delta^2}\,t}{2}",
         new=r"\mathbf{s}(t) = R\,e_3\,\tilde{R}, \quad R = \exp\left(-\frac{t}{2}\,\mathcal{I}"
             r"(\Omega_R e_1 + \Delta e_3)\right) \\ P = \frac{1 - \mathbf{s}\cdot e_3}{2}"),
    dict(id="P", title="Najmniejsze działanie", axes="temperatura", check=check_path_sum,
         classic=r"\frac{d}{dt}\frac{\partial L}{\partial\dot{x}} - \frac{\partial L}{\partial x} = 0 \\ "
                 r"S = \int_0^T L\,dt = \frac{mX^2}{2T}",
         new=r"S = \lim_{\hbar\to 0}\,\bigoplus_{x} S(x), \quad \oplus = \oplus_h, \quad h = -i\hbar"),
    dict(id="Q", title="Efektywny okres półtrwania", axes="skala", check=check_half_life,
         classic=r"\lambda = \frac{\ln 2}{T_1} + \frac{\ln 2}{T_2} \\ "
                 r"T = \frac{\ln 2}{\lambda} = \frac{T_1T_2}{T_1 + T_2}",
         new=r"T = T_1 \oplus T_2, \quad g(x) = \frac{1}{x}"),
    dict(id="R", title="Rozpraszanie Comptona", axes="jednostka", check=check_compton,
         classic=r"E + mc^2 = E' + \sqrt{p^2c^2 + m^2c^4} \\ "
                 r"\frac{E}{c} = \frac{E'}{c}\cos\theta + p\cos\phi \\ "
                 r"0 = \frac{E'}{c}\sin\theta - p\sin\phi \\ "
                 r"\frac{1}{E'} = \frac{1}{E} + \frac{1 - \cos\theta}{mc^2}",
         new=r"(p + k - k')^2 = m^2c^2 \\ E' = E \oplus \frac{mc^2}{1 - \cos\theta}, \quad g(x) = \frac{1}{x}"),
    dict(id="S", title="Fonony w łańcuchu atomów", axes="jednostka", check=check_phonons,
         classic=r"m\ddot{u}_n = K(u_{n+1} - 2u_n + u_{n-1}) \\ "
                 r"u_n = A\exp(i(kna - \omega t)), \quad u_{n+N} = u_n \\ "
                 r"\omega_k = 2\sqrt{\frac{K}{m}}\left|\sin\frac{\pi k}{N}\right|",
         new=r"D = \frac{K}{m}\left(2 - s - s^{-1}\right), \quad s^N = 1 \\ "
             r"\omega_k^2 = D\left(s = \exp\frac{2\pi ik}{N}\right) = \frac{4K}{m}\sin^2\frac{\pi k}{N}"),
    dict(id="T", title="Opływ walca z cyrkulacją", axes="jednostka", check=check_cylinder,
         classic=r"\phi = U\left(r + \frac{a^2}{r}\right)\cos\theta - \frac{\Gamma}{2\pi}\theta \\ "
                 r"p = p_\infty + \frac{\rho}{2}\left(U^2 - \left(2U\sin\theta + \frac{\Gamma}{2\pi a}\right)^2\right) \\ "
                 r"Y = -\int_0^{2\pi} p\sin\theta\,a\,d\theta = \rho U\Gamma",
         new=r"w = U\left(z + \frac{a^2}{z}\right) + \frac{i\Gamma}{2\pi}\ln z \\ "
             r"X - iY = \frac{i\rho}{2}\oint\left(\frac{dw}{dz}\right)^2 dz = -i\rho U\Gamma"),
]
for _i, _p in enumerate(PROBLEMS):
    _p["branch"] = BRANCHES[_i // 2]


# ==========================================================================
# length of a notation
# ==========================================================================

_FORMAT = (r"\\(?:left|right|[bB]igg?[lrm]?|q?quad|mathrm|mathbf|mathsf|mathcal|displaystyle|"
           r"end\{[a-z]+\}|[,;!: ]|\\)")


def tex_len(s):
    """Number of symbols a reader has to process in a LaTeX formula.

    One symbol: a command (\\frac, \\sin, \\omega, \\oplus), a letter, a number,
    an operator or a delimiter. Not counted: braces, the subscript sign, spacing
    and font commands. Both notations are counted by the same rule.
    """
    # ponytail: naive symbol count, not a description length; replace by an
    # encoder over a fixed grammar if the comparison has to carry more weight
    s = re.sub(r"\\begin\{[a-z]+\}", r"\\matrix ", s)
    s = re.sub(_FORMAT, " ", s)
    return len(re.findall(r"\\[A-Za-z]+|[A-Za-z]|\d+(?:[.,]\d+)?|[^\sA-Za-z\d{}_&\\]", s))


# ==========================================================================
# run, write, report
# ==========================================================================

def run(p):
    out = p["check"]()
    n_c, n_n = tex_len(p["classic"]), tex_len(p["new"])
    return dict(id=p["id"], branch=p["branch"], title=p["title"], axes=p["axes"],
                resid=float(out["resid"]), tol=out["tol"], ok=bool(out["resid"] <= out["tol"]),
                len_classic=n_c, len_new=n_n, ratio=n_n / n_c, facts=out.get("facts", {}))


def _pl(x, nd):
    """Polish decimal comma; nd=None gives scientific notation for math mode."""
    if nd is None:
        m, e = f"{x:.1e}".split("e")
        return m.replace(".", "{,}") + r"\cdot10^{" + str(int(e)) + "}"
    return f"{x:.{nd}f}".replace(".", "{,}")


def write_tex(results, path):
    ratios = sorted(r["ratio"] for r in results)
    median = (ratios[9] + ratios[10]) / 2
    shorter = sum(r["len_new"] < r["len_classic"] for r in results)
    longer = sum(r["len_new"] > r["len_classic"] for r in results)
    lines = ["% Generated by python -m src.problems. Do not edit.",
             rf"\newcommand{{\gcShorter}}{{{shorter}}}",
             rf"\newcommand{{\gcLonger}}{{{longer}}}",
             rf"\newcommand{{\gcEqual}}{{{len(results) - shorter - longer}}}",
             rf"\newcommand{{\gcMedian}}{{{_pl(median, 2)}}}",
             rf"\newcommand{{\gcRatioMin}}{{{_pl(ratios[0], 2)}}}",
             rf"\newcommand{{\gcRatioMax}}{{{_pl(ratios[-1], 2)}}}",
             rf"\newcommand{{\gcSumClassic}}{{{sum(r['len_classic'] for r in results)}}}",
             rf"\newcommand{{\gcSumNew}}{{{sum(r['len_new'] for r in results)}}}",
             rf"\newcommand{{\gcResidMax}}{{{_pl(max(r['resid'] for r in results if r['tol'] < 1e-6), None)}}}",
             rf"\newcommand{{\gcTight}}{{{sum(r['tol'] < 1e-6 for r in results)}}}",
             rf"\newcommand{{\gcRatioMinId}}{{{min(results, key=lambda r: r['ratio'])['id']}}}",
             rf"\newcommand{{\gcRatioMaxId}}{{{max(results, key=lambda r: r['ratio'])['id']}}}",
             rf"\newcommand{{\gcLongerIds}}{{{', '.join(r['id'] for r in results if r['len_new'] > r['len_classic'])}}}"]
    rows = []
    for p, r in zip(PROBLEMS, results):
        lines += [rf"\newcommand{{\gcClassic{p['id']}}}{{{p['classic']}}}",
                  rf"\newcommand{{\gcNew{p['id']}}}{{{p['new']}}}",
                  rf"\newcommand{{\gcLenClassic{p['id']}}}{{{r['len_classic']}}}",
                  rf"\newcommand{{\gcLenNew{p['id']}}}{{{r['len_new']}}}",
                  rf"\newcommand{{\gcResid{p['id']}}}{{{_pl(r['resid'], None)}}}"]
        for name, (value, nd) in r["facts"].items():
            assert name.isalpha(), name
            lines.append(rf"\newcommand{{\gc{name}}}{{{_pl(value, nd)}}}")
        rows.append(rf"{p['id']} & {p['title']} & {r['axes']} & {r['len_classic']} & {r['len_new']} & "
                    rf"${_pl(r['ratio'], 2)}$ & ${_pl(r['resid'], None)}$ \\")
    lines.append(r"\newcommand{\gcRows}{%")
    lines += rows
    lines.append("}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    results = [run(p) for p in PROBLEMS]
    with open(os.path.join(ROOT, "results_problems.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    write_tex(results, os.path.join(ROOT, "paper", "data", "problems_table.tex"))
    print(f"{'id':3s}{'problem':30s}{'axis':13s}{'classic':>8s}{'new':>5s}{'ratio':>7s}{'resid':>10s}{'tol':>9s}")
    for r in results:
        print(f"{r['id']:3s}{r['title']:30s}{r['axes']:13s}{r['len_classic']:8d}{r['len_new']:5d}"
              f"{r['ratio']:7.2f}{r['resid']:10.1e}{r['tol']:9.0e}  {'ok' if r['ok'] else 'FAIL'}")
    bad = [r["id"] for r in results if not r["ok"]]
    print(f"agreement: {len(results) - len(bad)} of {len(results)}; failed: {bad if bad else 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
