"""Numerical verification for the paper.

Checks the algebraic laws of every generator, classifies the induced structure,
measures the cost of the closed-form inverses, and writes the data files that
the pgfplots figures read.

The laws are evaluated twice: once in the chart (the linear coordinates y = g(x),
where the operation is ordinary addition) and once in the original coordinates.
The first number measures arithmetic round-off; the second also measures the
conditioning of g and g^-1, which is what an implementation actually pays.

Usage:  python scripts/verify.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

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
    lambert_w,
    solve_monotone,
)

DATA = os.path.join(ROOT, "paper", "data")
os.makedirs(DATA, exist_ok=True)


def rel(x, y):
    return abs(x - y) / max(1.0, abs(x), abs(y))


# --------------------------------------------------------------------------
# 1. algebraic laws
# --------------------------------------------------------------------------

def check_laws(model, lo, hi, structure, n=400, seed=7):
    """structure: 'group' | 'monoid' | 'semigroup'."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(lo, hi, n)
    b = rng.uniform(lo, hi, n)
    c = rng.uniform(lo, hi, n)

    chart_assoc = 0.0
    coord_assoc = 0.0
    for x, y, z in zip(a, b, c):
        left, right = model.add(model.add(x, y), z), model.add(x, model.add(y, z))
        coord_assoc = max(coord_assoc, rel(left, right))
        # the same comparison carried out in the chart
        chart_assoc = max(chart_assoc, rel(model.g(left), model.g(right)))

    dist = max(
        rel(model.mul(x, model.add(y, z)),
            model.add(model.mul(x, y), model.mul(x, z)))
        for x, y, z in zip(a, b, c)
    )

    neutral = float("nan")
    inverse = float("nan")
    inverse_chart = float("nan")
    z0 = float("nan")
    if structure in ("group", "monoid"):
        z0 = float(model.zero())
        neutral = max(rel(model.add(x, z0), x) for x in a)
    if structure == "group":
        inverse = max(rel(model.add(x, model.neg(x)), z0) for x in a)
        inverse_chart = max(
            abs(model.g(model.add(x, model.neg(x))) - model.g(z0))
            / max(1.0, abs(model.g(x)))
            for x in a
        )

    return {
        "name": model.name,
        "domain": model.domain,
        "structure": structure,
        "zero": z0,
        "one": float(model.one()) if structure != "semigroup" else float("nan"),
        "chart_associativity": chart_assoc,
        "coord_associativity": coord_assoc,
        "neutral_element": neutral,
        "inverse": inverse,
        "inverse_chart": inverse_chart,
        "distributivity": dist,
    }


MODELS = [
    (PowerArithmetic(2.0), 0.1, 50.0, "monoid"),
    (CubicArithmetic(), -8.0, 8.0, "group"),
    (GeometricArithmetic(), 0.05, 20.0, "group"),
    (LogisticArithmetic(), 0.02, 0.98, "group"),
    (SoftplusArithmetic(), 0.05, 20.0, "group"),
    (LambertArithmetic(), 0.02, 20.0, "group"),
    (RelativisticArithmetic(1.0), -0.99, 0.99, "group"),
    (KeplerArithmetic(0.85), -6.0, 6.0, "group"),
    (TsallisArithmetic(0.5), -1.5, 20.0, "group"),
    (HarmonicArithmetic(), 0.5, 9.0, "semigroup"),
]


def run_laws():
    return [check_laws(m, lo, hi, s) for m, lo, hi, s in MODELS]


# --------------------------------------------------------------------------
# 2. constants
# --------------------------------------------------------------------------

def run_constants():
    lam, kep = LambertArithmetic(), KeplerArithmetic(0.85)
    return {
        "omega_W1": lambert_w(1.0),
        "omega_textbook": 0.5671432904097838,
        "lambert_one": float(lam.one()),
        "kepler_zero": float(kep.zero()),
        "kepler_one": float(kep.one()),
        "kepler_g_of_one": float(kep.g(1.0)),
        "logistic_zero": float(LogisticArithmetic().zero()),
        "softplus_zero": float(SoftplusArithmetic().zero()),
        "softplus_one": float(SoftplusArithmetic().one()),
        "relativistic_one": float(RelativisticArithmetic(1.0).one()),
        "tsallis_zero": float(TsallisArithmetic(0.5).zero()),
    }


# --------------------------------------------------------------------------
# 3. Lambert inverse: closed form, log-domain solve, asymptotic branch
# --------------------------------------------------------------------------

def run_lambert():
    rows = []
    for y in [1.0, 10.0, 100.0, 500.0, 700.0, 709.0, 710.0, 800.0, 1e4, 1e6]:
        naive = lambert_w(math.exp(y)) if y < 709.78 else float("inf")
        exact = solve_monotone(lambda x: x + math.log(x), y,
                               max(y - math.log(max(y, 1.0)), 1e-12),
                               lambda x: 1.0 + 1.0 / x)
        asym = y - math.log(max(y, 1.0)) + math.log(max(y, 1.0)) / max(y, 1.0)
        rows.append({
            "y": y,
            "closed_form": naive,
            "log_domain": exact,
            "asymptotic": asym,
            "err_asymptotic": rel(asym, exact),
        })
    return rows


# --------------------------------------------------------------------------
# 4. phase behaviour of the Lambert arithmetic
# --------------------------------------------------------------------------

def run_phase(a_const=0.8):
    lam = LambertArithmetic()
    rows = []
    for expo in np.arange(-5.0, 5.01, 0.05):
        x = 10.0 ** expo
        val = float(lam.add(a_const, x))
        rows.append({
            "x": float(x),
            "add": val,
            "classic_sum": a_const + x,
            "classic_product": a_const * x,
            "rel_to_additive": rel(val, a_const + x),
            "rel_to_multiplicative": rel(val, a_const * x),
        })
    return rows


# --------------------------------------------------------------------------
# 5. small-argument regime of the Lambert arithmetic
# --------------------------------------------------------------------------

def run_small_args():
    lam = LambertArithmetic()
    rows = []
    for x in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
        val = float(lam.add(x, x))
        rows.append({"x": x, "add": val, "product": x * x,
                     "rel_to_product": abs(val - x * x) / (x * x)})
    return rows


# --------------------------------------------------------------------------
# 6. the non-associative variant
# --------------------------------------------------------------------------

def run_synergy():
    base = LambertArithmetic()
    rows = []
    for s in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        mod = SynergyArithmetic(base, s=s)
        for a in (1.3, 0.4):
            b, c = 1.7, 2.3
            left, right = mod.add(mod.add(a, b), c), mod.add(a, mod.add(b, c))
            rows.append({
                "s": s,
                "a": a,
                "assoc_defect": rel(left, right),
                "identity_defect": abs(mod.add(a, base.zero()) - a),
                "left": left,
                "right": right,
            })
    return rows


# --------------------------------------------------------------------------
# 7. conditioning near the boundary
# --------------------------------------------------------------------------

def run_cancellation():
    """Relative size of a (+) (-a) against a, per generator.

    The identity a (+) (-a) = 0_f holds exactly. In floating point the result is
    a p-th root of a cancelled difference, so the condition number grows like
    (a/R)^p, where R is the size of the result.
    """
    rows = []
    for model, lo, hi in [(CubicArithmetic(), -8.0, 8.0),
                          (LambertArithmetic(), 0.02, 20.0),
                          (LogisticArithmetic(), 0.02, 0.98),
                          (RelativisticArithmetic(1.0), -0.99, 0.99)]:
        rng = np.random.default_rng(11)
        xs = rng.uniform(lo, hi, 300)
        z0 = model.zero()
        rows.append({
            "model": model.name,
            "max_residual_over_a": max(
                abs(model.add(x, model.neg(x)) - z0) / max(abs(x), 1e-300)
                for x in xs),
        })
    return rows


# --------------------------------------------------------------------------
# 8. worked examples quoted in the text
# --------------------------------------------------------------------------

def run_examples():
    p2, p4 = PowerArithmetic(2.0), PowerArithmetic(4.0)
    harm = HarmonicArithmetic()
    lam = LambertArithmetic()
    logi = LogisticArithmetic()
    rel_ = RelativisticArithmetic(1.0)
    out = {
        "rms_230_45": float(p2.add(230.0, 45.0)),
        "stefan_boltzmann_800_1000": float(p4.add(800.0, 1000.0)),
        "parallel_6_3": float(harm.add(6.0, 3.0)),
        "focal_50_200": float(harm.add(50.0, 200.0)),
        "lambert_08_12": float(lam.add(0.8, 0.12)),
        "lambert_08_1e6": float(lam.add(0.8, 1e6)),
        "logistic_08_07": float(logi.add(0.8, 0.7)),
        "relativistic_05_05": float(rel_.add(0.5, 0.5)),
        "relativistic_09_09": float(rel_.add(0.9, 0.9)),
        "decibels_80_80": 10.0 * math.log10(10 ** 8 + 10 ** 8),
        "kepler_one": float(KeplerArithmetic(0.85).one()),
    }
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def write_dat(name, rows, columns):
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(columns) + "\n")
        for r in rows:
            fh.write(" ".join(repr(float(r[c])) if isinstance(r[c], (int, float, np.floating))
                              else str(r[c]) for c in columns) + "\n")
    return path


def main():
    laws = run_laws()
    constants = run_constants()
    lambert = run_lambert()
    phase = run_phase()
    small = run_small_args()
    synergy = run_synergy()
    cancel = run_cancellation()
    examples = run_examples()

    with open(os.path.join(ROOT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"laws": laws, "constants": constants, "lambert": lambert,
                   "synergy": synergy, "examples": examples}, fh, indent=2)

    write_dat("lambert_phase.dat", phase,
              ["x", "add", "classic_sum", "classic_product"])
    write_dat("synergy_assoc.dat",
              [r for r in synergy if r["a"] == 1.3],
              ["s", "assoc_defect", "identity_defect"])
    write_dat("lambert_error.dat",
              [{"y": r["y"], "asym": r["err_asymptotic"]} for r in lambert if r["y"] >= 10],
              ["y", "asym"])

    print("=" * 92)
    print("ALGEBRAIC LAWS  (relative deviations; chart = linear coordinates y = g(x))")
    print("=" * 92)
    hdr = f"{'model':20s} {'structure':9s} {'0_f':>8s} {'1_f':>8s} {'chart_assoc':>11s} {'coord_assoc':>11s} {'neutral':>10s} {'inverse':>10s} {'distr':>10s}"
    print(hdr)
    for r in laws:
        print(f"{r['name']:20s} {r['structure']:9s} {r['zero']:8.4f} {r['one']:8.4f} "
              f"{r['chart_associativity']:11.2e} {r['coord_associativity']:11.2e} "
              f"{r['neutral_element']:10.2e} {r['inverse']:10.2e} {r['distributivity']:10.2e}")

    print()
    print("=" * 92)
    print("CONSTANTS")
    print("=" * 92)
    for k, v in constants.items():
        print(f"  {k:26s} {v!r}")

    print()
    print("=" * 92)
    print("LAMBERT INVERSE: textbook closed form W(exp y) vs log-domain solve vs asymptotics")
    print("=" * 92)
    for r in lambert:
        cf = "overflow" if not math.isfinite(r["closed_form"]) else f"{r['closed_form']:.10f}"
        print(f"  y={r['y']:>9.0f}  W(exp y)={cf:>12s}  log-domain={r['log_domain']:.10f}  "
              f"asymptotic={r['asymptotic']:.10f}  rel.err={r['err_asymptotic']:.2e}")

    print()
    print("=" * 92)
    print("LAMBERT: small-argument regime, a (+) a against the product a*a")
    print("=" * 92)
    for r in small:
        print(f"  a={r['x']:.0e}   a(+)a={r['add']:.6e}   a*a={r['product']:.6e}   rel.dev={r['rel_to_product']:.2e}")

    print()
    print("=" * 92)
    print("SYNERGY OPERATOR: associativity and identity defects")
    print("=" * 92)
    for r in synergy:
        print(f"  s={r['s']:5.2f} a={r['a']:.1f}  assoc={r['assoc_defect']:.3e}  "
              f"identity={r['identity_defect']:.3e}  left={r['left']:.6f} right={r['right']:.6f}")

    print()
    print("=" * 92)
    print("WORKED EXAMPLES")
    print("=" * 92)
    for k, v in examples.items():
        print(f"  {k:28s} {v:.6f}")

    print()
    print("=" * 92)
    print("CROSSOVER: 0.8 (+) x approaches 0.8 + x")
    print("=" * 92)
    for target in (1e-1, 1e-2, 1e-3):
        hit = next((r for r in phase if r["rel_to_additive"] < target), None)
        if hit:
            print(f"  relative deviation below {target:g} for x > {hit['x']:.3g}")

    print()
    print("=" * 92)
    print("CANCELLATION: relative size of a (+) (-a), which is analytically zero")
    print("=" * 92)
    for r in cancel:
        print(f"  {r['model']:22s} max |a (+) (-a)| / |a| = {r['max_residual_over_a']:.2e}")

    print()
    bad = [r["name"] for r in laws if r["coord_associativity"] > 1e-6]
    print(f"models whose coordinate-space associativity exceeds 1e-6: {bad if bad else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
