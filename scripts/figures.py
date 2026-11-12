"""Figures and animations of the second paper (generator calculus).

Usage:  python scripts/figures.py            everything
        python scripts/figures.py 3 7        only branches 3 and 7
        python scripts/figures.py map len    only the two summary figures
Output: paper/figures/*.pdf for LaTeX, *.png and *.gif for the README.

One figure per branch of physics: two panels (one per problem) and three frames
of the simulation; the same simulation is saved as a GIF. Numbers in titles come
from results_problems.json or are computed here; none is typed by hand.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import animation, font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from src import problems as P  # noqa: E402
from src.gcalc import deriv, rk4, rotate, unit_exp  # noqa: E402

OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(ROOT, "results_problems.json"), encoding="utf-8") as _fh:
    RES = {r["id"]: r for r in json.load(_fh)}

# Categorical slots 1-4 of a palette checked for colour-vision deficiency
# (adjacent pairs, light surface). One colour per axis of the model.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
AXIS_COLOR = {"skala": BLUE, "jednostka": ORANGE, "temperatura": AQUA, "komutator": YELLOW}
INK, INK2, MUTED, GRID, RULE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#9ec5f4", "#3987e5", "#0d366b"]                      # one hue, light to dark, for ordered values
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

logging.getLogger("fontTools").setLevel(logging.ERROR)     # the OTF files carry odd timestamps
for _f in glob.glob("/usr/share/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf"):
    font_manager.fontManager.addfont(_f)                      # the typeface of the paper, when installed
sns.set_theme(context="paper", style="ticks")
plt.rcParams.update({
    "font.family": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 8.5,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5,
    "lines.markersize": 5, "lines.markeredgewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 150,
})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def pl(x, nd=2):
    """Number with a Polish decimal comma."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\cdot10^{{{int(e)}}}$"


def fact(pid, name):
    return RES[pid]["facts"][name][0]


def label(ax, finding, what, width=46):
    """Title = the finding; the line under it says what is drawn."""
    keep = re.sub(r"\$[^$]*\$", lambda m: m.group(0).replace(" ", "\x00"), finding)   # never break inside math
    ax.set_title(textwrap.fill(keep, width).replace("\x00", " "), pad=15)
    ax.annotate(what, (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=7.5, color=INK2, va="bottom")


def bare(ax, equal=True):
    """A drawing, not a chart: no ticks, no grid."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if equal:
        ax.set_aspect("equal")


def commas(ax):
    """Decimal commas on the linear axes of a chart."""
    if ax.name == "rectilinear" and not getattr(ax, "_keepfmt", False):
        for a in (ax.xaxis, ax.yaxis):
            if a.get_scale() == "linear":
                a.set_major_formatter(COMMA)


def finish(fig, name, source):
    for ax in fig.axes:
        commas(ax)
    fig.text(0.0, -0.012, f"Źródło: {source}", fontsize=7, color=MUTED, ha="left", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


# ==========================================================================
# 1. classical mechanics
# ==========================================================================

_TOP = {}


def _top_path():
    if not _TOP:
        L = np.array([0.6, 0.0, 1.0])
        t = np.linspace(0.0, 2 * 2 * math.pi / np.linalg.norm(L), 150)       # two turns about L
        _TOP.update(t=t, L=L / np.linalg.norm(L),
                    tips=np.array([rotate(P.top_rotor(x), [0, 0, 1]) for x in t]))
    return _TOP


def b1_panels(axA, axB):
    t = np.linspace(0, 10, 160)
    n = np.array([rotate(P.top_rotor(x), [0, 0, 1]) for x in t])
    tp = np.linspace(0.8, 9.6, 12)
    num = np.array([P.top_rk4(x, 2000)[:, 2] for x in tp])
    for i, (c, name) in enumerate(zip((BLUE, ORANGE, AQUA), "xyz")):
        axA.plot(t, n[:, i], color=c)
        axA.plot(tp, num[:, i], "o", color=c, mec="white")
        axA.text(10.2, n[-1, i], f"${name}$", va="center")
    axA.set(xlabel="czas $t$", ylabel="składowa osi symetrii bąka", xlim=(-0.3, 10.9))
    label(axA, f"Wzór z rotorem pokrywa się z całkowaniem równań Eulera (różnica {sci(RES['A']['resid'])})",
          "linie: rotor w postaci zamkniętej; punkty: RK4")

    h = np.logspace(-7, -1, 60)
    fd = np.abs((P.lj(P.LJ_R0 + h) - 2 * P.lj(P.LJ_R0) + P.lj(P.LJ_R0 - h)) / h ** 2 - P.LJ_K) / P.LJ_K
    jet = max(abs(deriv(P.lj, P.LJ_R0, 2)[2] - P.LJ_K) / P.LJ_K, 1e-16)
    axB.loglog(h, fd, color=INK2, label="iloraz różnicowy")
    axB.axhline(jet, color=ORANGE, label=r"dżet, $\varepsilon^3 = 0$")
    axB.set(xlabel="krok $h$", ylabel=r"błąd względny $V''(r_0)$")
    axB.legend(loc="center right")
    label(axB, f"Błąd drugiej pochodnej: dżet {sci(jet)}, iloraz różnicowy co najmniej {sci(fd.min())}",
          "potencjał Lennarda-Jonesa w minimum $r_0$")


def b1_sim(ax, u):
    ax.cla()
    ax.set_axis_off()
    T = _top_path()
    k = max(1, int(u * (len(T["t"]) - 1)))
    M = P.top_matrix(P.top_rotor(T["t"][k]))
    a = np.linspace(0, 2 * np.pi, 60)
    rim = (M @ np.array([0.55 * np.cos(a), 0.55 * np.sin(a), 0.6 + 0 * a])).T
    ax.plot(*np.array([[0, 0, 0], 1.25 * T["L"]]).T, color=INK2, lw=1.2)          # angular momentum
    ax.plot(*T["tips"][:k + 1].T, color=BLUE, lw=1.0)                              # path of the axis
    ax.plot(*np.array([[0, 0, 0], T["tips"][k]]).T, color=ORANGE, lw=2)            # symmetry axis
    ax.plot(*rim.T, color=ORANGE, lw=1.2)
    ax.plot(*np.array([M @ np.array([0, 0, 0.6]), rim[0]]).T, color=ORANGE, lw=1.2)
    ax.text(*(1.32 * T["L"]), r"$\mathbf{L}$", color=INK2)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.3, 1.15))
    ax.view_init(20, -60)
    ax.set_box_aspect((1, 1, 0.86), zoom=1.22)


# ==========================================================================
# 2. gravitation and celestial mechanics
# ==========================================================================

def b2_panels(axA, axB):
    n = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
    et = [P.kepler_period_error(int(x), regular=False) for x in n]
    es = [P.kepler_period_error(int(x), regular=True) for x in n]
    axA.loglog(n, et, "o-", color=INK2, mec="white", label="czas $t$ (równanie Newtona)")
    axA.loglog(n, es, "o-", color=ORANGE, mec="white", label="czas fikcyjny $s$ (spinor)")
    axA.set(xlabel="liczba kroków RK4 na okres", ylabel="błąd położenia po 1 okresie")
    axA.legend(loc="lower left")
    label(axA, f"Przy mimośrodzie 0,9 i 1000 kroków spinor zmniejsza błąd o "
               f"{round(math.log10(et[3] / es[3]))} rzędów wielkości",
          "ta sama metoda RK4, inna zmienna niezależna")

    dv = np.linspace(0, 0.5, 100)
    for r, c in zip((1, 2, 4), RAMP[::-1]):
        ve = math.sqrt(1 / r)
        axB.plot(dv, P.POW2.sub(ve + dv, ve), color=c, label=rf"zapłon w $r = {r}\,r_p$")
    axB.plot(dv, dv, color=INK2, ls="--", lw=1, label=r"zapłon w $r \to \infty$")
    axB.set(xlabel=r"impuls $\Delta v / v_e$", ylabel=r"prędkość końcowa $v_\infty / v_e$")
    axB.legend(loc="upper left")
    label(axB, rf"Impuls 0,1 $v_e$ w perycentrum daje $v_\infty$ = {pl(fact('D', 'Oberth'), 3)} $v_e$",
          r"odejmowanie w skali $x^2$; dolot po paraboli")


def b2_sim(ax, u):
    ax.cla()
    w_of = lambda s: math.sqrt(0.4) * np.cos(0.5 * s) + 1j * math.sqrt(1.6) * np.sin(0.5 * s)   # a = 1, e = 0.6
    wa, w = w_of(np.linspace(0, 4 * math.pi, 400)), w_of(u * 4 * math.pi)
    for curve, c in ((wa ** 2, BLUE), (wa, ORANGE)):
        ax.plot(curve.real, curve.imag, color=c, lw=1.2)
    for pt, c in ((w * w, BLUE), (w, ORANGE)):
        ax.plot([0, pt.real], [0, pt.imag], color=c, lw=0.8)
        ax.plot(pt.real, pt.imag, "o", color=c, mec="white", ms=7)
    ax.plot(0, 0, "o", color=INK, ms=4)
    ax.text(0.45, -1.5, "$w$", color=INK)
    ax.text(-1.75, 0.75, "$z = w^2$", color=INK)
    ax.set(xlim=(-1.9, 1.0), ylim=(-1.65, 1.65))
    bare(ax)


# ==========================================================================
# 3. oscillations, waves, acoustics
# ==========================================================================

KIND = ((-1.0, BLUE, r"$\mathbf{e}^2 = -1$: okrąg"), (0.0, ORANGE, r"$\mathbf{e}^2 = 0$: prosta"),
        (1.0, AQUA, r"$\mathbf{e}^2 = +1$: hiperbola"))


def b3_panels(axA, axB):
    for kap, c, name in KIND:
        C, S = unit_exp(kap, np.linspace(-np.pi, np.pi, 200) if kap < 0 else np.linspace(-1.6, 1.6, 200))
        axA.plot(C, S, color=c, label=name)
    axA.set_aspect("equal")
    axA.set(xlabel="$C$", ylabel="$S$", xlim=(-1.5, 2.8), ylim=(-2.4, 2.4))
    axA.legend(loc="upper left")
    label(axA, r"Jedna funkcja $\exp(\mathbf{e}t)$ kreśli okrąg, prostą i hiperbolę",
          r"punkty $(C, S)$, gdzie $C + \mathbf{e}S = \exp(\mathbf{e}t)$")

    d = np.linspace(0, 20, 200)
    axB.plot(d, P.db_correction(d), color=AQUA)
    for x in (0.0, 10.0):
        y = float(P.db_correction(x))
        axB.plot(x, y, "o", color=AQUA, mec="white", ms=7)
        axB.annotate(f"{pl(y)} dB", (x, y), xytext=(8, 5), textcoords="offset points")
    axB.set(xlabel="różnica poziomów $L_1 - L_2$ [dB]", ylabel="dodatek do głośniejszego źródła [dB]")
    label(axB, f"Źródło słabsze o 10 dB dodaje {pl(fact('F', 'DbTen'))} dB; "
               f"źródło równe dodaje {pl(float(P.db_correction(0.0)))} dB",
          rf"suma $\oplus_h$ z $h = 10/\ln 10$ = {pl(P.H_DB)} dB")


def b3_sim(ax, u):
    ax.cla()
    t, g = np.linspace(0, 12, 300), 0.15 + 2.05 * u
    for gg in np.arange(0.15, g, 0.25):
        ax.plot(t, P.damped(t, gg), color=GRID, lw=0.8)
    ax.plot(t, P.damped(t, 1.0), color=INK2, lw=0.9, ls="--")
    kap = g * g - 1
    ax.plot(t, P.damped(t, g), color=BLUE if kap < -0.03 else AQUA if kap > 0.03 else ORANGE, lw=2)
    ax.text(0.97, 0.95, rf"$\gamma/\omega_0$ = {pl(g)}" + "\n" + rf"$\mathbf{{e}}^2$ = {pl(kap)}",
            transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel=r"czas $\omega_0 t$", ylim=(-0.8, 1.05))


# ==========================================================================
# 4. thermodynamics and statistical physics
# ==========================================================================

def b4_panels(axA, axB):
    T = np.linspace(0.06, 2.0, 160)
    axA.plot(T, (1 / T) ** 2 * np.exp(1 / T) / (1 + np.exp(1 / T)) ** 2, color=INK2, label="wzór Schottky'ego")
    Tp = np.linspace(0.1, 1.9, 19)
    axA.plot(Tp, [P.schottky(x) for x in Tp], "o", color=AQUA, mec="white", label=r"dżet sumy $\oplus_h$")
    axA.set(xlabel=r"temperatura $k_BT/\Delta$", ylabel="ciepło właściwe $C/k_B$")
    axA.legend(loc="upper right")
    label(axA, rf"Ciepło właściwe z dżetu: maksimum {pl(fact('G', 'SchottkyC'), 3)} $k_B$ "
               rf"przy $k_BT$ = {pl(fact('G', 'SchottkyT'), 3)} $\Delta$",
          r"układ dwupoziomowy o przerwie $\Delta$")

    V, p, S, T = P.carnot_cycle(400)
    axB.fill(V, p, color=BLUE, alpha=0.22, lw=0)
    axB.plot(np.append(V, V[0]), np.append(p, p[0]), color=BLUE)
    axB.set(xlabel="objętość $V$", ylabel="ciśnienie $p$")
    ins = axB.inset_axes([0.5, 0.47, 0.47, 0.47])
    ins.fill(S, T, color=ORANGE, alpha=0.22, lw=0)
    ins.plot(np.append(S, S[0]), np.append(T, T[0]), color=ORANGE)
    ins.set_xlabel("entropia $S$", labelpad=1, fontsize=7.5)
    ins.set_ylabel("temperatura $T$", labelpad=1, fontsize=7.5)
    ins.tick_params(labelsize=7)
    ins.xaxis.set_major_formatter(COMMA)
    ins.yaxis.set_major_formatter(COMMA)
    label(axB, rf"Pole cyklu Carnota: {pl(abs(P.loop_area(V, p)), 4)} w płaszczyźnie $(V,p)$ "
               rf"i {pl(abs(P.loop_area(S, T)), 4)} w płaszczyźnie $(S,T)$",
          r"gaz doskonały; tożsamość $dT \wedge dS = dp \wedge dV$")


def b4_sim(ax, u):
    ax.cla()
    T, N = np.linspace(0.5, 1.5, 161), 10 ** (2.5 * u)
    entropy = lambda n: [-deriv(lambda x: P.f_two_phase(x, n), t, 1)[1] for t in T]
    ghosts = [n for n in (1, 3, 10, 30, 100) if n < N]
    for i, n in enumerate(ghosts):
        ax.plot(T, entropy(n), color=SEQ(0.15 + 0.6 * i / 4), lw=0.9)
    ax.plot([0.5, 1, 1, 1.5], [0, 0, 1, 1], color=INK2, lw=0.9, ls="--")
    ax.plot(T, entropy(N), color=AQUA, lw=2)
    ax.text(0.04, 0.95, f"$N$ = {N:.0f}" + "\n" + "$h = -T/N$", transform=ax.transAxes, va="top")
    ax.set(xlabel="temperatura $T/T_c$", ylim=(-0.05, 1.38))


# ==========================================================================
# 5. electricity and magnetism
# ==========================================================================

def b5_panels(axA, axB):
    four, one = P.maxwell_samples(40)
    lim = 1.08 * abs(four).max()
    axA.plot([-lim, lim], [-lim, lim], color=RULE, lw=1, zorder=0)
    for name, idx, c in ((r"skalar: $\nabla\cdot\mathbf{E}$", [0], BLUE), ("wektor: prawo Ampère'a", [1, 2, 3], ORANGE),
                         ("dwuwektor: prawo Faradaya", [4, 5, 6], AQUA), (r"trójwektor: $\nabla\cdot\mathbf{B}$", [7], YELLOW)):
        axA.plot(four[:, idx].ravel(), one[:, idx].ravel(), "o", color=c, mec="white", mew=0.6, label=name)
    axA.set(xlabel="lewe strony 4 równań klasycznych", ylabel="składowe 1 równania")
    axA.legend(loc="upper left")
    label(axA, "8 składowych jednego równania równa się 8 składowym czterech równań",
          "40 losowych punktów; pole testowe nie spełnia równań")

    zL, refl = 0.4 + 0.8j, lambda z: (z - 1) / (z + 1)
    a = np.linspace(0, 2 * np.pi, 200)
    axB.plot(np.cos(a), np.sin(a), color=RULE, lw=1)
    tr = refl(P.zin_new(zL, np.linspace(0, np.pi, 200)))
    axB.plot(tr.real, tr.imag, color=BLUE)
    for bl, name, off in ((0.0, "obciążenie $z_L$", (8, 4)), (np.pi / 2, "po ćwierci fali: $1/z_L$", (8, -10))):
        g = refl(P.zin_new(zL, bl))
        axB.plot(g.real, g.imag, "o", color=BLUE, mec="white", ms=7)
        axB.annotate(name, (g.real, g.imag), xytext=off, textcoords="offset points")
    axB.set_aspect("equal")
    axB.set(xlabel=r"Re $\Gamma$", ylabel=r"Im $\Gamma$", xlim=(-1.1, 1.9), ylim=(-1.1, 1.1))
    label(axB, "Odcinek linii obraca współczynnik odbicia; ćwierć fali zamienia $z$ na $1/z$",
          r"$\Gamma = (z - 1)/(z + 1)$, $z_L = 0{,}4 + 0{,}8j$; szary okrąg: $|\Gamma| = 1$")


def b5_sim(ax, u):
    ax.cla()
    ax.set_axis_off()
    z = np.linspace(0, 4 * np.pi, 200)
    ax.plot(z, np.cos(z - 2 * np.pi * u), np.sin(z - 2 * np.pi * u), color=ORANGE, lw=1.4)
    ax.plot([0, 4 * np.pi], [0, 0], [0, 0], color=INK2, lw=1)
    for zz in np.linspace(0, 4 * np.pi, 25):
        ph = zz - 2 * np.pi * u
        ax.plot([zz, zz], [0, np.cos(ph)], [0, np.sin(ph)], color=BLUE, lw=0.9)
    ax.set(ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))
    ax.set_box_aspect((2.4, 1, 1), zoom=1.0)
    ax.view_init(16, -72)


# ==========================================================================
# 6. optics
# ==========================================================================

_WF = {}


def _wavefront():
    """Arrival time from a point source across an interface: a (min, +) product per point."""
    if not _WF:
        X, Y = np.meshgrid(np.linspace(-2, 2, 161), np.linspace(-1.5, 1.5, 121))
        src, n2 = (-1.0, 1.0), 1.5
        T = np.hypot(X - src[0], Y - src[1])
        k = np.linspace(-5, 5, 1001)                                   # points of the interface y = 0
        t1 = np.hypot(k - src[0], src[1])
        low = Y < 0
        T[low] = (t1[None, :] + n2 * np.hypot(X[low][:, None] - k[None, :], Y[low][:, None])).min(axis=1)
        _WF.update(X=X, Y=Y, T=T, k=k, t1=t1, src=src, n2=n2)
    return _WF


def b6_panels(axA, axB):
    rng = np.random.default_rng(4)
    axA.plot([0, 1], [0, 1], color=RULE, lw=1, zorder=0)
    for n2, c in ((1.33, BLUE), (1.5, ORANGE), (2.4, AQUA)):
        xs, ys = [], []
        for _ in range(20):
            a, b, d = rng.uniform(0.5, 2), rng.uniform(0.5, 2), rng.uniform(0.3, 3)
            _, x = P.snell_tropical(a, b, d, 1.0, n2, 20001)
            xs.append(x / math.hypot(a, x))
            ys.append(n2 * (d - x) / math.hypot(b, d - x))
        axA.plot(xs, ys, "o", color=c, mec="white", mew=0.6, label=f"$n_2$ = {pl(n2)}")
    axA.set(xlabel=r"$n_1\sin\theta_1$", ylabel=r"$n_2\sin\theta_2$")
    axA.legend(loc="upper left")
    label(axA, "Minimum sumy czasów spełnia prawo Snella",
          "60 losowych geometrii; punkt przejścia z iloczynu w arytmetyce (min, +)")

    x = np.linspace(-1.25, 1.25, 1501) + 1e-9
    for N, c in zip((2, 5, 10), RAMP):
        axB.plot(x, P.slits_new(2 * np.pi * x, N) / N ** 2, color=c, label=f"$N$ = {N}")
    axB.set(xlabel=r"różnica faz $\delta / 2\pi$", ylabel="natężenie $I / (N^2 I_0)$", ylim=(-0.03, 1.27))
    axB.legend(loc="upper center", ncol=3)
    label(axB, "Więcej szczelin daje węższe maksima; wysokość maksimum rośnie jak $N^2$",
          r"suma $N$ fazorów $q^k$, $q = \exp(i\delta)$")


def b6_sim(ax, u):
    ax.cla()
    W = _wavefront()
    ax.axhspan(-1.5, 0, color="#cde2fb", lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    t = 0.3 + 3.6 * u
    lev = [t - 0.4 * j for j in range(6) if t - 0.4 * j > 0.05][::-1]
    ax.contour(W["X"], W["Y"], W["T"], levels=lev, colors=[BLUE], linewidths=np.linspace(0.5, 1.8, len(lev)))
    for xt in (-0.5, 0.6, 1.6):
        kk = W["k"][(W["t1"] + W["n2"] * np.hypot(xt - W["k"], 1.4)).argmin()]
        ax.plot([W["src"][0], kk, xt], [W["src"][1], 0, -1.4], color=ORANGE, lw=1)
    ax.plot(*W["src"], "o", color=ORANGE, mec="white", ms=7)
    for yy, name in ((1.28, "$n_1$ = 1"), (-1.4, "$n_2$ = 1,5")):
        ax.text(-1.95, yy, name, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    ax.set(xlim=(-2, 2), ylim=(-1.5, 1.5))
    bare(ax)


# ==========================================================================
# 7. relativity
# ==========================================================================

_ROS = {}


def _rosette():
    if not _ROS:
        lam, h = 0.04, 0.01
        f = lambda _, y: np.array([y[1], 1.0 + lam * y[0] ** 2 - y[0]])
        y, us = np.array([1.5, 0.0]), []
        for _ in range(int(10 * np.pi / h)):
            us.append(y[0])
            y = rk4(f, y, 0.0, h, 1)
        us, phi = np.array(us), np.arange(len(us)) * h
        peri = np.concatenate([[0], np.where((us[1:-1] > us[:-2]) & (us[1:-1] >= us[2:]))[0] + 1])
        _ROS.update(x=np.cos(phi) / us, y=np.sin(phi) / us, peri=peri)
    return _ROS


def b7_panels(axA, axB):
    b = np.linspace(0.02, 0.99, 120)
    eta = np.arctanh(b)
    axA.plot(b, [math.degrees(P.wigner_rotor(e, e, np.pi / 2)) for e in eta], color=YELLOW, lw=2,
             label="iloczyn dwóch wykładników")
    bp = np.linspace(0.1, 0.95, 9)
    axA.plot(bp, [math.degrees(P.wigner_matrix(math.atanh(x), math.atanh(x), np.pi / 2)) for x in bp], "o",
             color=INK2, mec="white", label="macierze 4×4")
    axA.plot(b, np.degrees(0.5 * eta ** 2), color=INK2, ls="--", lw=1, label=r"komutator: $\eta^2/2$")
    axA.set(xlabel="prędkość obu boostów $v/c$", ylabel="kąt obrotu [°]", ylim=(0, 95))
    axA.legend(loc="upper left")
    label(axA, f"Dwa prostopadłe boosty po 0,9$c$ obracają układ o {pl(fact('M', 'WignerDeg'), 1)}°",
          r"defekt addytywności pospieszności $\eta$")

    lam = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    xs = np.logspace(-8.6, -1.8, 50)
    axB.loglog(xs, 2 * np.pi * xs, color=ORANGE, label=r"arytmetyka $\lambda^2 = 0$: $2\pi\lambda$")
    axB.loglog(lam, [P.precession_numeric(x) for x in lam], "o", color=INK2, mec="white", label="całkowanie orbity")
    lm = P.mercury_lambda()
    axB.plot(lm, 2 * np.pi * lm, "D", color=ORANGE, mec="white", ms=7)
    axB.annotate("Merkury", (lm, 2 * np.pi * lm), xytext=(9, -4), textcoords="offset points")
    axB.set(xlabel=r"parametr $\lambda = 3\mu^2/c^2h^2$", ylabel="obrót peryhelium na obieg [rad]")
    axB.legend(loc="upper left")
    label(axB, rf"Peryhelium Merkurego: {pl(fact('N', 'MercuryArcsec'))}$''$ na stulecie "
               r"z arytmetyki $\lambda^2 = 0$",
          "parametry orbity: NASA; $GM$ Słońca: IAU 2015")


def b7_sim(ax, u):
    ax.cla()
    R = _rosette()
    n = max(2, int(u * len(R["x"])))
    ax.plot(R["x"][:n], R["y"][:n], color=ORANGE, lw=1.1)
    for i in R["peri"][R["peri"] < n]:
        ax.plot([0, R["x"][i]], [0, R["y"][i]], color=INK2, lw=0.8)
        ax.plot(R["x"][i], R["y"][i], "o", color=INK2, mec="white", ms=5)
    ax.plot(0, 0, "o", color=INK, ms=5)
    ax.plot(R["x"][n - 1], R["y"][n - 1], "o", color=ORANGE, mec="white", ms=7)
    ax.set(xlim=(-2.15, 2.15), ylim=(-2.15, 2.15))
    bare(ax)


# ==========================================================================
# 8. quantum mechanics
# ==========================================================================

def b8_panels(axA, axB):
    t, tp = np.linspace(0, 12, 140), np.linspace(1, 11, 9)
    for De, c in ((0.0, BLUE), (1.0, ORANGE), (2.0, AQUA)):
        axA.plot(t, [P.rabi_rotor(x, 1.0, De) for x in t], color=c, label=rf"$\Delta/\Omega_R$ = {De:g}")
        axA.plot(tp, [P.rabi_schrodinger(x, 1.0, De, 400) for x in tp], "o", color=c, mec="white")
    axA.set(xlabel=r"czas $\Omega_R t$", ylabel="prawdopodobieństwo odwrócenia spinu", ylim=(-0.03, 1.25))
    axA.legend(ncol=3, loc="upper center")
    label(axA, "Rotor i równanie Schrödingera dają tę samą krzywą Rabiego",
          "linie: rotor; punkty: całkowanie RK4")

    hb = np.logspace(math.log10(0.02), math.log10(0.5), 9)
    xs = np.logspace(-1.9, -0.2, 50)
    axB.loglog(xs, np.sqrt(np.pi * xs / 2), color=AQUA, label=r"faza stacjonarna: $\sqrt{\pi\hbar/a}$")
    axB.loglog(hb, [abs(P.path_sum(x)) for x in hb], "o", color=INK2, mec="white", label="suma po torach")
    axB.set(xlabel=r"$\hbar$", ylabel="moduł sumy fazorów")
    axB.legend(loc="upper left")
    axB.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    axB.set_yticks([0.2, 0.3, 0.5, 1.0])
    for a in (axB.xaxis, axB.yaxis):
        a.set_major_formatter(COMMA)
        a.set_minor_formatter(matplotlib.ticker.NullFormatter())
    label(axB, r"Strefa zgodnych faz zwęża się jak $\sqrt{\hbar}$",
          "tory łamane przez punkt $x$ w połowie czasu; w granicy zostaje tor klasyczny")


def b8_sim(ax, u):
    ax.cla()
    hb = 0.5 * (0.02 / 0.5) ** u
    x = np.linspace(-1.0, 3.0, 6000)
    y = np.cos((P.path_action(x) - P.S_CL) / hb)
    w = math.sqrt(math.pi * hb / 4)                  # phases within pi/2 of the classical path (a = 2)
    zone = np.abs(x - 1.0) <= w
    ax.axvspan(1 - w, 1 + w, color=AQUA, alpha=0.12, lw=0)
    ax.plot(x, y, color=RULE, lw=0.6)
    ax.plot(x[zone], y[zone], color=AQUA, lw=2.2)
    ax.text(0.03, 0.04, rf"$\hbar$ = {pl(hb)}; strefa $\pm${pl(w)}", transform=ax.transAxes,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set(xlabel="punkt pośredni toru $x$", ylim=(-1.3, 1.3))


# ==========================================================================
# 9. atomic, nuclear and particle physics
# ==========================================================================

_MC = {}


def b9_panels(axA, axB):
    t, T1, T2 = np.linspace(0, 40, 200), P.HALF["T1"], P.HALF["T2"]
    Te = fact("Q", "HalfEff")
    for T, c, name in zip((T2, T1, Te), RAMP, (f"tylko kanał 2: {T2:g} d", f"tylko kanał 1: {T1:g} d",
                                                 f"oba kanały: {pl(Te)} d")):
        axA.semilogy(t, 0.5 ** (t / T), color=c, label=name)
    axA.axhline(0.5, color=RULE, lw=1)
    axA.set(xlabel="czas [d]", ylabel="pozostała część jąder", ylim=(0.02, 1.2))
    axA.legend(loc="lower left")
    label(axA, rf"{T1:g} d $\oplus$ {T2:g} d = {pl(Te)} d: w skali $1/T$ okresy półtrwania się dodają",
          "wartości przykładowe; oś pionowa logarytmiczna")

    c1, th = np.linspace(0, 2, 50), np.linspace(0.3, 3.0, 8)
    for E, c in ((0.5, BLUE), (1.0, ORANGE), (2.0, AQUA)):
        axB.plot(c1, 1 / E + c1, color=c, label=rf"$E$ = {pl(E, 1)} $mc^2$")
        axB.plot(1 - np.cos(th), [1 / P.compton_classic(E, x) for x in th], "o", color=c, mec="white")
    axB.set(xlabel=r"$1 - \cos\theta$", ylabel=r"$mc^2 / E'$")
    axB.legend(loc="upper left")
    label(axB, "W skali $1/E$ wzór Comptona jest linią prostą",
          "linie: suma harmoniczna; punkty: 3 równania zachowania rozwiązane liczbowo")


def b9_sim(ax, u):
    ax.cla()
    if not _MC:
        rng, n = np.random.default_rng(1), 900
        t1 = rng.exponential(P.HALF["T1"] / math.log(2), n)
        t2 = rng.exponential(P.HALF["T2"] / math.log(2), n)
        gx, gy = np.meshgrid(np.arange(30), np.arange(30))
        _MC.update(t=np.minimum(t1, t2), ch=(t2 < t1).astype(int), x=gx.ravel(), y=gy.ravel())
    t = 30.0 * u
    alive = _MC["t"] > t
    for mask, c, name in ((alive, BLUE, "jądro"), (~alive & (_MC["ch"] == 0), ORANGE, "rozpad 1"),
                          (~alive & (_MC["ch"] == 1), AQUA, "rozpad 2")):
        ax.scatter(_MC["x"][mask], _MC["y"][mask], s=9, color=c, lw=0, label=name)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, handletextpad=0.1, columnspacing=0.8)
    ax.text(0.5, -0.03, f"$t$ = {pl(t, 1)} d: zostało {alive.mean() * 100:.0f} % "
                        f"(wzór: {0.5 ** (t / fact('Q', 'HalfEff')) * 100:.0f} %)",
            transform=ax.transAxes, ha="center", va="top")
    ax.set(xlim=(-1, 30), ylim=(-1, 30))
    bare(ax)


# ==========================================================================
# 10. condensed matter and fluids
# ==========================================================================

def b10_panels(axA, axB):
    x = np.linspace(0, 1, 300)
    axA.plot(x, 2 * np.abs(np.sin(np.pi * x)), color=INK2, lw=1, label=r"liczba cykliczna: $2\,|\sin(\pi k/N)|$")
    for N, c in zip((64, 24, 8), RAMP):
        k = np.arange(N)
        order = np.argsort(2 - 2 * np.cos(2 * np.pi * k / N), kind="stable")   # mode of each sorted eigenvalue
        axA.plot((k / N)[order], np.sqrt(np.abs(P.phonon_matrix(N))), "o", color=c, mec="white", mew=0.5,
                 ms=3.5 if N > 30 else 5.5, label=f"macierz, $N$ = {N}")
    axA.set(xlabel="numer modu $k/N$", ylabel=r"częstość $\omega_k / \sqrt{K/m}$", ylim=(0, 2.75))
    axA.legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.3)
    label(axA, "Widmo macierzy dynamicznej leży na krzywej liczby cyklicznej",
          r"punkty: wartości własne macierzy $N \times N$; linia: $D(s)$ dla $s^N = 1$")

    th = np.linspace(0, 2 * np.pi, 300)
    for G, c in zip((0.0, 2.0, 4.0), RAMP):
        axB.plot(np.degrees(th), 1 - (2 * np.sin(th) + G / (2 * np.pi)) ** 2, color=c, label=rf"$\Gamma$ = {G:g}")
    axB.set(xlabel="kąt na powierzchni walca [°]", ylabel="współczynnik ciśnienia $C_p$")
    axB.set_xticks([0, 90, 180, 270, 360])
    axB.set_ylim(-6.3, 2.6)
    axB.legend(loc="upper center", ncol=3)
    label(axB, rf"Całka ciśnienia daje siłę nośną {pl(P.cylinder_force(1.0, 4.0)[1], 3)} = $\rho U\Gamma$",
          r"$U = a = \rho = 1$; siła policzona dla $\Gamma = 4$; cyrkulacja łamie symetrię ciśnienia")


def b10_sim(ax, u):
    ax.cla()
    G = 0.9 * 4 * np.pi * u
    X, Y = np.meshgrid(np.linspace(-3, 3, 121), np.linspace(-2.4, 2.4, 97))
    with np.errstate(all="ignore"):
        dw = P.cylinder_dw(X + 1j * Y, 1.0, G)
    U, V = dw.real.copy(), -dw.imag
    U[np.hypot(X, Y) < 1.0] = np.nan
    ax.streamplot(X, Y, U, V, density=1.0, color=np.hypot(U, V), cmap=SEQ, linewidth=0.8, arrowsize=0.6)
    ax.add_patch(plt.Circle((0, 0), 1.0, color=INK2, zorder=3))
    ax.annotate("", (0, 1.02 + 1.2 * u), (0, 1.0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2), zorder=4)
    ax.text(0.03, 0.96, rf"$\Gamma$ = {pl(G, 1)}", transform=ax.transAxes, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set(xlim=(-3, 3), ylim=(-2.4, 2.4))
    bare(ax)


# ==========================================================================
# branch figures and animations
# ==========================================================================

BRANCH = {
    1: (b1_panels, b1_sim, "3d", "Oś bąka obiega stały moment pędu; rotor daje położenie bez całkowania"),
    2: (b2_panels, b2_sim, None, "Punkt $w$ drga harmonicznie wokół środka; kwadrat $z = w^2$ obiega elipsę Keplera"),
    3: (b3_panels, b3_sim, None, r"Jeden wzór przechodzi gładko przez tłumienie krytyczne ($\mathbf{e}^2 = 0$)"),
    4: (b4_panels, b4_sim, None, r"Gdy $h = -T/N$ dąży do 0, suma $\oplus_h$ staje się minimum i entropia skacze"),
    5: (b5_panels, b5_sim, "3d", r"Fala kołowa to wektor obracany rotorem o kąt $kz - \omega t$"),
    6: (b6_panels, b6_sim, None, "Front fali liczony w arytmetyce (min, +) załamuje się na granicy ośrodków"),
    7: (b7_panels, b7_sim, None, r"Orbita nie zamyka się: peryhelium obraca się o $2\pi\lambda$ na obieg (tu $\lambda$ = 0,04)"),
    8: (b8_panels, b8_sim, None, r"Część rzeczywista fazora: gdy $\hbar$ maleje, zgodne fazy mają tylko tory przy $x = 1$"),
    9: (b9_panels, b9_sim, None, r"Dwa kanały rozpadu działają jak jeden kanał o okresie $T_1 \oplus T_2$"),
    10: (b10_panels, b10_sim, None, "Rosnąca cyrkulacja zsuwa punkty spiętrzenia i zwiększa siłę nośną"),
}


def branch(k, frames=36):
    panels, sim, proj, finding = BRANCH[k]
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.12, 0.09, 0.92])
    panels(fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:]))
    head = fig.add_subplot(gs[1, :])
    head.set_axis_off()
    head.text(0, 0.5, "Symulacja. " + finding, fontsize=9, fontweight="bold", va="center")
    for i, u in enumerate((0.12, 0.5, 0.95)):
        sim(fig.add_subplot(gs[2, 2 * i:2 * i + 2], projection=proj), u)
    finish(fig, f"branch{k:02d}", f"scripts/figures.py, dział {k}: {P.BRANCHES[k - 1]}; liczby z results_problems.json")

    fig = plt.figure(figsize=(4.4, 4.4), constrained_layout=True)
    ax = fig.add_subplot(projection=proj)
    def frame(i):
        sim(ax, i / (frames - 1))
        commas(ax)

    anim = animation.FuncAnimation(fig, frame, frames=frames)
    anim.save(os.path.join(OUT, f"sim{k:02d}.gif"), writer=animation.PillowWriter(fps=10), dpi=90)
    plt.close(fig)


# ==========================================================================
# summary figures
# ==========================================================================

def fig_lengths():
    rs = list(RES.values())
    y = np.arange(len(rs))[::-1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 5.6), constrained_layout=True, sharey=True,
                                  gridspec_kw=dict(width_ratios=[2.1, 1]))
    for yy, r in zip(y, rs):
        ax.plot([r["len_classic"], r["len_new"]], [yy, yy], color=RULE, lw=1.3, zorder=1)
        ax.plot(r["len_classic"], yy, "o", color=INK2, mec="white", ms=6.5, zorder=2)
        ax.plot(r["len_new"], yy, "o", color=AXIS_COLOR[r["axes"]], mec="white", ms=7.5, zorder=3)
        ax2.plot([1e-18, max(r["resid"], 1e-17)], [yy, yy], color=RULE, lw=1.3, zorder=1)
        ax2.plot(max(r["resid"], 1e-17), yy, "o", color=INK2, mec="white", ms=6.5, zorder=2)
    for a in (ax, ax2):
        a._keepfmt = True
        a.grid(axis="y", visible=False)
        for yy in y[1::2][:-1]:
            a.axhline(yy - 0.5, color=GRID, lw=0.6)
    ax.set_yticks(y, [f"{r['id']}  {r['title'].replace('--', '–')}" for r in rs])
    ax.set(xlabel="liczba symboli w zapisie rozwiązania", ylim=(-0.7, len(rs) - 0.3))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, mec="white", label="zapis klasyczny")] + [
        Line2D([], [], marker="o", ls="", color=c, mec="white", label=f"zapis nowy, oś „{name}”")
        for name, c in AXIS_COLOR.items()], loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3)
    ratios = sorted(r["ratio"] for r in rs)
    shorter = sum(r["len_new"] < r["len_classic"] for r in rs)
    label(ax, f"Nowy zapis jest krótszy w {shorter} z {len(rs)} problemów; "
              f"mediana ilorazu długości wynosi {pl((ratios[9] + ratios[10]) / 2)}",
          "1 wiersz = 1 problem; odcinek łączy długość zapisu klasycznego i nowego", width=50)
    ax2.set_xscale("log")
    ax2.set(xlabel="odchyłka testu zgodności", xlim=(1e-18, 1e-1))
    tight = max(r["resid"] for r in rs if r["tol"] < 1e-6)
    label(ax2, f"{sum(r['tol'] < 1e-6 for r in rs)} testów zgadza się do {sci(tight)}",
          "oś logarytmiczna", width=24)
    finish(fig, "lengths", f"results_problems.json; n = {len(rs)} problemów; licznik symboli: src/problems.py, tex_len")


def fig_map():
    """Graph of number systems. Every node is connected; edges are horizontal or vertical."""
    fig, ax = plt.subplots(figsize=(7.2, 7.6), constrained_layout=True)
    ax.set_axis_off()
    ax.set(xlim=(0, 10.1), ylim=(-0.9, 18.6))

    def box(x, y, text, color, w, bold=False):
        ax.add_patch(matplotlib.patches.Rectangle((x, y - 0.36), w, 0.72, fc="white", ec=color, lw=1.4, zorder=2))
        ax.text(x + 0.12, y, text, va="center", ha="left", fontsize=8, zorder=3,
                fontweight="bold" if bold else "normal")

    def edge(x1, y1, x2, y2, xm, color):
        ax.plot([x1, xm, xm, x2], [y1, y1, y2, y2], color=color, lw=1.2, zorder=1, solid_joinstyle="miter")

    groups = [
        ("oś „skala”: generator $g$", BLUE, [
            r"$g = \ln$:  liczby dodatnie z mnożeniem",
            r"$g = 1/x$:  opory, ogniskowe, okresy półtrwania",
            r"$g = x^2$:  wartości skuteczne, prędkości ucieczki",
            r"$g = \mathrm{artanh}$:  prędkości, impedancje linii",
            r"$g = \mathrm{logit}$:  szanse, łączenie przesłanek"]),
        (r"oś „jednostka”: $\mathbf{e}^2$", ORANGE, [
            r"$\mathbf{e}^2 = -1$:  liczby zespolone, fazory",
            r"$\mathbf{e}^2 = +1$:  liczby podwójne, boosty",
            r"$\mathbf{e}^2 = 0$:  liczby dualne, dżety, pochodna",
            r"$Cl(3,0)$:  wektory, rotory, kwaterniony",
            r"$Cl(1,3)$:  czterowektory",
            r"$Cl(0,0,n)$:  liczby Grassmanna, formy",
            r"$s^N = 1$:  liczby cykliczne, sieć okresowa"]),
        ("oś „temperatura”: $h$", AQUA, [
            r"$h = 10/\ln 10$:  decybele",
            r"$h = -T/N$:  energia swobodna",
            r"$h = -i\hbar$:  suma po torach",
            r"$h \to 0$:  arytmetyka (min, +) i (max, +)"]),
        ("składanie nieprzemienne", YELLOW, [
            r"komutator $[X, Y]$:  obrót Thomasa–Wignera"]),
        ("poza modelem", MUTED, [
            r"oktoniony, liczby porządkowe, liczby $p$-adyczne"]),
    ]
    y, root_y = 18.0, 9.0
    box(0.1, root_y, r"$(\mathbb{R}, +)$", INK, 1.5, bold=True)
    for name, color, leaves in groups:
        ys = [y - i for i in range(len(leaves))]
        mid = (ys[0] + ys[-1]) / 2
        edge(1.6, root_y, 2.4, mid, 2.0, INK2)
        box(2.4, mid, name, color, 2.75, bold=True)
        for yy, text in zip(ys, leaves):
            edge(5.15, mid, 5.9, yy, 5.5, color)
            box(5.9, yy, text, color, 4.1)
        y = ys[-1] - 1.25
    n_in = sum(len(g[2]) for g in groups[:-1])
    ax.set_title(f"Trzy osie i komutator porządkują {n_in} systemów liczbowych i reguł składania", pad=15)
    ax.annotate("krawędź: system powstaje z liczb rzeczywistych przez wybór generatora, jednostki albo temperatury",
                (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=7.5, color=INK2, va="bottom")
    finish(fig, "map", "sekcja 2 pracy; zestawienie własne")


def main(argv):
    want = argv or [str(k) for k in BRANCH] + ["map", "len"]
    for a in want:
        if a == "map":
            fig_map()
        elif a == "len":
            fig_lengths()
        else:
            branch(int(a))
        print("done:", a, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
