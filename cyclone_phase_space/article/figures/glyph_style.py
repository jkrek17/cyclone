"""The live map's low marker (marker B), for the article's static figures.

One visual channel per axis of Hart's phase space, exactly as
`../live/glyphs.js` draws it:

    color     the lower core: red warm, blue cold
    shape     a triangle when asymmetric (B above 10 m), a circle when
              symmetric
    fill      filled with a dark edge when deep (the upper core agrees with
              the lower), a hollow ring in the lower core's color when
              shallow

Cells and quadrants of the phase space take the lower core's color, a deep
one stronger than a shallow one, as the live cube does (FACE_A in
`../live/space3d.js`). The gridded HCPSclass field and the CAVE colormap keep
their seven class colors (CPS_HartClass.cmap); only marks of a low use this.

Importing this module defines constants and functions only: no rcParams.
"""
from __future__ import annotations

import math

from matplotlib.lines import Line2D

# glyphs.js: GLYPH
WARM = "#e0362c"
COLD = "#2f74e0"
INK = "#0b1220"
HOLLOW = "#f2f5f9"
MISSING = "#8a8a86"
REF_GRAY = "#8f8e89"  # the other method (Hart's or the FSU page), not a class

B_ONSET_M = 10.0

# glyphs.js: CORES (lower, upper core, 1 warm) and ASYM by class code; class 6
# (cold lower, warm upper) spans the B wall, so its asymmetry is its own B.
CORES = [(1, 1), (1, 0), (1, 1), (1, 0), (0, 0), (0, 0), (0, 1)]
ASYM = [False, False, True, True, True, False, None]

# Cell / quadrant face alpha, deep and shallow, on a white page (the live
# cube's 0.11 / 0.045 are on a dark one).
CELL_ALPHA = (0.20, 0.08)


def core_hex(warm):
    return WARM if warm else COLD


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def axes_of_class(k, b=None):
    """(lower warm, upper warm, asymmetric) for class code `k`, B `b`, or
    None where the class is missing."""
    if not _finite(k) or int(round(k)) not in range(7):
        return None
    k = int(round(k))
    lo, up = CORES[k]
    asym = ASYM[k]
    if asym is None:
        asym = _finite(b) and b > B_ONSET_M
    return (lo, up, bool(asym))


def axes_of_values(b, vtl, vtu):
    """The three axes straight from B, -V_T^L and -V_T^U (Hart's signs:
    positive thermal wind is warm), or None where one is missing."""
    if not (_finite(b) and _finite(vtl) and _finite(vtu)):
        return None
    return (int(vtl > 0), int(vtu > 0), bool(b > B_ONSET_M))


def core_hex_of(axes):
    return MISSING if axes is None else core_hex(axes[0])


def cell_alpha(lo, up):
    return CELL_ALPHA[0] if lo == up else CELL_ALPHA[1]


def glyph_kw(axes, size_pt=7.0):
    """matplotlib `plot`/`Line2D` marker keywords for one glyph, `size_pt`
    the circle's diameter in points (the triangle is drawn 1.3 times larger,
    as glyphs.js draws it)."""
    if axes is None:
        return dict(marker="o", markersize=size_pt * 0.85, markerfacecolor=MISSING,
                    markeredgecolor=MISSING, markeredgewidth=0.0, linestyle="none")
    lo, up, asym = axes
    deep = lo == up
    marker = "^" if asym else "o"
    ms = size_pt * (1.3 if asym else 1.0)
    if deep:
        return dict(marker=marker, markersize=ms, markerfacecolor=core_hex(lo),
                    markeredgecolor=INK, markeredgewidth=max(0.6, size_pt * 0.08), linestyle="none")
    return dict(marker=marker, markersize=ms, markerfacecolor=HOLLOW,
                markeredgecolor=core_hex(lo), markeredgewidth=max(1.0, size_pt * 0.2), linestyle="none")


def draw_glyph(ax, x, y, axes, size_pt=7.0, zorder=5, alpha=1.0, **extra):
    """One glyph at data (x, y) (and z, via extra `zs`, on a 3-D axes)."""
    kw = glyph_kw(axes, size_pt)
    kw.update(extra)
    if "zs" in kw:
        z = kw.pop("zs")
        return ax.plot([x], [y], [z], zorder=zorder, alpha=alpha, **kw)
    return ax.plot([x], [y], zorder=zorder, alpha=alpha, **kw)


def draw_core_path(ax, xs, ys, axes_list, zs=None, lw=1.8, zorder=3, halo=True):
    """The path between consecutive points, each segment in the lower core
    color of its start (as the live map draws tracks), with a dark halo."""
    for i in range(len(xs) - 1):
        seg_x, seg_y = [xs[i], xs[i + 1]], [ys[i], ys[i + 1]]
        if not all(_finite(v) for v in seg_x + seg_y):
            continue
        args = (seg_x, seg_y) if zs is None else (seg_x, seg_y, [zs[i], zs[i + 1]])
        if halo:
            ax.plot(*args, color=INK, lw=lw + 1.2, alpha=0.3, zorder=zorder - 0.1,
                    solid_capstyle="round")
        ax.plot(*args, color=core_hex_of(axes_list[i]), lw=lw, zorder=zorder, solid_capstyle="round")


def glyph_strip(ax, hours, axes_list, y, size_pt=6.0, every=None, **extra):
    """A row of glyphs along an axis at height `y`: the class read as the
    map's markers, in place of a strip of class colors. `every` (h) thins
    them; None draws one per entry."""
    last = None
    for h, a in zip(hours, axes_list):
        if not _finite(h):
            continue
        if every is not None and last is not None and h - last < every - 1e-6:
            continue
        draw_glyph(ax, h, y, a, size_pt=size_pt, **extra)
        last = h


LEGEND_ENTRIES = [
    ((1, 1, False), "warm lower core, deep, symmetric"),
    ((1, 1, True), "warm, deep, asymmetric"),
    ((1, 0, False), "warm, shallow (hollow)"),
    ((0, 0, False), "cold lower core, deep, symmetric"),
    ((0, 0, True), "cold, deep, asymmetric"),
    ((0, 1, False), "cold, shallow (hollow)"),
]


def legend_handles(size_pt=7.0, entries=None):
    """Legend handles for the marker key: color the lower core, triangle
    asymmetric, hollow shallow."""
    out = []
    for axes, label in (entries or LEGEND_ENTRIES):
        out.append(Line2D([0], [0], label=label, **glyph_kw(axes, size_pt)))
    return out


# The class codes as the map's markers, for legends that list the classes.
def class_handles(names, size_pt=7.0):
    """`names`: a list or a {code: name} dict of the seven class names."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    hs = []
    for k, name in items:
        a = axes_of_class(k, b=0.0)
        hs.append(Line2D([0], [0], label=f"{k}  {name}", **glyph_kw(a, size_pt)))
    return hs
