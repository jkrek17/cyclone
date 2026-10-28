"""A rendering of Hart's (2003) three-dimensional cyclone phase space, with
the seven HCPSclass cells and the synthetic archetype life cycles of
`lifecycle_comparison.py`.

The space is Hart's own: three numbers per storm, B (thermal asymmetry,
m), minus V_T^L (lower thermal wind, m) and minus V_T^U (upper thermal
wind, m), which he presents as two cross-sections (B against -V_T^L and
-V_T^U against -V_T^L) for legibility. The gridded product's HCPSclass
(`cps_HartCPS.hart_class`) is a partition of that space by his own
thresholds: the planes B = 10 m, -V_T^L = 0 and -V_T^U = 0 cut it into
eight boxes, and the two boxes with a cold lower and warm upper layer
(either B) are merged into class 6. This script draws those seven cells in
the full space and, through them, the 168 h paths of the named life cycles
in `lifecycle_comparison.SCENARIOS`: the extratropical transition of
Figure 13 (`figD_lifecycle.png`, the same arrays), a tropical cyclone that
decays without transition, a warm seclusion, a cold-core low that
occludes, the tropical transition of a subtropical storm and a hybrid that
oscillates about B = 10 m. The interactive page also shows every path on
Hart's two diagrams.

The markers follow Hart's conventions: their size grows with the central
pressure depth (deeper is bigger) and the hour is labeled every 12 h. The
archetypes prescribe height perturbations, not a pressure, so the central
pressure of each frame is derived from the vortex (`central_pressure`):
its 1000 hPa amplitude, `Scenario.amp(t)[0]`, at 8 m per hPa (the
conversion `lifecycle_comparison.compute_frame` uses to build the
closed-low detector's MSLP from the 1000 hPa height) below the 1013 hPa
environment of `env_fields`.

The physics is not copied: every path comes from
`lifecycle_comparison.run_scenario`, the per-frame loop of that script's
`main()` (make_hours, build_track_and_motion, build_grid, env_fields,
compute_frame) applied to the scenario's own synthetic height fields, with
Hart's storm-centered method (`cps.hart`) and the gridded method
(`cps_HartCPS`) evaluated at every frame. Hart's method passing through
the same cells as the gridded method is then expected by construction
(the cells are his thresholds); the two can differ only where their
numbers put a frame on different sides of a plane.

Outputs (both in this directory):

    figJ_phase_space_3d.png   static small multiples, one panel per life
                              cycle from one viewing angle (matplotlib
                              mplot3d), 300 dpi, with a marker size legend
    phase_space_3d.html       interactive, self-contained Plotly page with
                              a button per life cycle and a switch between
                              the three-dimensional view and the same path
                              on Hart's two diagrams (Plotly itself is
                              loaded from the jsDelivr CDN); without
                              JavaScript the page shows its note text

Run from this directory:

    python3 phase_space_3d.py

It prints each life cycle's class sequence (both methods), its onset
(first frame with B > 10 m) and completion (first frame with -V_T^L < 0)
hours and its plane crossings, then the axis limits, the derived central
pressure range of each life cycle and the timings.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

import numpy as np

import lifecycle_comparison as lc  # sets up sys.path, Agg backend, hc.ORIENTATION_MODE
import diagram_style as ds

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "figJ_phase_space_3d.png"
OUT_HTML = HERE / "phase_space_3d.html"

B_THR = float(lc.hc.B_THRESHOLD_M)  # 10 m

# Axis limits: x = -V_T^L, y = B, z = -V_T^U (all m), shared by every panel
# and every example. Widened (never narrowed) by diagram_style.widen_limits
# plus a margin if any life cycle exceeds them.
BASE_XLIM = (-320.0, 320.0)
BASE_YLIM = (-20.0, 80.0)
BASE_ZLIM = (-320.0, 320.0)

# HCPSclass colors: D2D/colormaps/Grid/CPS_HartClass.cmap, identical to
# lifecycle_comparison.CLASS_PALETTE (the Figure 13 class strips).
CLASS_PALETTE = lc.CLASS_PALETTE
CLASS_NAMES = lc.CLASS_NAMES
CLASS_HEX = {k: to_hex(v) for k, v in CLASS_PALETTE.items()}

HART_COLOR = "#8f8e89"
PATH_EDGE = "#3a3a37"
CELL_ALPHA = 0.10
VIEW = (20, -40)  # elevation, azimuth (degrees) of every static panel
BOX_ASPECT = (1.0, 1.2, 0.9)  # -V_T^L, B, -V_T^U: B drawn a little longer, its range being narrow
HOUR_STEP = 12.0  # hour labels, as on Hart's diagrams (his are every 6 h frame; 12 h keeps them legible)
HOUR_MAJOR = 24.0  # ... the 24 h ones a little larger and placed first

# Central pressure derived from the vortex (see the module docstring and
# central_pressure): the 1013 hPa environment of lifecycle_comparison.env_fields
# and the 8 m per hPa of lifecycle_comparison.compute_frame.
P_ENV_HPA = 1013.0
M_PER_HPA = 8.0
# Marker diameter (points) grows with the depth below P_ENV_HPA: a floor so a
# filled low is still a marker, plus so much per hPa. The legend shows
# SIZE_LEGEND_HPA.
MARKER_D0_PT = 1.8
MARKER_K_PT = 0.20
SIZE_LEGEND_HPA = (1005.0, 995.0, 985.0, 975.0)
PX_PER_PT = 1.35  # the Plotly page sizes markers in pixels

# Short button labels for the interactive page (the panel titles otherwise).
BUTTON_LABELS = {
    "extratropical_transition": "Extratropical transition",
    "tropical_decay": "Tropical decay",
    "warm_seclusion": "Warm seclusion",
    "cold_occlusion": "Cold occlusion",
    "tropical_transition": "Tropical transition",
    "hybrid": "Hybrid",
}

# One or two sentences per life cycle for the interactive page, in terms of
# the fields; the class sequence and the event hours are appended from the
# computed arrays.
NOTES = {
    "extratropical_transition": (
        "A typhoon recurving into the westerlies (Figure 13): the deep warm core profile blends to a "
        "transitioning and then a cold core profile while a storm-attached thickness dipole, warm to "
        "the right of the track, grows and then wraps into a late seclusion."),
    "tropical_decay": (
        "A tropical cyclone that weakens over cooler water south of the baroclinic zone: the deep warm "
        "core profile keeps its shape while its amplitude falls to 30 percent, with no dipole and no "
        "background gradient, so it never leaves the symmetric deep warm core cell."),
    "warm_seclusion": (
        "A frontal low (Shapiro-Keyser): a deep cold core with a strong dipole develops a shallow warm "
        "core as warm air is wrapped into the center, the dipole then decays so the secluded core is "
        "symmetric, and the low finally fills and turns cold again."),
    "cold_occlusion": (
        "An extratropical low that occludes: a deep cold core throughout, with a frontal dipole that "
        "decays as the warm sector is lifted off the center, so the low goes from asymmetric to "
        "symmetric cold core."),
    "tropical_transition": (
        "A subtropical storm becoming tropical: a cutoff cold low with a weak dipole builds a "
        "lower-tropospheric warm core under the cold air aloft, loses its asymmetry, and then its warm "
        "core deepens through the column."),
    "hybrid": (
        "A hybrid storm with a shallow warm core whose thickness dipole pulses with a 56 h period as "
        "short-wave troughs pass, so B swings back and forth across 10 m while both thermal wind "
        "terms hold their sign and the storm slowly deepens."),
}


# ------------------------------------------------------------------ data
def central_pressure(d):
    """Central MSLP (hPa) at every frame of a synthetic life cycle, derived
    from its vortex: the archetypes prescribe height perturbations and no
    pressure, so the depth is the 1000 hPa amplitude Scenario.amp(t)[0]
    (m, positive for a depression) at M_PER_HPA below P_ENV_HPA. This is
    the central value of the MSLP field compute_frame hands the closed-low
    detector (1000 hPa + Z_1000 / 8), without the background's offset."""
    sc = d["scenario"]
    return np.array([P_ENV_HPA - float(np.asarray(sc.amp(float(t)), dtype=float)[0]) / M_PER_HPA
                     for t in d["hours"]])


def marker_diameter(mslp, unit_scale=1.0):
    """Marker diameter (points, or pixels with unit_scale=PX_PER_PT) for a
    central pressure: MARKER_D0_PT plus MARKER_K_PT per hPa of depth below
    P_ENV_HPA, so a deeper low is a bigger marker, as on Hart's diagrams."""
    depth = np.clip(P_ENV_HPA - np.asarray(mslp, dtype=float), 0.0, None)
    return unit_scale * (MARKER_D0_PT + MARKER_K_PT * depth)


def compute_all(keys=None):
    """[run_scenario result] for every scenario in lifecycle_comparison.SCENARIOS
    (or the given keys), in that order: the extratropical transition first,
    each with its derived central pressure series under "MSLP"."""
    out = []
    for k in (keys or list(lc.SCENARIOS)):
        d = lc.run_scenario(k)
        d["MSLP"] = central_pressure(d)
        out.append(d)
    return out


def events_of(d):
    """{"onset": i, "completion": i} for the gridded path, by the rule of
    lifecycle_comparison.main(): onset is the first frame with B > 10 m and
    completion the first with -V_T^L < 0. Each is kept only when it is a
    crossing, that is, when the storm did not start on that side of the
    plane: a low that is already asymmetric or cold core at 0 h has no
    onset or completion, and a later return across the plane (a seclusion
    turning cold again, say) is not one either."""
    hours = d["hours"]
    on, comp = lc.onset_completion(hours, d["B_grid"], d["VTL_grid"], B_THR)
    ev = {}
    for kind, h in (("onset", on), ("completion", comp)):
        if np.isfinite(h) and h > hours[0]:
            ev[kind] = int(np.argmin(np.abs(hours - h)))
    return ev


def class_sequence(d, method="grid"):
    return [c for _, _, c in lc.class_runs(d["hours"], d[f"CLS_{method}"])]


def sequence_text(d, method="grid", arrow=" → "):
    return arrow.join("none" if c is None else str(c) for c in class_sequence(d, method))


def axis_limits(ds_list):
    def widen(base, *arrs):
        lo, hi = ds.widen_limits(base, *arrs)
        pad = 0.05 * (base[1] - base[0])
        return (min(base[0], lo - pad) if lo < base[0] else base[0],
                max(base[1], hi + pad) if hi > base[1] else base[1])
    cat = {k: np.concatenate([d[f"{k}_{m}"] for d in ds_list for m in ("hart", "grid")])
           for k in ("VTL", "B", "VTU")}
    return widen(BASE_XLIM, cat["VTL"]), widen(BASE_YLIM, cat["B"]), widen(BASE_ZLIM, cat["VTU"])


def class_cells(xlim, ylim, zlim):
    """{class: (x0, x1, y0, y1, z0, z1)}: the seven HCPSclass cells clipped to
    the axis limits (x = -V_T^L, y = B, z = -V_T^U)."""
    (xa, xb), (ya, yb), (za, zb) = xlim, ylim, zlim
    return {
        0: (0.0, xb, ya, B_THR, 0.0, zb),
        1: (0.0, xb, ya, B_THR, za, 0.0),
        2: (0.0, xb, B_THR, yb, 0.0, zb),
        3: (0.0, xb, B_THR, yb, za, 0.0),
        4: (xa, 0.0, B_THR, yb, za, 0.0),
        5: (xa, 0.0, ya, B_THR, za, 0.0),
        6: (xa, 0.0, ya, yb, 0.0, zb),
    }


def label_position(c, box, ylim):
    """Where a cell's label goes in the interactive page: toward the cell's
    outer corner, 70 percent of the way out in -V_T^L and 85 percent in
    -V_T^U, and just inside its outer B face, where the life cycles (B
    about -10 to 40 m, thermal winds mostly inside +/-250 m) seldom go.
    Class 6 spans all B and is labeled at mid B."""
    x0, x1, y0, y1, z0, z1 = box
    if c == 6:
        y = (y0 + y1) / 2
    elif y1 <= B_THR:
        y = y0 + 0.08 * (ylim[1] - ylim[0])
    else:
        y = y1 - 0.08 * (ylim[1] - ylim[0])
    x = 0.7 * (x1 if abs(x1) > abs(x0) else x0)
    z = 0.85 * (z1 if abs(z1) > abs(z0) else z0)
    return x, y, z


def box_faces(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[v[i] for i in f] for f in idx]


def dividing_plane_outlines(xlim, ylim, zlim):
    """Polylines (lists of (x, y, z)) where each dividing plane meets the
    faces of the axis box, only where it actually separates two classes:
    -V_T^L = 0 and -V_T^U = 0 everywhere; B = 10 m everywhere except inside
    class 6 (cold lower, warm upper), which spans both sides of it."""
    (xa, xb), (ya, yb), (za, zb) = xlim, ylim, zlim
    return [
        [(0, ya, za), (0, yb, za), (0, yb, zb), (0, ya, zb), (0, ya, za)],   # -VTL = 0
        [(xa, ya, 0), (xb, ya, 0), (xb, yb, 0), (xa, yb, 0), (xa, ya, 0)],   # -VTU = 0
        # B = 10: L-shaped (the x < 0, z > 0 quadrant belongs to class 6)
        [(0, B_THR, zb), (xb, B_THR, zb), (xb, B_THR, za), (xa, B_THR, za), (xa, B_THR, 0), (0, B_THR, 0),
         (0, B_THR, zb)],
    ]


def class_segments(x, y, z, cls):
    """{class: [polyline]} splitting a path at the midpoints between frames, so
    each frame's stretch of path (half a segment either side of it) carries
    that frame's class color and a change of class shows exactly between the
    two frames where it happens."""
    p = np.column_stack([x, y, z]).astype(float)
    n = len(p)
    out = {}
    for i in range(n):
        if not np.isfinite(cls[i]) or not np.all(np.isfinite(p[i])):
            continue
        pts = []
        if i > 0 and np.all(np.isfinite(p[i - 1])):
            pts.append((p[i - 1] + p[i]) / 2)
        pts.append(p[i])
        if i < n - 1 and np.all(np.isfinite(p[i + 1])):
            pts.append((p[i] + p[i + 1]) / 2)
        if len(pts) > 1:
            out.setdefault(int(round(cls[i])), []).append(np.array(pts))
    return out


# --------------------------------------------------------------- figure
def _halo(width=2.2):
    return patheffects.withStroke(linewidth=width, foreground="white")


def _annotate3d(ax, xyz, text, offset, color=ds.TEXT_DARK, **kw):
    """Label a 3-D point at a fixed screen offset (points): the point is
    projected with this view's own projection matrix and annotated in the
    axes' 2-D projected coordinates (the view is fixed before this call)."""
    x2, y2, _ = proj3d.proj_transform(*xyz, ax.get_proj())
    dx, dy, ha = offset
    ax.annotate(text, xy=(x2, y2), xycoords="data", xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="bottom" if dy > 0 else "top", color=color, zorder=12,
                path_effects=[_halo()], **kw)


def _proj2d(ax, pts):
    x2, y2, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())
    return np.column_stack([x2, y2])


def place_cell_labels(ax, cells, traj, lims, extra_avoid=None):
    """{class: (x, y, z)} for the class numbers in this panel's view: of a
    5 x 5 x 5 lattice of candidate points inside each cell, the one nearest
    the cell's centroid whose screen projection keeps a set clearance from
    the paths (densified along their segments) and from the labels already
    placed; failing that, the candidate with the most clearance."""
    traj = traj[np.all(np.isfinite(traj), axis=1)]
    dense = np.concatenate([np.linspace(a, b, 8, endpoint=False) for a, b in zip(traj[:-1], traj[1:])]
                           + [traj[-1:]])
    avoid = _proj2d(ax, dense)
    if extra_avoid is not None and len(extra_avoid):
        avoid = np.vstack([avoid, extra_avoid])
    corners = np.array([[x, y, z] for x in lims[0] for y in lims[1] for z in lims[2]])
    span = np.ptp(_proj2d(ax, corners), axis=0).max()
    clearance = 0.06 * span
    fr = np.linspace(0.15, 0.85, 5)
    placed = {}
    for c, (x0, x1, y0, y1, z0, z1) in cells.items():
        g = np.array([[x0 + a * (x1 - x0), y0 + b * (y1 - y0), z0 + e * (z1 - z0)]
                      for a in fr for b in fr for e in fr])
        rel = np.array([[a - 0.5, b - 0.5, e - 0.5] for a in fr for b in fr for e in fr])
        g2 = _proj2d(ax, g)
        pts = avoid if not placed else np.vstack([avoid, _proj2d(ax, np.array(list(placed.values())))])
        dist = np.min(np.linalg.norm(g2[:, None, :] - pts[None, :, :], axis=2), axis=1)
        ok = dist >= clearance
        best = np.argmin(np.where(ok, np.linalg.norm(rel, axis=1), np.inf)) if ok.any() else np.argmax(dist)
        placed[c] = tuple(g[best])
    return placed


def _label_box(anchor, off, text, fontsize):
    """(x0, y0, x1, y1) in points of a label of `text` placed at `off`
    (dx, dy, ha) from `anchor`, approximating glyph widths."""
    dx, dy, ha = off
    w, h = 0.6 * fontsize * len(text), 1.1 * fontsize
    x = anchor[0] + dx
    x0 = x if ha == "left" else (x - w if ha == "right" else x - w / 2)
    y0 = anchor[1] + dy if dy > 0 else anchor[1] + dy - h
    return np.array([x0, y0, x0 + w, y0 + h])


LABEL_ANGLES_ABOVE = [45, 135, -45, -135, 90, -90, 0, 180]  # tried in this order: above the path first
LABEL_ANGLES_BELOW = [-45, -135, 45, 135, -90, 90, 0, 180]  # ... below first, for the in-between hours


def _choose_label(anchor, text, fontsize, dist, path_pts, boxes, angles=None):
    """(offset, box, clash) for a label near `anchor` (points): of eight
    directions at `dist` points, the one whose box covers the fewest path
    points and does not overlap a label already placed (`boxes`); `clash`
    is True when every direction overlaps one. `angles` orders the
    directions tried (ties go to the earlier one), so the 24 h labels
    prefer one side of the path and the 12 h labels between them the
    other."""
    best = None
    for k, ang in enumerate(np.radians(angles or LABEL_ANGLES_ABOVE)):
        ux, uy = np.cos(ang), np.sin(ang)
        off = (dist * ux, dist * uy if abs(uy) > 0.1 else 0.1 * dist, "left" if ux > 0.3 else
               ("right" if ux < -0.3 else "center"))
        box = _label_box(anchor, off, text, fontsize)
        pad = 1.5
        inside = np.sum((path_pts[:, 0] > box[0] - pad) & (path_pts[:, 0] < box[2] + pad)
                        & (path_pts[:, 1] > box[1] - pad) & (path_pts[:, 1] < box[3] + pad))
        clash = any(box[0] < b[2] + 2 and b[0] < box[2] + 2 and box[1] < b[3] + 2 and b[1] < box[3] + 2
                    for b in boxes)
        score = (1000 if clash else 0) + inside + 0.01 * k
        if best is None or score < best[0]:
            best = (score, off, box, clash)
    return best[1], best[2], best[3]


def draw_panel(ax, d, lims, letter):
    xlim, ylim, zlim = lims
    hours = d["hours"]
    cells = class_cells(*lims)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect(BOX_ASPECT, zoom=0.95)
    ax.view_init(elev=VIEW[0], azim=VIEW[1])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor(ds.GRID_COLOR)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color=ds.GRID_COLOR, linewidth=0.35)

    # -- the seven cells, translucent, edges light
    for c, box in cells.items():
        poly = Poly3DCollection(box_faces(*box), facecolor=CLASS_PALETTE[c], alpha=CELL_ALPHA,
                                edgecolor=(0.3, 0.3, 0.3, 0.12), linewidth=0.25)
        poly.set_zorder(1)
        ax.add_collection3d(poly)
    for line in dividing_plane_outlines(*lims):
        p = np.array(line, dtype=float)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=ds.TEXT_SECONDARY, lw=0.5, alpha=0.6, zorder=2)

    # -- hour labels (start, end, every 24 h, then the 12 h in between),
    # chosen first so the class numbers can keep clear of them too: each in
    # the direction that covers the least of the paths, the 24 h ones
    # preferring the side above the path and the 12 h ones below, skipped
    # where it would overlap one already placed (the ends are placed first)
    n = len(hours) - 1
    pts = np.column_stack([d["VTL_grid"], d["B_grid"], d["VTU_grid"]])
    to_pt = 72.0 / ax.figure.dpi
    screen = ax.transData.transform(_proj2d(ax, pts)) * to_pt  # points
    both = np.concatenate([np.column_stack([d[f"VTL_{m}"], d[f"B_{m}"], d[f"VTU_{m}"]]) for m in ("grid", "hart")])
    both = both[np.all(np.isfinite(both), axis=1)]
    dense = np.concatenate([np.linspace(a, b, 6, endpoint=False) for a, b in zip(both[:-1], both[1:])])
    path_pts = ax.transData.transform(_proj2d(ax, dense)) * to_pt
    radius = marker_diameter(d["MSLP"]) / 2.0  # the marker's own half width, points
    hour_labels, boxes = [], []
    order = ([0, n] + [i for i, h in enumerate(hours) if h % HOUR_MAJOR == 0 and 0 < i < n]
             + [i for i, h in enumerate(hours) if h % HOUR_STEP == 0 and h % HOUR_MAJOR != 0 and 0 < i < n])
    for i in order:
        bold = i in (0, n)
        minor = hours[i] % HOUR_MAJOR != 0
        text = f"{hours[i]:.0f}" + (" h" if bold else "")
        fs = 7.0 if bold else (5.6 if minor else 6.3)
        off, box, clash = _choose_label(screen[i], text, fs, radius[i] + (5.0 if bold else 2.5), path_pts, boxes,
                                        LABEL_ANGLES_BELOW if minor else LABEL_ANGLES_ABOVE)
        if clash:
            continue
        boxes.append(box)
        hour_labels.append((i, bold, minor, off))
    anchors = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    label_avoid = (ax.transData.inverted().transform(np.array(anchors) / to_pt) if anchors else None)

    traj = np.concatenate([np.column_stack([d[f"VTL_{m}"], d[f"B_{m}"], d[f"VTU_{m}"]]) for m in ("hart", "grid")])
    for c, pos in place_cell_labels(ax, cells, traj, lims, label_avoid).items():
        ax.text(*pos, str(c), color=CLASS_PALETTE[c], fontsize=9, fontweight="bold", ha="center", va="center",
                zorder=3, path_effects=[_halo(2.0)])

    # -- Hart's path: thin gray line and open circles, sized by the central
    # pressure like the gridded markers
    diam = marker_diameter(d["MSLP"])  # points
    ax.plot(d["VTL_hart"], d["B_hart"], d["VTU_hart"], "-", color=HART_COLOR, lw=0.8, alpha=0.9, zorder=4)
    ax.scatter(d["VTL_hart"], d["B_hart"], d["VTU_hart"], s=(0.85 * diam) ** 2, marker="o", facecolor="white",
               edgecolor=HART_COLOR, linewidth=0.6, depthshade=False, zorder=5)

    # -- gridded path colored by class: a dark casing, then class-colored
    # halves, then square markers whose size grows with the depth
    ax.plot(d["VTL_grid"], d["B_grid"], d["VTU_grid"], "-", color=PATH_EDGE, lw=2.6, alpha=0.85, zorder=6,
            solid_capstyle="round")
    for c, segs in class_segments(d["VTL_grid"], d["B_grid"], d["VTU_grid"], d["CLS_grid"]).items():
        lc3 = Line3DCollection(segs, colors=[CLASS_PALETTE[c]], linewidths=1.6, zorder=7)
        ax.add_collection3d(lc3)
    ok = np.isfinite(d["CLS_grid"])
    cols = [CLASS_PALETTE[int(round(c))] for c in d["CLS_grid"][ok]]
    ax.scatter(d["VTL_grid"][ok], d["B_grid"][ok], d["VTU_grid"][ok], s=diam[ok] ** 2, marker="s", c=cols,
               edgecolor=PATH_EDGE, linewidth=0.45, depthshade=False, zorder=8)

    # -- start and end markers (just outside the sized marker), then the
    # hour labels chosen above
    for i, marker in ((0, "o"), (n, "s")):
        ring = (diam[i] + 4.0) ** 2
        ax.scatter([d["VTL_grid"][i]], [d["B_grid"][i]], [d["VTU_grid"][i]], s=ring, marker=marker,
                   facecolor="none", edgecolor=ds.TEXT_DARK, linewidth=1.2, depthshade=False, zorder=9)
    for i, bold, minor, off in hour_labels:
        _annotate3d(ax, tuple(pts[i]), f"{hours[i]:.0f}" + (" h" if bold else ""), off,
                    fontsize=7.0 if bold else (5.6 if minor else 6.3), fontweight="bold" if bold else "normal",
                    color=ds.TEXT_SECONDARY if minor else ds.TEXT_DARK)

    # -- onset and completion crossings of the gridded path
    for kind, i in events_of(d).items():
        ax.scatter([d["VTL_grid"][i]], [d["B_grid"][i]], [d["VTU_grid"][i]], marker="x", s=48,
                   color=ds.TEXT_DARK, linewidth=1.4, depthshade=False, zorder=10)

    ax.set_xlabel(ds.AXIS_LABEL_VTL, labelpad=-4, fontsize=8)
    ax.set_ylabel(ds.AXIS_LABEL_B, labelpad=-4, fontsize=8)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(ds.AXIS_LABEL_VTU, labelpad=-2, rotation=90, fontsize=8)
    ax.set_xticks([-300, -150, 0, 150, 300])
    ax.set_yticks([-20, 10, 40, 70] if ylim == BASE_YLIM else ax.get_yticks())
    ax.set_zticks([-300, -150, 0, 150, 300])
    ax.tick_params(axis="both", which="major", pad=-2, labelsize=6.5)


def panel_title(d, letter):
    sc = d["scenario"]
    ev = events_of(d)
    parts = []
    if "onset" in ev:
        parts.append(f"onset {d['hours'][ev['onset']]:.0f} h")
    if "completion" in ev:
        parts.append(f"completion {d['hours'][ev['completion']]:.0f} h")
    line2 = "classes " + sequence_text(d)
    if parts:
        line2 += " (" + ", ".join(parts) + ")"
    return f"({letter}) {sc.title}", line2


def make_png(ds_list, lims):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "text.color": ds.TEXT_DARK,
        "axes.labelcolor": ds.TEXT_DARK, "xtick.color": ds.TEXT_SECONDARY, "ytick.color": ds.TEXT_SECONDARY,
        "axes.linewidth": 0.6, "legend.fontsize": 7.5,
    })
    ncol = 3
    nrow = int(np.ceil(len(ds_list) / ncol))
    fig_w, row_h, legend_h = 12.0, 4.05, 1.1
    fig_h = nrow * row_h + legend_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    pw, ph = 1.0 / ncol, row_h / fig_h
    for k, d in enumerate(ds_list):
        r, c = divmod(k, ncol)
        x0 = c * pw
        y0 = 1.0 - (r + 1) * ph
        ax = fig.add_axes([x0 - 0.022, y0 - 0.012, pw - 0.004, ph * 0.93], projection="3d", computed_zorder=False)
        letter = "abcdefghij"[k]
        draw_panel(ax, d, lims, letter)
        t1, t2 = panel_title(d, letter)
        fig.text(x0 + 0.018, y0 + ph - 0.012, t1, fontsize=9.5, fontweight="bold", ha="left", va="top")
        fig.text(x0 + 0.018, y0 + ph - 0.012 - 0.20 / fig_h, t2, fontsize=7.8, color=ds.TEXT_SECONDARY,
                 ha="left", va="top")

    class_handles = [Patch(facecolor=CLASS_PALETTE[c], alpha=0.6, edgecolor="none", label=f"{c}  {CLASS_NAMES[c]}")
                     for c in range(7)]
    fig.legend(handles=class_handles, loc="lower left", bbox_to_anchor=(0.015, 0.004), ncol=4, fontsize=7.3,
               frameon=False, title="HCPSclass cells (B = 10 m, $-V_T^L$ = 0, $-V_T^U$ = 0), path colored by class",
               title_fontsize=7.8, handlelength=1.3, columnspacing=1.1, alignment="left")
    path_handles = [
        Line2D([0], [0], marker="s", color=PATH_EDGE, markerfacecolor=CLASS_HEX[0], markeredgecolor=PATH_EDGE,
               lw=1.6, markersize=4.5, label="gridded, standard levels"),
        Line2D([0], [0], marker="o", color=HART_COLOR, markerfacecolor="white", lw=0.8, markersize=3.5,
               label="Hart, 50 hPa levels"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=ds.TEXT_DARK,
               markersize=6.5, markeredgewidth=1.1, label="start (0 h), hours every 12 h"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none", markeredgecolor=ds.TEXT_DARK,
               markersize=6, markeredgewidth=1.1, label="end (168 h)"),
        Line2D([0], [0], marker="x", color=ds.TEXT_DARK, lw=0, markersize=6, markeredgewidth=1.4,
               label="onset (B rises past 10 m), completion ($-V_T^L$ turns negative)"),
    ]
    fig.legend(handles=path_handles, loc="lower right", bbox_to_anchor=(0.995, 0.004), ncol=1, fontsize=7.3,
               frameon=False, title="Synthetic life cycles, 0 to 168 h, 6 h frames", title_fontsize=7.8,
               handlelength=1.8, columnspacing=1.1, alignment="left")
    # marker size legend: the central pressure derived from the vortex
    size_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CLASS_HEX[0], markeredgecolor=PATH_EDGE,
               markeredgewidth=0.45, markersize=float(marker_diameter(p)), label=f"{p:.0f} hPa")
        for p in SIZE_LEGEND_HPA]
    fig.legend(handles=size_handles, loc="lower left", bbox_to_anchor=(0.015, 0.082), ncol=len(size_handles),
               fontsize=7.3, frameon=False, handlelength=1.8, columnspacing=1.6, alignment="left",
               title="Marker size: central pressure, derived from the vortex (deeper is bigger)",
               title_fontsize=7.8)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------- html
def _r(a):
    return [None if not np.isfinite(v) else round(float(v), 1) for v in np.asarray(a, dtype=float)]


def _cls_label(c):
    if not np.isfinite(c):
        return "none"
    c = int(round(c))
    return f"{c} ({CLASS_NAMES[c]})"


def _vtl_html():
    return "&minus;V<sub>T</sub><sup>L</sup>"


def _vtu_html():
    return "&minus;V<sub>T</sub><sup>U</sup>"


def _rgba(color, alpha):
    r, g, b = (int(round(255 * v)) for v in to_rgb(color))
    return f"rgba({r},{g},{b},{alpha})"


def _hover(d, i, m):
    hours = d["hours"]
    return (f"hour {hours[i]:.0f}<br>B {d[f'B_{m}'][i]:.1f} m<br>{_vtl_html()} {d[f'VTL_{m}'][i]:.1f} m"
            f"<br>{_vtu_html()} {d[f'VTU_{m}'][i]:.1f} m<br>class {_cls_label(d[f'CLS_{m}'][i])}"
            f"<br>central pressure {d['MSLP'][i]:.0f} hPa (derived)")


def _hour_label_idx(d, scale):
    """Frame indices labeled with their hour: the ends, every 24 h, then
    every 12 h between, each kept only where it is at least 6 percent of
    the axis ranges (`scale`) from one already kept."""
    hours = d["hours"]
    n = len(hours) - 1
    pts = np.column_stack([d["VTL_grid"], d["B_grid"], d["VTU_grid"]]) / scale
    idx = []
    order = ([0, n] + [i for i, h in enumerate(hours) if h % HOUR_MAJOR == 0 and 0 < i < n]
             + [i for i, h in enumerate(hours) if h % HOUR_STEP == 0 and h % HOUR_MAJOR != 0 and 0 < i < n])
    for i in order:
        if all(np.linalg.norm(pts[i] - pts[j]) > 0.06 for j in idx):
            idx.append(i)
    return sorted(idx)


def scenario_traces(d, key):
    """The Plotly traces of one life cycle in the three-dimensional view,
    each tagged with meta = key. Markers are sized by the derived central
    pressure (marker_diameter), hours labeled every 12 h."""
    hours = d["hours"]
    n = len(hours) - 1
    diam = marker_diameter(d["MSLP"], PX_PER_PT)
    traces = []

    # Hart's storm-centered path: thin gray line, open circles
    traces.append(dict(
        type="scatter3d", mode="lines+markers", name="Hart, 50 hPa levels", legendgroup="hart",
        x=_r(d["VTL_hart"]), y=_r(d["B_hart"]), z=_r(d["VTU_hart"]),
        line=dict(color=HART_COLOR, width=2),
        marker=dict(size=_r(0.85 * diam), sizemode="diameter", symbol="circle", color="#ffffff",
                    line=dict(color=HART_COLOR, width=1)),
        text=["Hart, 50 hPa levels<br>" + _hover(d, i, "hart") for i in range(len(hours))],
        hovertemplate="%{text}<extra></extra>",
    ))
    # gridded path: dark casing, then one trace per class of half-segments
    traces.append(dict(
        type="scatter3d", mode="lines", name="gridded, standard levels", legendgroup="grid", showlegend=False,
        x=_r(d["VTL_grid"]), y=_r(d["B_grid"]), z=_r(d["VTU_grid"]), hoverinfo="skip",
        line=dict(color=PATH_EDGE, width=9),
    ))
    for c, segs in sorted(class_segments(d["VTL_grid"], d["B_grid"], d["VTU_grid"], d["CLS_grid"]).items()):
        xs, ys, zs = [], [], []
        for s in segs:
            xs += _r(s[:, 0]) + [None]
            ys += _r(s[:, 1]) + [None]
            zs += _r(s[:, 2]) + [None]
        traces.append(dict(type="scatter3d", mode="lines", legendgroup="grid", showlegend=False,
                           x=xs, y=ys, z=zs, hoverinfo="skip", line=dict(color=CLASS_HEX[c], width=6)))
    cols = [CLASS_HEX[int(round(c))] if np.isfinite(c) else "#ffffff" for c in d["CLS_grid"]]
    traces.append(dict(
        type="scatter3d", mode="markers", name="gridded, standard levels (colored by class, sized by depth)",
        legendgroup="grid",
        x=_r(d["VTL_grid"]), y=_r(d["B_grid"]), z=_r(d["VTU_grid"]),
        marker=dict(size=_r(diam), sizemode="diameter", symbol="square", color=cols,
                    line=dict(color=PATH_EDGE, width=1)),
        text=["gridded, standard levels<br>" + _hover(d, i, "grid") for i in range(len(hours))],
        hovertemplate="%{text}<extra></extra>",
    ))
    # start and end
    traces.append(dict(
        type="scatter3d", mode="markers", name="start (0 h) and end (168 h)", legendgroup="ends",
        x=_r(d["VTL_grid"][[0, n]]), y=_r(d["B_grid"][[0, n]]), z=_r(d["VTU_grid"][[0, n]]),
        marker=dict(size=_r(diam[[0, n]] + 6.0), sizemode="diameter", symbol=["circle-open", "square-open"],
                    color=ds.TEXT_DARK, line=dict(width=3)),
        text=[f"start, {hours[0]:.0f} h", f"end, {hours[n]:.0f} h"], hovertemplate="%{text}<extra></extra>",
    ))
    # hour labels every 12 h: the 24 h ones nudged up the -VTU axis and set
    # above the marker, the 12 h ones between them nudged down and set
    # below; a label is left out where it would sit on one already kept
    # (the ends first), measured in axis-range units
    span = np.array([BASE_XLIM[1] - BASE_XLIM[0], BASE_YLIM[1] - BASE_YLIM[0], BASE_ZLIM[1] - BASE_ZLIM[0]])
    idx = _hour_label_idx(d, span)
    minor = [hours[i] % HOUR_MAJOR != 0 for i in idx]
    traces.append(dict(
        type="scatter3d", mode="text", name="hours every 12 h", legendgroup="hours", showlegend=True,
        x=_r(d["VTL_grid"][idx]), y=_r(d["B_grid"][idx]),
        z=_r(d["VTU_grid"][idx] + np.where(minor, -14.0, 14.0)),
        text=[f"{hours[i]:.0f}" + ("" if m else " h") for i, m in zip(idx, minor)], hoverinfo="skip",
        textposition=["bottom center" if m else "top center" for m in minor],
        textfont=dict(color=[ds.TEXT_SECONDARY if m else ds.TEXT_DARK for m in minor],
                      size=[13 if i in (0, n) else (10 if m else 11) for i, m in zip(idx, minor)]),
    ))
    # onset and completion crossings
    ev = events_of(d)
    if ev:
        ii = [ev[k] for k in ("onset", "completion") if k in ev]
        labels = [f"{k} {hours[ev[k]]:.0f} h" for k in ("onset", "completion") if k in ev]
        traces.append(dict(
            type="scatter3d", mode="markers+text", name="onset and completion", legendgroup="events",
            x=_r(d["VTL_grid"][ii]), y=_r(d["B_grid"][ii]), z=_r(d["VTU_grid"][ii]),
            text=labels, textposition="middle right", textfont=dict(color=ds.TEXT_DARK, size=12),
            hovertext=[{"onset": "onset: B first rises above 10 m",
                        "completion": "completion: " + "-V<sub>T</sub><sup>L</sup> first turns negative"}[k]
                       + f", {hours[ev[k]]:.0f} h" for k in ("onset", "completion") if k in ev],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(symbol="x", size=6, color=ds.TEXT_DARK),
        ))
    for t in traces:
        t["meta"] = key
        t["visible"] = False
    return traces


def shared_traces(lims):
    xlim, ylim, zlim = lims
    cells = class_cells(xlim, ylim, zlim)
    traces = []
    tri_i = [0, 0, 4, 4, 0, 0, 3, 3, 1, 1, 0, 0]
    tri_j = [1, 2, 5, 6, 1, 5, 2, 6, 2, 6, 3, 7]
    tri_k = [2, 3, 6, 7, 5, 4, 6, 7, 6, 5, 7, 4]
    for c, (x0, x1, y0, y1, z0, z1) in cells.items():
        traces.append(dict(
            type="mesh3d", name=f"{c}  {CLASS_NAMES[c]}", legendgroup=f"c{c}", showlegend=True,
            x=[x0, x1, x1, x0, x0, x1, x1, x0], y=[y0, y0, y1, y1, y0, y0, y1, y1],
            z=[z0, z0, z0, z0, z1, z1, z1, z1], i=tri_i, j=tri_j, k=tri_k,
            color=CLASS_HEX[c], opacity=0.12, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0),
            hovertemplate=f"class {c}<br>{CLASS_NAMES[c]}<extra></extra>",
        ))
        x, y, z = label_position(c, (x0, x1, y0, y1, z0, z1), ylim)
        traces.append(dict(
            type="scatter3d", mode="text", legendgroup=f"c{c}", showlegend=False, hoverinfo="skip",
            x=[x], y=[y], z=[z], text=[f"<b>{c}</b> {CLASS_NAMES[c]}"],
            textfont=dict(color=CLASS_HEX[c] if c not in (2, 6) else {2: "#a88b00", 6: "#77766f"}[c], size=11),
        ))
    xs, ys, zs = [], [], []
    for line in dividing_plane_outlines(xlim, ylim, zlim):
        for p in line:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        xs.append(None); ys.append(None); zs.append(None)
    traces.append(dict(type="scatter3d", mode="lines", name="planes B = 10 m, -VTL = 0, -VTU = 0",
                       x=xs, y=ys, z=zs, hoverinfo="skip", line=dict(color=ds.TEXT_SECONDARY, width=2)))
    for t in traces:
        t["meta"] = "shared"
        t["visible"] = True
    return traces


# ---- the same paths on Hart's two diagrams (the interactive page's 2-D view)
PANELS_2D = (("x", "y", "B"), ("x2", "y2", "VTU"))  # (xaxis, yaxis, the vertical parameter) of each diagram


def scenario_traces_2d(d, key):
    """The Plotly traces of one life cycle on Hart's two diagrams, B against
    -V_T^L (axes x, y) and -V_T^U against -V_T^L (axes x2, y2), each
    tagged with meta = key: the same lines, class colors, marker sizes
    and hour labels as the three-dimensional view."""
    hours = d["hours"]
    n = len(hours) - 1
    diam = marker_diameter(d["MSLP"], PX_PER_PT)
    span = np.array([BASE_XLIM[1] - BASE_XLIM[0], BASE_YLIM[1] - BASE_YLIM[0], BASE_ZLIM[1] - BASE_ZLIM[0]])
    idx = _hour_label_idx(d, span)
    minor = [hours[i] % HOUR_MAJOR != 0 for i in idx]
    ev = events_of(d)
    cols = [CLASS_HEX[int(round(c))] if np.isfinite(c) else "#ffffff" for c in d["CLS_grid"]]
    traces = []
    for p, (xa, ya, vert) in enumerate(PANELS_2D):
        first = p == 0
        ax_kw = dict(xaxis=xa, yaxis=ya)
        traces.append(dict(
            type="scatter", mode="lines+markers", name="Hart, 50 hPa levels", legendgroup="hart",
            showlegend=first, x=_r(d["VTL_hart"]), y=_r(d[f"{vert}_hart"]),
            line=dict(color=HART_COLOR, width=1.2),
            marker=dict(size=_r(0.85 * diam), sizemode="diameter", symbol="circle", color="#ffffff",
                        line=dict(color=HART_COLOR, width=1)),
            text=["Hart, 50 hPa levels<br>" + _hover(d, i, "hart") for i in range(len(hours))],
            hovertemplate="%{text}<extra></extra>", **ax_kw,
        ))
        traces.append(dict(
            type="scatter", mode="lines", legendgroup="grid", showlegend=False, hoverinfo="skip",
            x=_r(d["VTL_grid"]), y=_r(d[f"{vert}_grid"]), line=dict(color=PATH_EDGE, width=4.5), **ax_kw,
        ))
        zero = np.zeros_like(d["VTL_grid"])
        for c, segs in sorted(class_segments(d["VTL_grid"], d[f"{vert}_grid"], zero, d["CLS_grid"]).items()):
            xs, ys = [], []
            for s in segs:
                xs += _r(s[:, 0]) + [None]
                ys += _r(s[:, 1]) + [None]
            traces.append(dict(type="scatter", mode="lines", legendgroup="grid", showlegend=False, x=xs, y=ys,
                               hoverinfo="skip", line=dict(color=CLASS_HEX[c], width=2.5), **ax_kw))
        traces.append(dict(
            type="scatter", mode="markers", name="gridded, standard levels (colored by class, sized by depth)",
            legendgroup="grid", showlegend=first, x=_r(d["VTL_grid"]), y=_r(d[f"{vert}_grid"]),
            marker=dict(size=_r(diam), sizemode="diameter", symbol="square", color=cols,
                        line=dict(color=PATH_EDGE, width=1)),
            text=["gridded, standard levels<br>" + _hover(d, i, "grid") for i in range(len(hours))],
            hovertemplate="%{text}<extra></extra>", **ax_kw,
        ))
        traces.append(dict(
            type="scatter", mode="markers", name="start (0 h) and end (168 h)", legendgroup="ends",
            showlegend=first, x=_r(d["VTL_grid"][[0, n]]), y=_r(d[f"{vert}_grid"][[0, n]]),
            marker=dict(size=_r(diam[[0, n]] + 6.0), sizemode="diameter", symbol=["circle-open", "square-open"],
                        color=ds.TEXT_DARK, line=dict(width=2)),
            text=[f"start, {hours[0]:.0f} h", f"end, {hours[n]:.0f} h"], hovertemplate="%{text}<extra></extra>",
            **ax_kw,
        ))
        traces.append(dict(
            type="scatter", mode="text", name="hours every 12 h", legendgroup="hours", showlegend=first,
            x=_r(d["VTL_grid"][idx]), y=_r(d[f"{vert}_grid"][idx]), hoverinfo="skip",
            text=[f"{hours[i]:.0f}" + ("" if m else " h") for i, m in zip(idx, minor)],
            textposition=["bottom center" if m else "top center" for m in minor],
            textfont=dict(color=[ds.TEXT_SECONDARY if m else ds.TEXT_DARK for m in minor],
                          size=[13 if i in (0, n) else (10 if m else 11) for i, m in zip(idx, minor)]),
            **ax_kw,
        ))
        if ev:
            ii = [ev[k] for k in ("onset", "completion") if k in ev]
            traces.append(dict(
                type="scatter", mode="markers+text", name="onset and completion", legendgroup="events",
                showlegend=first, x=_r(d["VTL_grid"][ii]), y=_r(d[f"{vert}_grid"][ii]),
                text=[f"{k} {hours[ev[k]]:.0f} h" for k in ("onset", "completion") if k in ev],
                textposition="middle right", textfont=dict(color=ds.TEXT_DARK, size=12),
                hovertext=[{"onset": "onset: B first rises above 10 m",
                            "completion": "completion: " + "-V<sub>T</sub><sup>L</sup> first turns negative"}[k]
                           + f", {hours[ev[k]]:.0f} h" for k in ("onset", "completion") if k in ev],
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(symbol="x", size=10, color=ds.TEXT_DARK, line=dict(width=2, color=ds.TEXT_DARK)),
                **ax_kw,
            ))
    for t in traces:
        t["meta"] = key
        t["visible"] = False
    return traces


def layout_2d(lims):
    """The Plotly layout of the two-diagram view: Hart's quadrants of
    diagram_style (shaded in its colors and alphas, labeled in their
    corners), the 10 m line on the first diagram and the zero lines on
    both, drawn as layout shapes and annotations so they need no trace.
    The axis domains are set by the page for the screen width."""
    xlim, ylim, zlim = lims
    shapes, notes = [], []
    # diagram 1: B against -VTL
    quads = [
        ((0, xlim[1]), (ylim[0], B_THR), "symmetric warm core"),
        ((0, xlim[1]), (B_THR, ylim[1]), "asymmetric warm core"),
        ((xlim[0], 0), (B_THR, ylim[1]), "asymmetric cold core"),
        ((xlim[0], 0), (ylim[0], B_THR), "symmetric cold core"),
    ]
    for (x0, x1), (y0, y1), name in quads:
        shapes.append(dict(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1, layer="below",
                           line=dict(width=0), fillcolor=_rgba(ds.B_VTL_QUADRANT_COLORS[name], ds.B_VTL_QUADRANT_ALPHA)))
        fx, fy, ha, va = ds.B_VTL_LABEL_POS[name]
        notes.append(dict(xref="x domain", yref="y domain", x=fx, y=fy, xanchor=ha, yanchor=va, showarrow=False,
                          text=f"<i>{name}</i>", font=dict(size=11, color=ds.TEXT_SECONDARY)))
    shapes.append(dict(type="line", xref="x", yref="y", x0=xlim[0], x1=xlim[1], y0=B_THR, y1=B_THR,
                       line=dict(color=ds.TEXT_DARK, width=1.2)))
    shapes.append(dict(type="line", xref="x", yref="y", x0=0, x1=0, y0=ylim[0], y1=ylim[1],
                       line=dict(color=ds.TEXT_DARK, width=1.2)))
    # diagram 2: -VTU against -VTL
    quads2 = [
        ((0, xlim[1]), (0, zlim[1]), "deep warm core", (0.55, 0.90)),
        ((0, xlim[1]), (zlim[0], 0), "shallow warm core", (0.55, 0.10)),
        ((xlim[0], 0), (zlim[0], 0), "deep cold core", (0.03, 0.10)),
        ((xlim[0], 0), (0, zlim[1]), "shallow cold core", (0.03, 0.84)),
    ]
    for (x0, x1), (y0, y1), name, (fx, fy) in quads2:
        shapes.append(dict(type="rect", xref="x2", yref="y2", x0=x0, x1=x1, y0=y0, y1=y1, layer="below",
                           line=dict(width=0), fillcolor=_rgba(ds.VTU_VTL_QUADRANT_COLORS[name], ds.VTU_VTL_QUADRANT_ALPHA)))
        notes.append(dict(xref="x2 domain", yref="y2 domain", x=fx, y=fy, xanchor="left", yanchor="middle",
                          showarrow=False, text=f"<i>{name}</i>", font=dict(size=11, color=ds.TEXT_SECONDARY)))
    shapes.append(dict(type="line", xref="x2", yref="y2", x0=xlim[0], x1=xlim[1], y0=0, y1=0,
                       line=dict(color=ds.TEXT_DARK, width=1.2)))
    shapes.append(dict(type="line", xref="x2", yref="y2", x0=0, x1=0, y0=zlim[0], y1=zlim[1],
                       line=dict(color=ds.TEXT_DARK, width=1.2)))
    for ref, title in (("x", "(a) B against the lower thermal wind"), ("x2", "(b) upper against lower thermal wind")):
        notes.append(dict(xref=f"{ref} domain", yref=f"{'y' if ref == 'x' else 'y2'} domain", x=0, y=1.0,
                          xanchor="left", yanchor="bottom", yshift=4, showarrow=False, text=f"<b>{title}</b>",
                          font=dict(size=12, color=ds.TEXT_DARK)))

    def axis(title, rng, dtick):
        return dict(title=dict(text=title, standoff=6), range=list(rng), dtick=dtick, gridcolor="#dedcd5",
                    zeroline=False, linecolor="#dedcd5", mirror=True, showline=True, ticks="outside",
                    tickcolor="#dedcd5")

    return dict(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", margin=dict(l=56, r=10, t=28, b=48), uirevision="keep",
        font=dict(family='-apple-system, "Segoe UI", Helvetica, Arial, sans-serif', size=12, color="#1a1a1a"),
        legend=dict(orientation="h", x=0, y=-0.12, yanchor="top", bgcolor="rgba(255,255,255,0.8)",
                    font=dict(size=11), itemsizing="constant"),
        shapes=shapes, annotations=notes, hovermode="closest",
        xaxis=dict(axis("-V<sub>T</sub><sup>L</sup>, lower thermal wind (m)", xlim, 100), anchor="y"),
        yaxis=dict(axis("B, thermal asymmetry (m)", ylim, 20), anchor="x"),
        xaxis2=dict(axis("-V<sub>T</sub><sup>L</sup>, lower thermal wind (m)", xlim, 100), anchor="y2"),
        yaxis2=dict(axis("-V<sub>T</sub><sup>U</sup>, upper thermal wind (m)", zlim, 100), anchor="x2"),
    )


def scenario_note(d):
    sc, hours = d["scenario"], d["hours"]
    runs = lc.class_runs(hours, d["CLS_grid"])
    seq = ", ".join(f"{c} from {h0:.0f} to {h1:.0f} h" if h1 > h0 else f"{c} at {h0:.0f} h"
                    for h0, h1, c in runs)
    ev = events_of(d)
    if ev:
        evs = " and ".join(f"{k} at {hours[i]:.0f} h" for k, i in ev.items())
        evs = f"The x marks {evs}."
    else:
        evs = "There is no onset or completion crossing to mark."
    same = class_sequence(d, "hart") == class_sequence(d, "grid")
    hart = ("Hart's method passes through the same classes in the same order, as it must by construction "
            "(the cells are his thresholds); the two methods can differ only where their numbers put a "
            "frame on different sides of a plane."
            if same else
            f"Hart's method reads {sequence_text(d, 'hart', ', ')}: the cells are his thresholds, so the "
            "difference is a frame whose numbers fall on different sides of a plane in the two methods.")
    p = d["MSLP"]
    press = (f"Central pressure (derived from the vortex) {p[0]:.0f} hPa at 0 h, "
             f"{p.min():.0f} hPa at its deepest ({hours[int(np.argmin(p))]:.0f} h), {p[-1]:.0f} hPa at 168 h.")
    return f"{NOTES.get(sc.key, sc.summary)} Gridded classes: {seq}. {evs} {hart} {press}"


def make_html(ds_list, lims):
    xlim, ylim, zlim = lims
    traces = shared_traces(lims)
    traces2 = []
    examples = []
    for d in ds_list:
        key = d["scenario"].key
        traces += scenario_traces(d, key)
        traces2 += scenario_traces_2d(d, key)
        ev = events_of(d)
        evs = "; ".join(f"{k} {d['hours'][i]:.0f} h" for k, i in ev.items()) or "no onset or completion crossing"
        examples.append(dict(key=key, title=d["scenario"].title,
                             summary=f"Gridded classes {sequence_text(d)}; {evs}.",
                             note=scenario_note(d)))

    def axis(title, rng, dtick):
        return dict(title=dict(text=title), range=list(rng), dtick=dtick, backgroundcolor="#ffffff",
                    gridcolor="#dedcd5", zerolinecolor="#dedcd5", showbackground=False)

    layout = dict(
        paper_bgcolor="#ffffff", margin=dict(l=0, r=0, t=0, b=0), uirevision="keep",
        font=dict(family='-apple-system, "Segoe UI", Helvetica, Arial, sans-serif', size=12, color="#1a1a1a"),
        legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.8)", font=dict(size=11), itemsizing="constant"),
        scene=dict(
            xaxis=axis("-V<sub>T</sub><sup>L</sup>, lower thermal wind (m)", xlim, 100),
            yaxis=axis("B, thermal asymmetry (m)", ylim, 20),
            zaxis=axis("-V<sub>T</sub><sup>U</sup>, upper thermal wind (m)", zlim, 100),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.9),
            camera=dict(eye=dict(x=1.35, y=-1.65, z=0.8)),
            uirevision="keep",
        ),
    )
    config = dict(responsive=True, displaylogo=False)
    sizes = [dict(hpa=p, px=round(float(marker_diameter(p, PX_PER_PT)), 1)) for p in SIZE_LEGEND_HPA]
    payload = dict(data=traces, layout=layout, data2=traces2, layout2=layout_2d(lims), config=config,
                   examples=examples, sizes=sizes)
    fig_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    buttons = "\n".join(
        f'  <button type="button" data-key="{html.escape(e["key"])}" aria-pressed="false">'
        f'{html.escape(BUTTON_LABELS.get(e["key"], e["title"]))}</button>' for e in examples)
    title = "Hart's cyclone phase space: the seven HCPSclass cells and six synthetic life cycles"
    intro = (
        "The space is Hart's (2003): his three phase-space parameters on three axes, the lower thermal wind "
        f"{_vtl_html()}, the thermal asymmetry B and the upper thermal wind {_vtu_html()}, all in meters. Hart "
        "presents it as two cross-sections for legibility, B against the lower term and the upper term "
        "against the lower; the three-dimensional view here is a rendering of the same space, and the "
        f"two-diagram view shows the same path on his two diagrams. The planes B = 10 m, {_vtl_html()} = 0 and "
        f"{_vtu_html()} = 0 are his thresholds; they divide the space into the seven colored HCPSclass cells, "
        "class 6 (cold below, warm above) spanning both sides of B = 10 m, and in the two-diagram view they "
        "are the lines between his shaded quadrants. Choose a life cycle with the buttons: each is a synthetic "
        "168 h storm whose height fields are built from a vortex profile, a thermal dipole across the track, "
        "a background gradient and a track, and whose path is computed frame by frame by the gridded method "
        "(squares, colored by class) and by Hart's storm-centered method (gray circles). As on Hart's "
        "diagrams, the marker grows with the central pressure depth (the archetypes prescribe no pressure, "
        "so it is derived from the vortex's 1000 hPa amplitude at 8 m per hPa below 1013 hPa; the sizes "
        + ", ".join(f"{s['hpa']:.0f}" for s in sizes) + " hPa are shown beside the buttons) and the hour "
        "is labeled every 12 h. Hover over a point for its hour, values, class and central pressure; in the "
        "three-dimensional view drag to rotate and scroll to zoom. The view is kept when you switch life "
        "cycles."
    )
    size_key = "".join(
        f'<span class="sz"><i style="width:{s["px"]}px;height:{s["px"]}px"></i>{s["hpa"]:.0f}</span>' for s in sizes)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cyclone phase space 3D</title>
<style>
  html, body {{ background: #ffffff; color: #1a1a1a; margin: 0; }}
  body {{ font-family: Georgia, "Times New Roman", serif; font-size: 17px; line-height: 1.55; padding: 0 16px; }}
  h1 {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 1.05em;
        line-height: 1.3; margin: 0.7em 0 0.45em; }}
  .controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin: 0 0 0.35em; }}
  .examples, .views {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .controls button {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 13px;
        line-height: 1.2; color: #1a1a1a; background: #f7f6f2; border: 1px solid #d8d5cc; border-radius: 4px;
        padding: 5px 9px; cursor: pointer; }}
  .controls button:hover {{ border-color: #8f8e89; }}
  .controls button:focus-visible {{ outline: 2px solid #1a4f8b; outline-offset: 1px; }}
  .controls button[aria-pressed="true"] {{ background: #1a1a1a; border-color: #1a1a1a; color: #ffffff; }}
  .views {{ margin-left: auto; }}
  .views button[aria-pressed="true"] {{ background: #1a4f8b; border-color: #1a4f8b; }}
  .sizes {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 12px;
        color: #52514e; display: flex; align-items: center; gap: 8px; }}
  .sizes .sz {{ display: inline-flex; align-items: center; gap: 3px; }}
  .sizes .sz i {{ display: inline-block; background: #e34948; border: 1px solid #3a3a37; box-sizing: border-box; }}
  p.current {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 13px;
        color: #52514e; margin: 0 0 0.2em; }}
  #plot {{ width: 100%; height: 72vh; min-height: 380px; }}
  @media (max-width: 899px) {{ #plot {{ height: 92vh; min-height: 560px; }} }}
  body.embedded h1 {{ display: none; }}
  body.embedded .controls {{ margin-top: 8px; }}
  body.embedded #plot {{ min-height: 360px; }}
  p.about {{ max-width: 46em; margin: 0.6em 0 0; font-size: 0.9em; color: #1a1a1a; }}
  p.note {{ max-width: 46em; margin: 0.6em 0 1.2em; font-size: 0.9em; color: #3a3a37; }}
  noscript p {{ max-width: 46em; font-size: 0.9em; color: #3a3a37; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="controls">
<div class="examples" role="group" aria-label="Life cycle shown">
{buttons}
</div>
<div class="sizes" aria-label="Marker size: central pressure, hPa"><span>marker size, hPa:</span>{size_key}</div>
<div class="views" role="group" aria-label="View">
  <button type="button" data-view="3d" aria-pressed="true">Three dimensions</button>
  <button type="button" data-view="2d" aria-pressed="false">Hart's two diagrams</button>
</div>
</div>
<p class="current" id="current" aria-live="polite"></p>
<div id="plot"><noscript><p>The interactive figure needs JavaScript. The static Figure 7 of the article shows
the same six life cycles in the three-dimensional view, and Figure 1 shows Hart's two diagrams.</p></noscript></div>
<p class="about" id="about"></p>
<p class="note">{intro}</p>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.3/plotly.min.js"></script>
<script>
  (function () {{
    var fig = {fig_json};
    var el = document.getElementById("plot");
    var current = document.getElementById("current");
    var about = document.getElementById("about");
    // Inside the article's iframe the article supplies the title, and the
    // plot fills the frame below the buttons.
    var embedded = window.self !== window.top;
    if (embedded) {{ document.body.className = "embedded"; }}
    // On a narrow screen the plot is kept not much taller than it is wide,
    // since the scene scales to the shorter side; the two diagrams are
    // stacked instead of side by side.
    var view = "3d";
    function fitPlot() {{
      if (embedded) {{
        var top = el.getBoundingClientRect().top + window.pageYOffset;
        el.style.height = Math.max(360, window.innerHeight - top - 6) + "px";
      }} else if (el.clientWidth < 900) {{
        var h = view === "2d" ? el.clientWidth * 1.9 : el.clientWidth * 1.25;
        el.style.height = Math.round(Math.min(window.innerHeight * 0.92, Math.max(420, h))) + "px";
      }} else {{
        el.style.height = "";
      }}
    }}
    fitPlot();
    var buttons = Array.prototype.slice.call(document.querySelectorAll(".examples button"));
    var viewButtons = Array.prototype.slice.call(document.querySelectorAll(".views button"));
    var keys = fig.examples.map(function (e) {{ return e.key; }});
    // Legend beside the scene on wide screens, across the top of it on
    // narrow ones (phones, or the page inside the article's iframe), where
    // the scene leaves a band free.
    var wideLegend = fig.layout.legend;
    var narrowLegend = {{orientation: "h", x: 0, y: 1, yanchor: "top", font: {{size: 10}},
                         itemsizing: "constant", bgcolor: "rgba(255,255,255,0.8)"}};
    function legendFor(w) {{ return w < 900 ? narrowLegend : wideLegend; }}
    // On narrow screens the cells are named in the scene only, which keeps
    // the legend below the plot to the path symbols.
    var meshIdx = [];
    fig.data.forEach(function (t, i) {{ if (t.type === "mesh3d") {{ meshIdx.push(i); }} }});
    function visibleFor(data, key) {{
      return data.map(function (t) {{ return t.meta === "shared" || t.meta === key; }});
    }}
    function domains2d(narrow) {{
      var L = fig.layout2;
      if (narrow) {{
        L.xaxis.domain = [0, 1]; L.yaxis.domain = [0.58, 1];
        L.xaxis2.domain = [0, 1]; L.yaxis2.domain = [0, 0.42];
        L.legend.y = -0.07;
      }} else {{
        L.xaxis.domain = [0, 0.46]; L.yaxis.domain = [0, 1];
        L.xaxis2.domain = [0.54, 1]; L.yaxis2.domain = [0, 1];
        L.legend.y = -0.12;
      }}
    }}
    var camera = null;
    function dataFor(v) {{ return v === "2d" ? fig.data2 : fig.data; }}
    function layoutFor(v) {{
      if (v === "2d") {{ domains2d(el.clientWidth < 700); return fig.layout2; }}
      if (camera) {{ fig.layout.scene.camera = camera; }}
      return fig.layout;
    }}
    var key = keys[0];
    function show(k, push) {{
      if (keys.indexOf(k) < 0) {{ k = keys[0]; }}
      key = k;
      var ex = fig.examples[keys.indexOf(k)];
      buttons.forEach(function (b) {{ b.setAttribute("aria-pressed", b.dataset.key === k ? "true" : "false"); }});
      current.textContent = ex.summary;
      about.innerHTML = "<strong>" + ex.title + ".<\\/strong> " + ex.note;
      if (window.Plotly && el.data) {{ Plotly.restyle(el, {{visible: visibleFor(dataFor(view), k)}}); }}
      if (push && window.history && history.replaceState) {{
        history.replaceState(null, "", "#" + k + (view === "2d" ? "/diagrams" : ""));
      }}
    }}
    function draw() {{
      var data = dataFor(view), vis = visibleFor(data, key);
      data.forEach(function (t, i) {{ t.visible = vis[i]; }});
      fitPlot();
      Plotly.react(el, data, layoutFor(view), fig.config);
    }}
    function setView(v, push) {{
      view = v === "2d" ? "2d" : "3d";
      viewButtons.forEach(function (b) {{ b.setAttribute("aria-pressed", b.dataset.view === view ? "true" : "false"); }});
      if (window.Plotly && el.data) {{ draw(); }}
      if (push && window.history && history.replaceState) {{
        history.replaceState(null, "", "#" + key + (view === "2d" ? "/diagrams" : ""));
      }}
    }}
    var hash = (location.hash || "").replace("#", "").split("/");
    var start = hash[0];
    if (keys.indexOf(start) < 0) {{ start = keys[0]; }}
    if (hash[1] === "diagrams") {{ view = "2d"; }}
    key = start;
    buttons.forEach(function (b) {{
      b.addEventListener("click", function () {{ show(b.dataset.key, true); }});
    }});
    viewButtons.forEach(function (b) {{
      b.addEventListener("click", function () {{ setView(b.dataset.view, true); }});
      b.setAttribute("aria-pressed", b.dataset.view === view ? "true" : "false");
    }});
    if (window.Plotly) {{
      var narrow = el.clientWidth < 900;
      fig.layout.legend = legendFor(el.clientWidth);
      if (narrow) {{
        var eye = fig.layout.scene.camera.eye, f = el.clientWidth < 520 ? 1.75 : 1.15;
        fig.layout.scene.camera.eye = {{x: eye.x * f, y: eye.y * f, z: eye.z * f}};
      }}
      meshIdx.forEach(function (i) {{ fig.data[i].showlegend = !narrow; }});
      draw();
      el.on("plotly_relayout", function (e) {{
        if (e && e["scene.camera"]) {{ camera = e["scene.camera"]; }}
      }});
      window.addEventListener("resize", function () {{
        fitPlot();
        if (view === "2d") {{ domains2d(el.clientWidth < 700); Plotly.relayout(el, fig.layout2); return; }}
        Plotly.Plots.resize(el);
        var nowNarrow = el.clientWidth < 900;
        if (nowNarrow !== narrow) {{
          narrow = nowNarrow;
          fig.layout.legend = legendFor(el.clientWidth);
          Plotly.relayout(el, {{legend: fig.layout.legend}});
          meshIdx.forEach(function (i) {{ fig.data[i].showlegend = !narrow; }});
          Plotly.restyle(el, {{showlegend: !narrow}}, meshIdx);
        }}
      }});
    }} else {{
      el.textContent = "The interactive plot needs Plotly, loaded from cdn.jsdelivr.net, which did not load.";
    }}
    show(start, false);
  }})();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(page, encoding="utf-8")


# ----------------------------------------------------------------- main
def main(keys=None):
    t0 = time.perf_counter()
    ds_list = compute_all(keys)
    t_compute = time.perf_counter() - t0
    for d in ds_list:
        lc.print_scenario_summary(d)
    lims = axis_limits(ds_list)
    print(f"\naxis limits: -VTL {lims[0]}, B {lims[1]}, -VTU {lims[2]}")
    for d in ds_list:
        p = d["MSLP"]
        print(f"central pressure (derived) {d['scenario'].key}: {p[0]:.0f} hPa at 0 h, min {p.min():.0f} hPa at "
              f"{d['hours'][int(np.argmin(p))]:.0f} h, {p[-1]:.0f} hPa at 168 h")
    make_png(ds_list, lims)
    make_html(ds_list, lims)
    print(f"wrote {OUT_PNG}\nwrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")
    print(f"compute {t_compute:.1f} s, total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
