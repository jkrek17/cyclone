"""One real storm-cycle on Hart's two diagrams: our trajectory against the
Florida State page's digitized values, for the western Pacific typhoon of
the 00 UTC 2 October 2026 GFS run (the archive's `2026100200/WPAC`, FSU
cyclone 7).

Our side is the storm's `track.csv` from `realtime/data/<cycle>/<storm>/`:
Hart's layers (`hvtl_hart`, `hvtu_hart`) and the track-motion B on the
900-600 hPa thickness (`hb_hart`), smoothed with the 24 h centered running
mean of `track_cps.running_mean`, the smoothing the Florida State page
applies to its own trajectory. The Florida State side is the verified rows
of `realtime/data/validation/fsu_pairs.csv` for the same storm-cycle
(`fsu_b`, `fsu_vtl`, `fsu_vtu` by forecast hour), the hours the digitizer
timed independently on both of the page's diagrams with the same lower
thermal wind.

The figure follows Hart's conventions: marker size grows with the central
pressure depth (`mslp_hpa`, the GFS minimum at the fix; the Florida State
markers, hollow, use our MSLP at the same hour so the two sides are on one
scale), the hour is labeled every 12 h on our trajectory, the quadrants are
shaded in the house style of `diagram_style.py` with the 10 m onset line
and the zero lines, and a third panel shows the three parameters against
forecast hour for both sides under a class strip for each.

Outputs (in this directory):

    figM_fsu_case.png              the figure, 300 dpi

The Florida State page's own diagrams for the storm-cycle are in the
archive folder (fsu_phase1.png, fsu_phase2.png) for comparison.

Run from this directory:

    python3 fsu_case.py [cycle storm]

It prints the verified hours, the onset and completion hours of both sides
(`track_cps.onset_completion`: the first hour B exceeds 10 m, then the
first hour at or after it that the lower term falls below zero), the
per-hour values and the mean and RMS difference (ours minus FSU) of each
parameter over the verified hours.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "realtime"))
sys.path.insert(0, str(HERE))

import track_cps as tc  # noqa: E402  (selects the Agg backend through gfs_cps)
import diagram_style as ds  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CYCLE, STORM = "2026100200", "WPAC"
DATA = PKG / "realtime" / "data"
OUT_PNG = HERE / "figM_fsu_case.png"

B_THR = float(tc.B_THR)
CLASS_PALETTE = tc.CLASS_PALETTE
PATH_EDGE = "#3a3a37"
FSU_EDGE = "#0b0b0b"
PURPLE, RED, BLUE = "#8e44ad", "#d62d2d", "#2a78d6"
HOUR_STEP, HOUR_MAJOR = 12.0, 24.0

# Marker diameter (points): a floor plus so much per hPa of depth below
# P_ENV_HPA, deeper is bigger, as on Hart's diagrams.
P_ENV_HPA = 1013.0
MARKER_D0_PT = 2.5
MARKER_K_PT = 0.11
SIZE_LEGEND_HPA = (990.0, 970.0, 950.0, 930.0)


# ------------------------------------------------------------------ data
def load_track(cycle, storm):
    """Our side: hours, valid times, the raw and 24 h smoothed Hart-layer
    series, the MSLP, the class of the smoothed triple (blank where the
    archive's class is blank, that is, outside a closed low) and the onset
    and completion hours of the smoothed series."""
    fixes = tc.read_fixes(DATA / cycle / storm / "track.csv")
    hours = np.array([f["fhr"] for f in fixes], dtype=float)
    col = {k: np.array([f[k] for f in fixes], dtype=float)
           for k in ("hb_hart", "hvtl_hart", "hvtu_hart", "mslp_hpa", "class")}
    sm = {k: tc.running_mean(hours, col[k]) for k in ("hb_hart", "hvtl_hart", "hvtu_hart")}
    cls = np.array([tc.class_hart_value(b, l, u, bool(np.isfinite(c)))
                    for b, l, u, c in zip(sm["hb_hart"], sm["hvtl_hart"], sm["hvtu_hart"], col["class"])])
    on, comp = tc.onset_completion(hours, sm["hb_hart"], sm["hvtl_hart"])
    return dict(hours=hours, valid=[f["valid"] for f in fixes], raw=col, B=sm["hb_hart"], VTL=sm["hvtl_hart"],
                VTU=sm["hvtu_hart"], MSLP=col["mslp_hpa"], CLS=cls, onset=on, completion=comp)


def load_fsu(cycle, storm):
    """The Florida State side: the verified rows of fsu_pairs.csv for the
    storm-cycle, by forecast hour, with the class of the digitized triple
    and the onset and completion hours over those rows."""
    rows = []
    with (DATA / "validation" / "fsu_pairs.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["cycle"] == cycle and r["storm"] == storm and r["verified"] == "1":
                rows.append(r)
    rows.sort(key=lambda r: int(r["fhr"]))
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])  # noqa: E731
    hours = f("fhr")
    b, vtl, vtu = f("fsu_b"), f("fsu_vtl"), f("fsu_vtu")
    cls = np.array([tc.class_hart_value(x, y, z, True) for x, y, z in zip(b, vtl, vtu)])
    on, comp = tc.onset_completion(hours, b, vtl)
    ours = dict(B=f("hb_hart"), VTL=f("hvtl_hart"), VTU=f("hvtu_hart"))  # the pairs file's own smoothed values
    return dict(hours=hours, B=b, VTL=vtl, VTU=vtu, CLS=cls, onset=on, completion=comp, number=rows[0]["fsu_number"],
                ours_in_pairs=ours)


def marker_diameter(mslp):
    depth = np.clip(P_ENV_HPA - np.asarray(mslp, dtype=float), 0.0, None)
    return MARKER_D0_PT + MARKER_K_PT * depth


# --------------------------------------------------------------- labels
def _halo(width=2.2):
    return patheffects.withStroke(linewidth=width, foreground="white")


LABEL_ANGLES_ABOVE = [45, 135, -45, -135, 90, -90, 0, 180]
LABEL_ANGLES_BELOW = [-45, -135, 45, 135, -90, 90, 0, 180]


def _label_box(anchor, off, text, fontsize):
    dx, dy, ha = off
    w, h = 0.58 * fontsize * len(text), 1.1 * fontsize
    x = anchor[0] + dx
    x0 = x if ha == "left" else (x - w if ha == "right" else x - w / 2)
    y0 = anchor[1] + dy if dy > 0 else anchor[1] + dy - h
    return np.array([x0, y0, x0 + w, y0 + h])


def _choose_label(anchor, text, fontsize, dist, avoid_pts, boxes, angles):
    """(offset, box, clash): of eight directions at `dist` points from the
    anchor, the one whose label box covers the fewest of `avoid_pts` (the
    paths and markers, in points) and overlaps no box already placed."""
    best = None
    for k, ang in enumerate(np.radians(angles)):
        ux, uy = np.cos(ang), np.sin(ang)
        off = (dist * ux, dist * uy if abs(uy) > 0.1 else 0.1 * dist,
               "left" if ux > 0.3 else ("right" if ux < -0.3 else "center"))
        box = _label_box(anchor, off, text, fontsize)
        pad = 1.0
        inside = np.sum((avoid_pts[:, 0] > box[0] - pad) & (avoid_pts[:, 0] < box[2] + pad)
                        & (avoid_pts[:, 1] > box[1] - pad) & (avoid_pts[:, 1] < box[3] + pad))
        clash = any(box[0] < b[2] + 2 and b[0] < box[2] + 2 and box[1] < b[3] + 2 and b[1] < box[3] + 2
                    for b in boxes)
        score = (1000 if clash else 0) + inside + 0.01 * k
        if best is None or score < best[0]:
            best = (score, off, box, clash)
    return best[1], best[2], best[3]


def hour_labels(ax, hours, x, y, diam, extra_xy=()):
    """Hour labels every 12 h along (x, y): the ends bold with "h", the 24 h
    ones above the path by preference and the 12 h ones between them
    below, in a smaller gray type; a label is left out where every
    direction would overlap one already placed. `extra_xy` are other
    points (the FSU markers) the labels keep off."""
    fig = ax.figure
    to_pt = 72.0 / fig.dpi
    ok = np.isfinite(x) & np.isfinite(y)
    screen = ax.transData.transform(np.column_stack([x, y])) * to_pt
    pts = np.column_stack([x, y])[ok]
    dense = np.concatenate([np.linspace(a, b, 8, endpoint=False) for a, b in zip(pts[:-1], pts[1:])] + [pts[-1:]])
    avoid = ax.transData.transform(dense) * to_pt
    if len(extra_xy):
        avoid = np.vstack([avoid, ax.transData.transform(np.asarray(extra_xy)) * to_pt])
    n = len(hours) - 1
    order = ([0, n] + [i for i, h in enumerate(hours) if h % HOUR_MAJOR == 0 and 0 < i < n]
             + [i for i, h in enumerate(hours) if h % HOUR_STEP == 0 and h % HOUR_MAJOR != 0 and 0 < i < n])
    boxes, placed = [], 0
    for i in order:
        if not ok[i]:
            continue
        bold, minor = i in (0, n), hours[i] % HOUR_MAJOR != 0
        text = f"{hours[i]:.0f}" + (" h" if bold else "")
        fs = 7.4 if bold else (5.8 if minor else 6.6)
        off, box, clash = _choose_label(screen[i], text, fs, diam[i] / 2 + (4.5 if bold else 2.5), avoid, boxes,
                                        LABEL_ANGLES_BELOW if minor else LABEL_ANGLES_ABOVE)
        if clash:
            continue
        boxes.append(box)
        placed += 1
        dx, dy, ha = off
        ax.annotate(text, (x[i], y[i]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="bottom" if dy > 0 else "top", fontsize=fs, fontweight="bold" if bold else "normal",
                    color=ds.TEXT_SECONDARY if minor else ds.TEXT_DARK, zorder=12, path_effects=[_halo()])
    return placed


# --------------------------------------------------------------- figure
def draw_diagram(ax, ours, fsu, vert, xlim, ylim):
    """One of Hart's diagrams: `vert` is "B" or "VTU" against -V_T^L. Our
    trajectory as a dark line with class-colored squares sized by MSLP and
    hour labels; the FSU values as hollow circles at the same hours and
    sizes, joined by a dashed line where consecutive verified hours are
    6 h apart."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if vert == "B":
        ds.draw_b_vtl_quadrants(ax, xlim, ylim, B_THR, fontsize=6.8)
        ax.axhline(B_THR, color=ds.TEXT_DARK, lw=1.0, zorder=2)
    else:
        # the deep warm core label moves to its quadrant's inner corner: this
        # storm's first four days sit where diagram_style's default puts it
        ds.draw_vtu_vtl_quadrants(ax, xlim, ylim, fontsize=6.8,
                                  label_overrides={"deep warm core": (xlim[1] * 0.03, ylim[1] * 0.07)})
        ax.axhline(0.0, color=ds.TEXT_DARK, lw=1.0, zorder=2)
    ax.axvline(0.0, color=ds.TEXT_DARK, lw=1.0, zorder=2)
    ax.grid(True, color=ds.GRID_COLOR, lw=0.4, zorder=1)
    ax.set_axisbelow(True)

    x, y, h = ours["VTL"], ours[vert], ours["hours"]
    diam = marker_diameter(ours["MSLP"])
    ok = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[ok], y[ok], "-", color=PATH_EDGE, lw=1.5, zorder=5, solid_capstyle="round")
    cols = [CLASS_PALETTE[int(c)] if np.isfinite(c) else "white" for c in ours["CLS"]]
    ax.scatter(x, y, s=diam ** 2, marker="s", c=cols, edgecolor=PATH_EDGE, linewidth=0.5, zorder=6)
    for hour, marker in ((ours["onset"], "x"), (ours["completion"], "x")):
        if np.isfinite(hour):
            i = int(np.argmin(np.abs(h - hour)))
            ax.scatter([x[i]], [y[i]], marker=marker, s=40, color=ds.TEXT_DARK, linewidth=1.4, zorder=9)
    n = len(h) - 1
    for i, marker in ((0, "o"), (n, "s")):
        ax.scatter([x[i]], [y[i]], s=(diam[i] + 4.5) ** 2, marker=marker, facecolor="none",
                   edgecolor=ds.TEXT_DARK, linewidth=1.1, zorder=8)

    # FSU: hollow circles sized with our MSLP at the same hour
    fh = fsu["hours"]
    idx = np.array([int(np.argmin(np.abs(h - t))) for t in fh])
    fd = diam[idx]
    fx, fy = fsu["VTL"], fsu[vert]
    for a, b in zip(range(len(fh) - 1), range(1, len(fh))):
        if fh[b] - fh[a] <= 6.0 + 1e-6:
            ax.plot(fx[[a, b]], fy[[a, b]], "--", color=FSU_EDGE, lw=0.9, zorder=7)
    ax.scatter(fx, fy, s=fd ** 2, marker="o", facecolor="none", edgecolor=FSU_EDGE, linewidth=1.1, zorder=10)
    for hour in (fsu["onset"], fsu["completion"]):
        if np.isfinite(hour):
            i = int(np.argmin(np.abs(fh - hour)))
            ax.scatter([fx[i]], [fy[i]], marker="+", s=46, color=FSU_EDGE, linewidth=1.3, zorder=11)

    ax.set_xlabel(ds.AXIS_LABEL_VTL)
    ax.set_ylabel(ds.AXIS_LABEL_B if vert == "B" else ds.AXIS_LABEL_VTU)
    return hour_labels(ax, h, x, y, diam, extra_xy=np.column_stack([fx, fy]))


def draw_series(ax, ours, fsu):
    """Panel (c): the three parameters against forecast hour, ours as lines
    and the FSU values as hollow markers (joined where consecutive), the
    MSLP as a thin gray line on a right axis, and the class strips of both
    sides above."""
    h = ours["hours"]
    series = (("B", PURPLE, "o", "B"), ("VTL", RED, "s", r"$-V_T^L$"), ("VTU", BLUE, "^", r"$-V_T^U$"))
    for key, color, marker, label in series:
        ax.plot(h, ours[key], "-", color=color, lw=1.5, zorder=5, label=f"{label}, ours")
        fh = fsu["hours"]
        for a, b in zip(range(len(fh) - 1), range(1, len(fh))):
            if fh[b] - fh[a] <= 6.0 + 1e-6:
                ax.plot(fh[[a, b]], fsu[key][[a, b]], "--", color=color, lw=0.9, zorder=6)
        ax.scatter(fh, fsu[key], s=22, marker=marker, facecolor="white", edgecolor=color, linewidth=1.0,
                   zorder=7, label=f"{label}, FSU")
    ax.axhline(0.0, color=ds.TEXT_DARK, lw=0.8, zorder=3)
    ax.axhline(B_THR, color=PURPLE, lw=0.7, ls=":", zorder=3)
    ax.text(h[-1], B_THR, " 10 m", color=PURPLE, fontsize=6.5, va="bottom", ha="right", zorder=4)
    ax.set_xlim(h[0] - 3.0, h[-1] + 3.0)
    ax.set_xlabel("forecast hour")
    ax.set_ylabel("m")
    ax.grid(True, color=ds.GRID_COLOR, lw=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(0, h[-1] + 1, 24))

    ax2 = ax.twinx()
    ax2.plot(h, ours["MSLP"], "-", color="#8f8e89", lw=0.9, zorder=2)
    ax2.set_ylabel("MSLP (hPa), ours", color="#52514e", fontsize=8)
    ax2.tick_params(axis="y", colors="#52514e", labelsize=7)
    lo, hi = np.nanmin(ours["MSLP"]), np.nanmax(ours["MSLP"])
    ax2.set_ylim(lo - 0.35 * (hi - lo), hi + 2.2 * (hi - lo))  # the pressure keeps to the lower part

    # class strips: ours over the FSU's (verified hours only), then the
    # onset and completion of each side as labeled ticks between them
    s_ours = tc.class_strip(ax, h, ours["CLS"], [0.0, 1.145, 1.0, 0.05], "ours ")
    s_ours.set_ylabel("ours", rotation=0, ha="right", va="center", fontsize=7.5)
    s_fsu = tc.class_strip(ax, fsu["hours"], fsu["CLS"], [0.0, 1.075, 1.0, 0.05], "FSU ")
    s_fsu.set_ylabel("FSU", rotation=0, ha="right", va="center", fontsize=7.5)
    for strip in (s_ours, s_fsu):
        for sp in strip.spines.values():
            sp.set_edgecolor(ds.GRID_COLOR)
    # our events labeled above our strip, the FSU's below its strip, the
    # onset to the left of its tick and the completion to the right
    for side, color, yfrac in (("ours", ds.TEXT_DARK, 1.0), ("FSU", FSU_EDGE, 0.0)):
        src = ours if side == "ours" else fsu
        strip = s_ours if side == "ours" else s_fsu
        for kind, hour in (("onset", src["onset"]), ("completion", src["completion"])):
            if np.isfinite(hour):
                strip.plot([hour, hour], [0, 1], color=color, lw=1.6, zorder=5)
                left = kind == "onset"
                strip.annotate(f"{kind} {hour:.0f} h", (hour, yfrac), xytext=(-2 if left else 2, 2 if yfrac else -2),
                               textcoords="offset points", ha="right" if left else "left",
                               va="bottom" if yfrac else "top", fontsize=6.3, color=color, zorder=6,
                               path_effects=[_halo(2.0)])
    return ax2


def make_figure(ours, fsu, cycle, storm):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "text.color": ds.TEXT_DARK,
        "axes.labelcolor": ds.TEXT_DARK, "xtick.color": ds.TEXT_SECONDARY, "ytick.color": ds.TEXT_SECONDARY,
        "axes.linewidth": 0.6, "legend.fontsize": 7.2, "axes.titlesize": 9, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.labelsize": 8.5,
    })
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.28], left=0.045, right=0.945, bottom=0.175, top=0.775,
                          wspace=0.27)
    axa, axb, axc = (fig.add_subplot(gs[0, k]) for k in range(3))

    xlim = tc.limits(ds.FIG10_VTL_LIM, ours["VTL"], fsu["VTL"])
    # B's lower limit is opened from the house -20 m to -30 m: the typhoon's
    # first four days run along B = 0 at the diagram's foot, where the hour
    # labels and the quadrant label need the room
    ylim_b = tc.limits((-30.0, ds.FIG10_B_LIM[1]), ours["B"], fsu["B"])
    ylim_u = tc.limits(ds.FIG10_VTU_LIM, ours["VTU"], fsu["VTU"])
    na = draw_diagram(axa, ours, fsu, "B", xlim, ylim_b)
    nb = draw_diagram(axb, ours, fsu, "VTU", xlim, ylim_u)
    draw_series(axc, ours, fsu)
    axa.set_title("(a) B against $-V_T^L$", loc="left")
    axb.set_title("(b) $-V_T^U$ against $-V_T^L$", loc="left")
    axc.set_title("(c) the three parameters by forecast hour", loc="left", pad=58)

    valid0 = ours["valid"][0]
    fig.suptitle(f"GFS {valid0:%H} UTC {valid0:%-d %B %Y} run, western Pacific typhoon (archive {cycle}/{storm}, "
                 f"Florida State cyclone {fsu['number']}): ours against the Florida State diagram",
                 x=0.045, y=0.985, ha="left", fontsize=10, fontweight="bold")

    # legends: the two sides and the marks, then the marker size scale
    side_handles = [
        Line2D([0], [0], marker="s", color=PATH_EDGE, markerfacecolor=CLASS_PALETTE[0], markeredgecolor=PATH_EDGE,
               lw=1.5, markersize=6, label="ours: Hart's layers, track-motion B, 24 h mean (square colored by class)"),
        Line2D([0], [0], marker="o", color=FSU_EDGE, markerfacecolor="none", markeredgecolor=FSU_EDGE, lw=0.9,
               ls="--", markersize=6, label="Florida State page, digitized (verified hours; hollow)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=ds.TEXT_DARK,
               markersize=7, markeredgewidth=1.1, label="start (0 h), hours every 12 h"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none", markeredgecolor=ds.TEXT_DARK,
               markersize=6.5, markeredgewidth=1.1, label=f"end ({ours['hours'][-1]:.0f} h)"),
        Line2D([0], [0], marker="x", color=ds.TEXT_DARK, lw=0, markersize=6, markeredgewidth=1.4,
               label="onset and completion, ours (x) and FSU (+)"),
    ]
    fig.legend(handles=side_handles, loc="upper left", bbox_to_anchor=(0.045, 0.952), ncol=2, frameon=False,
               handlelength=2.0, columnspacing=1.6, fontsize=7.2)
    size_handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=CLASS_PALETTE[0],
                           markeredgecolor=PATH_EDGE, markeredgewidth=0.5, markersize=float(marker_diameter(p)),
                           label=f"{p:.0f}") for p in SIZE_LEGEND_HPA]
    fig.legend(handles=size_handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=len(size_handles),
               frameon=False, handlelength=1.6, columnspacing=1.0, fontsize=7.2,
               title="Marker size: MSLP at the fix (hPa), deeper is bigger; the Florida State markers take ours "
                     "at the same hour", title_fontsize=7.4)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    plt.close(fig)
    return na, nb


# ----------------------------------------------------------------- main
def report(ours, fsu):
    h = ours["hours"]
    idx = np.array([int(np.argmin(np.abs(h - t))) for t in fsu["hours"]])
    print(f"verified hours ({len(idx)}): " + ", ".join(f"{t:.0f}" for t in fsu["hours"]))
    chk = max(float(np.nanmax(np.abs(ours[k][idx] - fsu["ours_in_pairs"][k]))) for k in ("B", "VTL", "VTU"))
    print(f"our smoothed values against the pairs file's: largest difference {chk:.2f} m")
    print(f"ours: onset {ours['onset']:.0f} h, completion {ours['completion']:.0f} h; "
          f"FSU: onset {fsu['onset']:.0f} h, completion {fsu['completion']:.0f} h")
    print(f"ours: MSLP {ours['MSLP'][0]:.1f} hPa at 0 h, minimum {np.nanmin(ours['MSLP']):.1f} hPa at "
          f"{h[int(np.nanargmin(ours['MSLP']))]:.0f} h, {ours['MSLP'][-1]:.1f} hPa at {h[-1]:.0f} h")
    print(f"{'fhr':>4} {'B ours':>7} {'B fsu':>7} {'VTL ours':>9} {'VTL fsu':>8} {'VTU ours':>9} {'VTU fsu':>8} "
          f"{'mslp':>6} {'cls ours':>8} {'cls fsu':>7}")
    for j, i in enumerate(idx):
        print(f"{h[i]:4.0f} {ours['B'][i]:7.1f} {fsu['B'][j]:7.1f} {ours['VTL'][i]:9.1f} {fsu['VTL'][j]:8.1f} "
              f"{ours['VTU'][i]:9.1f} {fsu['VTU'][j]:8.1f} {ours['MSLP'][i]:6.1f} {ours['CLS'][i]:8.0f} "
              f"{fsu['CLS'][j]:7.0f}")
    for k in ("B", "VTL", "VTU"):
        d = ours[k][idx] - fsu[k]
        d = d[np.isfinite(d)]
        print(f"{k}: ours minus FSU over verified hours: mean {d.mean():+.1f} m, RMS {np.sqrt((d ** 2).mean()):.1f} m, "
              f"largest {d[np.argmax(np.abs(d))]:+.1f} m")
    same = int(np.sum(ours["CLS"][idx] == fsu["CLS"]))
    print(f"class agreement on verified hours: {same} of {len(idx)}")


def main(cycle=CYCLE, storm=STORM):
    ours = load_track(cycle, storm)
    fsu = load_fsu(cycle, storm)
    report(ours, fsu)
    na, nb = make_figure(ours, fsu, cycle, storm)
    print(f"hour labels placed: {na} on (a), {nb} on (b) of {int(sum(1 for t in ours['hours'] if t % HOUR_STEP == 0))}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
