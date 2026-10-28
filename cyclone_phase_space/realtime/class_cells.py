#!/usr/bin/env python3
"""Class agreement with the Florida State diagrams, cell by cell: every
combination of the thermal wind bands (standard 925-700 and 500-300 hPa,
or Hart's 900-600 and 600-300 hPa) with the three B choices (steering-flow
B on the 925-700 hPa thickness scaled to 900-600, track-motion B on the
same scaled thickness, track-motion B on the 900-600 hPa thickness), so
the effect of the bands and the effect of the B method are separated.

Reads data/validation/fsu_pairs.csv (fsu_stats.py) and, for the verified
hours, gives per cell: the joint class agreement, Cohen's kappa, the
agreement of each of the three signs that make the class (B side, lower
and upper thermal wind sign), the 7 x 7 confusion matrix, the class
frequencies on both sides, the majority-class baseline, the split of the
disagreements into B-side only, one thermal wind sign only, or more than
one, and the share of disagreements within 5 m of a boundary on either
side; and the onset and completion timing of fsu_stats.timing_stats.

    python3 class_cells.py [--pairs FILE] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fsu_stats as fs  # noqa: E402

BANDS = {"standard": ("hvtl", "hvtu"), "hart": ("hvtl_hart", "hvtu_hart")}
BS = {"steer_925": "hb", "track_925": "hb_track", "track_900": "hb_hart"}
BAND_LABEL = {"standard": "standard bands (925-700, 500-300 hPa)", "hart": "Hart's bands (900-600, 600-300 hPa)"}
B_LABEL = {"steer_925": "steering-flow B, 925-700 hPa scaled", "track_925": "track-motion B, 925-700 hPa scaled",
           "track_900": "track-motion B, 900-600 hPa"}
NEAR_M = 5.0
NAMES = ["sym deep warm", "sym shallow warm", "asym deep warm", "asym shallow warm", "asym cold", "sym cold",
         "shallow cold"]


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            rec = {"cycle": r["cycle"], "storm": r["storm"], "fhr": int(r["fhr"]), "verified": r["verified"] == "1",
                   "motion_kt": fnum(r["motion_kt"]) if fnum(r["motion_kt"]) is not None else float("nan")}
            for k in ("fsu_b", "fsu_vtl", "fsu_vtu", "hb", "hb_track", "hb_hart", "hvtl", "hvtu", "hvtl_hart", "hvtu_hart"):
                rec[k] = fnum(r.get(k))
            rows.append(rec)
    return rows


def kappa(conf: np.ndarray) -> float:
    n = conf.sum()
    if n == 0:
        return float("nan")
    po = np.trace(conf) / n
    pe = float((conf.sum(0) * conf.sum(1)).sum()) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def cell(rows: list[dict], bcol: str, vcols: tuple[str, str]) -> dict:
    vtl_c, vtu_c = vcols
    conf = np.zeros((7, 7), int)
    n = agree = bside = sl = su = 0
    dis = {"b_only": 0, "vtl_only": 0, "vtu_only": 0, "several": 0}
    near = 0
    for r in rows:
        ours = fs.hart_class(r[bcol], r[vtl_c], r[vtu_c])
        theirs = fs.hart_class(r["fsu_b"], r["fsu_vtl"], r["fsu_vtu"])
        if ours is None or theirs is None:
            continue
        n += 1
        conf[ours, theirs] += 1
        same_b = (r[bcol] > fs.B_THR) == (r["fsu_b"] > fs.B_THR)
        same_l = (r[vtl_c] >= 0) == (r["fsu_vtl"] >= 0)
        same_u = (r[vtu_c] >= 0) == (r["fsu_vtu"] >= 0)
        bside += same_b
        sl += same_l
        su += same_u
        if ours == theirs:
            agree += 1
            continue
        wrong = [not same_b, not same_l, not same_u]
        if sum(wrong) > 1:
            dis["several"] += 1
        elif not same_b:
            dis["b_only"] += 1
        elif not same_l:
            dis["vtl_only"] += 1
        elif not same_u:
            dis["vtu_only"] += 1
        else:  # class 6 merge: a sign agrees but the class rule differs through row 6
            dis["several"] += 1
        # within NEAR_M of the boundary that was crossed, on either side
        close = False
        if not same_b:
            close |= min(abs(r[bcol] - fs.B_THR), abs(r["fsu_b"] - fs.B_THR)) <= NEAR_M
        if not same_l:
            close |= min(abs(r[vtl_c]), abs(r["fsu_vtl"])) <= NEAR_M
        if not same_u:
            close |= min(abs(r[vtu_c]), abs(r["fsu_vtu"])) <= NEAR_M
        near += close
    freq_ours = conf.sum(1)
    freq_fsu = conf.sum(0)
    return {"n": int(n), "agree": int(agree), "rate": round(agree / n, 3) if n else None,
            "kappa": round(kappa(conf), 3) if n else None,
            "majority_baseline": round(float(freq_fsu.max()) / n, 3) if n else None,
            "b_side_agree": round(bside / n, 3) if n else None, "vtl_sign_agree": round(sl / n, 3) if n else None,
            "vtu_sign_agree": round(su / n, 3) if n else None,
            "disagreements": dis, "disagreements_near_boundary": int(near),
            "confusion_ours_rows_fsu_cols": conf.tolist(),
            "freq_ours": freq_ours.tolist(), "freq_fsu": freq_fsu.tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=Path, default=HERE / "data" / "validation" / "fsu_pairs.csv")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or a.pairs.parent
    rows = load(a.pairs)
    ver = [r for r in rows if r["verified"]]
    cells = {}
    for band, vcols in BANDS.items():
        for bk, bcol in BS.items():
            cells[f"{band}+{bk}"] = cell(ver, bcol, vcols)
    # timing per cell through fsu_stats.timing_stats, which reads fs.BANDS
    saved = fs.BANDS
    fs.BANDS = {f"{band}+{bk}": {"b": bcol, "vtl": vcols[0], "vtu": vcols[1]}
                for band, vcols in BANDS.items() for bk, bcol in BS.items()}
    try:
        trows = [dict(r, fsu_b=r["fsu_b"], fsu_vtl=r["fsu_vtl"]) for r in rows]  # usable rows, as fsu_stats
        timing = fs.timing_stats(trows)
    finally:
        fs.BANDS = saved
    for k, t in timing.items():
        cells[k]["timing"] = {"onset_n": t["onset_diff_h"]["n"], "onset_within_6h": t["onset_within_6h"],
                              "onset_median_h": t["onset_diff_h"]["median"],
                              "completion_n": t["completion_diff_h"]["n"],
                              "completion_within_6h": t["completion_within_6h"],
                              "completion_median_h": t["completion_diff_h"]["median"]}
    S = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "pairs": str(a.pairs),
         "verified_rows": len(ver), "near_boundary_m": NEAR_M, "cells": cells}
    (out / "class_cells.json").write_text(json.dumps(S, indent=1) + "\n")
    L = ["# Class agreement with the FSU diagrams, bands by B method", "",
         f"Generated {S['generated']} by `realtime/class_cells.py` from `{a.pairs.name}` ({len(ver)} verified hours; "
         "timing on the usable hours as in fsu_stats.py). Joint class by the D2D rule; kappa is Cohen's; the "
         "majority baseline is the share of FSU's most common class; the sign columns are the agreement of each "
         f"of the three signs that make the class; near is the share of disagreements within {NEAR_M:g} m of the "
         "boundary crossed, on either side.", "",
         "| Bands | B | n | class agree | kappa | majority | B side | VTL sign | VTU sign | B only | VTL only | VTU only | several | near | onset within 6 h (n, median) | completion within 6 h (n, median) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for band in BANDS:
        for bk in BS:
            c = cells[f"{band}+{bk}"]
            d = c["disagreements"]
            nd = max(1, c["n"] - c["agree"])
            t = c["timing"]
            L.append(f"| {BAND_LABEL[band]} | {B_LABEL[bk]} | {c['n']} | {c['rate']:.0%} | {c['kappa']:.2f} | "
                     f"{c['majority_baseline']:.0%} | {c['b_side_agree']:.0%} | {c['vtl_sign_agree']:.0%} | "
                     f"{c['vtu_sign_agree']:.0%} | {d['b_only']} | {d['vtl_only']} | {d['vtu_only']} | {d['several']} | "
                     f"{c['disagreements_near_boundary'] / nd:.0%} | {t['onset_within_6h']} of {t['onset_n']}, "
                     f"{t['onset_median_h']:+.0f} h | {t['completion_within_6h']} of {t['completion_n']}, "
                     f"{t['completion_median_h']:+.0f} h |")
    for key in ("hart+track_900", "standard+steer_925"):
        c = cells[key]
        L += ["", f"## Confusion matrix, {key} (rows ours, columns FSU)", "",
              "| ours \\ FSU | " + " | ".join(f"{k} {NAMES[k]}" for k in range(7)) + " | total |",
              "|---|" + "---|" * 8]
        for i in range(7):
            L.append(f"| {i} {NAMES[i]} | " + " | ".join(str(v) for v in c["confusion_ours_rows_fsu_cols"][i])
                     + f" | {c['freq_ours'][i]} |")
        L.append("| total | " + " | ".join(str(v) for v in c["freq_fsu"]) + f" | {c['n']} |")
    (out / "class_cells.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
