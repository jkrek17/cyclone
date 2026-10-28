"""class_cells: the joint class agreement per band and B choice, with kappa,
the three sign agreements and the split of the disagreements."""

from __future__ import annotations

import numpy as np
import pytest

import class_cells as cc


def _row(b, vtl, vtu, fb, fvtl, fvtu):
    return {"hb": b, "hvtl": vtl, "hvtu": vtu, "fsu_b": fb, "fsu_vtl": fvtl, "fsu_vtu": fvtu}


def test_cell_counts_agreement_signs_and_the_split():
    rows = [
        _row(5, 50, 50, 5, 60, 40),      # both 0: agree
        _row(15, 50, 50, 5, 60, 40),     # ours 2, FSU 0: B only, within 5 m of 10 on our side
        _row(5, -20, -50, 5, 30, -50),   # ours 5, FSU 1: VTL only, not near
        _row(30, 50, -3, 30, 50, 3),     # ours 3, FSU 2: VTU only, near
        _row(30, -20, -50, 2, 30, -50),  # ours 4, FSU 1: B and VTL: several, not near
    ]
    c = cc.cell(rows, "hb", ("hvtl", "hvtu"))
    assert c["n"] == 5 and c["agree"] == 1 and c["rate"] == pytest.approx(0.2)
    assert c["disagreements"] == {"b_only": 1, "vtl_only": 1, "vtu_only": 1, "several": 1}
    assert c["disagreements_near_boundary"] == 2
    assert c["b_side_agree"] == pytest.approx(3 / 5)
    assert c["vtl_sign_agree"] == pytest.approx(3 / 5)
    assert c["vtu_sign_agree"] == pytest.approx(4 / 5)
    conf = np.array(c["confusion_ours_rows_fsu_cols"])
    assert conf.sum() == 5 and conf[0, 0] == 1 and conf[2, 0] == 1 and conf[5, 1] == 1 and conf[3, 2] == 1 and conf[4, 1] == 1
    assert c["freq_fsu"][0] == 2 and c["majority_baseline"] == pytest.approx(2 / 5)


def test_kappa_is_one_for_perfect_agreement_and_zero_at_chance():
    assert cc.kappa(np.diag([3, 4, 5])) == pytest.approx(1.0)
    chance = np.array([[1, 1], [1, 1]])
    assert cc.kappa(chance) == pytest.approx(0.0)


def test_missing_values_are_skipped():
    rows = [_row(None, 50, 50, 5, 60, 40), _row(5, 50, 50, None, 60, 40), _row(5, 50, 50, 5, 60, 40)]
    c = cc.cell(rows, "hb", ("hvtl", "hvtu"))
    assert c["n"] == 1 and c["agree"] == 1
