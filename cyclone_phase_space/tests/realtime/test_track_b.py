"""export_web's track-motion B: hart_class_point and track_motion_b (see
export_web.py, the LAYER_SCALE block and build_storms)."""

from __future__ import annotations

import math

import pytest

import export_web as ew


def test_class_rule_matches_the_d2d_table():
    f = ew.hart_class_point
    assert f(0.0, 50.0, 50.0) == 0 and f(30.0, 50.0, 50.0) == 2
    assert f(0.0, 50.0, -50.0) == 1 and f(30.0, 50.0, -50.0) == 3
    assert f(30.0, -50.0, -50.0) == 4 and f(0.0, -50.0, -50.0) == 5
    assert f(0.0, -50.0, 50.0) == 6 and f(30.0, -50.0, 50.0) == 6
    # ties: B exactly at the threshold is symmetric, a term at zero is warm
    assert f(10.0, 0.0, 0.0) == 0 and f(10.01, 0.0, 0.0) == 2
    assert f(None, 1.0, 1.0) is None and f(float("nan"), 1.0, 1.0) is None


def _pts(positions, bq):
    return [{"fhr": 6 * k, "lat": la, "lon": lo, "hvtl": 50.0, "hvtu": -50.0, "bq": list(bq)}
            for k, (la, lo) in enumerate(positions)]


def test_northward_motion_reads_east_minus_west():
    pts = _pts([(40.0, 0.0), (41.0, 0.0), (42.0, 0.0)], bq=(0.0, 0.0, 30.0, 10.0))  # N S E W
    ew.track_motion_b(pts)
    for p in pts:
        assert p["hbt"] == pytest.approx(20.0)
        assert p["hdg"] == pytest.approx(0.0) and p["spd"] > 0
        assert p["clst"] == 3  # asymmetric shallow warm core


def test_eastward_motion_reads_south_minus_north_and_flips_in_the_south():
    north = _pts([(40.0, 0.0), (40.0, 1.0), (40.0, 2.0)], bq=(5.0, 25.0, 0.0, 0.0))
    ew.track_motion_b(north)
    assert north[1]["hbt"] == pytest.approx(20.0) and north[1]["hdg"] == pytest.approx(90.0)
    south = _pts([(-40.0, 0.0), (-40.0, 1.0), (-40.0, 2.0)], bq=(5.0, 25.0, 0.0, 0.0))
    ew.track_motion_b(south)
    assert south[1]["hbt"] == pytest.approx(-20.0)


def test_single_point_or_missing_means_gives_none():
    one = _pts([(40.0, 0.0)], bq=(0.0, 0.0, 1.0, 0.0))
    ew.track_motion_b(one)
    assert one[0]["hbt"] is None and one[0]["clst"] is None
    two = _pts([(40.0, 0.0), (41.0, 0.0)], bq=(0.0, 0.0, 1.0, 0.0))
    two[1].pop("bq")
    ew.track_motion_b(two)
    assert two[0]["hbt"] is not None and two[1]["hbt"] is None and two[1]["spd"] is not None


def test_speed_matches_the_distance():
    pts = _pts([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)], bq=(0.0, 0.0, 0.0, 0.0))
    ew.track_motion_b(pts)
    km = 111.195 * 2  # centered over two degrees of longitude at the equator in 12 h
    assert pts[1]["spd"] == pytest.approx(km / 12.0 / 1.852, abs=0.2)


# ------------------------------------------------------------------ hysteresis class and margin
# (export_web.hysteresis_class, class_margin; see the HYST_* constants)


def _track(bs, vtl=50.0, vtu=-50.0):
    """A linked track with Hart's B from the motion already set: hbt per
    point, the raw class clst from it, the thermal winds fixed."""
    pts = [{"fhr": 6 * k, "hbt": b, "hvtl": vtl, "hvtu": vtu} for k, b in enumerate(bs)]
    for p in pts:
        p["clst"] = ew.hart_class_point(p["hbt"], p["hvtl"], p["hvtu"])
    return pts


def test_b_oscillating_across_10m_flickers_raw_but_not_hysteresis():
    bs = [8.0, 12.0, 9.0, 11.0, 8.5, 12.5, 9.5]  # every step crosses B = 10 m
    pts = _track(bs)
    ew.hysteresis_class(pts)
    raw = [p["clst"] for p in pts]
    assert raw == [1, 3, 1, 3, 1, 3, 1]  # the raw class alternates
    assert [p["clsh"] for p in pts] == [1] * len(pts)  # the hysteresis class does not


def test_b_must_pass_15m_to_become_asymmetric_and_5m_to_return():
    pts = _track([8.0, 14.9, 15.1, 9.0, 5.1, 4.9, 12.0])
    ew.hysteresis_class(pts)
    assert [p["clsh"] for p in pts] == [1, 1, 3, 3, 3, 1, 1]


def test_starts_from_the_first_raw_class_even_when_asymmetric():
    pts = _track([12.0, 11.0, 8.0, 4.0])
    ew.hysteresis_class(pts)
    assert pts[0]["clsh"] == 3 and [p["clsh"] for p in pts] == [3, 3, 3, 1]


def test_thermal_wind_term_must_pass_10m_beyond_zero():
    pts = _track([0.0] * 6)
    for p, vtl in zip(pts, [30.0, -5.0, -9.9, -10.1, 5.0, 10.1]):
        p["hvtl"] = vtl
        p["clst"] = ew.hart_class_point(p["hbt"], p["hvtl"], p["hvtu"])
    assert [p["clst"] for p in pts] == [1, 5, 5, 5, 1, 1]  # raw: warm at and above zero
    ew.hysteresis_class(pts)
    assert [p["clsh"] for p in pts] == [1, 1, 1, 5, 5, 1]  # cold only below -10, warm only above +10


def test_class_6_rule_unchanged_and_its_asymmetry_read_from_b():
    # cold lower under a warm upper layer is class 6 whatever B; once the
    # lower layer turns warm the asymmetry reading, kept from B, applies
    pts = _track([20.0, 20.0, 20.0], vtl=-50.0, vtu=50.0)
    pts[2]["hvtl"] = 50.0
    pts[2]["clst"] = ew.hart_class_point(20.0, 50.0, 50.0)
    ew.hysteresis_class(pts)
    assert [p["clsh"] for p in pts] == [6, 6, 2]


def test_missing_raw_class_gives_none_and_resumes():
    pts = _track([8.0, None, 12.0, 11.0])
    assert pts[1]["clst"] is None
    ew.hysteresis_class(pts)
    assert [p["clsh"] for p in pts] == [1, None, 1, 1]
    # a track with no raw class at all
    none = _track([None, None])
    ew.hysteresis_class(none)
    assert [p["clsh"] for p in none] == [None, None]
    # the first classed point seeds the state, wherever it is
    late = _track([None, 12.0, 9.0])
    ew.hysteresis_class(late)
    assert [p["clsh"] for p in late] == [None, 3, 3]


def test_margin_is_the_nearest_wall_and_b_does_not_enter_class_6():
    m = ew.class_margin
    assert m(1, 8.0, 50.0, -50.0) == pytest.approx(2.0)      # |B - 10|
    assert m(1, 30.0, 7.5, -50.0) == pytest.approx(7.5)      # |VTL|
    assert m(0, 30.0, 50.0, 3.0) == pytest.approx(3.0)       # |VTU|
    assert m(6, 10.2, -40.0, 60.0) == pytest.approx(40.0)    # class 6: B is 0.2 m from the wall but does not count
    assert m(5, 10.0, -50.0, -50.0) == pytest.approx(0.0)    # on the wall
    assert m(2, 12.34, 50.0, 50.0) == pytest.approx(2.3)     # one decimal


def test_margin_none_when_a_value_is_missing():
    m = ew.class_margin
    assert m(None, 8.0, 50.0, -50.0) is None
    assert m(1, None, 50.0, -50.0) is None
    assert m(1, 8.0, None, -50.0) is None
    assert m(1, 8.0, 50.0, float("nan")) is None
    assert m(6, None, -40.0, 60.0) == pytest.approx(40.0)    # class 6 needs no B
