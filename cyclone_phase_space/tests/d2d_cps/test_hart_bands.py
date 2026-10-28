"""Tests for the seven-level (Hart's own bands, GFS) and two-slope-mean
(standard levels) entry points of cps_HartCPS.py, and for the XML
definitions that select between them, the two-slope mean, and the
925/850/700 fallback per model.

Expectations are derived from the specification, not copied from the
implementation: a synthetic vortex whose height-perturbation amplitude
is a prescribed function of ln p has a known band slope at its center
(the Gaussian bump's max-minus-min over the 500 km disk is the
amplitude itself to within exp(-(500/150)**2), about 1.5e-5), the
two-slope mean is by definition the mean of the two three-level slopes,
and executeBLayer on a 900-600 hPa thickness with layerScale 1 must
equal the semicircle difference executeB computes on the same thickness
array. The XML test parses every definition and checks the method order,
that every referenced function exists with the right arity, and that the
level Fields and pressure constants match.

See tests/d2d_cps/conftest.py for the sys.path setup (`import cps_HartCPS`
as a bare module, `import synthetic` from tests/cps).
"""

from __future__ import annotations

import glob
import inspect
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import cps_HartCPS as hc
import synthetic

EARTH_RADIUS_KM = 6371.0

DEFINITIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "D2D", "derivedParameters", "definitions")
)

#: Every level a test vortex is built on: 1000 hPa (for the MSLP field the
#: closed-low mask reads), the six standard levels, and Hart's 50 hPa set.
ALL_LEVELS = (1000.0, 925.0) + hc.HART_LOWER_BAND + hc.HART_UPPER_BAND[1:]
HART_13 = hc.HART_LOWER_BAND + hc.HART_UPPER_BAND[1:]  # 600 hPa once


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_and_dx_dy(clat, clon, half_width_deg, dlat):
    lat2d, lon2d = synthetic.make_grid(clat, clon, half_width_deg=half_width_deg, dlat=dlat, dlon=dlat)
    lat_vals = lat2d[:, 0]
    dx_row = EARTH_RADIUS_KM * np.cos(np.radians(lat_vals)) * np.radians(dlat) * 1000.0
    dx2d = np.repeat(dx_row[:, np.newaxis], lon2d.shape[1], axis=1)
    dy_m = EARTH_RADIUS_KM * np.radians(dlat) * 1000.0
    return lat2d, lon2d, dx2d, dy_m


def _pmsl_from_z1000(z1000):
    """MSLP (hPa) from a synthetic 1000 hPa height field at about 8 m per hPa."""
    return 1000.0 + np.asarray(z1000, dtype=float) / 8.0


def _ocean_psfc(shape, hpa=1013.0):
    return np.full(shape, hpa)


def _anchor_amp(anchor_p, anchor_amp):
    """Amplitude as a function of pressure, interpolated linearly in ln p
    between anchor levels -- the same construction as
    article/figures/band_comparison.py's archetype profiles.
    """
    lnp = np.log(np.asarray(anchor_p, dtype=float))[::-1]
    amp = np.asarray(anchor_amp, dtype=float)[::-1]

    def amp_fn(p):
        return float(np.interp(math.log(p), lnp, amp))

    return amp_fn


def _vortex(amp_fn, clat=20.0, clon=0.0, half_width_deg=20.0, dlat=0.5, scale_km=150.0):
    """A Gaussian vortex on every level in ALL_LEVELS with per-level
    amplitude `amp_fn(p)`; returns the grid, the heights keyed by level,
    the grid spacing fields and the center indices.
    """
    lat2d, lon2d, dx2d, dy_m = _grid_and_dx_dy(clat, clon, half_width_deg, dlat)
    z_stack = synthetic.warm_core_heights(lat2d, lon2d, clat, clon, ALL_LEVELS, amp_fn, scale_km=scale_km)
    z = {p: z_stack[i] for i, p in enumerate(ALL_LEVELS)}
    ci, cj = lat2d.shape[0] // 2, lat2d.shape[1] // 2
    return lat2d, z, dx2d, dy_m, ci, cj


def _four_level_steering(u, v):
    return [u, v, u, v, u, v, u, v]


def _class_args_7(z, pmsl, winds, psfc, coriolis, dx2d, dy_m):
    return [pmsl] + [z[p] for p in HART_13] + winds + [psfc, coriolis, dx2d, dy_m]


def _class_args_std(z, pmsl, winds, psfc, coriolis, dx2d, dy_m):
    return [pmsl] + [z[p] for p in (925.0, 850.0, 700.0, 500.0, 400.0, 300.0)] + winds + [psfc, coriolis, dx2d, dy_m]


def _slope_1d(pressures, amp_fn):
    """The least-squares slope of amp(p) against ln p over `pressures`:
    what the gridded band slope at the vortex center must reproduce.
    """
    x = np.log(np.asarray(pressures, dtype=float))
    y = np.array([amp_fn(p) for p in pressures])
    return float(np.polyfit(x, y, 1)[0])


@pytest.fixture
def standard_orientation(monkeypatch):
    """ORIENTATION_MODE 0 for the synthetic grids here (row index
    increasing northward), as test_hart_cps.py's fixture of the same
    purpose does; cps_HartCPS's real default is 1.
    """
    monkeypatch.setattr(hc, "ORIENTATION_MODE", 0)


# ---------------------------------------------------------------------------
# executeBand7 on a known warm core
# ---------------------------------------------------------------------------


def test_execute_band7_reproduces_prescribed_slope():
    # Amplitude linear in ln p with a different slope in each band: the
    # lower slope is +60 m (warm), the upper -80 m (cold aloft). The
    # seven-level slope at the center must reproduce each prescribed
    # value to the disk's own truncation (exp(-(500/150)**2) ~ 1.5e-5).
    def amp_fn(p):
        if p >= 600.0:
            return 100.0 + 60.0 * math.log(p / 600.0)
        return 100.0 - 80.0 * math.log(p / 600.0)

    lat2d, z, dx2d, dy_m, ci, cj = _vortex(amp_fn, half_width_deg=8.0, dlat=0.25)
    psfc = _ocean_psfc(lat2d.shape)

    vtl7 = hc.executeBand7(*[z[p] for p in hc.HART_LOWER_BAND], psfc, dx2d, dy_m, 500.0, *hc.HART_LOWER_BAND)
    vtu7 = hc.executeBand7(*[z[p] for p in hc.HART_UPPER_BAND], psfc, dx2d, dy_m, 500.0, *hc.HART_UPPER_BAND)

    assert vtl7.dtype == np.float32 and vtu7.dtype == np.float32
    np.testing.assert_allclose(vtl7[ci, cj], 60.0, rtol=1e-3)
    np.testing.assert_allclose(vtu7[ci, cj], -80.0, rtol=1e-3)
    # The analytic slope through the seven anchor values is the same number.
    np.testing.assert_allclose(_slope_1d(hc.HART_LOWER_BAND, amp_fn), 60.0, rtol=1e-12)

    # The 925/850/700 fallback sits entirely inside the linear lower part
    # of this profile and reproduces the same +60 m (the two-slope mean
    # does not here, since its 850/700/500 band straddles the kink at
    # 600 hPa; it agrees with the others on a profile linear in ln p, see
    # test_class7_and_class_mean2_match_standard_class_on_linear_profile).
    vtl3 = hc.executeBand3(z[925.0], z[850.0], z[700.0], psfc, dx2d, dy_m, 500.0, 925.0, 850.0, 700.0)
    np.testing.assert_allclose(vtl3[ci, cj], 60.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# executeBandMean2 is the mean of the two three-level slopes
# ---------------------------------------------------------------------------


def test_execute_band_mean2_equals_mean_of_two_band3_slopes():
    # The seclusion archetype of band_comparison.py: the 925/850/700 and
    # 850/700/500 slopes differ by about 150 m here, so the test is not
    # trivially satisfied by equal slopes.
    amp_fn = _anchor_amp([1000, 925, 850, 700, 600, 500, 400, 300], [150, 140, 125, 100, 105, 130, 180, 230])
    lat2d, z, dx2d, dy_m, ci, cj = _vortex(amp_fn, half_width_deg=8.0, dlat=0.25)
    psfc = _ocean_psfc(lat2d.shape)

    slope_a = hc.executeBand3(z[925.0], z[850.0], z[700.0], psfc, dx2d, dy_m, 500.0, 925.0, 850.0, 700.0)
    slope_b = hc.executeBand3(z[850.0], z[700.0], z[500.0], psfc, dx2d, dy_m, 500.0, 850.0, 700.0, 500.0)
    mean2 = hc.executeBandMean2(z[925.0], z[850.0], z[700.0], z[500.0], psfc, dx2d, dy_m, 500.0)

    assert mean2.dtype == np.float32
    assert abs(slope_a[ci, cj] - slope_b[ci, cj]) > 100.0
    expected = 0.5 * (slope_a.astype(np.float64) + slope_b.astype(np.float64))
    np.testing.assert_allclose(mean2, expected, rtol=1e-5, atol=1e-3, equal_nan=True)
    # The center value is the mean of the two analytic slopes.
    analytic = 0.5 * (_slope_1d((925.0, 850.0, 700.0), amp_fn) + _slope_1d((850.0, 700.0, 500.0), amp_fn))
    np.testing.assert_allclose(mean2[ci, cj], analytic, rtol=1e-3)

    # Constants may arrive as 1-element arrays (AWIPS ConstantField values).
    mean2_arr = hc.executeBandMean2(
        z[925.0], z[850.0], z[700.0], z[500.0], psfc, dx2d, dy_m, np.array([500.0]), np.array([900.0]),
    )
    np.testing.assert_array_equal(mean2_arr, mean2)


def test_execute_band_mean2_nan_where_a_band_level_is_below_ground():
    # A terrain block with surface pressure 850 hPa masks 925 and 850 hPa
    # (both under the 900 hPa cap) and so both three-level slopes; the
    # mean must be NaN there, not a one-band value.
    amp_fn = _anchor_amp([1000, 925, 850, 700, 600, 500, 400, 300], [150, 140, 125, 100, 105, 130, 180, 230])
    lat2d, z, dx2d, dy_m, ci, cj = _vortex(amp_fn, half_width_deg=8.0, dlat=0.25)
    psfc = _ocean_psfc(lat2d.shape)
    psfc[:4, :4] = 850.0

    mean2 = hc.executeBandMean2(z[925.0], z[850.0], z[700.0], z[500.0], psfc, dx2d, dy_m, 500.0)
    assert np.all(np.isnan(mean2[:4, :4]))
    assert np.isfinite(mean2[ci, cj])


# ---------------------------------------------------------------------------
# executeBLayer on the 900-600 hPa thickness
# ---------------------------------------------------------------------------


def _thickness_gradient_case():
    ny, nx = 41, 21
    lat_vals = np.linspace(10.0, 30.0, ny)
    lon_vals = np.linspace(-5.0, 5.0, nx)
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    dx_row = EARTH_RADIUS_KM * np.cos(np.radians(lat_vals)) * math.radians(lon_vals[1] - lon_vals[0]) * 1000.0
    dx2d = np.repeat(dx_row[:, np.newaxis], nx, axis=1)
    dy_m = EARTH_RADIUS_KM * math.radians(lat_vals[1] - lat_vals[0]) * 1000.0
    y_km = EARTH_RADIUS_KM * np.radians(lat2d - 20.0)
    z900 = np.full((ny, nx), 900.0)
    z600 = z900 + 3300.0 - 0.02 * y_km  # thicker (warmer) to the south
    u = np.full((ny, nx), 15.0)  # moving east: the right-hand half-disk is the south one
    v = np.zeros((ny, nx))
    return lat2d, dx2d, dy_m, z900, z600, u, v


def test_execute_b_layer_900_600_scale_1_equals_execute_b_semicircle_difference(standard_orientation):
    lat2d, dx2d, dy_m, z900, z600, u, v = _thickness_gradient_case()
    psfc = _ocean_psfc(lat2d.shape)
    coriolis = np.full(lat2d.shape, 1.0)
    winds = _four_level_steering(u, v)

    b_layer = hc.executeBLayer(z600, z900, *winds, psfc, coriolis, dx2d, dy_m, 500.0, 600.0, 900.0, 1.0)
    # executeB's semicircle difference on the same thickness array (its
    # own 925/700 hPa masks blank nothing over this open-ocean psfc).
    b_ref = hc.executeB(z900, z600, *winds, psfc, coriolis, dx2d, dy_m, 500.0, 1.0)

    assert b_layer.dtype == np.float32
    np.testing.assert_allclose(b_layer, b_ref, rtol=1e-6, atol=1e-4, equal_nan=True)
    ci, cj = lat2d.shape[0] // 2, lat2d.shape[1] // 2
    assert b_layer[ci, cj] > 0  # warm air on the right of eastward motion in the NH

    # Same as parameter_b_grid applied directly to the thickness.
    u_s, v_s = hc.steering_window_mean(*hc.steering([u] * 4, [v] * 4), dx2d, dy_m, 500.0)
    b_direct = hc.parameter_b_grid(z600 - z900, u_s, v_s, dx2d, dy_m, coriolis, 500.0, 1.0)
    np.testing.assert_allclose(b_layer, b_direct.astype(np.float32), rtol=1e-6, atol=1e-4, equal_nan=True)

    # layerScale multiplies the difference; constants may be 1-element arrays.
    b_scaled = hc.executeBLayer(
        z600, z900, *winds, psfc, coriolis, dx2d, dy_m,
        np.array([500.0]), np.array([600.0]), np.array([900.0]), np.array([2.0]), np.array([900.0]),
    )
    np.testing.assert_allclose(b_scaled, 2.0 * b_layer, rtol=1e-6, atol=1e-4, equal_nan=True)


def test_execute_b_layer_masks_its_own_levels_below_ground(standard_orientation):
    lat2d, dx2d, dy_m, z900, z600, u, v = _thickness_gradient_case()
    coriolis = np.full(lat2d.shape, 1.0)
    winds = _four_level_steering(u, v)

    # Surface pressure 880 hPa: below the 900 hPa cap, so the 900 hPa
    # height is below ground and B is NaN there; 600 hPa is untouched.
    psfc = _ocean_psfc(lat2d.shape)
    psfc[:3, :] = 880.0
    b = hc.executeBLayer(z600, z900, *winds, psfc, coriolis, dx2d, dy_m, 500.0, 600.0, 900.0, 1.0)
    assert np.all(np.isnan(b[:3, :]))
    assert np.isfinite(b[lat2d.shape[0] // 2, lat2d.shape[1] // 2])

    # The same surface pressure leaves a 700-500 hPa layer alone (700 and
    # 500 hPa are masked only where the surface is at or below them).
    b_mid = hc.executeBLayer(z600, z900, *winds, psfc, coriolis, dx2d, dy_m, 500.0, 500.0, 700.0, 1.0)
    assert np.all(np.isfinite(b_mid[:3, 5:-5]))


# ---------------------------------------------------------------------------
# The seven-level and two-slope-mean class and index
# ---------------------------------------------------------------------------


def test_class7_and_class_mean2_match_standard_class_on_linear_profile():
    # Deep warm core, amplitude linear in ln p (slope +120 m from 1000 to
    # 300 hPa): every band reads the same +120 m, so the three class
    # entry points agree (code 0 at the center with weak steering and no
    # thickness gradient, NaN outside the closed low) and so do the
    # three index entry points.
    def amp_fn(p):
        return 20.0 + 120.0 * math.log(p / 300.0)

    lat2d, z, dx2d, dy_m, ci, cj = _vortex(amp_fn)
    pmsl = _pmsl_from_z1000(z[1000.0])
    psfc = _ocean_psfc(lat2d.shape)
    coriolis = np.full(lat2d.shape, 1.0)
    winds = _four_level_steering(np.full(lat2d.shape, 5.0), np.zeros(lat2d.shape))

    cls7 = hc.executeHartClass7(*_class_args_7(z, pmsl, winds, psfc, coriolis, dx2d, dy_m))
    cls_m2 = hc.executeHartClassMean2(*_class_args_std(z, pmsl, winds, psfc, coriolis, dx2d, dy_m))
    cls_std = hc.executeHartClass(*_class_args_std(z, pmsl, winds, psfc, coriolis, dx2d, dy_m))

    for cls in (cls7, cls_m2, cls_std):
        assert cls.dtype == np.float32
        assert cls[ci, cj] == 0.0
        assert np.isnan(cls[0, 0])
    np.testing.assert_array_equal(np.isnan(cls7), np.isnan(cls_std))
    np.testing.assert_array_equal(cls7[np.isfinite(cls7)], cls_std[np.isfinite(cls_std)])
    np.testing.assert_array_equal(cls_m2, cls_std)

    idx7 = hc.executeIndex7(pmsl, *[z[p] for p in HART_13], psfc, dx2d, dy_m)
    idx_m2 = hc.executeIndexMean2(pmsl, *[z[p] for p in (925.0, 850.0, 700.0, 500.0, 400.0, 300.0)], psfc, dx2d, dy_m)
    idx_std = hc.executeIndexStd(pmsl, *[z[p] for p in (925.0, 850.0, 700.0, 500.0, 400.0, 300.0)], psfc, dx2d, dy_m)
    expected = 2.0 * math.tanh(1.2) + math.tanh(1.2)
    for idx in (idx7, idx_m2, idx_std):
        assert idx.dtype == np.float32
        np.testing.assert_allclose(idx[ci, cj], expected, rtol=1e-3)
        assert np.isnan(idx[0, 0])


def test_class7_reads_shallow_warm_core_cold_where_standard_class_reads_warm():
    # A warm anomaly confined below 850 hPa under a cold core from 700 hPa
    # up (anchors at 1000, 925, 850, 700, 600, 500, 400, 300 hPa). The
    # 925/850/700 band, which sits entirely below Hart's layer, reads it
    # strongly warm (+335 m analytic); Hart's own 900-600 hPa band reads
    # it cold (-21 m); the two-slope mean lies between (+32 m). So the
    # fallback and the two-slope mean give class 1 (symmetric shallow
    # warm core) and the seven-level class gives 5 (symmetric cold core):
    # the standard band overstates a shallow warm core, and Hart's layers
    # correct it in the cold direction, which is the direction the
    # Florida State comparison found (the 925-700 band about 17 m too
    # warm on the GFS cases).
    anchors_p = [1000, 925, 850, 700, 600, 500, 400, 300]
    amp_fn = _anchor_amp(anchors_p, [240, 225, 140, 110, 200, 270, 330, 390])
    vtl7_analytic = _slope_1d(hc.HART_LOWER_BAND, amp_fn)
    vtl3_analytic = _slope_1d((925.0, 850.0, 700.0), amp_fn)
    mean2_analytic = 0.5 * (vtl3_analytic + _slope_1d((850.0, 700.0, 500.0), amp_fn))
    assert vtl7_analytic < 0.0 < mean2_analytic < vtl3_analytic

    lat2d, z, dx2d, dy_m, ci, cj = _vortex(amp_fn)
    pmsl = _pmsl_from_z1000(z[1000.0])
    psfc = _ocean_psfc(lat2d.shape)
    coriolis = np.full(lat2d.shape, 1.0)
    winds = _four_level_steering(np.full(lat2d.shape, 5.0), np.zeros(lat2d.shape))

    vtl7 = hc.executeBand7(*[z[p] for p in hc.HART_LOWER_BAND], psfc, dx2d, dy_m, 500.0, *hc.HART_LOWER_BAND)
    vtl_m2 = hc.executeBandMean2(z[925.0], z[850.0], z[700.0], z[500.0], psfc, dx2d, dy_m, 500.0)
    vtl3 = hc.executeBand3(z[925.0], z[850.0], z[700.0], psfc, dx2d, dy_m, 500.0, 925.0, 850.0, 700.0)
    np.testing.assert_allclose(vtl7[ci, cj], vtl7_analytic, rtol=1e-3)
    np.testing.assert_allclose(vtl_m2[ci, cj], mean2_analytic, rtol=1e-3)
    np.testing.assert_allclose(vtl3[ci, cj], vtl3_analytic, rtol=1e-3)
    assert vtl7[ci, cj] < vtl_m2[ci, cj] < vtl3[ci, cj]

    cls7 = hc.executeHartClass7(*_class_args_7(z, pmsl, winds, psfc, coriolis, dx2d, dy_m))
    cls_m2 = hc.executeHartClassMean2(*_class_args_std(z, pmsl, winds, psfc, coriolis, dx2d, dy_m))
    cls_std = hc.executeHartClass(*_class_args_std(z, pmsl, winds, psfc, coriolis, dx2d, dy_m))
    assert cls7[ci, cj] == 5.0
    assert cls_m2[ci, cj] == 1.0
    assert cls_std[ci, cj] == 1.0

    idx7 = hc.executeIndex7(pmsl, *[z[p] for p in HART_13], psfc, dx2d, dy_m)
    idx_m2 = hc.executeIndexMean2(pmsl, *[z[p] for p in (925.0, 850.0, 700.0, 500.0, 400.0, 300.0)], psfc, dx2d, dy_m)
    idx_std = hc.executeIndexStd(pmsl, *[z[p] for p in (925.0, 850.0, 700.0, 500.0, 400.0, 300.0)], psfc, dx2d, dy_m)
    assert idx7[ci, cj] < idx_m2[ci, cj] < idx_std[ci, cj]


def test_class7_constants_as_one_element_arrays_and_pmsl_kind_fallback():
    def amp_fn(p):
        return 20.0 + 120.0 * math.log(p / 300.0)

    lat2d, z, dx2d, dy_m, ci, cj = _vortex(amp_fn)
    pmsl = _pmsl_from_z1000(z[1000.0])
    psfc = _ocean_psfc(lat2d.shape)
    coriolis = np.full(lat2d.shape, 1.0)
    winds = _four_level_steering(np.full(lat2d.shape, 5.0), np.zeros(lat2d.shape))
    args = _class_args_7(z, pmsl, winds, psfc, coriolis, dx2d, dy_m)

    cls_plain = hc.executeHartClass7(*args)
    cls_arrays = hc.executeHartClass7(
        *args,
        np.array([500.0]), np.array([10.0]), np.array([1.0]), np.array([5.0]), np.array([200.0]),
        np.array([900.0]), np.array([0.0]),
    )
    np.testing.assert_array_equal(cls_arrays, cls_plain)

    # pmslKind 1: the first argument is 1000 hPa height instead of MSLP.
    args_z1000 = [z[1000.0]] + args[1:]
    cls_z1000 = hc.executeHartClass7(*args_z1000, 500.0, 10.0, 1.0, 5.0, 200.0, 900.0, 1.0)
    assert cls_z1000[ci, cj] == 0.0
    assert np.isnan(cls_z1000[0, 0])


# ---------------------------------------------------------------------------
# The XML definitions: method order, functions, arity and levels
# ---------------------------------------------------------------------------

#: The method order every definition must carry: Hart's seven-level bands
#: first (GFS), the two-slope mean second (standard levels), the shipped
#: 925/850/700 band last (fallback). HVTU and HB have no two-slope-mean
#: block (500/400/300 and the rescaled 925-700 thickness serve as both
#: the standard-level method and the fallback). HCPSclass and HCPSidx
#: repeat each band method for the four MSLP inputs, band-major.
EXPECTED_METHODS = {
    "HVTL": ["executeBand7", "executeBandMean2", "executeBand3"],
    "HVTU": ["executeBand7", "executeBand3"],
    "HB": ["executeBLayer", "executeB"],
    "HCPSclass": ["executeHartClass7"] * 4 + ["executeHartClassMean2"] * 4 + ["executeHartClass"] * 4,
    "HCPSidx": ["executeIndex7"] * 4 + ["executeIndexMean2"] * 4 + ["executeIndexStd"] * 4,
}

MSLP_ORDER = [("MSLP", "MSL"), ("PMSL", "MSL"), ("MSLP", "Surface"), ("GH", "1000MB")]


def _level_hpa(level_attr):
    assert level_attr.endswith("MB"), level_attr
    return float(level_attr[:-2])


def _gh_levels(method, skip_first=False):
    fields = [f for f in method if f.tag == "Field"]
    if skip_first:
        fields = fields[1:]
    return [_level_hpa(f.get("level")) for f in fields if f.get("abbreviation") == "GH"]


def _constants(method):
    return [float(c.get("value")) for c in method if c.tag == "ConstantField"]


def _definitions():
    paths = sorted(glob.glob(os.path.join(DEFINITIONS_DIR, "cps_*.xml")))
    assert len(paths) == 5, paths
    return paths


@pytest.mark.parametrize("path", _definitions(), ids=os.path.basename)
def test_xml_definition_method_order_functions_and_arity(path):
    root = ET.parse(path).getroot()
    assert root.tag == "DerivedParameter"
    abbreviation = root.get("abbreviation")
    assert os.path.basename(path) == f"cps_{abbreviation}.xml"
    assert root.get("unit") == ""

    methods = [m for m in root if m.tag == "Method"]
    names = [m.get("name") for m in methods]
    assert all(n.startswith("cps_HartCPS.") for n in names)
    assert [n.split(".", 1)[1] for n in names] == EXPECTED_METHODS[abbreviation]

    for m in methods:
        assert m.get("levels") == "Surface"
        func_name = m.get("name").split(".", 1)[1]
        func = getattr(hc, func_name)
        assert callable(func)
        # AWIPS passes every Field and ConstantField positionally, in file
        # order: the count must equal the function's parameter count.
        inputs = [c for c in m if c.tag in ("Field", "ConstantField")]
        params = inspect.signature(func).parameters
        assert len(inputs) == len(params), (path, func_name, len(inputs), len(params))
        # Fields come before ConstantFields, as the signatures put arrays
        # before constants.
        tags = [c.tag for c in inputs]
        assert tags == sorted(tags, key=lambda t: t == "ConstantField")
        for c in m:
            if c.tag == "Field" and c.get("level") is not None and c.get("level") != "Surface":
                assert c.get("level").endswith("MB") or c.get("level") == "MSL"


@pytest.mark.parametrize("path", _definitions(), ids=os.path.basename)
def test_xml_definition_levels_match_their_method(path):
    root = ET.parse(path).getroot()
    abbreviation = root.get("abbreviation")
    methods = [m for m in root if m.tag == "Method"]

    for i, m in enumerate(methods):
        func_name = m.get("name").split(".", 1)[1]
        consts = _constants(m)
        assert consts[0] == 500.0  # radiusKm, Hart's circle, in every block

        if func_name == "executeBand7":
            levels = _gh_levels(m)
            expected = list(hc.HART_LOWER_BAND if abbreviation == "HVTL" else hc.HART_UPPER_BAND)
            assert levels == expected
            assert consts[1:8] == expected  # p1..p7 match z1..z7 by position
            assert consts[8] == 900.0  # capHpa
        elif func_name == "executeBandMean2":
            assert _gh_levels(m) == [925.0, 850.0, 700.0, 500.0]
            assert consts == [500.0, 900.0]
        elif func_name == "executeBand3":
            levels = _gh_levels(m)
            expected = list(hc.LOWER_BAND if abbreviation == "HVTL" else hc.UPPER_BAND)
            assert levels == expected
            assert consts[1:4] == expected
        elif func_name == "executeBLayer":
            assert _gh_levels(m) == [600.0, 900.0]  # zTop, zBot
            assert consts[1:4] == [600.0, 900.0, 1.0]  # pTop, pBot, layerScale
        elif func_name == "executeB":
            assert _gh_levels(m) == [925.0, 700.0]
            assert consts[1] == pytest.approx(hc.HART_B_LAYER_SCALE, abs=1e-4)
        elif func_name in ("executeHartClass7", "executeIndex7"):
            assert _gh_levels(m, skip_first=True) == list(HART_13)
            if func_name == "executeHartClass7":
                assert consts[2] == 1.0  # layerScale on Hart's own layer
        elif func_name in ("executeHartClassMean2", "executeHartClass", "executeIndexMean2", "executeIndexStd"):
            assert _gh_levels(m, skip_first=True) == [925.0, 850.0, 700.0, 500.0, 400.0, 300.0]
            if func_name.startswith("executeHartClass"):
                assert consts[2] == pytest.approx(hc.HART_B_LAYER_SCALE, abs=1e-4)
        else:
            raise AssertionError(func_name)

        if abbreviation in ("HCPSclass", "HCPSidx"):
            first = [f for f in m if f.tag == "Field"][0]
            assert (first.get("abbreviation"), first.get("level")) == MSLP_ORDER[i % 4]
            assert consts[-1] == (1.0 if i % 4 == 3 else 0.0)  # pmslKind
