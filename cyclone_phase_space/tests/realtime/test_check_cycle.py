"""check_cycle, the 30-minute test for a new GFS cycle on the live map.

No network: the probe is replaced by a set of URLs that count as posted.
The constants are checked against gfs_cps and export_web, which the check
mirrors so it can run without the pip install.
"""

from __future__ import annotations

import datetime as dt

import check_cycle as cc
import export_web as ew
import gfs_cps as g

NOW = dt.datetime(2026, 11, 19, 19, 30, tzinfo=dt.timezone.utc)


def test_mirrors_export_and_gfs_modules():
    assert cc.NOMADS == g.NOMADS
    assert cc.AWS == g.AWS
    assert cc.LAST_FHR == ew.LAST_FHR
    assert cc.PROBE_CYCLES == ew.PROBE_CYCLES
    assert cc.gfs_name("2026111912", 198) == g.gfs_name("2026111912", 198)


def test_candidates_newest_first():
    c = cc.candidates(NOW)
    assert c[:3] == ["2026111918", "2026111912", "2026111906"]
    assert len(c) == cc.PROBE_CYCLES


def test_newest_cycle_falls_back_and_uses_aws():
    aws_12z = dict(cc.idx_urls("2026111912"))["aws"]
    assert cc.newest_cycle(NOW, probe=lambda u: u == aws_12z) == "2026111912"


def test_newest_cycle_none_when_nothing_posted():
    assert cc.newest_cycle(NOW, probe=lambda u: False) is None


def test_is_new():
    assert cc.is_new("2026111912", "2026111906")
    assert not cc.is_new("2026111912", "2026111912")
    assert not cc.is_new("2026111906", "2026111912")
    assert cc.is_new("2026111912", None)
    assert not cc.is_new(None, "2026111906")
