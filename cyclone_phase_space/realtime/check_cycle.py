"""Is there a GFS cycle newer than the one the live map shows?

    python3 check_cycle.py [--published YYYYMMDDHH]

Finds the newest cycle whose 0.25 degree f198 index is posted (NOMADS,
else the AWS bucket), probing back from the current 6 h cycle, the same
rule export_web.py uses for its default cycle, and compares it with the
cycle the map shows now (--published, the "cycle" of the cps-live
branch's latest/index.json). Prints the result and, under GitHub Actions,
writes cycle=<newest> and new=true|false to $GITHUB_OUTPUT.

The "CPS live check" workflow runs this every 30 minutes and starts the
"CPS live map data" export only when new is true, so a cycle reaches the
map soon after its f198 is posted, even when GitHub delays or drops some
scheduled runs. Standard library only, so the check needs no pip install
and takes a few seconds. Exit status 0 either way; 1 if no cycle in the
last PROBE_CYCLES has f198 (nothing is started then).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import urllib.error
import urllib.request

# The same archive roots, file names and probe depth as gfs_cps.py and
# export_web.py (tests/realtime/test_check_cycle.py keeps them in step).
NOMADS = "https://nomads.ncep.noaa.gov"
AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
LAST_FHR = 198
PROBE_CYCLES = 8
TIMEOUT = 20


def gfs_name(cycle: str, fhr: int) -> tuple[str, str]:
    """(directory, file name) of the 0.25 degree pgrb2 file, as gfs_cps.gfs_name."""
    ymd, hh = cycle[:8], cycle[8:]
    return f"gfs.{ymd}/{hh}/atmos", f"gfs.t{hh}z.pgrb2.0p25.f{fhr:03d}"


def idx_urls(cycle: str) -> list[tuple[str, str]]:
    """(source, URL) of the cycle's f198 .idx on NOMADS, then AWS."""
    d, f = gfs_name(cycle, LAST_FHR)
    return [("nomads", f"{NOMADS}/pub/data/nccf/com/gfs/prod/{d}/{f}.idx"),
            ("aws", f"{AWS}/{d}/{f}.idx")]


def posted(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def candidates(now: dt.datetime) -> list[str]:
    """The cycles to probe, newest first: the current 6 h cycle and PROBE_CYCLES - 1 before it."""
    t = now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0)
    return [(t - dt.timedelta(hours=6 * i)).strftime("%Y%m%d%H") for i in range(PROBE_CYCLES)]


def newest_cycle(now: dt.datetime | None = None, probe=posted) -> str | None:
    now = now or dt.datetime.now(dt.timezone.utc)
    for cyc in candidates(now):
        for src, url in idx_urls(cyc):
            if probe(url):
                print(f"cycle {cyc}: f{LAST_FHR:03d} found on {src}", flush=True)
                return cyc
        print(f"cycle {cyc}: f{LAST_FHR:03d} not posted", flush=True)
    return None


def is_new(newest: str | None, published: str | None) -> bool:
    """True when newest is a later cycle than published (or nothing is published)."""
    if not newest:
        return False
    return not published or newest > published


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--published", default="", help="cycle the map shows now, YYYYMMDDHH (blank: none)")
    a = ap.parse_args(argv)
    published = a.published.strip() or None
    newest = newest_cycle()
    new = is_new(newest, published)
    print(f"published {published or 'none'}, newest {newest or 'none'}: "
          f"{'export' if new else 'nothing to do'}", flush=True)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"cycle={newest or ''}\nnew={'true' if new else 'false'}\n")
    return 0 if newest else 1


if __name__ == "__main__":
    sys.exit(main())
