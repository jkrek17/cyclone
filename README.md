# Cyclone phase space

Hart's (2003) cyclone phase space computed at every grid point of a model
run, with no storm tracker: every closed low on every forecast frame gets
its thermal asymmetry, its lower and upper thermal wind and one of seven
classes. The engine is delivered as a live global map of the current GFS
run with every tracked low's path through the three-dimensional phase
space, and as AWIPS II derived parameters for CAVE.

- Live map: https://jkrek17.github.io/cyclone/
- Article: https://jkrek17.github.io/cyclone/article/
- Forecaster guide: https://jkrek17.github.io/cyclone/forecasters/

## Contents

| Path | What it is |
|---|---|
| `cyclone_phase_space/D2D/` | The AWIPS install set: one Python module, five derived-parameter definitions and two colormaps. |
| `cyclone_phase_space/cps/` | The storm-centered reference implementation of Hart's method. |
| `cyclone_phase_space/realtime/` | The engine outside AWIPS: the six-hourly export behind the live map, the tracker, the daily collection against the Florida State diagrams, the digitizer and the archive statistics, with the archive under `realtime/data/`. |
| `cyclone_phase_space/article/` | The article, its figure scripts, the forecasters' guide and the live map page. |
| `cyclone_phase_space/docs/` | Technical, user, installation and publishing guides. |
| `cyclone_phase_space/tests/` | The test suite: `python3 -m pytest cyclone_phase_space/tests -q`. |
| `nwp/` | The GRIB fetcher the realtime tools use (NOMADS, then the AWS bucket). |
| `globe_kit/` | The MapLibre globe kit the live map loads. |

## Workflows

- `CPS live map data` exports the newest GFS cycle four times a day to the
  orphan branch `cps-live`.
- `CPS daily collection` follows the watched storms in the 12 UTC run and
  commits their tracks and diagrams under `cyclone_phase_space/realtime/data/`.
- `Publish cyclone site` assembles the map, its data, the article and the
  guide and deploys them with GitHub Pages after each of the above.

Dependencies for the realtime tools: `python3 -m pip install -r
cyclone_phase_space/realtime/requirements.txt`.
