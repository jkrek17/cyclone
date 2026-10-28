# Class agreement with the FSU diagrams, bands by B method

Generated 2026-10-28T01:35Z by `realtime/class_cells.py` from `fsu_pairs.csv` (517 verified hours; timing on the usable hours as in fsu_stats.py). Joint class by the D2D rule; kappa is Cohen's; the majority baseline is the share of FSU's most common class; the sign columns are the agreement of each of the three signs that make the class; near is the share of disagreements within 5 m of the boundary crossed, on either side.

| Bands | B | n | class agree | kappa | majority | B side | VTL sign | VTU sign | B only | VTL only | VTU only | several | near | onset within 6 h (n, median) | completion within 6 h (n, median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard bands (925-700, 500-300 hPa) | steering-flow B, 925-700 hPa scaled | 505 | 55% | 0.43 | 35% | 68% | 84% | 93% | 115 | 46 | 22 | 47 | 54% | 2 of 6, -9 h | 7 of 10, +6 h |
| standard bands (925-700, 500-300 hPa) | track-motion B, 925-700 hPa scaled | 515 | 63% | 0.54 | 35% | 78% | 85% | 93% | 75 | 53 | 20 | 41 | 60% | 4 of 9, +0 h | 7 of 9, +6 h |
| standard bands (925-700, 500-300 hPa) | track-motion B, 900-600 hPa | 515 | 66% | 0.57 | 35% | 82% | 85% | 93% | 59 | 59 | 23 | 32 | 50% | 5 of 6, +3 h | 7 of 9, +6 h |
| Hart's bands (900-600, 600-300 hPa) | steering-flow B, 925-700 hPa scaled | 505 | 61% | 0.51 | 35% | 68% | 90% | 97% | 136 | 24 | 9 | 29 | 59% | 2 of 6, -9 h | 11 of 12, +0 h |
| Hart's bands (900-600, 600-300 hPa) | track-motion B, 925-700 hPa scaled | 515 | 70% | 0.62 | 35% | 78% | 91% | 97% | 94 | 31 | 6 | 25 | 64% | 4 of 9, +0 h | 11 of 11, +0 h |
| Hart's bands (900-600, 600-300 hPa) | track-motion B, 900-600 hPa | 515 | 75% | 0.68 | 35% | 82% | 91% | 97% | 69 | 30 | 7 | 25 | 57% | 5 of 6, +3 h | 10 of 11, +0 h |

## Confusion matrix, hart+track_900 (rows ours, columns FSU)

| ours \ FSU | 0 sym deep warm | 1 sym shallow warm | 2 asym deep warm | 3 asym shallow warm | 4 asym cold | 5 sym cold | 6 shallow cold | total |
|---|---|---|---|---|---|---|---|---|
| 0 sym deep warm | 67 | 2 | 3 | 1 | 0 | 0 | 0 | 73 |
| 1 sym shallow warm | 4 | 142 | 0 | 8 | 7 | 2 | 0 | 163 |
| 2 asym deep warm | 2 | 3 | 15 | 0 | 0 | 0 | 0 | 20 |
| 3 asym shallow warm | 3 | 13 | 1 | 37 | 7 | 0 | 0 | 61 |
| 4 asym cold | 0 | 4 | 0 | 10 | 74 | 26 | 0 | 114 |
| 5 sym cold | 0 | 11 | 0 | 4 | 17 | 49 | 0 | 81 |
| 6 shallow cold | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| total | 76 | 178 | 19 | 60 | 105 | 77 | 0 | 515 |

## Confusion matrix, standard+steer_925 (rows ours, columns FSU)

| ours \ FSU | 0 sym deep warm | 1 sym shallow warm | 2 asym deep warm | 3 asym shallow warm | 4 asym cold | 5 sym cold | 6 shallow cold | total |
|---|---|---|---|---|---|---|---|---|
| 0 sym deep warm | 49 | 14 | 2 | 1 | 0 | 0 | 0 | 66 |
| 1 sym shallow warm | 4 | 91 | 0 | 9 | 14 | 11 | 0 | 129 |
| 2 asym deep warm | 9 | 0 | 15 | 1 | 0 | 0 | 0 | 25 |
| 3 asym shallow warm | 5 | 59 | 2 | 41 | 25 | 10 | 0 | 142 |
| 4 asym cold | 0 | 9 | 0 | 8 | 67 | 36 | 0 | 120 |
| 5 sym cold | 0 | 2 | 0 | 0 | 0 | 12 | 0 | 14 |
| 6 shallow cold | 0 | 0 | 0 | 0 | 1 | 8 | 0 | 9 |
| total | 67 | 175 | 19 | 60 | 107 | 77 | 0 | 505 |
