# RLACO Results — best conflicts & convergence time

Per cell: 10 independent runs. **Min conflicts** = best coloring found across the 10 runs (0 = legal k-coloring). **Conv. time** = median time-to-best (wall-clock at which a run first reached its own final best).

| Instance | k | Method | n | Min | Median | Max | Mean±SD | Conv. time (med) | Total (med) | Eps (med) |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| DSJC250.5 | 28 | baseline | 10 | **70** | 77 | 80 | 76.6±2.9 | 1h39m | 4h00m | 224 |
| DSJC250.5 | 28 | soft | 10 | **16** | 24 | 28 | 22.7±3.1 | 2h42m | 4h00m | 236 |
| DSJC500.1 | 12 | baseline | 10 | **136** | 142 | 145 | 141.4±2.3 | 1h09m | 2h00m | 202 |
| DSJC500.1 | 12 | soft | 10 | **48** | 59 | 81 | 62.9±11.6 | 1h41m | 2h00m | 198 |
| DSJC500.5 | 48 | baseline | 10 | **225** | 230 | 235 | 230±3.5 | 1h36m | 5h01m | 112 |
| DSJC500.5 | 48 | soft | 10 | **74** | 77 | 83 | 78.1±3.4 | 4h46m | 5h01m | 108 |
| DSJC500.9 | 126 | baseline | 10 | **196** | 201 | 206 | 202.1±2.8 | 2h15m | 6h01m | 84 |
| DSJC500.9 | 126 | soft | 10 | **42** | 46 | 54 | 47.4±3.4 | 5h40m | 6h01m | 76 |
| flat300_26_0 | 26 | baseline | 10 | **187** | 195 | 200 | 194.7±3.6 | 2h26m | 4h00m | 212 |
| flat300_26_0 | 26 | soft | 10 | **93** | 96 | 103 | 96.8±3.2 | 2h59m | 4h00m | 141 |
| flat300_28_0 | 28 | baseline | 10 | **150** | 156 | 158 | 155.2±2.6 | 2h28m | 4h00m | 172 |
| flat300_28_0 | 28 | soft | 10 | **58** | 62 | 68 | 62.4±3.1 | 2h54m | 4h00m | 176 |
| le450_15c | 15 | baseline | 10 | **295** | 298 | 300 | 298±1.6 | 1h05m | 3h00m | 282 |
| le450_15c | 15 | soft | 10 | **222** | 246 | 257 | 243.2±11.8 | 2h14m | 3h00m | 252 |
| le450_15d | 15 | baseline | 10 | **290** | 302 | 309 | 301.4±5.3 | 40m16s | 3h01m | 180 |
| le450_15d | 15 | soft | 10 | **229** | 248 | 275 | 250±14.4 | 2h51m | 3h01m | 97 |
| le450_25c | 25 | baseline | 10 | **63** | 68 | 71 | 68±2.6 | 1h25m | 3h00m | 519 |
| le450_25c | 25 | soft | 10 | **43** | 58 | 67 | 57.4±7.0 | 1h49m | 3h02m | 86 |
| le450_25d | 25 | baseline | 10 | **66** | 70 | 72 | 69.9±1.6 | 57m29s | 3h00m | 227 |
| le450_25d | 25 | soft | 10 | **44** | 48 | 61 | 50.5±6.4 | 1h52m | 3h00m | 514 |
