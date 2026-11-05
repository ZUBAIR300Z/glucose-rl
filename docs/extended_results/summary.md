**120 days per controller**

| Controller | TIR | TBR<70 | TBR<54 | GRI | Survived |
|----|----|----|----|----|----|
| PID (repo) | 53.8 | 11.1 | 7.6 | 67.2 | 77% |
| PID scaled to basal, tuned | 75.6 | 2.7 | 0.8 | 31.2 | 93% |
| Prescribed basal only | 42.0 | 0.0 | 0.0 | 72.6 | 68% |
| Basal-bolus, meals announced | 80.1 | 5.9 | 2.3 | 30.2 | 98% |
| v2-res-sac-zone | 77.3 | 3.5 | 1.3 | 30.7 | 94% |
| v2-res-sac | 75.3 | 2.2 | 1.0 | 30.3 | 93% |
| v2-sac | 72.1 | 1.0 | 0.1 | 31.9 | 90% |

**Paired vs PID scaled to basal, tuned**: mean difference (Wilcoxon p; days better / worse)

| Controller | TIR | TBR<70 | TBR<54 | GRI |
|----|----|----|----|----|
| v2-res-sac-zone (n=120) | +1.73 (p=0.000; 85/31) | +0.82 (p=0.000; 10/36) | +0.49 (p=0.001; 2/17) | -0.45 (p=0.002; 79/34) |
| v2-res-sac (n=120) | -0.30 (p=0.022; 44/64) | -0.56 (p=0.125; 25/16) | +0.13 (p=0.182; 5/10) | -0.92 (p=0.281; 50/58) |
| v2-sac (n=120) | -3.52 (p=0.000; 26/93) | -1.74 (p=0.000; 37/4) | -0.70 (p=0.002; 13/1) | +0.69 (p=0.006; 32/85) |

**Paired vs Basal-bolus, meals announced**: mean difference (Wilcoxon p; days better / worse)

| Controller | TIR | TBR<70 | TBR<54 | GRI |
|----|----|----|----|----|
| v2-res-sac-zone (n=120) | -2.71 (p=0.001; 42/78) | -2.34 (p=0.001; 39/22) | -0.97 (p=0.026; 21/13) | +0.48 (p=0.014; 37/79) |
| v2-res-sac (n=120) | -4.74 (p=0.000; 38/81) | -3.72 (p=0.000; 46/11) | -1.33 (p=0.000; 23/8) | +0.01 (p=0.076; 38/79) |
| v2-sac (n=120) | -7.96 (p=0.000; 34/86) | -4.90 (p=0.000; 52/1) | -2.16 (p=0.000; 30/0) | +1.61 (p=0.027; 40/77) |
