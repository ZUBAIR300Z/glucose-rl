**Held-out test patients (6 patients x 5 days), no shield**

Paired over the same patient-days (agents averaged over seeds); p from a Wilcoxon signed-rank test, * = p < 0.05.

| Controller | Seeds | TIR 70-180 % ↑ | TBR <70 % ↓ | TBR <54 % ↓ | GRI ↓ | Survived | Seeds with TBR <4% | vs PID (repo): mean paired diff | vs tuned basal PID: mean paired diff |
|----|----|----|----|----|----|----|----|----|----|
| PID (repo) | - | 54.5 | 11.2 | 8.0 | 68 | 73% | - | - | - |
| PID scaled to basal, tuned | - | 76.1 | 2.1 | 0.3 | 30 | 90% | - | - | - |
| Prescribed basal only | - | 46.3 | 0.0 | 0.0 | 69 | 70% | - | - | - |
| Basal-bolus, meals announced* | - | 79.7 | 5.7 | 1.7 | 32 | 97% | - | - | - |
| SAC, v1 env (original setup) | 3 | 52.8 ± 0.7 | 30.2 ± 1.0 | 22.9 | 83 | 36% | 0/3 | TIR -1.7 (p=0.984); TBR +19.0 (p=0.000)* | TIR -23.2 (p=0.000)*; TBR +28.1 (p=0.000)* |
| SAC, v2 env | 3 | 71.2 ± 2.2 | 0.8 ± 0.5 | 0.0 | 32 | 87% | 3/3 | TIR +16.7 (p=0.000)*; TBR -10.4 (p=0.000)* | TIR -4.8 (p=0.001)*; TBR -1.3 (p=0.139) |
| SAC, v2 env, dose cap 8x basal | 1 | 72.5 ± 0.0 | 1.1 ± 0.0 | 0.3 | 31 | 90% | 1/1 | TIR +17.9 (p=0.000)*; TBR -10.1 (p=0.001)* | TIR -3.6 (p=0.038)*; TBR -1.0 (p=0.310) |
| TQC, v2 env | 2 | 67.5 ± 1.0 | 0.1 ± 0.1 | 0.0 | 34 | 85% | 2/2 | TIR +13.0 (p=0.001)*; TBR -11.1 (p=0.000)* | TIR -8.6 (p=0.000)*; TBR -2.0 (p=0.028)* |
| PPO, v2 env | 2 | 65.8 ± 1.9 | 0.3 ± 0.3 | 0.0 | 37 | 83% | 2/2 | TIR +11.3 (p=0.001)*; TBR -10.9 (p=0.000)* | TIR -10.2 (p=0.000)*; TBR -1.7 (p=0.028)* |

**By patient cohort** (no shield)

| Controller | child TIR / TBR | adolescent TIR / TBR | adult TIR / TBR |
|----|----|----|----|
| PID (repo) | 60% / 14.7% | 48% / 18.1% | 55% / 0.9% |
| PID scaled to basal, tuned | 74% / 5.0% | 73% / 1.2% | 80% / 0.0% |
| Prescribed basal only | 60% / 0.0% | 38% / 0.0% | 41% / 0.0% |
| Basal-bolus, meals announced | 68% / 10.9% | 81% / 3.0% | 90% / 3.0% |
| SAC, v1 env (original setup) | 50% / 34.4% | 47% / 34.7% | 61% / 21.5% |
| SAC, v2 env | 73% / 2.0% | 68% / 0.4% | 73% / 0.0% |
| SAC, v2 env, dose cap 8x basal | 74% / 3.3% | 70% / 0.0% | 74% / 0.0% |
| TQC, v2 env | 76% / 0.3% | 64% / 0.0% | 62% / 0.0% |
| PPO, v2 env | 76% / 1.0% | 62% / 0.0% | 59% / 0.0% |

**Held-out test patients (6 patients x 5 days), every controller behind the safety shield**

Paired over the same patient-days (agents averaged over seeds); p from a Wilcoxon signed-rank test, * = p < 0.05.

| Controller | Seeds | TIR 70-180 % ↑ | TBR <70 % ↓ | TBR <54 % ↓ | GRI ↓ | Survived | Seeds with TBR <4% | vs PID (repo): mean paired diff | vs tuned basal PID: mean paired diff |
|----|----|----|----|----|----|----|----|----|----|
| PID (repo) | - | 55.0 | 11.0 | 7.4 | 67 | 80% | - | - | - |
| PID scaled to basal, tuned | - | 76.1 | 2.0 | 0.3 | 29 | 90% | - | - | - |
| Prescribed basal only | - | 46.2 | 0.0 | 0.0 | 69 | 70% | - | - | - |
| Basal-bolus, meals announced* | - | 79.6 | 2.8 | 0.9 | 28 | 93% | - | - | - |
| SAC, v1 env (original setup) | 3 | 55.7 ± 1.2 | 22.2 ± 0.9 | 12.9 | 77 | 67% | 0/3 | TIR +0.7 (p=0.262); TBR +11.2 (p=0.000)* | TIR -20.3 (p=0.000)*; TBR +20.2 (p=0.000)* |
| SAC, v2 env | 3 | 71.0 ± 2.2 | 0.7 ± 0.5 | 0.0 | 32 | 87% | 3/3 | TIR +16.0 (p=0.000)*; TBR -10.3 (p=0.000)* | TIR -5.1 (p=0.001)*; TBR -1.3 (p=0.139) |
| SAC, v2 env, dose cap 8x basal | 1 | 72.2 ± 0.0 | 1.0 ± 0.0 | 0.3 | 31 | 90% | 1/1 | TIR +17.2 (p=0.000)*; TBR -10.0 (p=0.001)* | TIR -3.8 (p=0.028)*; TBR -1.0 (p=0.310) |
| TQC, v2 env | 2 | 67.4 ± 1.0 | 0.1 ± 0.1 | 0.0 | 35 | 85% | 2/2 | TIR +12.4 (p=0.001)*; TBR -10.9 (p=0.000)* | TIR -8.6 (p=0.000)*; TBR -1.9 (p=0.028)* |
| PPO, v2 env | 2 | 65.8 ± 1.8 | 0.4 ± 0.4 | 0.0 | 38 | 83% | 2/2 | TIR +10.7 (p=0.001)*; TBR -10.7 (p=0.000)* | TIR -10.3 (p=0.000)*; TBR -1.6 (p=0.028)* |

**By patient cohort** (shielded)

| Controller | child TIR / TBR | adolescent TIR / TBR | adult TIR / TBR |
|----|----|----|----|
| PID (repo) | 61% / 15.1% | 49% / 17.1% | 55% / 0.9% |
| PID scaled to basal, tuned | 74% / 4.9% | 74% / 1.0% | 80% / 0.0% |
| Prescribed basal only | 60% / 0.0% | 38% / 0.0% | 41% / 0.0% |
| Basal-bolus, meals announced | 73% / 4.6% | 78% / 2.8% | 88% / 1.0% |
| SAC, v1 env (original setup) | 46% / 32.0% | 54% / 24.7% | 68% / 9.9% |
| SAC, v2 env | 73% / 1.9% | 67% / 0.3% | 72% / 0.0% |
| SAC, v2 env, dose cap 8x basal | 74% / 3.0% | 70% / 0.0% | 73% / 0.0% |
| TQC, v2 env | 76% / 0.3% | 64% / 0.0% | 62% / 0.0% |
| PPO, v2 env | 76% / 1.1% | 62% / 0.0% | 59% / 0.0% |
