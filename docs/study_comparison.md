**Held-out test patients (6 patients x 5 days), no shield**

Paired over the same patient-days (agents averaged over seeds); p from a Wilcoxon signed-rank test, * = p < 0.05.

| Controller | Seeds | TIR 70-180 % ↑ | TBR <70 % ↓ | TBR <54 % ↓ | GRI ↓ | Survived | Seeds with TBR <4% | vs PID (repo): mean paired diff | vs tuned basal PID: mean paired diff |
|----|----|----|----|----|----|----|----|----|----|
| PID (repo) | - | 54.5 | 11.2 | 8.0 | 68 | 73% | - | - | - |
| PID scaled to basal, tuned | - | 76.1 | 2.1 | 0.3 | 30 | 90% | - | - | - |
| Prescribed basal only | - | 46.3 | 0.0 | 0.0 | 69 | 70% | - | - | - |
| Basal-bolus, meals announced* | - | 79.7 | 5.7 | 1.7 | 32 | 97% | - | - | - |
| SAC, v1 env (original setup) | 5 | 53.8 ± 1.3 | 25.6 ± 7.2 | 19.4 | 79 | 38% | 0/5 | TIR -0.7 (p=0.919); TBR +14.4 (p=0.000)* | TIR -22.3 (p=0.000)*; TBR +23.5 (p=0.000)* |
|   ↳ ensemble of 5 seeds (median dose) | - | 53.2 | 28.5 | 21.1 | 78 | 37% | - | TIR -1.3 (p=0.935); TBR +17.2 (p=0.000)* | TIR -22.9 (p=0.000)*; TBR +26.4 (p=0.000)* |
| SAC, v2 env | 4 | 71.6 ± 2.0 | 1.0 ± 0.6 | 0.0 | 32 | 88% | 4/4 | TIR +17.0 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -4.5 (p=0.002)*; TBR -1.0 (p=0.248) |
|   ↳ ensemble of 4 seeds (median dose) | - | 72.1 | 0.9 | 0.0 | 31 | 90% | - | TIR +17.6 (p=0.000)*; TBR -10.3 (p=0.000)* | TIR -4.0 (p=0.006)*; TBR -1.1 (p=0.161) |
| SAC, v2 env, dose cap 8x basal | 4 | 72.2 ± 1.6 | 1.0 ± 0.6 | 0.3 | 31 | 89% | 4/4 | TIR +17.7 (p=0.000)*; TBR -10.3 (p=0.000)* | TIR -3.8 (p=0.012)*; TBR -1.1 (p=0.161) |
|   ↳ ensemble of 4 seeds (median dose) | - | 71.6 | 0.7 | 0.1 | 31 | 93% | - | TIR +17.0 (p=0.000)*; TBR -10.5 (p=0.000)* | TIR -4.5 (p=0.009)*; TBR -1.3 (p=0.128) |
| TQC, v2 env | 3 | 67.9 ± 1.0 | 0.5 ± 0.6 | 0.0 | 35 | 84% | 3/3 | TIR +13.4 (p=0.000)*; TBR -10.8 (p=0.000)* | TIR -8.2 (p=0.000)*; TBR -1.6 (p=0.051) |
|   ↳ ensemble of 3 seeds (median dose) | - | 68.6 | 0.0 | 0.0 | 33 | 83% | - | TIR +14.1 (p=0.000)*; TBR -11.2 (p=0.000)* | TIR -7.4 (p=0.001)*; TBR -2.0 (p=0.028)* |
| PPO, v2 env | 3 | 67.2 ± 2.4 | 0.5 ± 0.3 | 0.0 | 36 | 83% | 3/3 | TIR +12.6 (p=0.000)*; TBR -10.8 (p=0.000)* | TIR -8.9 (p=0.000)*; TBR -1.6 (p=0.028)* |
|   ↳ ensemble of 3 seeds (median dose) | - | 69.4 | 0.5 | 0.0 | 34 | 83% | - | TIR +14.9 (p=0.000)*; TBR -10.8 (p=0.000)* | TIR -6.6 (p=0.001)*; TBR -1.6 (p=0.028)* |
| RecurrentPPO, v2 env | 2 | 65.9 ± 8.5 | 2.5 ± 2.1 | 0.6 | 42 | 87% | 1/2 | TIR +11.3 (p=0.000)*; TBR -8.7 (p=0.000)* | TIR -10.2 (p=0.000)*; TBR +0.4 (p=0.532) |

**By patient cohort** (no shield)

| Controller | child TIR / TBR | adolescent TIR / TBR | adult TIR / TBR |
|----|----|----|----|
| PID (repo) | 60% / 14.7% | 48% / 18.1% | 55% / 0.9% |
| PID scaled to basal, tuned | 74% / 5.0% | 73% / 1.2% | 80% / 0.0% |
| Prescribed basal only | 60% / 0.0% | 38% / 0.0% | 41% / 0.0% |
| Basal-bolus, meals announced | 68% / 10.9% | 81% / 3.0% | 90% / 3.0% |
| SAC, v1 env (original setup) | 53% / 30.7% | 50% / 28.6% | 58% / 17.5% |
| SAC, v1 env (original setup): ensemble | 52% / 32.4% | 47% / 34.7% | 61% / 18.3% |
| SAC, v2 env | 72% / 2.8% | 69% / 0.3% | 73% / 0.0% |
| SAC, v2 env: ensemble | 71% / 2.7% | 71% / 0.0% | 75% / 0.0% |
| SAC, v2 env, dose cap 8x basal | 75% / 2.9% | 70% / 0.0% | 72% / 0.0% |
| SAC, v2 env, dose cap 8x basal: ensemble | 74% / 2.1% | 70% / 0.0% | 71% / 0.0% |
| TQC, v2 env | 75% / 0.9% | 65% / 0.5% | 64% / 0.0% |
| TQC, v2 env: ensemble | 76% / 0.1% | 66% / 0.0% | 64% / 0.0% |
| PPO, v2 env | 75% / 1.4% | 65% / 0.0% | 62% / 0.0% |
| PPO, v2 env: ensemble | 77% / 1.4% | 67% / 0.0% | 64% / 0.0% |
| RecurrentPPO, v2 env | 69% / 4.8% | 62% / 2.6% | 67% / 0.1% |

**Held-out test patients (6 patients x 5 days), every controller behind the safety shield**

Paired over the same patient-days (agents averaged over seeds); p from a Wilcoxon signed-rank test, * = p < 0.05.

| Controller | Seeds | TIR 70-180 % ↑ | TBR <70 % ↓ | TBR <54 % ↓ | GRI ↓ | Survived | Seeds with TBR <4% | vs PID (repo): mean paired diff | vs tuned basal PID: mean paired diff |
|----|----|----|----|----|----|----|----|----|----|
| PID (repo) | - | 55.0 | 11.0 | 7.4 | 67 | 80% | - | - | - |
| PID scaled to basal, tuned | - | 76.1 | 2.0 | 0.3 | 29 | 90% | - | - | - |
| Prescribed basal only | - | 46.2 | 0.0 | 0.0 | 69 | 70% | - | - | - |
| Basal-bolus, meals announced* | - | 79.6 | 2.8 | 0.9 | 28 | 93% | - | - | - |
| SAC, v1 env (original setup) | 5 | 57.5 ± 2.4 | 18.1 ± 6.1 | 10.2 | 70 | 71% | 0/5 | TIR +2.5 (p=0.114); TBR +7.1 (p=0.002)* | TIR -18.6 (p=0.000)*; TBR +16.1 (p=0.000)* |
|   ↳ ensemble of 5 seeds (median dose) | - | 58.8 | 17.8 | 9.4 | 70 | 67% | - | TIR +3.8 (p=0.055); TBR +6.8 (p=0.004)* | TIR -17.3 (p=0.000)*; TBR +15.8 (p=0.000)* |
| SAC, v2 env | 4 | 71.3 ± 2.0 | 0.9 ± 0.6 | 0.0 | 32 | 88% | 4/4 | TIR +16.3 (p=0.000)*; TBR -10.1 (p=0.000)* | TIR -4.7 (p=0.001)*; TBR -1.0 (p=0.248) |
|   ↳ ensemble of 4 seeds (median dose) | - | 72.0 | 0.9 | 0.0 | 32 | 90% | - | TIR +17.0 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -4.1 (p=0.006)*; TBR -1.1 (p=0.161) |
| SAC, v2 env, dose cap 8x basal | 4 | 71.9 ± 1.5 | 0.9 ± 0.7 | 0.3 | 31 | 89% | 4/4 | TIR +16.9 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -4.2 (p=0.005)*; TBR -1.1 (p=0.123) |
|   ↳ ensemble of 4 seeds (median dose) | - | 71.6 | 0.6 | 0.1 | 31 | 90% | - | TIR +16.6 (p=0.000)*; TBR -10.4 (p=0.000)* | TIR -4.5 (p=0.014)*; TBR -1.3 (p=0.128) |
| TQC, v2 env | 3 | 67.9 ± 1.1 | 0.3 ± 0.3 | 0.0 | 35 | 84% | 3/3 | TIR +12.9 (p=0.000)*; TBR -10.7 (p=0.000)* | TIR -8.2 (p=0.000)*; TBR -1.7 (p=0.028)* |
|   ↳ ensemble of 3 seeds (median dose) | - | 68.6 | 0.0 | 0.0 | 33 | 83% | - | TIR +13.6 (p=0.000)*; TBR -11.0 (p=0.000)* | TIR -7.5 (p=0.001)*; TBR -1.9 (p=0.028)* |
| PPO, v2 env | 3 | 67.1 ± 2.4 | 0.5 ± 0.3 | 0.0 | 37 | 83% | 3/3 | TIR +12.1 (p=0.000)*; TBR -10.5 (p=0.000)* | TIR -9.0 (p=0.000)*; TBR -1.5 (p=0.028)* |
|   ↳ ensemble of 3 seeds (median dose) | - | 69.4 | 0.5 | 0.0 | 34 | 83% | - | TIR +14.4 (p=0.000)*; TBR -10.5 (p=0.000)* | TIR -6.7 (p=0.000)*; TBR -1.5 (p=0.028)* |
| RecurrentPPO, v2 env | 2 | 65.5 ± 8.3 | 2.4 ± 1.9 | 0.6 | 43 | 87% | 1/2 | TIR +10.5 (p=0.000)*; TBR -8.7 (p=0.000)* | TIR -10.6 (p=0.000)*; TBR +0.4 (p=0.570) |

**By patient cohort** (shielded)

| Controller | child TIR / TBR | adolescent TIR / TBR | adult TIR / TBR |
|----|----|----|----|
| PID (repo) | 61% / 15.1% | 49% / 17.1% | 55% / 0.9% |
| PID scaled to basal, tuned | 74% / 4.9% | 74% / 1.0% | 80% / 0.0% |
| Prescribed basal only | 60% / 0.0% | 38% / 0.0% | 41% / 0.0% |
| Basal-bolus, meals announced | 73% / 4.6% | 78% / 2.8% | 88% / 1.0% |
| SAC, v1 env (original setup) | 50% / 28.2% | 58% / 18.8% | 65% / 7.4% |
| SAC, v1 env (original setup): ensemble | 51% / 27.6% | 58% / 18.7% | 67% / 7.1% |
| SAC, v2 env | 72% / 2.6% | 69% / 0.2% | 73% / 0.0% |
| SAC, v2 env: ensemble | 71% / 2.6% | 71% / 0.0% | 74% / 0.0% |
| SAC, v2 env, dose cap 8x basal | 74% / 2.6% | 70% / 0.0% | 72% / 0.0% |
| SAC, v2 env, dose cap 8x basal: ensemble | 75% / 1.9% | 70% / 0.0% | 70% / 0.0% |
| TQC, v2 env | 74% / 0.9% | 65% / 0.0% | 64% / 0.0% |
| TQC, v2 env: ensemble | 75% / 0.1% | 66% / 0.0% | 64% / 0.0% |
| PPO, v2 env | 75% / 1.4% | 65% / 0.0% | 62% / 0.0% |
| PPO, v2 env: ensemble | 77% / 1.5% | 67% / 0.0% | 64% / 0.0% |
| RecurrentPPO, v2 env | 68% / 4.7% | 62% / 2.3% | 66% / 0.1% |
