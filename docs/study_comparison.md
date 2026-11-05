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
| SAC, v2 env | 5 | 71.1 ± 2.0 | 0.9 ± 0.6 | 0.0 | 32 | 87% | 5/5 | TIR +16.6 (p=0.000)*; TBR -10.3 (p=0.000)* | TIR -4.9 (p=0.002)*; TBR -1.2 (p=0.155) |
|   ↳ ensemble of 5 seeds (median dose) | - | 71.7 | 1.0 | 0.0 | 32 | 90% | - | TIR +17.2 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -4.4 (p=0.004)*; TBR -1.0 (p=0.161) |
| SAC, v2 env, dose cap 8x basal | 5 | 72.3 ± 1.4 | 1.0 ± 0.6 | 0.2 | 31 | 89% | 5/5 | TIR +17.8 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -3.7 (p=0.007)*; TBR -1.1 (p=0.123) |
|   ↳ ensemble of 5 seeds (median dose) | - | 71.8 | 0.9 | 0.3 | 31 | 93% | - | TIR +17.3 (p=0.000)*; TBR -10.4 (p=0.001)* | TIR -4.2 (p=0.007)*; TBR -1.2 (p=0.310) |
| TQC, v2 env | 3 | 67.9 ± 1.0 | 0.5 ± 0.6 | 0.0 | 35 | 84% | 3/3 | TIR +13.4 (p=0.000)*; TBR -10.8 (p=0.000)* | TIR -8.2 (p=0.000)*; TBR -1.6 (p=0.051) |
|   ↳ ensemble of 3 seeds (median dose) | - | 68.6 | 0.0 | 0.0 | 33 | 83% | - | TIR +14.1 (p=0.000)*; TBR -11.2 (p=0.000)* | TIR -7.4 (p=0.001)*; TBR -2.0 (p=0.028)* |
| PPO, v2 env | 3 | 67.2 ± 2.4 | 0.5 ± 0.3 | 0.0 | 36 | 83% | 3/3 | TIR +12.6 (p=0.000)*; TBR -10.8 (p=0.000)* | TIR -8.9 (p=0.000)*; TBR -1.6 (p=0.028)* |
|   ↳ ensemble of 3 seeds (median dose) | - | 69.4 | 0.5 | 0.0 | 34 | 83% | - | TIR +14.9 (p=0.000)*; TBR -10.8 (p=0.000)* | TIR -6.6 (p=0.001)*; TBR -1.6 (p=0.028)* |
| RecurrentPPO, v2 env | 2 | 65.9 ± 8.5 | 2.5 ± 2.1 | 0.6 | 42 | 87% | 1/2 | TIR +11.3 (p=0.000)*; TBR -8.7 (p=0.000)* | TIR -10.2 (p=0.000)*; TBR +0.4 (p=0.532) |
| Residual SAC on tuned PID (scale 0.5) | 5 | 76.0 ± 1.2 | 1.1 ± 0.3 | 0.4 | 28 | 91% | 5/5 | TIR +21.5 (p=0.000)*; TBR -10.1 (p=0.000)* | TIR -0.1 (p=0.484); TBR -0.9 (p=0.263) |
|   ↳ ensemble of 5 seeds (median dose) | - | 76.1 | 1.0 | 0.3 | 28 | 90% | - | TIR +21.5 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR +0.0 (p=0.353); TBR -1.0 (p=0.116) |
| Residual SAC on tuned PID (scale 1.0) | 3 | 74.7 ± 1.2 | 0.8 ± 0.1 | 0.3 | 29 | 90% | 3/3 | TIR +20.1 (p=0.000)*; TBR -10.4 (p=0.000)* | TIR -1.4 (p=0.041)*; TBR -1.2 (p=0.075) |
|   ↳ ensemble of 3 seeds (median dose) | - | 75.2 | 0.9 | 0.2 | 28 | 90% | - | TIR +20.7 (p=0.000)*; TBR -10.4 (p=0.000)* | TIR -0.9 (p=0.063); TBR -1.2 (p=0.075) |
| Residual SAC on tuned PID, zone reward | 4 | 77.5 ± 0.8 | 2.5 ± 0.5 | 0.8 | 29 | 92% | 4/4 | TIR +23.0 (p=0.000)*; TBR -8.8 (p=0.001)* | TIR +1.4 (p=0.001)*; TBR +0.4 (p=0.333) |
|   ↳ ensemble of 4 seeds (median dose) | - | 77.8 | 2.2 | 0.9 | 28 | 93% | - | TIR +23.3 (p=0.000)*; TBR -9.0 (p=0.001)* | TIR +1.7 (p=0.000)*; TBR +0.2 (p=0.735) |
| Residual SAC on tuned PID, GRI reward | 5 | 77.7 ± 0.6 | 1.5 ± 0.8 | 0.3 | 27 | 91% | 5/5 | TIR +23.2 (p=0.000)*; TBR -9.7 (p=0.000)* | TIR +1.7 (p=0.027)*; TBR -0.6 (p=0.657) |
|   ↳ ensemble of 5 seeds (median dose) | - | 78.4 | 1.1 | 0.2 | 26 | 90% | - | TIR +23.9 (p=0.000)*; TBR -10.1 (p=0.000)* | TIR +2.4 (p=0.008)*; TBR -1.0 (p=0.128) |
| Residual PPO on tuned PID (scale 0.5) | 3 | 72.4 ± 1.1 | 1.1 ± 0.2 | 0.2 | 32 | 84% | 3/3 | TIR +17.9 (p=0.000)*; TBR -10.1 (p=0.000)* | TIR -3.7 (p=0.002)*; TBR -1.0 (p=0.028)* |
|   ↳ ensemble of 3 seeds (median dose) | - | 72.2 | 1.2 | 0.2 | 32 | 83% | - | TIR +17.7 (p=0.000)*; TBR -10.0 (p=0.000)* | TIR -3.9 (p=0.001)*; TBR -0.8 (p=0.028)* |
| Residual SAC on basal-bolus, meals announced* | 4 | 72.0 ± 5.8 | 1.2 ± 0.9 | 0.3 | 32 | 94% | 4/4 | TIR +17.4 (p=0.000)*; TBR -10.0 (p=0.000)* | TIR -4.1 (p=0.032)*; TBR -0.8 (p=0.422) |
|   ↳ ensemble of 4 seeds (median dose) | - | 69.9 | 0.5 | 0.1 | 33 | 93% | - | TIR +15.4 (p=0.000)*; TBR -10.7 (p=0.000)* | TIR -6.1 (p=0.008)*; TBR -1.5 (p=0.176) |
| Residual SAC on basal-bolus, zone reward, meals announced* | 5 | 78.3 ± 3.3 | 3.0 ± 1.7 | 1.0 | 29 | 95% | 2/5 | TIR +23.8 (p=0.000)*; TBR -8.2 (p=0.002)* | TIR +2.3 (p=0.318); TBR +1.0 (p=0.492) |
|   ↳ ensemble of 5 seeds (median dose) | - | 79.5 | 3.6 | 0.9 | 29 | 97% | - | TIR +25.0 (p=0.000)*; TBR -7.7 (p=0.004)* | TIR +3.5 (p=0.177); TBR +1.5 (p=0.382) |
| Residual SAC on basal-bolus, GRI reward, meals announced* | 3 | 78.0 ± 1.0 | 3.5 ± 1.3 | 1.0 | 30 | 94% | 1/3 | TIR +23.5 (p=0.000)*; TBR -7.8 (p=0.002)* | TIR +1.9 (p=0.465); TBR +1.4 (p=0.246) |
|   ↳ ensemble of 3 seeds (median dose) | - | 78.8 | 3.1 | 1.2 | 28 | 93% | - | TIR +24.3 (p=0.000)*; TBR -8.2 (p=0.004)* | TIR +2.7 (p=0.328); TBR +1.0 (p=0.753) |

**By patient cohort** (no shield)

| Controller | child TIR / TBR | adolescent TIR / TBR | adult TIR / TBR |
|----|----|----|----|
| PID (repo) | 60% / 14.7% | 48% / 18.1% | 55% / 0.9% |
| PID scaled to basal, tuned | 74% / 5.0% | 73% / 1.2% | 80% / 0.0% |
| Prescribed basal only | 60% / 0.0% | 38% / 0.0% | 41% / 0.0% |
| Basal-bolus, meals announced | 68% / 10.9% | 81% / 3.0% | 90% / 3.0% |
| SAC, v1 env (original setup) | 53% / 30.7% | 50% / 28.6% | 58% / 17.5% |
| SAC, v1 env (original setup): ensemble | 52% / 32.4% | 47% / 34.7% | 61% / 18.3% |
| SAC, v2 env | 73% / 2.5% | 68% / 0.2% | 72% / 0.0% |
| SAC, v2 env: ensemble | 71% / 3.1% | 70% / 0.0% | 74% / 0.0% |
| SAC, v2 env, dose cap 8x basal | 75% / 3.0% | 70% / 0.0% | 72% / 0.0% |
| SAC, v2 env, dose cap 8x basal: ensemble | 74% / 2.6% | 70% / 0.0% | 72% / 0.0% |
| TQC, v2 env | 75% / 0.9% | 65% / 0.5% | 64% / 0.0% |
| TQC, v2 env: ensemble | 76% / 0.1% | 66% / 0.0% | 64% / 0.0% |
| PPO, v2 env | 75% / 1.4% | 65% / 0.0% | 62% / 0.0% |
| PPO, v2 env: ensemble | 77% / 1.4% | 67% / 0.0% | 64% / 0.0% |
| RecurrentPPO, v2 env | 69% / 4.8% | 62% / 2.6% | 67% / 0.1% |
| Residual SAC on tuned PID (scale 0.5) | 75% / 3.4% | 74% / 0.0% | 78% / 0.0% |
| Residual SAC on tuned PID (scale 0.5): ensemble | 76% / 3.1% | 74% / 0.0% | 78% / 0.0% |
| Residual SAC on tuned PID (scale 1.0) | 76% / 2.5% | 73% / 0.0% | 75% / 0.0% |
| Residual SAC on tuned PID (scale 1.0): ensemble | 77% / 2.6% | 74% / 0.0% | 75% / 0.0% |
| Residual SAC on tuned PID, zone reward | 72% / 6.8% | 76% / 0.6% | 85% / 0.0% |
| Residual SAC on tuned PID, zone reward: ensemble | 71% / 6.8% | 77% / 0.0% | 85% / 0.0% |
| Residual SAC on tuned PID, GRI reward | 74% / 4.1% | 76% / 0.3% | 83% / 0.1% |
| Residual SAC on tuned PID, GRI reward: ensemble | 76% / 3.3% | 76% / 0.0% | 83% / 0.0% |
| Residual PPO on tuned PID (scale 0.5) | 75% / 2.9% | 72% / 0.4% | 70% / 0.0% |
| Residual PPO on tuned PID (scale 0.5): ensemble | 75% / 3.2% | 71% / 0.4% | 70% / 0.0% |
| Residual SAC on basal-bolus, meals announced | 73% / 1.5% | 65% / 2.0% | 77% / 0.3% |
| Residual SAC on basal-bolus, meals announced: ensemble | 75% / 0.0% | 61% / 1.6% | 74% / 0.0% |
| Residual SAC on basal-bolus, zone reward, meals announced | 73% / 4.6% | 75% / 3.0% | 87% / 1.6% |
| Residual SAC on basal-bolus, zone reward, meals announced: ensemble | 74% / 4.2% | 78% / 3.2% | 87% / 3.3% |
| Residual SAC on basal-bolus, GRI reward, meals announced | 73% / 4.9% | 75% / 3.4% | 86% / 2.1% |
| Residual SAC on basal-bolus, GRI reward, meals announced: ensemble | 75% / 3.5% | 76% / 3.3% | 86% / 2.4% |

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
| SAC, v2 env | 5 | 70.8 ± 2.1 | 0.8 ± 0.6 | 0.0 | 33 | 87% | 5/5 | TIR +15.8 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -5.3 (p=0.001)*; TBR -1.1 (p=0.155) |
|   ↳ ensemble of 5 seeds (median dose) | - | 71.4 | 1.0 | 0.0 | 32 | 90% | - | TIR +16.4 (p=0.000)*; TBR -10.1 (p=0.000)* | TIR -4.7 (p=0.004)*; TBR -1.0 (p=0.161) |
| SAC, v2 env, dose cap 8x basal | 5 | 72.0 ± 1.3 | 0.9 ± 0.6 | 0.3 | 31 | 89% | 5/5 | TIR +17.0 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -4.0 (p=0.003)*; TBR -1.1 (p=0.093) |
|   ↳ ensemble of 5 seeds (median dose) | - | 71.7 | 0.8 | 0.3 | 31 | 93% | - | TIR +16.7 (p=0.000)*; TBR -10.3 (p=0.001)* | TIR -4.3 (p=0.005)*; TBR -1.2 (p=0.236) |
| TQC, v2 env | 3 | 67.9 ± 1.1 | 0.3 ± 0.3 | 0.0 | 35 | 84% | 3/3 | TIR +12.9 (p=0.000)*; TBR -10.7 (p=0.000)* | TIR -8.2 (p=0.000)*; TBR -1.7 (p=0.028)* |
|   ↳ ensemble of 3 seeds (median dose) | - | 68.6 | 0.0 | 0.0 | 33 | 83% | - | TIR +13.6 (p=0.000)*; TBR -11.0 (p=0.000)* | TIR -7.5 (p=0.001)*; TBR -1.9 (p=0.028)* |
| PPO, v2 env | 3 | 67.1 ± 2.4 | 0.5 ± 0.3 | 0.0 | 37 | 83% | 3/3 | TIR +12.1 (p=0.000)*; TBR -10.5 (p=0.000)* | TIR -9.0 (p=0.000)*; TBR -1.5 (p=0.028)* |
|   ↳ ensemble of 3 seeds (median dose) | - | 69.4 | 0.5 | 0.0 | 34 | 83% | - | TIR +14.4 (p=0.000)*; TBR -10.5 (p=0.000)* | TIR -6.7 (p=0.000)*; TBR -1.5 (p=0.028)* |
| RecurrentPPO, v2 env | 2 | 65.5 ± 8.3 | 2.4 ± 1.9 | 0.6 | 43 | 87% | 1/2 | TIR +10.5 (p=0.000)*; TBR -8.7 (p=0.000)* | TIR -10.6 (p=0.000)*; TBR +0.4 (p=0.570) |
| Residual SAC on tuned PID (scale 0.5) | 5 | 76.0 ± 1.2 | 1.1 ± 0.3 | 0.4 | 28 | 91% | 5/5 | TIR +21.0 (p=0.000)*; TBR -9.9 (p=0.000)* | TIR -0.1 (p=0.434); TBR -0.8 (p=0.314) |
|   ↳ ensemble of 5 seeds (median dose) | - | 76.0 | 1.0 | 0.3 | 28 | 90% | - | TIR +21.0 (p=0.000)*; TBR -10.0 (p=0.000)* | TIR -0.0 (p=0.354); TBR -0.9 (p=0.116) |
| Residual SAC on tuned PID (scale 1.0) | 3 | 74.6 ± 1.2 | 0.8 ± 0.2 | 0.2 | 29 | 90% | 3/3 | TIR +19.6 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -1.5 (p=0.039)*; TBR -1.2 (p=0.075) |
|   ↳ ensemble of 3 seeds (median dose) | - | 75.2 | 0.9 | 0.2 | 28 | 90% | - | TIR +20.2 (p=0.000)*; TBR -10.2 (p=0.000)* | TIR -0.9 (p=0.063); TBR -1.1 (p=0.075) |
| Residual SAC on tuned PID, zone reward | 4 | 77.4 ± 0.8 | 2.5 ± 0.4 | 0.8 | 29 | 92% | 4/4 | TIR +22.4 (p=0.000)*; TBR -8.5 (p=0.001)* | TIR +1.3 (p=0.003)*; TBR +0.5 (p=0.333) |
|   ↳ ensemble of 4 seeds (median dose) | - | 77.7 | 2.3 | 0.9 | 29 | 93% | - | TIR +22.7 (p=0.000)*; TBR -8.7 (p=0.002)* | TIR +1.6 (p=0.001)*; TBR +0.3 (p=0.678) |
| Residual SAC on tuned PID, GRI reward | 5 | 77.7 ± 0.7 | 1.5 ± 0.8 | 0.3 | 27 | 90% | 5/5 | TIR +22.7 (p=0.000)*; TBR -9.5 (p=0.000)* | TIR +1.7 (p=0.019)*; TBR -0.5 (p=0.657) |
|   ↳ ensemble of 5 seeds (median dose) | - | 78.6 | 1.2 | 0.2 | 26 | 87% | - | TIR +23.5 (p=0.000)*; TBR -9.8 (p=0.000)* | TIR +2.5 (p=0.004)*; TBR -0.8 (p=0.236) |
| Residual PPO on tuned PID (scale 0.5) | 3 | 72.4 ± 1.1 | 1.1 ± 0.2 | 0.2 | 32 | 83% | 3/3 | TIR +17.3 (p=0.000)*; TBR -9.9 (p=0.000)* | TIR -3.7 (p=0.002)*; TBR -0.8 (p=0.043)* |
|   ↳ ensemble of 3 seeds (median dose) | - | 72.2 | 1.2 | 0.2 | 32 | 83% | - | TIR +17.1 (p=0.000)*; TBR -9.8 (p=0.000)* | TIR -3.9 (p=0.001)*; TBR -0.8 (p=0.028)* |
| Residual SAC on basal-bolus, meals announced* | 4 | 71.2 ± 6.0 | 1.0 ± 0.7 | 0.3 | 33 | 92% | 4/4 | TIR +16.2 (p=0.000)*; TBR -10.0 (p=0.000)* | TIR -4.9 (p=0.026)*; TBR -1.0 (p=0.382) |
|   ↳ ensemble of 4 seeds (median dose) | - | 68.7 | 0.4 | 0.0 | 34 | 93% | - | TIR +13.7 (p=0.000)*; TBR -10.6 (p=0.000)* | TIR -7.4 (p=0.003)*; TBR -1.5 (p=0.176) |
| Residual SAC on basal-bolus, zone reward, meals announced* | 5 | 76.2 ± 3.1 | 2.4 ± 1.3 | 0.9 | 31 | 93% | 5/5 | TIR +21.2 (p=0.000)*; TBR -8.6 (p=0.003)* | TIR +0.1 (p=0.808); TBR +0.4 (p=0.831) |
|   ↳ ensemble of 5 seeds (median dose) | - | 79.0 | 2.2 | 0.8 | 27 | 93% | - | TIR +24.0 (p=0.000)*; TBR -8.8 (p=0.002)* | TIR +2.9 (p=0.213); TBR +0.2 (p=0.925) |
| Residual SAC on basal-bolus, GRI reward, meals announced* | 3 | 76.5 ± 1.7 | 2.5 ± 0.9 | 1.0 | 31 | 92% | 3/3 | TIR +21.5 (p=0.000)*; TBR -8.5 (p=0.002)* | TIR +0.4 (p=0.781); TBR +0.5 (p=0.831) |
|   ↳ ensemble of 3 seeds (median dose) | - | 78.6 | 1.9 | 0.8 | 28 | 90% | - | TIR +23.6 (p=0.000)*; TBR -9.2 (p=0.001)* | TIR +2.5 (p=0.371); TBR -0.1 (p=0.583) |

**By patient cohort** (shielded)

| Controller | child TIR / TBR | adolescent TIR / TBR | adult TIR / TBR |
|----|----|----|----|
| PID (repo) | 61% / 15.1% | 49% / 17.1% | 55% / 0.9% |
| PID scaled to basal, tuned | 74% / 4.9% | 74% / 1.0% | 80% / 0.0% |
| Prescribed basal only | 60% / 0.0% | 38% / 0.0% | 41% / 0.0% |
| Basal-bolus, meals announced | 73% / 4.6% | 78% / 2.8% | 88% / 1.0% |
| SAC, v1 env (original setup) | 50% / 28.2% | 58% / 18.8% | 65% / 7.4% |
| SAC, v1 env (original setup): ensemble | 51% / 27.6% | 58% / 18.7% | 67% / 7.1% |
| SAC, v2 env | 73% / 2.3% | 68% / 0.2% | 72% / 0.0% |
| SAC, v2 env: ensemble | 71% / 2.9% | 70% / 0.0% | 74% / 0.0% |
| SAC, v2 env, dose cap 8x basal | 74% / 2.6% | 70% / 0.0% | 72% / 0.0% |
| SAC, v2 env, dose cap 8x basal: ensemble | 74% / 2.3% | 70% / 0.0% | 71% / 0.0% |
| TQC, v2 env | 74% / 0.9% | 65% / 0.0% | 64% / 0.0% |
| TQC, v2 env: ensemble | 75% / 0.1% | 66% / 0.0% | 64% / 0.0% |
| PPO, v2 env | 75% / 1.4% | 65% / 0.0% | 62% / 0.0% |
| PPO, v2 env: ensemble | 77% / 1.5% | 67% / 0.0% | 64% / 0.0% |
| RecurrentPPO, v2 env | 68% / 4.7% | 62% / 2.3% | 66% / 0.1% |
| Residual SAC on tuned PID (scale 0.5) | 75% / 3.4% | 74% / 0.0% | 78% / 0.0% |
| Residual SAC on tuned PID (scale 0.5): ensemble | 76% / 3.1% | 74% / 0.0% | 78% / 0.0% |
| Residual SAC on tuned PID (scale 1.0) | 76% / 2.5% | 73% / 0.0% | 75% / 0.0% |
| Residual SAC on tuned PID (scale 1.0): ensemble | 77% / 2.6% | 74% / 0.0% | 75% / 0.0% |
| Residual SAC on tuned PID, zone reward | 71% / 7.0% | 76% / 0.5% | 85% / 0.0% |
| Residual SAC on tuned PID, zone reward: ensemble | 71% / 7.0% | 77% / 0.0% | 85% / 0.0% |
| Residual SAC on tuned PID, GRI reward | 75% / 4.2% | 75% / 0.3% | 83% / 0.1% |
| Residual SAC on tuned PID, GRI reward: ensemble | 76% / 3.5% | 76% / 0.0% | 83% / 0.0% |
| Residual PPO on tuned PID (scale 0.5) | 75% / 3.1% | 72% / 0.3% | 70% / 0.0% |
| Residual PPO on tuned PID (scale 0.5): ensemble | 75% / 3.2% | 71% / 0.4% | 70% / 0.0% |
| Residual SAC on basal-bolus, meals announced | 73% / 1.1% | 65% / 1.8% | 76% / 0.1% |
| Residual SAC on basal-bolus, meals announced: ensemble | 73% / 0.0% | 61% / 1.3% | 72% / 0.0% |
| Residual SAC on basal-bolus, zone reward, meals announced | 72% / 3.1% | 72% / 2.9% | 84% / 1.2% |
| Residual SAC on basal-bolus, zone reward, meals announced: ensemble | 78% / 1.4% | 75% / 3.0% | 84% / 2.2% |
| Residual SAC on basal-bolus, GRI reward, meals announced | 73% / 3.0% | 72% / 3.3% | 85% / 1.3% |
| Residual SAC on basal-bolus, GRI reward, meals announced: ensemble | 77% / 1.9% | 72% / 3.1% | 86% / 0.5% |
