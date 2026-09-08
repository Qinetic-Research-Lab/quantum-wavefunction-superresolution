# Sigma-space reanalysis of the canonical test set

Source: `outputs/analysis/test_metrics_canonical.csv` (200 samples). No retraining, no data generation. ω midpoint taken from `summary_canonical.json` (2.7498664701), matching the already-reported Spearman vs `|ω − ω_mid|`.

## Spearman correlation with CNN infidelity

| predictor | ρ | p |
|---|---|---|
| ω | 0.026086 | 7.138750e-01 |
| \|ω − ω_mid\| | 0.861259 | 4.040720e-60 |
| σ = 1/√ω | -0.026086 | 7.138750e-01 |
| log(ω) | 0.026086 | 7.138750e-01 |

## Spearman correlation with spline infidelity (context)

| predictor | ρ | p |
|---|---|---|
| ω | 1.000000 | 0.000000e+00 |
| \|ω − ω_mid\| | 0.031052 | 6.624753e-01 |
| σ = 1/√ω | -1.000000 | 0.000000e+00 |
| log(ω) | 1.000000 | 0.000000e+00 |

## Table 1 edge-bin CNN infidelity asymmetry

- Low bin [0.5, 1.4): mean CNN infidelity = 3.199805e-04 (n = 36; paper 3.20e-4)
- High bin [4.1, 5.0): mean CNN infidelity = 2.233262e-04 (n = 40; paper 2.23e-4)
- Ratio low/high = 1.4328 (paper ~1.4×)

The already-reported vs-ω and vs-|ω−ω_mid| CNN Spearman values are reproduced from the same CSV; any tiny difference vs the JSON would indicate a read error (none expected).
