# Oracle-family baseline

For each 64-point test input, ω is **fitted from the coarse wavefunction** (second-moment identity `⟨x²⟩ = 1/(2ω)` for the HO ground state; nonlinear least-squares Gaussian fallback if that estimate is unstable). The analytic ground state `ψ(x) = (ω/π)^{1/4} exp(−ω x² / 2)` is then evaluated on the 1024-point grid and scored with `utils.quantum_fidelity` against the **numerical** eigsh high-resolution state.

Stored ω values are used only to bin results, never to build the prediction.

Fit methods used: {'moment': 200}
|Δω| mean/median/max = 1.296198e-01 / 1.025376e-01 / 3.548000e-01

## Overall fidelity (n = 200)

| method | mean | median | min | max |
|---|---|---|---|---|
| oracle (fitted ω) | 0.999789 | 0.999835 | 0.999416 | 0.999995 |
| analytic @ true ω (diagnostic, not a baseline) | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Per-ω-bin (Table 1 bins)

| ω bin | n | mean F | median F | min F | max F | mean infid |
|---|---|---|---|---|---|---|
| [0.5, 1.4) | 36 | 0.999981 | 0.999985 | 0.999963 | 0.999995 | 1.903701e-05 |
| [1.4, 2.3) | 36 | 0.999936 | 0.999940 | 0.999897 | 0.999959 | 6.425050e-05 |
| [2.3, 3.2) | 46 | 0.999840 | 0.999844 | 0.999776 | 0.999884 | 1.599069e-04 |
| [3.2, 4.1) | 42 | 0.999709 | 0.999708 | 0.999621 | 0.999774 | 2.913372e-04 |
| [4.1, 5.0) | 40 | 0.999511 | 0.999499 | 0.999416 | 0.999617 | 4.885413e-04 |

CNN/spline Table 1 numbers are unchanged; see `outputs/analysis/summary_canonical.json`.
