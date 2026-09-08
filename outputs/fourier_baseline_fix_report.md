# Fourier baseline grid-handling fix

Canonical seed-42 CNN/spline numbers were **not** rewritten.
Test set: last 200 samples of the cached n=2000 run (`test_metrics_canonical.csv` omegas matched).

## Hypothesis

The 64-point array lives on `linspace(-10.0, 10.0, 64)` with `dx = 20/63` (endpoint-inclusive). `np.fft.rfft`/`irfft` to 1024 points interpolates in **index space**, equivalent to treating the samples as period `64` rather than `2L`, and maps LR index 32 (x = 0.158730) onto HR index 512 (x = 0.009775).

**Predicted rigid shift at domain center: 0.148955 length units** (prompt: ~0.15). **Confirmed.**

Fix: DFT on the unique periodic points `psi[:-1]` (63 samples, period 2L), zero-pad 63 → 1023, restore the endpoint so the interpolant sits on the solver's 1024-point inclusive grid. This remains a Fourier zero-padding baseline; DST was not substituted.

## Peak-location check (test sample 0)

- HR peak at x = -0.009775
- Old FFT peak at x = -0.146628 (shift -0.136852)
- Fixed FFT peak at x = 0.009775 (shift +0.019550)

## Overall fidelity (n = 200)

| method | mean | median | min | max | std |
|---|---|---|---|---|---|
| old FFT (index-space rFFT) | 0.969008 | 0.969256 | 0.944092 | 0.994355 | 0.014494 |
| fixed FFT (endpoint-corrected) | 0.999770 | 0.999821 | 0.999361 | 0.999995 | 0.000187 |

## Per-ω-bin mean fidelity (Table 1 bins)

| ω bin | n | old FFT F | fixed FFT F | old infid | fixed infid |
|---|---|---|---|---|---|
| [0.5, 1.4) | 36 | 0.989648 | 0.999979 | 1.035193e-02 | 2.065835e-05 |
| [1.4, 2.3) | 36 | 0.980681 | 0.999930 | 1.931888e-02 | 6.981217e-05 |
| [2.3, 3.2) | 46 | 0.969822 | 0.999826 | 3.017819e-02 | 1.740732e-04 |
| [3.2, 4.1) | 42 | 0.959795 | 0.999682 | 4.020473e-02 | 3.177805e-04 |
| [4.1, 5.0) | 40 | 0.948664 | 0.999466 | 5.133612e-02 | 5.341040e-04 |

CNN and spline columns from the canonical run were not recomputed or overwritten.
