# Oracle inversion baseline

Same canonical 200-sample test slice as `analysis/oracle_baseline.py` (last 10% of `data_cache/harmonic_n2000_omega0.5-5.0_seed42.npz`). Stored ω is used only to bin results and to compute |Δω|; it is never the inversion target.

For each 64-point input, recover ω by 1-D minimization of
`r(w) = 1 − fidelity(ψ_LR, coarse FD ground state at w)`,
where the coarse state is `QuantumSolver._solve(n_low=64, w)`.
`scipy.optimize.minimize_scalar` is called with `method='bounded'`, `bounds=(0.4, 5.1)`, `xatol=1e-08`. Bounded Brent has no `x0`; the existing second-moment estimate is clipped into the interval, evaluated first, and used as a fallback if the optimizer fails. The 1024-point FD ground state at the recovered ω is then scored against numerical `psi_hr` with `utils.quantum_fidelity`.

Optimizer failures (fell back to moment seed): 0/200
Function evaluations per sample: mean 16.2, max 30

## ω recovery vs true ω

|Δω| mean = 6.086432e-08
|Δω| median = 4.727867e-08
|Δω| max = 2.812597e-07
mean |recovered ω − moment seed| = 1.215666e-01

## Overall fidelity (n = 200)

| method | mean | median | min | max |
|---|---|---|---|---|
| inversion (FD @ recovered ω) | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Per-ω-bin (Table 1 bins)

| ω bin | n | mean F | median F | min F | max F | mean infid |
|---|---|---|---|---|---|---|
| [0.5, 1.4) | 36 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 3.083953e-17 |
| [1.4, 2.3) | 36 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000e+00 |
| [2.3, 3.2) | 46 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 4.827057e-17 |
| [3.2, 4.1) | 42 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.057355e-16 |
| [4.1, 5.0) | 40 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.110223e-17 |

`analysis/oracle_baseline.py` and all canonical seed-42 artifacts were not modified.
