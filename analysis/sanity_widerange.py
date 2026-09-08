"""Task D pre-flight: points-per-width and domain-truncation sanity checks."""

from __future__ import annotations

import math
import sys

L = 10.0
N_LOW = 64
DX_LR = 2.0 * L / (N_LOW - 1)  # 20/63


def sigma_psi(omega: float) -> float:
    return 1.0 / math.sqrt(omega)


def main() -> None:
    print(f"dx_LR = {DX_LR:.8f}  (20/63 = {20.0/63.0:.8f})")
    ok = True
    for om in (0.3, 7.0):
        sig = sigma_psi(om)
        ppw = sig / DX_LR
        print(f"omega={om:.1f}:  sigma_psi={sig:.6f}  points_per_width={ppw:.4f}")
    sig03 = sigma_psi(0.3)
    five = 5.0 * sig03
    print(f"5 * sigma_psi(0.3) = {five:.6f}   L = {L}")
    if five >= L:
        print("FAIL: 5*sigma_psi is not < 10; domain truncation is unsafe.")
        ok = False
    else:
        print("PASS: 5*sigma_psi < 10 (domain truncation safety).")

    # Expected from the plan; stop if we disagree.
    exp_ppw_lo, exp_ppw_hi, exp_five = 5.75, 1.19, 9.13
    ppw_lo = sigma_psi(0.3) / DX_LR
    ppw_hi = sigma_psi(7.0) / DX_LR
    if abs(ppw_lo - exp_ppw_lo) > 0.1 or abs(ppw_hi - exp_ppw_hi) > 0.1:
        print("FAIL: points-per-width disagrees with plan reasoning "
              f"(got {ppw_lo:.3f} and {ppw_hi:.3f}, expected ~{exp_ppw_lo} and ~{exp_ppw_hi}).")
        ok = False
    if abs(five - exp_five) > 0.05:
        print(f"FAIL: 5*sigma disagrees with plan reasoning (got {five:.3f}, expected ~{exp_five}).")
        ok = False
    if not ok:
        sys.exit(1)
    print("Sanity checks match the plan. Safe to start widerange training.")


if __name__ == "__main__":
    main()
