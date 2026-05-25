"""
Numerical solver for the 1D time-independent Schrödinger equation
with configurable potentials (harmonic, double-well, anharmonic).

Uses atomic units (hbar = m = 1) and scipy.sparse.linalg.eigsh for
the ground-state eigenpair on uniform grids of configurable size.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class QuantumSolver:
    """Solve the 1D TISE on a uniform grid for configurable potentials.

    Supported potentials
    --------------------
    "harmonic"     : V(x) = 0.5 * omega^2 * x^2
    "double_well"  : V(x) = omega^2 * (x^2 - a^2)^2 / (4*a^2)   [a=2]
    "anharmonic"   : V(x) = 0.5 * omega^2 * x^2 + 0.1 * omega * x^4

    Parameters
    ----------
    L : float
        Half-width of the spatial domain [-L, L].
    n_high : int
        Number of grid points for the high-resolution solution.
    n_low : int
        Number of grid points for the low-resolution solution.
    potential : str
        One of "harmonic", "double_well", "anharmonic".
    """

    POTENTIALS = ("harmonic", "double_well", "anharmonic")

    def __init__(self, L: float = 10.0, n_high: int = 1024, n_low: int = 64,
                 potential: str = "harmonic"):
        if potential not in self.POTENTIALS:
            raise ValueError(f"Unknown potential '{potential}'. "
                             f"Choose from {self.POTENTIALS}")
        self.L = L
        self.n_high = n_high
        self.n_low = n_low
        self.potential = potential

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _potential(self, x: np.ndarray, omega: float) -> np.ndarray:
        """Evaluate V(x) for the configured potential type."""
        if self.potential == "harmonic":
            return 0.5 * omega**2 * x**2
        elif self.potential == "double_well":
            a = 2.0
            return omega**2 * (x**2 - a**2)**2 / (4.0 * a**2)
        elif self.potential == "anharmonic":
            return 0.5 * omega**2 * x**2 + 0.1 * omega * x**4
        raise ValueError(f"Unknown potential: {self.potential}")

    def _build_hamiltonian(self, N: int, omega: float):
        """Construct the sparse Hamiltonian H = T + V on N grid points.

        Kinetic energy uses the standard 3-point finite-difference stencil
        for -0.5 * d²/dx² with Dirichlet boundary conditions (psi=0 at
        domain edges).
        """
        dx = 2.0 * self.L / (N - 1)
        x = np.linspace(-self.L, self.L, N)

        diag_main = np.full(N, 1.0 / dx**2)
        diag_off = np.full(N - 1, -0.5 / dx**2)
        T = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csc")

        V = sparse.diags([self._potential(x, omega)], [0], format="csc")

        return T + V, x, dx

    def _solve(self, N: int, omega: float):
        """Return (x, psi, energy) for the ground state on an N-point grid."""
        H, x, dx = self._build_hamiltonian(N, omega)
        eigenvalue, eigenvector = eigsh(H, k=1, which="SA")
        energy = eigenvalue[0]
        psi = eigenvector[:, 0]

        # Enforce a consistent sign convention (positive lobe first)
        if psi[np.argmax(np.abs(psi))] < 0:
            psi = -psi

        # L² normalisation: ∫|ψ|² dx ≈ Σ|ψ_i|² Δx = 1
        norm = np.sqrt(np.sum(psi**2) * dx)
        psi /= norm

        return x, psi, energy

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def solve(self, omega: float):
        """Solve for a given omega, returning both HR and LR solutions.

        Returns
        -------
        dict with keys:
            omega    – the angular frequency used
            x_hr     – high-res grid (n_high,)
            psi_hr   – high-res ground-state wavefunction (n_high,)
            E_hr     – ground-state energy on the high-res grid
            x_lr     – low-res grid (n_low,)
            psi_lr   – low-res ground-state wavefunction (n_low,)
            E_lr     – ground-state energy on the low-res grid
        """
        x_hr, psi_hr, E_hr = self._solve(self.n_high, omega)
        x_lr, psi_lr, E_lr = self._solve(self.n_low, omega)
        return dict(
            omega=omega,
            x_hr=x_hr, psi_hr=psi_hr, E_hr=E_hr,
            x_lr=x_lr, psi_lr=psi_lr, E_lr=E_lr,
        )

    def solve_random_omega(self, omega_min: float = 0.5, omega_max: float = 5.0,
                           rng: np.random.Generator | None = None):
        """Sample a random omega ∈ [omega_min, omega_max] and solve."""
        if rng is None:
            rng = np.random.default_rng()
        omega = rng.uniform(omega_min, omega_max)
        return self.solve(omega)
