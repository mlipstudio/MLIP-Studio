"""Analytical-Hessian-preconditioned L-BFGS optimizer.

This optimizer is intended for calculators, such as MACE, that expose an
analytical Cartesian Hessian through ``calculator.get_hessian(atoms=atoms)``.
It can use the optimized atoms' calculator directly, or a separate
Hessian-only calculator while the optimized atoms keep another target
potential for energies and forces. It supports molecular, periodic, and ASE
cell-filter geometry optimization. For cell filters, the MACE Cartesian
Hessian preconditions the atomic-position block and the added cell degrees of
freedom receive a regularized diagonal block. It does not take a raw Newton
step. Instead, it uses the regularized analytical Hessian as the
preconditioner inside ASE's ``PreconLBFGS`` two-loop recursion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import numpy as np
from ase import Atoms
from ase.optimize.precon import PreconLBFGS
from ase.optimize.precon.precon import Precon

from .fixed_step import AdaptiveStepPreconLBFGSMixin, FixedStepPreconLBFGSMixin
from .lindh import active_cartesian_mask

LOGGER = logging.getLogger(__name__)


class AnalyticalHessianError(ValueError):
    """Raised when analytical-Hessian LBFGS cannot be constructed."""


def _calculator_identity(calculator: object) -> str:
    cls = calculator.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _is_mace_calculator(calculator: object, seen: set[int] | None = None) -> bool:
    """Return True when a calculator or wrapped base calculator is MACE-backed."""
    if calculator is None:
        return False
    if seen is None:
        seen = set()
    calc_id = id(calculator)
    if calc_id in seen:
        return False
    seen.add(calc_id)

    identity = _calculator_identity(calculator).lower()
    if identity.startswith("mace.") or ".mace" in identity or "mace" in calculator.__class__.__name__.lower():
        return True
    for attr in ("base", "base_calc", "calculator", "calc"):
        wrapped = getattr(calculator, attr, None)
        if wrapped is not None and _is_mace_calculator(wrapped, seen):
            return True
    return False


def _underlying_atoms(optimizable: object) -> object:
    """Return the wrapped Atoms object for ASE filters, otherwise the input."""
    return getattr(optimizable, "atoms", optimizable)


def _calculator_label(calculator: object | None) -> str | None:
    if calculator is None:
        return None
    return _calculator_identity(calculator)


def _active_optimizable_mask(optimizable: object) -> np.ndarray:
    """Return active Cartesian/cell components for Atoms or ASE filters."""
    atoms = _underlying_atoms(optimizable)
    atomic_mask = active_cartesian_mask(atoms)
    n_atoms = len(atoms)
    n_optimizable = len(optimizable)
    extra_rows = n_optimizable - n_atoms
    if extra_rows <= 0:
        return atomic_mask

    extra_size = 3 * extra_rows
    cell_mask = getattr(optimizable, "mask", None)
    if cell_mask is not None:
        cell_active = np.asarray(cell_mask, dtype=float).reshape(-1) != 0.0
        if cell_active.size != extra_size:
            cell_active = np.ones(extra_size, dtype=bool)
    else:
        cell_active = np.ones(extra_size, dtype=bool)
    return np.concatenate([atomic_mask, cell_active])


def _to_numpy_hessian(raw_hessian: object, size: int) -> np.ndarray:
    """Convert a calculator Hessian return value to a dense square array."""
    if hasattr(raw_hessian, "detach"):
        raw_hessian = raw_hessian.detach().cpu().numpy()
    hessian = np.asarray(raw_hessian, dtype=float)
    if hessian.size != size * size:
        raise AnalyticalHessianError(
            f"Expected Hessian with {size * size} elements for {size} Cartesian "
            f"coordinates, got shape {hessian.shape}."
        )
    hessian = hessian.reshape(size, size)
    hessian = 0.5 * (hessian + hessian.T)
    if not np.all(np.isfinite(hessian)):
        raise FloatingPointError("Analytical Hessian contains non-finite values.")
    return hessian


class AnalyticalHessianPreconditioner(Precon):
    """ASE preconditioner built from a regularized analytical Hessian."""

    def __init__(
        self,
        eigenvalue_floor: float = 0.10,
        rebuild_interval: int = 1,
        fallback_alpha: float = 70.0,
        diagnostic_logging: bool = False,
        require_mace: bool = True,
        hessian_calculator: object | None = None,
        hessian_calculator_label: str | None = None,
    ) -> None:
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.rebuild_interval = max(1, int(rebuild_interval))
        self.fallback_alpha = float(fallback_alpha)
        self.diagnostic_logging = bool(diagnostic_logging)
        self.require_mace = bool(require_mace)
        self.hessian_calculator = hessian_calculator
        self.hessian_calculator_label = hessian_calculator_label
        self.P: np.ndarray | None = None
        self._active_mask: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None
        self._raw_eigenvalues: np.ndarray | None = None
        self._calls = 0
        self.n_builds = 0
        self.n_fallbacks = 0
        self.last_fallback_reason: str | None = None

    def make_precon(self, atoms: Atoms, reinitialize: bool | None = None) -> np.ndarray:
        self._calls += 1
        should_rebuild = (
            reinitialize is True
            or self.P is None
            or (self._calls - 1) % self.rebuild_interval == 0
        )
        if not should_rebuild:
            return self.asarray()

        try:
            hessian = self._build_hessian(atoms)
            active = _active_optimizable_mask(atoms)
            self._set_from_hessian(hessian, active)
            self.n_builds += 1
            self.last_fallback_reason = None
        except Exception as exc:  # noqa: BLE001 - optimizer fallback must be robust.
            self._set_fallback(atoms, str(exc))
        return self.asarray()

    def _build_hessian(self, atoms: Atoms) -> np.ndarray:
        base_atoms = _underlying_atoms(atoms)
        calculator = self.hessian_calculator if self.hessian_calculator is not None else base_atoms.calc
        if calculator is None or not hasattr(calculator, "get_hessian"):
            raise AnalyticalHessianError(
                "The selected calculator does not expose get_hessian(atoms=...). "
                "Use a MACE calculator or another calculator with analytical "
                "Cartesian Hessian support."
            )
        if self.require_mace and not _is_mace_calculator(calculator):
            raise AnalyticalHessianError(
                "MACE Hessian-LBFGS requires a MACE calculator. The selected "
                f"calculator appears to be {_calculator_identity(calculator)}."
            )
        atomic_size = 3 * len(base_atoms)
        if self.hessian_calculator is None:
            hessian_atoms = base_atoms
        else:
            hessian_atoms = base_atoms.copy()
            hessian_atoms.calc = calculator
        raw_hessian = calculator.get_hessian(atoms=hessian_atoms)
        atomic_hessian = _to_numpy_hessian(raw_hessian, atomic_size)

        optimizable_size = 3 * len(atoms)
        if optimizable_size == atomic_size:
            return atomic_hessian
        if optimizable_size < atomic_size:
            raise AnalyticalHessianError(
                "The optimizer coordinate vector is smaller than the wrapped "
                "Atoms Cartesian vector, so the MACE Hessian cannot be embedded."
            )

        hessian = np.eye(optimizable_size, dtype=float) * self.fallback_alpha
        hessian[:atomic_size, :atomic_size] = atomic_hessian
        return hessian

    def _set_from_hessian(self, hessian: np.ndarray, active: np.ndarray) -> None:
        if not np.any(active):
            self._active_mask = active
            self._eigenvalues = np.empty(0)
            self._raw_eigenvalues = np.empty(0)
            self._eigenvectors = np.empty((0, 0))
            self.P = np.eye(hessian.shape[0], dtype=float) * self.fallback_alpha
            return

        active_hessian = hessian[np.ix_(active, active)]
        active_hessian = 0.5 * (active_hessian + active_hessian.T)
        if not np.all(np.isfinite(active_hessian)):
            raise FloatingPointError("Active analytical Hessian contains non-finite values.")
        raw_eigenvalues, eigenvectors = np.linalg.eigh(active_hessian)
        eigenvalues = np.maximum(raw_eigenvalues, self.eigenvalue_floor)
        self._active_mask = active
        self._raw_eigenvalues = raw_eigenvalues
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        regularized = np.eye(hessian.shape[0], dtype=float) * self.fallback_alpha
        regularized[np.ix_(active, active)] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        self.P = 0.5 * (regularized + regularized.T)
        if self.diagnostic_logging:
            LOGGER.info(
                "Analytical Hessian preconditioner: min raw eigenvalue %.6g, "
                "max raw eigenvalue %.6g, clipped %d mode(s).",
                float(np.min(raw_eigenvalues)),
                float(np.max(raw_eigenvalues)),
                int(np.count_nonzero(raw_eigenvalues < self.eigenvalue_floor)),
            )

    def _set_fallback(self, atoms: Atoms, reason: str) -> None:
        size = 3 * len(atoms)
        active = _active_optimizable_mask(atoms)
        self.P = np.eye(size, dtype=float) * self.fallback_alpha
        self._active_mask = active
        n_active = int(np.count_nonzero(active))
        self._raw_eigenvalues = np.full(n_active, self.fallback_alpha, dtype=float)
        self._eigenvalues = np.full(n_active, self.fallback_alpha, dtype=float)
        self._eigenvectors = np.eye(n_active, dtype=float)
        self.n_fallbacks += 1
        self.last_fallback_reason = reason
        LOGGER.warning("Analytical Hessian preconditioner fallback: %s", reason)

    def solve(self, x: np.ndarray) -> np.ndarray:
        vector = np.asarray(x, dtype=float).reshape(-1)
        if self._active_mask is None or self._eigenvalues is None or self._eigenvectors is None:
            raise RuntimeError("Analytical Hessian preconditioner has not been built.")
        result = np.zeros_like(vector)
        if self._eigenvalues.size == 0:
            return result
        active_vector = vector[self._active_mask]
        projected = self._eigenvectors.T @ active_vector
        solved = self._eigenvectors @ (projected / self._eigenvalues)
        result[self._active_mask] = solved
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Analytical Hessian solve produced non-finite values.")
        return result

    def Pdot(self, x: np.ndarray) -> np.ndarray:
        return self.asarray().dot(np.asarray(x).reshape(-1))

    def asarray(self) -> np.ndarray:
        if self.P is None:
            raise RuntimeError("Analytical Hessian preconditioner has not been built.")
        return np.asarray(self.P)

    def copy(self) -> "AnalyticalHessianPreconditioner":
        return AnalyticalHessianPreconditioner(
            eigenvalue_floor=self.eigenvalue_floor,
            rebuild_interval=self.rebuild_interval,
            fallback_alpha=self.fallback_alpha,
            diagnostic_logging=self.diagnostic_logging,
            require_mace=self.require_mace,
            hessian_calculator=self.hessian_calculator,
            hessian_calculator_label=self.hessian_calculator_label,
        )


class _MACEHessianPreconditionedLBFGS(PreconLBFGS, ABC):
    """Abstract L-BFGS state and analytical-Hessian integration.

    The optimizer uses the MACE Cartesian Hessian for atom-position degrees of
    freedom. If constructed with an ASE cell filter, the additional cell degrees
    of freedom are handled with a regularized diagonal preconditioner block.
    """

    def __init__(
        self,
        atoms: Atoms,
        restart: str | None = None,
        logfile: str | object = "-",
        trajectory: str | None = None,
        maxstep: float = 0.20,
        memory: int = 20,
        eigenvalue_floor: float = 0.10,
        rebuild_interval: int = 1,
        diagnostic_logging: bool = False,
        require_mace: bool = True,
        hessian_calculator: object | None = None,
        hessian_calculator_label: str | None = None,
        **kwargs: object,
    ) -> None:
        kwargs.pop("use_armijo", None)
        kwargs["use_armijo"] = False
        base_atoms = _underlying_atoms(atoms)
        hessian_source = hessian_calculator if hessian_calculator is not None else base_atoms.calc
        if hessian_source is None or not hasattr(hessian_source, "get_hessian"):
            raise AnalyticalHessianError(
                "Analytical-Hessian LBFGS requires a calculator with "
                "get_hessian(atoms=...), such as the MACE calculator. For "
                "non-MACE target potentials, pass a MACE calculator through "
                "hessian_calculator=...."
            )
        if require_mace and not _is_mace_calculator(hessian_source):
            raise AnalyticalHessianError(
                "MACE Hessian-LBFGS requires a MACE calculator. The selected "
                f"Hessian calculator appears to be {_calculator_identity(hessian_source)}."
            )
        self.target_calculator_label = _calculator_label(base_atoms.calc)
        self.hessian_calculator_label = hessian_calculator_label or _calculator_label(hessian_source)
        self.uses_separate_hessian_calculator = hessian_calculator is not None
        self.periodic_system = bool(np.any(base_atoms.pbc))
        self.optimizes_cell = len(atoms) > len(base_atoms)
        self.analytical_hessian_precon = AnalyticalHessianPreconditioner(
            eigenvalue_floor=eigenvalue_floor,
            rebuild_interval=rebuild_interval,
            diagnostic_logging=diagnostic_logging,
            require_mace=require_mace,
            hessian_calculator=hessian_calculator,
            hessian_calculator_label=self.hessian_calculator_label,
        )
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            memory=memory,
            precon=self.analytical_hessian_precon,
            variable_cell=False,
            **kwargs,
        )
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.rebuild_interval = int(rebuild_interval)

    @abstractmethod
    def step(self, f=None) -> None:
        """Implemented by a released line-search-free stepping mixin."""
        raise NotImplementedError

    @property
    def n_hessian_builds(self) -> int:
        return self.analytical_hessian_precon.n_builds

    @property
    def n_hessian_fallbacks(self) -> int:
        return self.analytical_hessian_precon.n_fallbacks

    def update(
        self,
        r: np.ndarray,
        f: np.ndarray,
        r0: np.ndarray | None,
        f0: np.ndarray | None,
    ) -> None:
        """Update L-BFGS history while skipping unstable curvature pairs."""
        if not self._just_reset_hessian and r0 is not None and f0 is not None:
            s0 = r.reshape(-1) - r0.reshape(-1)
            y0 = f0.reshape(-1) - f.reshape(-1)
            curvature = float(np.dot(y0, s0))
            scale = max(1.0, float(np.linalg.norm(s0) * np.linalg.norm(y0)))
            if curvature > 1.0e-10 * scale and np.isfinite(curvature):
                self.s.append(s0)
                self.y.append(y0)
                self.rho.append(1.0 / curvature)
            else:
                LOGGER.info(
                    "Skipped analytical-Hessian LBFGS curvature pair with s.y=%g.",
                    curvature,
                )
        self._just_reset_hessian = False

        while len(self.y) > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def get_hessian_metadata(self) -> dict[str, int | float | str | None]:
        raw = self.analytical_hessian_precon._raw_eigenvalues
        clipped = None
        min_raw = None
        if raw is not None and raw.size:
            clipped = int(np.count_nonzero(raw < self.eigenvalue_floor))
            min_raw = float(np.min(raw))
        return {
            "Optimizer": "MACE Hessian LBFGS",
            "L-BFGS memory": self.memory,
            "Analytical Hessian rebuild interval": self.rebuild_interval,
            "Hessian regularization floor": self.eigenvalue_floor,
            "Line search": "None",
            "Target calculator": self.target_calculator_label,
            "Hessian calculator": self.hessian_calculator_label,
            "Separate Hessian calculator": self.uses_separate_hessian_calculator,
            "Periodic system": self.periodic_system,
            "Cell degrees of freedom optimized": self.optimizes_cell,
            "Cell Hessian block": "regularized diagonal" if self.optimizes_cell else None,
            "Number of analytical Hessian builds": self.n_hessian_builds,
            "Number of fallback preconditioner uses": self.n_hessian_fallbacks,
            "Last fallback reason": self.analytical_hessian_precon.last_fallback_reason,
            "Minimum raw Hessian eigenvalue": min_raw,
            "Number of clipped Hessian modes": clipped,
        }


class MACEHessianLBFGS(FixedStepPreconLBFGSMixin, _MACEHessianPreconditionedLBFGS):
    """MACE-Hessian-preconditioned L-BFGS without a line search."""

    def get_hessian_metadata(self) -> dict[str, int | float | str | None]:
        metadata = super().get_hessian_metadata()
        metadata.update(self.get_fixed_step_metadata())
        metadata["Optimizer"] = "MACE Hessian LBFGS"
        return metadata


class MACESeedLBFGS(AdaptiveStepPreconLBFGSMixin, _MACEHessianPreconditionedLBFGS):
    """One-shot MACE-Hessian L-BFGS with one target evaluation per cycle.

    The optimizer starts from one regularized MACE Hessian, then uses L-BFGS
    force-difference updates.  It does not perform a line search.  Instead, the
    response at each newly evaluated geometry controls the next step radius and
    the retained weight of the MACE inverse-Hessian seed.
    """

    _ONE_SHOT_REBUILD_INTERVAL = 1_000_000_000

    def __init__(
        self,
        atoms: Atoms,
        restart: str | None = None,
        logfile: str | object = "-",
        trajectory: str | None = None,
        maxstep: float = 0.20,
        memory: int = 20,
        eigenvalue_floor: float = 0.10,
        diagnostic_logging: bool = False,
        require_mace: bool = True,
        hessian_calculator: object | None = None,
        hessian_calculator_label: str | None = None,
        **kwargs: object,
    ) -> None:
        kwargs.pop("rebuild_interval", None)
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            memory=memory,
            eigenvalue_floor=eigenvalue_floor,
            rebuild_interval=self._ONE_SHOT_REBUILD_INTERVAL,
            diagnostic_logging=diagnostic_logging,
            require_mace=require_mace,
            hessian_calculator=hessian_calculator,
            hessian_calculator_label=hessian_calculator_label,
            **kwargs,
        )

    def get_hessian_metadata(self) -> dict[str, int | float | str | None]:
        metadata = super().get_hessian_metadata()
        metadata.update(self.get_adaptive_step_metadata())
        metadata["Optimizer"] = "MACE-Seed LBFGS"
        metadata["Analytical Hessian rebuild interval"] = "initial build only"
        metadata["Hessian usage"] = "one initial MACE Hessian only"
        return metadata
