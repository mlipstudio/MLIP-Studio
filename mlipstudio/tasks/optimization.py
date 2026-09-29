"""Geometry and cell optimization tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, BFGSLineSearch, FIRE, GPMin, LBFGS, LBFGSLineSearch, MDMin

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, prepare_atoms


_BUILTIN_OPTIMIZERS = {
    "BFGS": BFGS,
    "BFGSLineSearch": BFGSLineSearch,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "FIRE": FIRE,
    "GPMin": GPMin,
    "MDMin": MDMin,
}

_SPECIAL_OPTIMIZERS = {
    "Lindh Hessian LBFGS": ("optimizers.lindh", "LindhHessianLBFGS"),
    "MACE Hessian LBFGS": (
        "optimizers.analytical_hessian",
        "MACEHessianLBFGS",
    ),
    "MACE-Seed LBFGS": ("optimizers.analytical_hessian", "MACESeedLBFGS"),
}

# Optional packages, not dependencies: offered only when installed.
_OPTIONAL_OPTIMIZERS = {
    "Sella": ("sella", "make_sella"),
    "geomeTRIC": ("geometric", "GeomeTRICOptimizer"),
}

_FIXED_CELL_ONLY = ("Lindh Hessian LBFGS", "Sella")


def _optimizer_class(name: str) -> type:
    if name in _BUILTIN_OPTIMIZERS:
        return _BUILTIN_OPTIMIZERS[name]
    if name in _SPECIAL_OPTIMIZERS or name in _OPTIONAL_OPTIMIZERS:
        if name in _OPTIONAL_OPTIMIZERS:
            package, class_name = _OPTIONAL_OPTIMIZERS[name]
            if find_spec(package) is None:
                raise TaskCalculationError(
                    f"The '{name}' optimizer needs the optional '{package}' package, "
                    "which is not installed."
                )
            module_name = "optimizers.external"
        else:
            module_name, class_name = _SPECIAL_OPTIMIZERS[name]
        try:
            from importlib import import_module

            return getattr(import_module(module_name), class_name)
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise TaskCalculationError(
                f"The '{name}' optimizer could not be imported: {exc}"
            ) from exc
    available = (
        tuple(_BUILTIN_OPTIMIZERS)
        + tuple(_SPECIAL_OPTIMIZERS)
        + tuple(n for n, (package, _) in _OPTIONAL_OPTIMIZERS.items() if find_spec(package))
    )
    raise TaskValidationError(
        f"Unknown optimizer '{name}'. Available: {', '.join(available)}."
    )


def _maximum_force(optimizable: Any) -> float:
    forces = np.asarray(optimizable.get_forces(), dtype=float)
    if forces.size == 0:
        return 0.0
    return float(np.linalg.norm(forces.reshape((-1, 3)), axis=1).max(initial=0.0))


@dataclass(frozen=True)
class OptimizationResult:
    """Raw optimization output in ASE units."""

    atoms: Atoms
    trajectory: tuple[Atoms, ...]
    energy: float
    forces: np.ndarray
    converged: bool
    steps: int
    optimizer: str
    optimize_cell: bool
    optimizer_max_force: float
    atomic_max_force: float
    runtime_seconds: float
    calculator_name: str
    optimizer_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class OptimizationTask(Task):
    """Run a standard ASE or Hessian-guided optimization."""

    optimizer: str = "LBFGS"
    fmax: float = 0.01
    steps: int = 50
    optimize_cell: bool = False
    optimizer_parameters: Mapping[str, Any] = field(default_factory=dict)
    hessian_calculator: Any | None = None
    hessian_calculator_label: str | None = None
    logfile: Any = None
    copy_atoms: bool = True
    record_trajectory: bool = True

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> OptimizationResult:
        optimizer_class = _optimizer_class(self.optimizer)
        if self.fmax <= 0:
            raise TaskValidationError("fmax must be greater than zero.")
        if self.steps < 0:
            raise TaskValidationError("steps must be non-negative.")
        if self.optimize_cell and self.optimizer in _FIXED_CELL_ONLY:
            raise TaskValidationError(
                f"{self.optimizer} supports fixed-cell atomic optimization only."
            )
        if self.optimizer == "geomeTRIC" and (self.optimize_cell or np.any(atoms.pbc)):
            raise TaskValidationError("geomeTRIC supports non-periodic molecules only.")

        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        if self.optimize_cell:
            if not np.all(working_atoms.pbc):
                raise TaskValidationError(
                    "Cell optimization requires periodic boundary conditions in all directions."
                )
            if working_atoms.cell.volume <= 0:
                raise TaskValidationError("Cell optimization requires a valid non-zero cell.")
            optimizable: Any = FrechetCellFilter(working_atoms)
        else:
            optimizable = working_atoms

        kwargs = dict(self.optimizer_parameters)
        is_mace_hessian_optimizer = self.optimizer in {
            "MACE Hessian LBFGS",
            "MACE-Seed LBFGS",
        }
        if self.hessian_calculator is not None:
            if not is_mace_hessian_optimizer:
                raise TaskValidationError(
                    "hessian_calculator is only valid with MACE Hessian LBFGS "
                    "or MACE-Seed LBFGS."
                )
            if "hessian_calculator" in kwargs:
                raise TaskValidationError(
                    "Pass hessian_calculator as a task argument or in "
                    "optimizer_parameters, not both."
                )
            kwargs["hessian_calculator"] = self.hessian_calculator
        if self.hessian_calculator_label is not None:
            if not is_mace_hessian_optimizer:
                raise TaskValidationError(
                    "hessian_calculator_label is only valid with a MACE Hessian optimizer."
                )
            if "hessian_calculator_label" in kwargs:
                raise TaskValidationError(
                    "Pass hessian_calculator_label as a task argument or in "
                    "optimizer_parameters, not both."
                )
            kwargs["hessian_calculator_label"] = self.hessian_calculator_label
        kwargs.setdefault("logfile", self.logfile)
        frames: list[Atoms] = []
        if self.record_trajectory:
            frames.append(detached_copy(working_atoms))

        try:
            optimizer = optimizer_class(optimizable, **kwargs)
            if self.record_trajectory:
                optimizer.attach(
                    lambda: frames.append(detached_copy(working_atoms)),
                    interval=1,
                )
            started = time.perf_counter()
            converged = bool(optimizer.run(fmax=float(self.fmax), steps=int(self.steps)))
            runtime = time.perf_counter() - started
            energy = float(working_atoms.get_potential_energy())
            forces = np.asarray(working_atoms.get_forces(), dtype=float).copy()
            optimizer_max_force = _maximum_force(optimizable)
            atomic_max_force = _maximum_force(working_atoms)
            metadata_getter = getattr(optimizer, "get_hessian_metadata", None)
            if not callable(metadata_getter):
                metadata_getter = getattr(optimizer, "get_lindh_metadata", None)
            optimizer_metadata = (
                dict(metadata_getter()) if callable(metadata_getter) else {}
            )
        except Exception as exc:
            raise TaskCalculationError(f"Optimization failed: {exc}") from exc

        return OptimizationResult(
            atoms=detached_copy(working_atoms),
            trajectory=tuple(frames),
            energy=energy,
            forces=forces,
            converged=converged,
            steps=optimizer.get_number_of_steps(),
            optimizer=self.optimizer,
            optimize_cell=self.optimize_cell,
            optimizer_max_force=optimizer_max_force,
            atomic_max_force=atomic_max_force,
            runtime_seconds=runtime,
            calculator_name=calculator_name(resolved_calculator),
            optimizer_metadata=optimizer_metadata,
        )
