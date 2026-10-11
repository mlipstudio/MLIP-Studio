"""Analytical Cartesian Hessians from compatible MACE or AIMNet2 calculators."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, prepare_atoms


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy- and torch-like values without importing torch."""

    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    try:
        return np.asarray(converted, dtype=float)
    except Exception as exc:
        raise TaskCalculationError(
            "The analytical Hessian could not be converted to a numerical array."
        ) from exc


@dataclass(frozen=True)
class AnalyticalHessianResult:
    """Cartesian Hessian in eV/Angstrom^2 and basic diagnostics."""

    atoms: Atoms
    hessian_eV_per_A2: np.ndarray
    symmetry_error_eV_per_A2: float
    maximum_absolute_element_eV_per_A2: float
    runtime_seconds: float
    calculator_name: str


@dataclass(frozen=True)
class AnalyticalHessianTask(Task):
    """Evaluate the exact model Hessian exposed by a MACE or AIMNet2 calculator.

    The calculator must provide ``get_hessian(atoms=...)``.  The task checks
    the capability instead of importing MACE, which keeps the public API
    importable when the optional MACE runtime is absent.
    """

    copy_atoms: bool = True

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> AnalyticalHessianResult:
        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        get_hessian = getattr(resolved_calculator, "get_hessian", None)
        if not callable(get_hessian):
            raise TaskValidationError(
                "AnalyticalHessianTask requires a calculator with "
                "get_hessian(atoms=...), such as a compatible MACE or AIMNet2 calculator."
            )

        started = time.perf_counter()
        try:
            raw_hessian = get_hessian(atoms=working_atoms)
            array = _as_numpy(raw_hessian)
        except TaskCalculationError:
            raise
        except Exception as exc:
            raise TaskCalculationError(
                f"Analytical Hessian calculation failed: {exc}"
            ) from exc

        expected_size = 9 * len(working_atoms) ** 2
        if array.size != expected_size:
            raise TaskCalculationError(
                "The calculator returned an analytical Hessian with "
                f"{array.size} elements; expected {expected_size} for "
                f"{len(working_atoms)} atoms."
            )
        hessian = array.reshape((3 * len(working_atoms), 3 * len(working_atoms))).copy()
        if not np.all(np.isfinite(hessian)):
            raise TaskCalculationError("The analytical Hessian contains non-finite values.")

        symmetry_error = float(np.max(np.abs(hessian - hessian.T), initial=0.0))
        maximum_element = float(np.max(np.abs(hessian), initial=0.0))
        return AnalyticalHessianResult(
            atoms=detached_copy(working_atoms),
            hessian_eV_per_A2=hessian,
            symmetry_error_eV_per_A2=symmetry_error,
            maximum_absolute_element_eV_per_A2=maximum_element,
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(resolved_calculator),
        )


# Explicit aliases make the MACE-specific feature easy to discover while the
# capability-based class remains useful for compatible calculator wrappers.
MACEHessianTask = AnalyticalHessianTask
MACEHessianResult = AnalyticalHessianResult

