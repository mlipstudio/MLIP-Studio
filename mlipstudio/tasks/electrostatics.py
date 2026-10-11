"""Dipole moment and partial-charge calculations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, prepare_atoms


def _result_array(value: Any, name: str) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    try:
        array = np.asarray(converted, dtype=float).reshape(-1)
    except Exception as exc:
        raise TaskCalculationError(f"Could not convert calculator result '{name}'.") from exc
    if not np.all(np.isfinite(array)):
        raise TaskCalculationError(f"Calculator result '{name}' is not finite.")
    return array


@dataclass(frozen=True)
class DipoleMomentResult:
    """Molecular dipole and per-atom partial charges."""

    atoms: Atoms
    energy_eV: float
    dipole_eA: np.ndarray
    dipole_magnitude_eA: float
    partial_charges_e: np.ndarray
    total_partial_charge_e: float
    runtime_seconds: float
    calculator_name: str


@dataclass(frozen=True)
class DipoleMomentTask(Task):
    """Calculate dipole and partial charges using a MACE-POLAR or AIMNet2 model."""

    charge: int = 0
    spin_multiplicity: int = 1
    external_field: tuple[float, float, float] = (0.0, 0.0, 0.0)
    copy_atoms: bool = True

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> DipoleMomentResult:
        if self.spin_multiplicity < 1:
            raise TaskValidationError("spin_multiplicity must be at least one.")
        field = np.asarray(self.external_field, dtype=float)
        if field.shape != (3,) or not np.all(np.isfinite(field)):
            raise TaskValidationError("external_field must contain three finite values.")
        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        model_name = getattr(resolved_calculator, "_mlipstudio_model_name", None)
        if model_name is not None and not model_name.startswith(("MACE POLAR", "AIMNet2")):
            raise TaskValidationError(
                f"{model_name} is not registered for dipole/partial-charge prediction."
            )
        if getattr(resolved_calculator, "_mlipstudio_model_family", None) == "AIMNet2":
            if self.spin_multiplicity != 1:
                raise TaskValidationError(
                    "AIMNet2 models are closed-shell; spin_multiplicity must be 1."
                )
            if np.any(field != 0.0):
                raise TaskValidationError(
                    "AIMNet2 models do not support an external electric field."
                )

        working_atoms.info["charge"] = self.charge
        working_atoms.info["total_charge"] = self.charge
        working_atoms.info["spin"] = self.spin_multiplicity
        working_atoms.info["total_spin"] = self.spin_multiplicity
        working_atoms.info["external_field"] = field.tolist()
        reset = getattr(resolved_calculator, "reset", None)
        if callable(reset):
            reset()
        started = time.perf_counter()
        try:
            energy = float(working_atoms.get_potential_energy())
            results = getattr(resolved_calculator, "results", {})
            dipole = _result_array(results["dipole"], "dipole")
            charges = _result_array(results["charges"], "charges")
        except KeyError as exc:
            raise TaskCalculationError(
                f"Calculator did not return required result '{exc.args[0]}'."
            ) from exc
        except TaskCalculationError:
            raise
        except Exception as exc:
            raise TaskCalculationError(
                f"Dipole/partial-charge calculation failed: {exc}"
            ) from exc
        if dipole.shape != (3,):
            raise TaskCalculationError(
                f"Dipole must have shape (3,), got {dipole.shape}."
            )
        if charges.shape != (len(working_atoms),):
            raise TaskCalculationError(
                f"Partial charges must have shape ({len(working_atoms)},), "
                f"got {charges.shape}."
            )
        if not np.isfinite(energy):
            raise TaskCalculationError("Calculated energy is not finite.")
        return DipoleMomentResult(
            atoms=detached_copy(working_atoms),
            energy_eV=energy,
            dipole_eA=dipole.copy(),
            dipole_magnitude_eA=float(np.linalg.norm(dipole)),
            partial_charges_e=charges.copy(),
            total_partial_charge_e=float(np.sum(charges)),
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(resolved_calculator),
        )
