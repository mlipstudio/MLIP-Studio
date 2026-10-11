"""Atomization and cohesive-energy calculations."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from ..exceptions import TaskCalculationError, TaskValidationError
from ..references import get_element_reference_set
from .base import Task, calculator_name, detached_copy, prepare_atoms


def _normalise_reference_mapping(
    references: Mapping[int | str, float],
) -> dict[int, float]:
    normalised: dict[int, float] = {}
    for key, value in references.items():
        if isinstance(key, str):
            try:
                atomic_number = atomic_numbers[key]
            except KeyError as exc:
                raise TaskValidationError(
                    f"Unknown element symbol in isolated-atom references: {key}."
                ) from exc
        elif isinstance(key, int) and not isinstance(key, bool):
            atomic_number = key
        else:
            raise TaskValidationError(
                "Isolated-atom reference keys must be symbols or atomic numbers."
            )
        energy = float(value)
        if atomic_number < 1 or not np.isfinite(energy):
            raise TaskValidationError(
                f"Invalid isolated-atom reference for atomic number {atomic_number}."
            )
        normalised[atomic_number] = energy
    return normalised


@dataclass(frozen=True)
class AtomizationCohesiveEnergyResult:
    """Raw atomization or cohesive energy and its isolated-atom provenance."""

    atoms: Atoms
    calculation_type: str
    system_energy_eV: float
    isolated_atoms_energy_eV: float
    atomization_energy_eV: float | None
    cohesive_energy_eV_per_atom: float | None
    isolated_atom_energies_eV: Mapping[int, float]
    stoichiometry: Mapping[int, int]
    reference_source: str
    runtime_seconds: float
    calculator_name: str


@dataclass(frozen=True)
class AtomizationCohesiveEnergyTask(Task):
    """Calculate molecular atomization or periodic cohesive energy.

    Explicit isolated-atom energies take precedence. Otherwise a named bundled
    reference set may be requested. In automatic mode, FairChem calculators use
    the table matching their task name and other calculators evaluate isolated
    atoms directly with the same loaded calculator. AIMNet2 models are
    closed-shell, so they need explicit or bundled references.
    """

    calculation_type: str = "auto"
    isolated_atom_energies: Mapping[int | str, float] | None = None
    reference_set: str | None = None
    isolated_atom_cell_A: float = 20.0
    system_metadata: Mapping[str, Any] = field(
        default_factory=lambda: {
            "charge": 0,
            "total_charge": 0,
            "spin": 1,
            "total_spin": 1,
            "external_field": [0.0, 0.0, 0.0],
        }
    )
    isolated_atom_metadata: Mapping[str, Any] = field(
        default_factory=lambda: {
            "charge": 0,
            "total_charge": 0,
            "spin": 0,
            "total_spin": 0,
            "external_field": [0.0, 0.0, 0.0],
        }
    )
    copy_atoms: bool = True

    def _resolve_type(self, atoms: Atoms) -> str:
        if self.calculation_type not in {"auto", "atomization", "cohesive"}:
            raise TaskValidationError(
                "calculation_type must be 'auto', 'atomization', or 'cohesive'."
            )
        periodic = bool(np.any(atoms.pbc))
        if self.calculation_type == "auto":
            resolved = "cohesive" if periodic else "atomization"
        else:
            resolved = self.calculation_type
        if resolved == "atomization" and periodic:
            raise TaskValidationError(
                "Atomization energy requires a non-periodic structure."
            )
        if resolved == "cohesive" and not periodic:
            raise TaskValidationError("Cohesive energy requires a periodic structure.")
        return resolved

    @staticmethod
    def _references_from_table(
        table_name: str,
        atomic_numbers_needed: tuple[int, ...],
    ) -> dict[int, float]:
        table = get_element_reference_set(table_name)
        missing = [number for number in atomic_numbers_needed if number >= len(table)]
        if missing:
            raise TaskValidationError(
                f"Reference set '{table_name}' has no entries for atomic number(s): "
                + ", ".join(map(str, missing))
            )
        references = {number: float(table[number]) for number in atomic_numbers_needed}
        if not all(np.isfinite(value) for value in references.values()):
            raise TaskValidationError(
                f"Reference set '{table_name}' contains non-finite values."
            )
        return references

    def _calculate_isolated_references(
        self,
        calculator: Any,
        atomic_numbers_needed: tuple[int, ...],
    ) -> dict[int, float]:
        if self.isolated_atom_cell_A <= 0:
            raise TaskValidationError("isolated_atom_cell_A must be greater than zero.")
        references: dict[int, float] = {}
        for atomic_number in atomic_numbers_needed:
            isolated = Atoms(
                numbers=[atomic_number],
                positions=[[0.0, 0.0, 0.0]],
                cell=[self.isolated_atom_cell_A] * 3,
                pbc=False,
            )
            isolated.info.update(dict(self.isolated_atom_metadata))
            isolated.calc = calculator
            reset = getattr(calculator, "reset", None)
            if callable(reset):
                reset()
            try:
                energy = float(isolated.get_potential_energy())
            except Exception as exc:
                raise TaskCalculationError(
                    f"Isolated-atom calculation failed for Z={atomic_number}: {exc}"
                ) from exc
            if not np.isfinite(energy):
                raise TaskCalculationError(
                    f"Isolated-atom energy for Z={atomic_number} is not finite."
                )
            references[atomic_number] = energy
        return references

    def _resolve_references(
        self,
        calculator: Any,
        atomic_numbers_needed: tuple[int, ...],
    ) -> tuple[dict[int, float], str]:
        if self.isolated_atom_energies is not None and self.reference_set is not None:
            raise TaskValidationError(
                "Specify isolated_atom_energies or reference_set, not both."
            )
        if self.isolated_atom_energies is not None:
            references = _normalise_reference_mapping(self.isolated_atom_energies)
            source = "user-supplied"
        elif self.reference_set is not None:
            references = self._references_from_table(
                self.reference_set,
                atomic_numbers_needed,
            )
            source = f"bundled:{self.reference_set}"
        elif getattr(calculator, "_mlipstudio_model_family", None) == "FairChem":
            parameters = getattr(calculator, "_mlipstudio_parameters", {})
            task_name = parameters.get("task_name")
            if not task_name:
                raise TaskValidationError(
                    "FairChem atomization/cohesive energy requires a task_name-specific "
                    "elemental reference table."
                )
            table_name = f"{task_name}_elem_refs"
            references = self._references_from_table(table_name, atomic_numbers_needed)
            source = f"bundled:{table_name}"
        elif getattr(calculator, "_mlipstudio_model_family", None) == "AIMNet2":
            raise TaskValidationError(
                "AIMNet2 models are closed-shell and cannot evaluate isolated atoms; "
                "pass isolated_atom_energies or reference_set."
            )
        else:
            references = self._calculate_isolated_references(
                calculator,
                atomic_numbers_needed,
            )
            source = "calculated-with-system-calculator"

        missing = [number for number in atomic_numbers_needed if number not in references]
        if missing:
            raise TaskValidationError(
                "Missing isolated-atom reference energy for atomic number(s): "
                + ", ".join(map(str, missing))
            )
        return {number: references[number] for number in atomic_numbers_needed}, source

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> AtomizationCohesiveEnergyResult:
        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        working_atoms.info.update(dict(self.system_metadata))
        resolved_type = self._resolve_type(working_atoms)
        counts = Counter(int(number) for number in working_atoms.get_atomic_numbers())
        unique_numbers = tuple(sorted(counts))
        started = time.perf_counter()
        try:
            system_energy = float(working_atoms.get_potential_energy())
        except Exception as exc:
            raise TaskCalculationError(f"System-energy calculation failed: {exc}") from exc
        if not np.isfinite(system_energy):
            raise TaskCalculationError("System energy is not finite.")

        references, reference_source = self._resolve_references(
            resolved_calculator,
            unique_numbers,
        )
        isolated_total = float(
            sum(references[number] * count for number, count in counts.items())
        )
        energy_difference = isolated_total - system_energy
        atomization = energy_difference if resolved_type == "atomization" else None
        cohesive = (
            energy_difference / len(working_atoms)
            if resolved_type == "cohesive"
            else None
        )
        return AtomizationCohesiveEnergyResult(
            atoms=detached_copy(working_atoms),
            calculation_type=resolved_type,
            system_energy_eV=system_energy,
            isolated_atoms_energy_eV=isolated_total,
            atomization_energy_eV=atomization,
            cohesive_energy_eV_per_atom=cohesive,
            isolated_atom_energies_eV=dict(references),
            stoichiometry=dict(counts),
            reference_source=reference_source,
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(resolved_calculator),
        )
