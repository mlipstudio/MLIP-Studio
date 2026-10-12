"""Finite-difference vibrational mode analysis."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from ..vibrational_analysis import (
    RIGID_BODY,
    analyze_vibrational_modes,
    vibrated_atom_indices,
)
from .base import Task, calculator_name, detached_copy, prepare_atoms


@dataclass(frozen=True)
class VibrationalModeResult:
    """Normal-mode frequencies, energies, displacements, and harmonic thermodynamics.

    ``mode_types`` labels each mode "rigid-body", "vibration" or "imaginary".
    ``imaginary_mode_indices`` lists imaginary vibrations only; the ZPE and
    entropy include only real vibrations.
    """

    atoms: Atoms
    frequencies_cm_minus1: np.ndarray
    energies_eV: np.ndarray
    modes: np.ndarray
    mode_types: tuple[str, ...]
    rigid_body_mode_count: int
    rigid_body_reason: str
    imaginary_mode_indices: tuple[int, ...]
    zero_point_energy_eV: float
    vibrational_entropy_eV_per_K: float
    temperature_K: float
    warnings: tuple[str, ...]
    runtime_seconds: float
    calculator_name: str

    @property
    def has_imaginary_modes(self) -> bool:
        return bool(self.imaginary_mode_indices)

    @property
    def rigid_body_mode_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, mode_type in enumerate(self.mode_types)
            if mode_type == RIGID_BODY
        )


@dataclass(frozen=True)
class VibrationalModeTask(Task):
    """Calculate harmonic vibrational modes with ASE finite differences.

    Temporary displacement files are always removed.  Complex frequencies
    are retained so imaginary modes are not silently discarded.

    The ``rigid_body_modes`` modes with the smallest |frequency| are treated
    as translations and rotations.  ``None`` detects the count: 6 for a
    nonlinear molecule, 5 for a linear one, 3 for a single atom or periodic
    cell, and 0 for a partial Hessian (``indices`` or ``FixAtoms``).
    """

    delta_A: float = 0.01
    nfree: int = 2
    indices: tuple[int, ...] | None = None
    temperature_K: float = 298.15
    imaginary_tolerance: float = 1.0e-10
    rigid_body_modes: int | None = None
    copy_atoms: bool = True

    def _validated_indices(self, atom_count: int) -> tuple[int, ...] | None:
        if self.indices is None:
            return None
        indices = tuple(self.indices)
        if not indices:
            raise TaskValidationError("indices must contain at least one atom index.")
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= atom_count
            for index in indices
        ):
            raise TaskValidationError(
                f"Vibrational atom indices must be integers from 0 to {atom_count - 1}."
            )
        if len(set(indices)) != len(indices):
            raise TaskValidationError("Vibrational atom indices must be unique.")
        return indices

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> VibrationalModeResult:
        if self.delta_A <= 0:
            raise TaskValidationError("delta_A must be greater than zero.")
        if self.nfree not in {2, 4}:
            raise TaskValidationError("nfree must be 2 or 4.")
        if self.temperature_K <= 0:
            raise TaskValidationError("temperature_K must be greater than zero.")
        if self.imaginary_tolerance < 0:
            raise TaskValidationError("imaginary_tolerance must be non-negative.")
        if self.rigid_body_modes is not None and (
            not isinstance(self.rigid_body_modes, int)
            or isinstance(self.rigid_body_modes, bool)
            or self.rigid_body_modes < 0
        ):
            raise TaskValidationError(
                "rigid_body_modes must be None or a non-negative integer."
            )

        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        indices = self._validated_indices(len(working_atoms))
        mode_count = 3 * len(vibrated_atom_indices(working_atoms, indices))
        if self.rigid_body_modes is not None and self.rigid_body_modes > mode_count:
            raise TaskValidationError(
                f"rigid_body_modes cannot exceed the {mode_count} modes of the "
                "vibrated atoms."
            )
        started = time.perf_counter()

        try:
            from ase.vibrations import Vibrations

            with tempfile.TemporaryDirectory(prefix="mlipstudio-vibrations-") as temp_dir:
                vibrations = Vibrations(
                    working_atoms,
                    indices=indices,
                    name=str(Path(temp_dir) / "vib"),
                    delta=float(self.delta_A),
                    nfree=int(self.nfree),
                )
                try:
                    vibrations.run()
                    frequencies = np.asarray(
                        vibrations.get_frequencies(), dtype=complex
                    ).reshape(-1)
                    energies = np.asarray(
                        vibrations.get_energies(), dtype=complex
                    ).reshape(-1)
                    modes = np.asarray(
                        [vibrations.get_mode(index) for index in range(len(frequencies))]
                    )
                finally:
                    vibrations.clean()
        except Exception as exc:
            raise TaskCalculationError(
                f"Vibrational mode calculation failed: {exc}"
            ) from exc

        if frequencies.size == 0 or energies.shape != frequencies.shape:
            raise TaskCalculationError(
                "Vibrational analysis returned empty or inconsistent mode arrays."
            )
        if not (
            np.all(np.isfinite(frequencies.real))
            and np.all(np.isfinite(frequencies.imag))
            and np.all(np.isfinite(energies.real))
            and np.all(np.isfinite(energies.imag))
        ):
            raise TaskCalculationError("Vibrational analysis returned non-finite values.")

        try:
            analysis = analyze_vibrational_modes(
                working_atoms,
                frequencies,
                indices=indices,
                temperature_K=float(self.temperature_K),
                rigid_body_modes=self.rigid_body_modes,
                imaginary_tolerance=float(self.imaginary_tolerance),
            )
        except ValueError as exc:
            raise TaskCalculationError(
                f"Vibrational mode classification failed: {exc}"
            ) from exc

        return VibrationalModeResult(
            atoms=detached_copy(working_atoms),
            frequencies_cm_minus1=frequencies.copy(),
            energies_eV=energies.copy(),
            modes=modes.copy(),
            mode_types=analysis.mode_types,
            rigid_body_mode_count=analysis.rigid_body_mode_count,
            rigid_body_reason=analysis.rigid_body_reason,
            imaginary_mode_indices=analysis.imaginary_mode_indices,
            zero_point_energy_eV=analysis.zero_point_energy_eV,
            vibrational_entropy_eV_per_K=analysis.vibrational_entropy_eV_per_K,
            temperature_K=analysis.temperature_K,
            warnings=analysis.warnings,
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(resolved_calculator),
        )
