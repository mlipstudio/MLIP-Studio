"""Rigid-body mode separation and harmonic thermochemistry for normal modes.

Both the MLIP Studio GUI and ``VibrationalModeTask`` call
:func:`analyze_vibrational_modes`, so they report the same zero-point energy
and vibrational entropy.  The module needs only NumPy and ASE.

A Cartesian finite-difference Hessian contains the translations and rotations
of the whole system as near-zero modes.  Their residual frequencies come from
numerical noise and incomplete geometry convergence, not from vibrations.
Instead of a frequency cutoff, the modes are sorted by |frequency| and the
``n`` smallest are treated as rigid-body motion, with ``n`` set by the system
type (see :func:`count_rigid_body_modes`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.units import _Nav, _e, invcm, kB

RIGID_BODY = "rigid-body"
VIBRATION = "vibration"
IMAGINARY = "imaginary"

# A rigid-body mode this far from zero usually means the structure is not a
# stationary point of the model.  Finite-difference error adds a part that
# grows with the step: about 50 cm^-1 for converged water at ASE's 0.01 A.
RIGID_BODY_WARNING_CM_MINUS1 = 50.0

# Linear when I_min <= tolerance * I_max.  CO2 bent by up to about 7 degrees
# still counts as linear, which covers loosely optimized linear molecules.
LINEAR_INERTIA_TOLERANCE = 1.0e-3


@dataclass(frozen=True)
class VibrationalAnalysis:
    """Mode classification and harmonic thermochemistry.

    Per-mode values keep the order in which the frequencies were supplied.
    Imaginary frequencies are stored as negative numbers.
    """

    frequencies_cm_minus1: np.ndarray
    mode_types: tuple[str, ...]
    rigid_body_mode_count: int
    rigid_body_reason: str
    zero_point_energy_eV: float
    vibrational_entropy_eV_per_K: float
    temperature_K: float
    warnings: tuple[str, ...]

    @property
    def rigid_body_mode_indices(self) -> tuple[int, ...]:
        return self._indices_of(RIGID_BODY)

    @property
    def imaginary_mode_indices(self) -> tuple[int, ...]:
        """Imaginary vibrations; imaginary rigid-body modes are not included."""
        return self._indices_of(IMAGINARY)

    @property
    def vibrational_mode_indices(self) -> tuple[int, ...]:
        """All modes that are not rigid-body motion, imaginary ones included."""
        return tuple(
            index
            for index, mode_type in enumerate(self.mode_types)
            if mode_type != RIGID_BODY
        )

    @property
    def vibrational_entropy_J_per_mol_K(self) -> float:
        return self.vibrational_entropy_eV_per_K * _e * _Nav

    def _indices_of(self, mode_type: str) -> tuple[int, ...]:
        return tuple(
            index for index, value in enumerate(self.mode_types) if value == mode_type
        )


def signed_frequencies(
    frequencies_cm_minus1: Any,
    imaginary_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return real frequencies with imaginary modes as negative values.

    ASE reports an imaginary mode as a complex number with a positive
    imaginary part.  Real input is taken to use the negative-value convention.
    """

    values = np.asarray(frequencies_cm_minus1).reshape(-1)
    if np.iscomplexobj(values):
        imaginary = np.abs(values.imag) > imaginary_tolerance
        return np.where(imaginary, -np.abs(values.imag), values.real)
    return values.astype(float)


def _fixed_atom_indices(atoms: Atoms) -> set[int]:
    return {
        int(index)
        for constraint in atoms.constraints
        if isinstance(constraint, FixAtoms)
        for index in constraint.get_indices()
    }


def vibrated_atom_indices(
    atoms: Atoms,
    indices: Sequence[int] | None = None,
) -> list[int]:
    """Atoms that ``ase.vibrations.Vibrations`` displaces.

    Like ASE, ``indices=None`` means every atom not fixed by ``FixAtoms``.
    """

    if indices is not None:
        return [int(index) for index in indices]
    fixed = _fixed_atom_indices(atoms)
    return [index for index in range(len(atoms)) if index not in fixed]


def is_linear(atoms: Atoms, tolerance: float = LINEAR_INERTIA_TOLERANCE) -> bool:
    """Whether two or more atoms have a vanishing smallest moment of inertia."""

    if len(atoms) < 2:
        return False
    moments = np.sort(atoms.get_moments_of_inertia())
    return bool(moments[0] <= tolerance * moments[-1])


def count_rigid_body_modes(
    atoms: Atoms,
    indices: Sequence[int] | None = None,
) -> tuple[int, str]:
    """Return the number of rigid-body modes in the Hessian and the reason.

    6 for a nonlinear molecule, 5 for a linear one, and 3 for a single atom
    or a periodic cell, where only the translations remain.  A partial Hessian
    has none: the atoms left out through ``indices`` or ``FixAtoms`` hold the
    others in place.  A cell periodic along one axis also keeps the rotation
    about that axis; pass an explicit count for such systems.
    """

    vibrated = vibrated_atom_indices(atoms, indices)
    if len(set(vibrated)) < len(atoms) or _fixed_atom_indices(atoms):
        return 0, "partial Hessian"
    if atoms.pbc.any():
        return 3, "periodic cell"
    if len(atoms) == 1:
        return 3, "single atom"
    if is_linear(atoms):
        return 5, "linear molecule"
    return 6, "nonlinear molecule"


def analyze_vibrational_modes(
    atoms: Atoms,
    frequencies_cm_minus1: Any,
    *,
    indices: Sequence[int] | None = None,
    temperature_K: float = 298.15,
    rigid_body_modes: int | None = None,
    imaginary_tolerance: float = 1.0e-10,
) -> VibrationalAnalysis:
    """Classify normal modes and evaluate the harmonic ZPE and entropy.

    ``frequencies_cm_minus1`` are the 3 * n_vibrated frequencies of a
    Cartesian Hessian, such as ``Vibrations.get_frequencies()``; ``atoms`` and
    ``indices`` must describe the same calculation.  The ``rigid_body_modes``
    modes with the smallest |frequency| are rigid-body motion.  ``None``
    detects the count with :func:`count_rigid_body_modes`.  The remaining
    modes are vibrations; imaginary ones are reported in ``warnings`` and left
    out of the ZPE and entropy.
    """

    if temperature_K <= 0:
        raise ValueError("temperature_K must be greater than zero.")
    frequencies = signed_frequencies(frequencies_cm_minus1, imaginary_tolerance)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Vibrational frequencies must be finite.")
    expected = 3 * len(vibrated_atom_indices(atoms, indices))
    if frequencies.size != expected:
        raise ValueError(
            f"Expected {expected} frequencies for the vibrated atoms, "
            f"got {frequencies.size}."
        )

    if rigid_body_modes is None:
        rigid_count, reason = count_rigid_body_modes(atoms, indices)
    elif (
        not isinstance(rigid_body_modes, int)
        or isinstance(rigid_body_modes, bool)
        or not 0 <= rigid_body_modes <= frequencies.size
    ):
        raise ValueError(
            f"rigid_body_modes must be an integer from 0 to {frequencies.size}."
        )
    else:
        rigid_count, reason = rigid_body_modes, "user override"

    rigid = np.zeros(frequencies.size, dtype=bool)
    rigid[np.argsort(np.abs(frequencies), kind="stable")[:rigid_count]] = True
    imaginary = ~rigid & (frequencies < 0)
    mode_types = tuple(
        RIGID_BODY if is_rigid else IMAGINARY if is_imaginary else VIBRATION
        for is_rigid, is_imaginary in zip(rigid, imaginary)
    )

    energies = frequencies[~rigid & (frequencies > 0)] * invcm
    x = energies / (kB * temperature_K)
    with np.errstate(over="ignore"):
        entropy_terms = x / np.expm1(x) - np.log1p(-np.exp(-x))

    warnings = []
    if rigid_count:
        largest = float(np.max(np.abs(frequencies[rigid])))
        if largest > RIGID_BODY_WARNING_CM_MINUS1:
            warnings.append(
                f"A rigid-body mode has |frequency| {largest:.1f} cm^-1, above "
                f"{RIGID_BODY_WARNING_CM_MINUS1:g} cm^-1. The geometry is probably "
                "not converged (a coarse finite-difference step can also cause "
                "this); check it before trusting the ZPE and entropy."
            )
    imaginary_count = int(np.count_nonzero(imaginary))
    if imaginary_count:
        listed = ", ".join(f"{abs(value):.2f}i" for value in frequencies[imaginary])
        if imaginary_count == 1:
            summary = (
                f"1 imaginary vibrational mode ({listed} cm^-1): the structure "
                "is a first-order saddle point (transition state)"
            )
        else:
            summary = (
                f"{imaginary_count} imaginary vibrational modes ({listed} cm^-1): "
                f"the structure is a saddle point of order {imaginary_count}"
            )
        warnings.append(
            f"{summary}, not a minimum. Imaginary modes are excluded from the "
            "ZPE and entropy."
        )

    return VibrationalAnalysis(
        frequencies_cm_minus1=frequencies,
        mode_types=mode_types,
        rigid_body_mode_count=rigid_count,
        rigid_body_reason=reason,
        zero_point_energy_eV=0.5 * float(np.sum(energies)),
        vibrational_entropy_eV_per_K=float(kB * np.sum(entropy_terms)),
        temperature_K=float(temperature_K),
        warnings=tuple(warnings),
    )
