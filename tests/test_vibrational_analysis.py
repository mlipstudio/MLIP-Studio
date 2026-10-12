import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.units import invcm

import mlipstudio
from mlipstudio.vibrational_analysis import (
    analyze_vibrational_modes,
    count_rigid_body_modes,
)

# MACE-OFF24 frequencies of H2O in ASE's order (cm^-1).  The six smallest
# |frequencies| are translations and rotations; 63.2 is a poorly converged one.
WATER_FREQUENCIES = np.array(
    [31.31j, 5.61j, 0.11j, 0.49, 3.66, 63.2, 1626.09, 3838.91, 3945.45]
)


def test_water_rigid_body_modes_are_excluded_from_zpe_and_entropy():
    analysis = analyze_vibrational_modes(molecule("H2O"), WATER_FREQUENCIES)

    assert analysis.rigid_body_mode_count == 6
    assert analysis.rigid_body_reason == "nonlinear molecule"
    assert analysis.mode_types == ("rigid-body",) * 6 + ("vibration",) * 3
    assert analysis.imaginary_mode_indices == ()
    np.testing.assert_allclose(
        analysis.frequencies_cm_minus1,
        [-31.31, -5.61, -0.11, 0.49, 3.66, 63.2, 1626.09, 3838.91, 3945.45],
    )
    assert analysis.zero_point_energy_eV == pytest.approx(
        0.5 * (1626.09 + 3838.91 + 3945.45) * invcm
    )
    assert analysis.zero_point_energy_eV == pytest.approx(0.58337, abs=1e-5)
    assert analysis.vibrational_entropy_J_per_mol_K == pytest.approx(0.0288, abs=1e-4)
    assert isinstance(analysis.vibrational_entropy_eV_per_K, float)
    assert len(analysis.warnings) == 1
    assert "63.2 cm^-1" in analysis.warnings[0]


def test_small_rigid_body_frequencies_do_not_warn():
    frequencies = WATER_FREQUENCIES.copy()
    frequencies[5] = 12.0

    analysis = analyze_vibrational_modes(molecule("H2O"), frequencies)

    assert analysis.warnings == ()


@pytest.mark.parametrize(
    ("atoms", "indices", "expected"),
    [
        (molecule("H2O"), None, (6, "nonlinear molecule")),
        (molecule("CO2"), None, (5, "linear molecule")),
        (molecule("N2"), None, (5, "linear molecule")),
        (Atoms("Ar"), None, (3, "single atom")),
        (bulk("Cu", "fcc", a=3.6), None, (3, "periodic cell")),
        (bulk("NaCl", "rocksalt", a=5.64), None, (3, "periodic cell")),
        (molecule("H2O"), [0], (0, "partial Hessian")),
        (molecule("H2O"), [2, 1, 0], (6, "nonlinear molecule")),
    ],
)
def test_rigid_body_mode_count_detection(atoms, indices, expected):
    assert count_rigid_body_modes(atoms, indices) == expected


def test_fix_atoms_constraint_makes_a_partial_hessian():
    atoms = molecule("H2O")
    atoms.set_constraint(FixAtoms(indices=[0]))

    assert count_rigid_body_modes(atoms) == (0, "partial Hessian")


def test_slightly_bent_linear_molecule_is_still_linear():
    atoms = molecule("CO2")
    atoms.positions[0, 1] += 0.02

    assert count_rigid_body_modes(atoms) == (5, "linear molecule")


def test_rigid_body_override_and_validation():
    analysis = analyze_vibrational_modes(
        molecule("H2O"), WATER_FREQUENCIES, rigid_body_modes=0
    )
    assert analysis.rigid_body_reason == "user override"
    assert analysis.mode_types[:3] == ("imaginary",) * 3

    with pytest.raises(ValueError, match="rigid_body_modes"):
        analyze_vibrational_modes(
            molecule("H2O"), WATER_FREQUENCIES, rigid_body_modes=10
        )
    with pytest.raises(ValueError, match="Expected 9 frequencies"):
        analyze_vibrational_modes(molecule("H2O"), WATER_FREQUENCIES[:6])


def test_imaginary_vibrations_are_reported_and_excluded():
    # 0.3i is a rigid-body mode; 520i is an imaginary vibration.
    frequencies = np.array([0.3j, 0.1, 0.2, 1.5, 2.0, 4.0, 520.0j, 1600.0, 3700.0])

    analysis = analyze_vibrational_modes(molecule("H2O"), frequencies)

    assert analysis.mode_types == ("rigid-body",) * 6 + (
        "imaginary",
        "vibration",
        "vibration",
    )
    assert analysis.imaginary_mode_indices == (6,)
    assert analysis.vibrational_mode_indices == (6, 7, 8)
    assert len(analysis.warnings) == 1
    assert "520.00i cm^-1" in analysis.warnings[0]
    assert "first-order saddle point" in analysis.warnings[0]
    assert analysis.zero_point_energy_eV == pytest.approx(0.5 * (1600.0 + 3700.0) * invcm)

    frequencies[8] = 75.0j
    second_order = analyze_vibrational_modes(molecule("H2O"), frequencies)

    assert second_order.imaginary_mode_indices == (6, 8)
    assert "saddle point of order 2" in second_order.warnings[0]


class SaddleCalculator(Calculator):
    """E = 0.5 * sum(k * x**2) with negative curvature along x."""

    implemented_properties = ["energy", "forces"]
    curvatures = np.array([-1.0, 2.0, 3.0])

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results = {
            "energy": 0.5 * float(np.sum(self.curvatures * positions**2)),
            "forces": -self.curvatures * positions,
        }


def test_vibrational_mode_task_reports_imaginary_vibration():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    result = mlipstudio.VibrationalModeTask(rigid_body_modes=0).calculate(
        atoms, SaddleCalculator()
    )

    assert result.rigid_body_mode_count == 0
    assert result.imaginary_mode_indices == (0,)
    assert result.has_imaginary_modes
    assert result.mode_types == ("imaginary", "vibration", "vibration")
    assert "first-order saddle point" in result.warnings[-1]
    real_energies = result.energies_eV.real[1:]
    assert result.zero_point_energy_eV == pytest.approx(0.5 * real_energies.sum())


def test_vibrational_mode_task_detects_single_atom_translations():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    result = mlipstudio.VibrationalModeTask().calculate(atoms, SaddleCalculator())

    assert result.rigid_body_mode_count == 3
    assert result.rigid_body_reason == "single atom"
    assert result.rigid_body_mode_indices == (0, 1, 2)
    assert result.zero_point_energy_eV == 0.0
    assert result.vibrational_entropy_eV_per_K == 0.0


def test_vibrational_mode_task_rejects_bad_rigid_body_override():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(mlipstudio.TaskValidationError, match="cannot exceed"):
        mlipstudio.VibrationalModeTask(rigid_body_modes=4).calculate(
            atoms, SaddleCalculator()
        )
    with pytest.raises(mlipstudio.TaskValidationError, match="non-negative"):
        mlipstudio.VibrationalModeTask(rigid_body_modes=-1).calculate(
            atoms, SaddleCalculator()
        )
