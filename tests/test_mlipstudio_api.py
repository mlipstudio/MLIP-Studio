import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import mlipstudio


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results = {
            "energy": 0.5 * float(np.sum(positions**2)),
            "forces": -positions.copy(),
            "stress": np.zeros(6),
        }


def test_model_catalog_is_lightweight_and_contains_all_universal_models():
    models = mlipstudio.list_models()

    assert len(models) == 65
    assert len([model for model in models if model.family not in {"In-House", "AIMNet2"}]) == 62
    assert len(mlipstudio.list_models("aimnet2")) == 2
    assert mlipstudio.get_model_spec("MACE MPA Medium").family == "MACE"
    assert len(mlipstudio.list_models("orb")) == 10


def test_unknown_model_suggests_a_close_supported_name():
    with pytest.raises(mlipstudio.UnknownModelError, match="MACE MPA Medium"):
        mlipstudio.get_model_spec("MACE MPA Medum")


def test_model_specific_required_parameters_fail_before_loading_dependencies():
    with pytest.raises(mlipstudio.ModelConfigurationError, match="task_name"):
        mlipstudio.create_calculator("UMA Small 1.2")

    with pytest.raises(mlipstudio.ModelConfigurationError, match="modal"):
        mlipstudio.create_calculator("7net-omni")


def test_task_specific_models_are_marked_in_the_catalog():
    assert (
        mlipstudio.get_model_spec("PET-MAD-DOS").calculator_kind
        == "electronic_structure"
    )
    assert mlipstudio.get_model_spec("QM9-Gap").calculator_kind == "property_predictor"


def test_pet_mad_dos_factory_builds_its_task_specific_provider(monkeypatch):
    calculator_module = types.ModuleType("upet.calculator")

    class FakePETMADDOSCalculator:
        def __init__(self, *, version, device):
            self.version = version
            self.device = device

    calculator_module.PETMADDOSCalculator = FakePETMADDOSCalculator
    monkeypatch.setitem(sys.modules, "upet.calculator", calculator_module)

    calculator = mlipstudio.create_calculator("PET-MAD-DOS", device="cpu")

    assert calculator.version == "latest"
    assert calculator.device == "cpu"
    assert calculator._mlipstudio_model_name == "PET-MAD-DOS"


def test_single_point_returns_raw_values_without_mutating_input():
    atoms = Atoms("H", positions=[[1.0, 2.0, 3.0]])
    original_positions = atoms.get_positions().copy()

    result = mlipstudio.SinglePointTask(
        properties=("energy", "forces"),
    ).calculate(atoms, HarmonicCalculator())

    assert result.energy == pytest.approx(7.0)
    np.testing.assert_allclose(result.forces, [[-1.0, -2.0, -3.0]])
    assert result.max_force == pytest.approx(np.sqrt(14.0))
    assert result.atoms.calc is None
    assert atoms.calc is None
    assert atoms.get_positions() == pytest.approx(original_positions)


def test_single_point_rejects_nonperiodic_stress():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(mlipstudio.TaskValidationError, match="periodic"):
        mlipstudio.SinglePointTask(properties=("stress",)).calculate(
            atoms,
            HarmonicCalculator(),
        )


def test_optimization_converges_and_records_detached_frames():
    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])

    result = mlipstudio.OptimizationTask(
        optimizer="LBFGS",
        fmax=1.0e-4,
        steps=30,
    ).calculate(atoms, HarmonicCalculator())

    assert result.converged
    assert result.atomic_max_force <= 1.0e-4
    assert len(result.trajectory) >= 2
    assert all(frame.calc is None for frame in result.trajectory)
    assert result.atoms.calc is None
    assert atoms.positions[0, 0] == pytest.approx(1.0)


class FakeDOSCalculator:
    def calculate_dos(self, atoms):
        return np.array([-2.0, -1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.5, 0.0])

    def calculate_bandgap(self, atoms, dos):
        return np.array([1.25])

    def calculate_efermi(self, atoms, dos):
        return np.array([-0.5])


def test_band_gap_dos_task_returns_absolute_and_fermi_shifted_grid():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)

    result = mlipstudio.BandGapDOSTask().calculate(atoms, FakeDOSCalculator())

    assert result.band_gap_eV == pytest.approx(1.25)
    assert result.fermi_level_eV == pytest.approx(-0.5)
    np.testing.assert_allclose(result.energies_eV, [-2.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        result.energies_relative_to_fermi_eV,
        [-1.5, -0.5, 0.5, 1.5],
    )


class CurrentPETMADDOSCalculator:
    energy_interval = 0.05

    def __init__(self):
        self.requested_properties = None

    def calculate(self, atoms, properties=("dos_raw", "bandgap", "fermi_level")):
        self.requested_properties = tuple(properties)
        return {
            "dos_denoised": np.array([0.0, 1.0, 0.5, 0.0]),
            "bandgap": np.array([1.25]),
            "fermi_level": np.array([0.10]),
        }


def test_band_gap_dos_task_supports_current_upet_result_dictionary_api():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)
    calculator = CurrentPETMADDOSCalculator()

    result = mlipstudio.BandGapDOSTask().calculate(atoms, calculator)

    assert calculator.requested_properties == (
        "dos_denoised",
        "bandgap",
        "fermi_level",
    )
    assert result.band_gap_eV == pytest.approx(1.25)
    assert result.fermi_level_eV == pytest.approx(0.10)
    np.testing.assert_allclose(result.energies_eV, [0.0, 0.05, 0.10, 0.15])
    np.testing.assert_allclose(
        result.energies_relative_to_fermi_eV,
        [-0.10, -0.05, 0.0, 0.05],
    )
    np.testing.assert_allclose(result.density_of_states, [0.0, 1.0, 0.5, 0.0])


class FakeQM9GapCalculator:
    def calculate_homo_lumo_gap(self, atoms):
        return 4.2


def test_homo_lumo_task_validates_domain_and_returns_gap():
    molecule = Atoms("CH4", positions=np.zeros((5, 3)))
    result = mlipstudio.HOMOLUMOGapTask().calculate(
        molecule,
        FakeQM9GapCalculator(),
    )
    assert result.gap_eV == pytest.approx(4.2)

    unsupported = Atoms("NaH", positions=np.zeros((2, 3)))
    with pytest.raises(mlipstudio.TaskValidationError, match="Na"):
        mlipstudio.HOMOLUMOGapTask().calculate(
            unsupported,
            FakeQM9GapCalculator(),
        )


def test_bundled_qm9_gap_does_not_require_repository_level_modules(monkeypatch):
    for legacy_module in ("predict", "data", "model"):
        monkeypatch.setitem(sys.modules, legacy_module, None)

    calculator = mlipstudio.create_calculator("QM9-Gap", device="cpu")
    result = mlipstudio.HOMOLUMOGapTask().calculate(
        Atoms(
            "CH4",
            positions=[
                [0.0, 0.0, 0.0],
                [0.629118, 0.629118, 0.629118],
                [-0.629118, -0.629118, 0.629118],
                [0.629118, -0.629118, -0.629118],
                [-0.629118, 0.629118, -0.629118],
            ],
        ),
        calculator,
    )

    assert result.gap_eV == pytest.approx(13.629583, abs=1.0e-5)


class SpinSensitiveCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self):
        super().__init__()
        self.evaluated_multiplicities = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        multiplicity = int(atoms.info["spin"])
        self.evaluated_multiplicities.append(multiplicity)
        self.results = {"energy": float((multiplicity - 3) ** 2)}


def test_spin_determination_scans_distinct_states_without_mutating_input():
    atoms = Atoms("O", positions=[[0.0, 0.0, 0.0]])
    calculator = SpinSensitiveCalculator()

    result = mlipstudio.SpinDeterminationTask(
        charge=0,
        multiplicities=(1, 3, 5),
    ).calculate(atoms, calculator)

    assert calculator.evaluated_multiplicities == [1, 3, 5]
    assert result.optimal_state.multiplicity == 3
    assert result.optimal_state.energy_eV == pytest.approx(0.0)
    assert [state.energy_eV for state in result.states] == pytest.approx([4.0, 0.0, 4.0])
    assert "spin" not in atoms.info


def test_spin_determination_rejects_unphysical_multiplicity_parity():
    atoms = Atoms("O", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(mlipstudio.TaskValidationError, match="parity"):
        mlipstudio.SpinDeterminationTask(
            multiplicities=(2,),
        ).calculate(atoms, SpinSensitiveCalculator())


class ReferenceEnergyCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, system_energy, isolated_energies):
        super().__init__()
        self.system_energy = float(system_energy)
        self.isolated_energies = dict(isolated_energies)
        self.evaluated_atomic_numbers = []
        self.system_metadata_seen = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if len(atoms) == 1:
            atomic_number = int(atoms.numbers[0])
            self.evaluated_atomic_numbers.append(atomic_number)
            energy = self.isolated_energies[atomic_number]
        else:
            self.system_metadata_seen = dict(atoms.info)
            energy = self.system_energy
        self.results = {"energy": float(energy)}


def test_atomization_task_calculates_unique_isolated_atom_references():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])
    calculator = ReferenceEnergyCalculator(system_energy=-5.0, isolated_energies={1: -1.0})

    result = mlipstudio.AtomizationCohesiveEnergyTask().calculate(atoms, calculator)

    assert result.calculation_type == "atomization"
    assert result.system_energy_eV == pytest.approx(-5.0)
    assert result.isolated_atoms_energy_eV == pytest.approx(-2.0)
    assert result.atomization_energy_eV == pytest.approx(3.0)
    assert result.cohesive_energy_eV_per_atom is None
    assert result.reference_source == "calculated-with-system-calculator"
    assert calculator.evaluated_atomic_numbers == [1]
    assert atoms.calc is None


def test_atomization_task_applies_system_metadata_to_its_working_copy():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])
    calculator = ReferenceEnergyCalculator(system_energy=-5.0, isolated_energies={1: -1.0})

    result = mlipstudio.AtomizationCohesiveEnergyTask(
        system_metadata={"charge": 0, "spin": 1},
    ).calculate(atoms, calculator)

    assert calculator.system_metadata_seen == {"charge": 0, "spin": 1}
    assert result.atoms.info == {"charge": 0, "spin": 1}
    assert atoms.info == {}


def test_cohesive_task_uses_per_atom_normalization():
    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [0, 0, 0.75]],
        cell=[4, 4, 4],
        pbc=True,
    )
    calculator = ReferenceEnergyCalculator(system_energy=-5.0, isolated_energies={1: -1.0})

    result = mlipstudio.AtomizationCohesiveEnergyTask(
        isolated_atom_energies={"H": -1.0},
    ).calculate(atoms, calculator)

    assert result.calculation_type == "cohesive"
    assert result.cohesive_energy_eV_per_atom == pytest.approx(1.5)
    assert result.atomization_energy_eV is None
    assert result.reference_source == "user-supplied"


def test_atomization_task_rejects_missing_explicit_reference():
    atoms = Atoms("HF", positions=[[0, 0, 0], [0, 0, 1]])
    calculator = ReferenceEnergyCalculator(
        system_energy=-4.0,
        isolated_energies={1: -1.0, 9: -2.0},
    )

    with pytest.raises(mlipstudio.TaskValidationError, match="atomic number.*9"):
        mlipstudio.AtomizationCohesiveEnergyTask(
            isolated_atom_energies={"H": -1.0},
        ).calculate(atoms, calculator)


def test_fairchem_atomization_uses_task_specific_bundled_table():
    from mlipstudio.references import get_element_reference_set

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])
    calculator = ReferenceEnergyCalculator(system_energy=-5.0, isolated_energies={})
    calculator._mlipstudio_model_family = "FairChem"
    calculator._mlipstudio_model_name = "UMA Small 1.2"
    calculator._mlipstudio_parameters = {"task_name": "omol"}

    result = mlipstudio.AtomizationCohesiveEnergyTask().calculate(atoms, calculator)

    hydrogen_reference = get_element_reference_set("omol_elem_refs")[1]
    assert result.isolated_atoms_energy_eV == pytest.approx(2 * hydrogen_reference)
    assert result.reference_source == "bundled:omol_elem_refs"
    assert calculator.evaluated_atomic_numbers == []


class DipoleCalculator(Calculator):
    implemented_properties = ["energy"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": -2.5,
            "dipole": np.array([1.0, 2.0, 2.0]),
            "charges": np.array([-0.4, 0.2, 0.2]),
        }


def test_dipole_task_returns_vector_magnitude_and_partial_charges():
    atoms = Atoms("OH2", positions=np.zeros((3, 3)))
    calculator = DipoleCalculator()
    calculator._mlipstudio_model_name = "MACE POLAR 1 S"

    result = mlipstudio.DipoleMomentTask(
        charge=0,
        spin_multiplicity=1,
    ).calculate(atoms, calculator)

    np.testing.assert_allclose(result.dipole_eA, [1.0, 2.0, 2.0])
    assert result.dipole_magnitude_eA == pytest.approx(3.0)
    np.testing.assert_allclose(result.partial_charges_e, [-0.4, 0.2, 0.2])
    assert result.total_partial_charge_e == pytest.approx(0.0)
    assert "external_field" not in atoms.info


def test_dipole_task_rejects_known_incompatible_model():
    atoms = Atoms("OH2", positions=np.zeros((3, 3)))
    calculator = DipoleCalculator()
    calculator._mlipstudio_model_name = "MACE MPA Medium"

    with pytest.raises(mlipstudio.TaskValidationError, match="not registered"):
        mlipstudio.DipoleMomentTask().calculate(atoms, calculator)
