import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import mlipstudio
from mlipstudio.tasks.eos import birch_murnaghan


class ScaledHarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = float(scale)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results = {
            "energy": 0.5 * self.scale * float(np.sum(positions**2)),
            "forces": -self.scale * positions.copy(),
            "stress": np.zeros(6),
        }

    def get_hessian(self, atoms=None):
        return np.eye(3 * len(atoms)) * self.scale


def test_analytical_hessian_task_returns_matrix_and_diagnostics():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])

    result = mlipstudio.AnalyticalHessianTask().calculate(
        atoms, ScaledHarmonicCalculator(scale=2.0)
    )

    np.testing.assert_allclose(result.hessian_eV_per_A2, np.eye(6) * 2.0)
    assert result.symmetry_error_eV_per_A2 == pytest.approx(0.0)
    assert result.maximum_absolute_element_eV_per_A2 == pytest.approx(2.0)
    assert result.atoms.calc is None


def test_vibrational_mode_task_returns_modes_and_cleans_scratch_files():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    result = mlipstudio.VibrationalModeTask().calculate(
        atoms, ScaledHarmonicCalculator()
    )

    assert result.frequencies_cm_minus1.shape == (3,)
    assert result.energies_eV.shape == (3,)
    assert result.modes.shape == (3, 1, 3)
    assert result.imaginary_mode_indices == ()
    assert result.zero_point_energy_eV > 0
    assert result.vibrational_entropy_eV_per_K >= 0


class EOSCalculator(Calculator):
    implemented_properties = ["energy"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = birch_murnaghan(
            np.asarray(atoms.get_volume()), -3.0, 27.0, 0.8, 4.0
        )
        self.results = {"energy": float(energy)}


def test_equation_of_state_recovers_known_parameters():
    atoms = Atoms(
        "Si", positions=[[0, 0, 0]], cell=[3.0, 3.0, 3.0], pbc=True
    )

    result = mlipstudio.EquationOfStateTask(
        num_points=7, volume_range_percent=8
    ).calculate(atoms, EOSCalculator())

    assert result.equilibrium_energy_eV == pytest.approx(-3.0, abs=1.0e-7)
    assert result.equilibrium_volume_A3 == pytest.approx(27.0, abs=1.0e-6)
    assert result.bulk_modulus_eV_per_A3 == pytest.approx(0.8, rel=1.0e-5)
    assert result.pressure_derivative == pytest.approx(4.0, rel=1.0e-5)
    assert result.rmse_eV < 1.0e-8
    assert all(sample.calc is None for sample in result.sampled_structures)


def test_model_consensus_returns_disagreement_arrays():
    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])

    result = mlipstudio.ModelConsensusTask().calculate(
        atoms,
        {
            "model-a": ScaledHarmonicCalculator(scale=1.0),
            "model-b": ScaledHarmonicCalculator(scale=2.0),
        },
    )

    assert result.mean_energy_eV == pytest.approx(0.75)
    assert result.energy_range_eV_per_atom == pytest.approx(0.5)
    np.testing.assert_allclose(result.mean_forces_eV_per_A, [[-1.5, 0.0, 0.0]])
    np.testing.assert_allclose(result.force_disagreement_eV_per_A, [0.5])
    assert result.pairwise_force_rmse_eV_per_A[0, 1] == pytest.approx(
        np.sqrt(1.0 / 3.0)
    )


class FakeGapCalculator:
    def calculate_homo_lumo_gap(self, atoms):
        return float(len(atoms))


class MolecularEnergyCalculator(Calculator):
    implemented_properties = ["energy"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": -1.0 if len(atoms) == 1 else -5.0}


def test_batch_tasks_preserve_order_and_report_failures():
    structures = [
        Atoms("H", positions=[[1, 0, 0]]),
        Atoms("H", positions=[[2, 0, 0]]),
    ]
    single_points = mlipstudio.BatchSinglePointTask(
        properties=("energy", "forces")
    ).calculate(structures, ScaledHarmonicCalculator())
    assert [result.energy for result in single_points.results] == pytest.approx(
        [0.5, 2.0]
    )

    gaps = mlipstudio.BatchHOMOLUMOGapTask().calculate(
        [Atoms("CH4", positions=np.zeros((5, 3))), Atoms("NaH", positions=np.zeros((2, 3)))],
        FakeGapCalculator(),
    )
    assert len(gaps.successful_items) == 1
    assert len(gaps.failed_items) == 1
    assert "Na" in gaps.failed_items[0].error

    atomization = mlipstudio.BatchAtomizationEnergyTask(
        calculation_type="atomization",
        isolated_atom_energies={"H": -1.0},
    ).calculate(
        [
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]]),
        ],
        MolecularEnergyCalculator(),
    )
    assert [result.atomization_energy_eV for result in atomization.results] == pytest.approx(
        [3.0, 3.0]
    )


@pytest.mark.parametrize(
    ("optimizer", "parameters", "metadata_key"),
    [
        ("Lindh Hessian LBFGS", {}, "Number of Lindh Hessian builds"),
        ("MACE Hessian LBFGS", {"require_mace": False}, "Number of analytical Hessian builds"),
        ("MACE-Seed LBFGS", {"require_mace": False}, "Number of analytical Hessian builds"),
    ],
)
def test_optimization_task_loads_specialist_optimizers(
    optimizer, parameters, metadata_key
):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])

    result = mlipstudio.OptimizationTask(
        optimizer=optimizer,
        steps=0,
        optimizer_parameters=parameters,
        record_trajectory=False,
    ).calculate(atoms, ScaledHarmonicCalculator())

    assert result.optimizer == optimizer
    assert metadata_key in result.optimizer_metadata


def test_lindh_optimizer_rejects_cell_optimization():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(mlipstudio.TaskValidationError, match="fixed-cell"):
        mlipstudio.OptimizationTask(
            optimizer="Lindh Hessian LBFGS", optimize_cell=True
        ).calculate(atoms, ScaledHarmonicCalculator())


@pytest.mark.parametrize(("optimizer", "package"), [("Sella", "sella"), ("geomeTRIC", "geometric")])
def test_optional_optimizers_run_when_installed(optimizer, package):
    pytest.importorskip(package)
    from ase.build import molecule
    from ase.calculators.emt import EMT

    result = mlipstudio.OptimizationTask(optimizer=optimizer, fmax=0.05, steps=100).calculate(
        molecule("H2O"), EMT()
    )

    assert result.converged
    assert result.atomic_max_force < 0.05
    # Same frame count as the ASE optimizers: the stored start plus one per observer call.
    assert len(result.trajectory) == result.steps + 2
    assert {len(frame) for frame in result.trajectory} == {3}


def test_missing_optional_optimizer_reports_not_installed(monkeypatch):
    from mlipstudio.tasks import optimization

    monkeypatch.setattr(optimization, "find_spec", lambda name: None)
    with pytest.raises(mlipstudio.TaskCalculationError, match="not installed"):
        optimization._optimizer_class("Sella")
    with pytest.raises(mlipstudio.TaskValidationError) as unknown:
        optimization._optimizer_class("Unknown")
    assert "Sella" not in str(unknown.value) and "geomeTRIC" not in str(unknown.value)


def test_geometric_rejects_periodic_structures():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(mlipstudio.TaskValidationError, match="non-periodic"):
        mlipstudio.OptimizationTask(optimizer="geomeTRIC").calculate(
            atoms, ScaledHarmonicCalculator()
        )

