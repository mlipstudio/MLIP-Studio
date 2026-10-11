"""AIMNet2 support in the mlipstudio API.

The fast tests use fakes; the end-to-end test runs only when aimnet is installed.
"""
import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes

import aimnet2_legacy
import mlipstudio

_A = 1.03 / np.sqrt(3)
AMMONIUM_POSITIONS = [[0, 0, 0], [_A, _A, _A], [-_A, -_A, _A], [-_A, _A, -_A], [_A, -_A, -_A]]


def test_aimnet2_models_are_in_the_catalog():
    models = mlipstudio.list_models("AIMNet2")

    assert [model.name for model in models] == ["AIMNet2 (wB97M-D3)", "AIMNet2 (B97-3c)"]
    assert [model.identifier for model in models] == ["aimnet2", "aimnet2_b973c"]
    for model in models:
        assert model.default_parameters == {"charge": 0}
        assert model.allowed_parameters == ("charge",)
        assert "Chemical Science 16" in model.citation


@pytest.fixture
def fake_aimnet(monkeypatch):
    """Stand in for the aimnet package and the legacy wrapper's model loading."""
    monkeypatch.setitem(sys.modules, "aimnet", types.ModuleType("aimnet"))
    created = []

    class FakeLegacyCalculator:
        def __init__(self, *, model, charge):
            self.model = model
            self.charge = charge
            created.append(self)

    monkeypatch.setattr(aimnet2_legacy, "AIMNet2LegacyCalculator", FakeLegacyCalculator)
    return created


def _set_cuda_available(monkeypatch, available):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    return "cuda" if available else "cpu"


def test_aimnet2_factory_passes_model_and_charge(monkeypatch, fake_aimnet):
    device = _set_cuda_available(monkeypatch, False)

    calculator = mlipstudio.create_calculator("AIMNet2 (B97-3c)", device=device, charge=-1)

    assert calculator.model == "aimnet2_b973c"
    assert calculator.charge == -1
    assert calculator._mlipstudio_model_family == "AIMNet2"
    assert calculator._mlipstudio_parameters == {"charge": -1}


@pytest.mark.parametrize("cuda_available, device", [(True, "cpu"), (False, "cuda")])
def test_aimnet2_device_must_match_where_aimnet_runs(monkeypatch, fake_aimnet, cuda_available, device):
    _set_cuda_available(monkeypatch, cuda_available)

    with pytest.raises(mlipstudio.ModelConfigurationError, match="pass device="):
        mlipstudio.create_calculator("AIMNet2 (wB97M-D3)", device=device)
    assert fake_aimnet == []


def test_aimnet2_charge_must_be_an_integer():
    with pytest.raises(mlipstudio.ModelConfigurationError, match="charge must be an integer"):
        mlipstudio.create_calculator("AIMNet2 (wB97M-D3)", charge=0.5)


def test_aimnet2_reports_missing_aimnet_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "aimnet", None)

    with pytest.raises(mlipstudio.ModelDependencyError, match="'aimnet'"):
        mlipstudio.create_calculator("AIMNet2 (wB97M-D3)")


class ChargeAwareCalculator(Calculator):
    implemented_properties = ["energy", "charges", "dipole"]

    def __init__(self):
        super().__init__()
        self.evaluated_sizes = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.evaluated_sizes.append(len(atoms))
        charges = np.full(len(atoms), float(atoms.info.get("charge", 0)) / len(atoms))
        self.results = {"energy": -5.0, "charges": charges, "dipole": charges @ atoms.positions}


def _as_aimnet2(calculator):
    calculator._mlipstudio_model_name = "AIMNet2 (wB97M-D3)"
    calculator._mlipstudio_model_family = "AIMNet2"
    return calculator


def test_dipole_task_accepts_aimnet2_and_passes_the_charge():
    atoms = Atoms("NH4", positions=AMMONIUM_POSITIONS)

    result = mlipstudio.DipoleMomentTask(charge=1).calculate(atoms, _as_aimnet2(ChargeAwareCalculator()))

    assert result.total_partial_charge_e == pytest.approx(1.0)


@pytest.mark.parametrize(
    "options, message",
    [({"spin_multiplicity": 2}, "closed-shell"), ({"external_field": (0.0, 0.0, 0.1)}, "electric field")],
)
def test_dipole_task_rejects_open_shell_or_field_for_aimnet2(options, message):
    atoms = Atoms("NH4", positions=AMMONIUM_POSITIONS)

    with pytest.raises(mlipstudio.TaskValidationError, match=message):
        mlipstudio.DipoleMomentTask(charge=1, **options).calculate(atoms, _as_aimnet2(ChargeAwareCalculator()))


def test_aimnet2_atomization_needs_explicit_references():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])

    with pytest.raises(mlipstudio.TaskValidationError, match="isolated_atom_energies"):
        mlipstudio.AtomizationCohesiveEnergyTask().calculate(atoms, _as_aimnet2(ChargeAwareCalculator()))

    calculator = _as_aimnet2(ChargeAwareCalculator())
    result = mlipstudio.AtomizationCohesiveEnergyTask(
        isolated_atom_energies={"H": -1.0},
    ).calculate(atoms, calculator)

    assert result.atomization_energy_eV == pytest.approx(3.0)
    assert calculator.evaluated_sizes == [2]


def test_aimnet2_end_to_end_with_the_real_model():
    pytest.importorskip("aimnet")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = mlipstudio.create_calculator("AIMNet2 (wB97M-D3)", device=device)
    water = molecule("H2O")

    single_point = mlipstudio.SinglePointTask(properties=("energy", "forces")).calculate(water, calculator)
    hessian = mlipstudio.AnalyticalHessianTask().calculate(water, calculator)
    dipole = mlipstudio.DipoleMomentTask(charge=1).calculate(
        Atoms("NH4", positions=AMMONIUM_POSITIONS),
        calculator,
    )

    assert single_point.energy == pytest.approx(-2081.0, abs=0.5)
    assert single_point.forces.shape == (3, 3)
    assert hessian.hessian_eV_per_A2.shape == (9, 9)
    assert hessian.symmetry_error_eV_per_A2 < 1.0e-3
    assert dipole.total_partial_charge_e == pytest.approx(1.0, abs=1.0e-4)
    assert dipole.energy_eV == pytest.approx(-1549.2070, abs=1.0e-3)
