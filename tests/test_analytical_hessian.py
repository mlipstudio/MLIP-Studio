import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter

from optimizers import (
    AnalyticalHessianError,
    AnalyticalHessianPreconditioner,
    MACEHessianLBFGS,
    MACESeedLBFGS,
)


class QuadraticHessianCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(self, target, curvature=2.5):
        super().__init__()
        self.target = np.asarray(target, dtype=float)
        self.curvature = float(curvature)
        self.hessian_calls = 0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        delta = atoms.get_positions() - self.target
        self.results["energy"] = 0.5 * self.curvature * float(np.sum(delta * delta))
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -self.curvature * delta
        self.results["stress"] = np.zeros(6)

    def get_hessian(self, atoms=None):
        self.hessian_calls += 1
        size = 3 * len(atoms)
        return np.eye(size) * self.curvature


class MACEQuadraticHessianCalculator(QuadraticHessianCalculator):
    """Test double whose class name exercises the default MACE-only guard."""


def test_analytical_hessian_preconditioner_uses_calculator_hessian():
    target = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    calc = QuadraticHessianCalculator(target, curvature=3.0)
    atoms.calc = calc

    precon = AnalyticalHessianPreconditioner(eigenvalue_floor=0.02, require_mace=False)
    matrix = precon.make_precon(atoms)

    assert matrix.shape == (6, 6)
    assert np.allclose(matrix, np.eye(6) * 3.0)
    assert calc.hessian_calls == 1
    assert precon.n_builds == 1
    assert precon.n_fallbacks == 0


def test_mace_hessian_lbfgs_converges_quadratic():
    target = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    calc = QuadraticHessianCalculator(target, curvature=2.0)
    atoms.calc = calc

    opt = MACEHessianLBFGS(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=0.2,
        memory=5,
        rebuild_interval=1,
        require_mace=False,
    )
    converged = opt.run(fmax=1.0e-4, steps=20)

    assert converged
    assert np.allclose(atoms.get_positions(), target, atol=1.0e-3)
    assert opt.n_hessian_builds >= 1


def test_mace_hessian_lbfgs_accepts_separate_mace_hessian_calculator():
    target = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    target_calc = QuadraticHessianCalculator(target, curvature=1.4)
    hessian_calc = MACEQuadraticHessianCalculator(target, curvature=2.0)
    atoms.calc = target_calc

    opt = MACEHessianLBFGS(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=0.2,
        memory=5,
        rebuild_interval=1,
        hessian_calculator=hessian_calc,
        hessian_calculator_label="MACE test Hessian",
    )
    converged = opt.run(fmax=1.0e-4, steps=20)

    assert converged
    assert np.allclose(atoms.get_positions(), target, atol=1.0e-3)
    assert hessian_calc.hessian_calls >= 1
    assert target_calc.hessian_calls == 0
    assert opt.get_hessian_metadata()["Separate Hessian calculator"] is True
    assert opt.get_hessian_metadata()["Hessian calculator"] == "MACE test Hessian"


def test_mace_hessian_lbfgs_converges_without_line_search():
    target = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    atoms.calc = MACEQuadraticHessianCalculator(target, curvature=2.0)
    opt = MACEHessianLBFGS(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=0.2,
        memory=5,
    )

    converged = opt.run(fmax=1.0e-4, steps=20)

    assert converged
    assert np.allclose(atoms.get_positions(), target, atol=1.0e-3)
    assert opt.get_hessian_metadata()["Line search"] == "None"
    assert opt.n_fixed_step_fallbacks == 0


def test_mace_hessian_lbfgs_respects_maxstep():
    target = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    atoms.calc = MACEQuadraticHessianCalculator(target, curvature=2.0)
    initial = atoms.get_positions().copy()
    opt = MACEHessianLBFGS(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=0.05,
        memory=5,
    )

    opt.run(fmax=0.0, steps=1)

    displacement = np.linalg.norm(atoms.get_positions() - initial, axis=1)
    assert displacement.max() <= 0.05 + 1.0e-12


def test_mace_seed_lbfgs_uses_one_hessian_and_no_line_search():
    target = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    atoms.calc = MACEQuadraticHessianCalculator(target, curvature=2.0)

    opt = MACESeedLBFGS(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=0.2,
        memory=5,
    )
    converged = opt.run(fmax=1.0e-4, steps=30)

    assert converged
    assert opt.n_hessian_builds == 1
    assert atoms.calc.hessian_calls == 1
    assert np.allclose(atoms.get_positions(), target, atol=1.0e-3)
    metadata = opt.get_hessian_metadata()
    assert metadata["Optimizer"] == "MACE-Seed LBFGS"
    assert metadata["Line search"] == "None"
    assert metadata["Step strategy"] == "response-controlled radius; one trial per cycle"


def test_mace_seed_lbfgs_contracts_after_a_bad_seeded_step():
    target = np.zeros((1, 3))
    atoms = Atoms("H", positions=[[0.05, 0.0, 0.0]])
    target_calc = QuadraticHessianCalculator(target, curvature=20.0)
    misleading_hessian = MACEQuadraticHessianCalculator(target, curvature=0.1)
    atoms.calc = target_calc
    opt = MACESeedLBFGS(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=0.2,
        initial_step_radius=0.2,
        minimum_step_radius=0.01,
        hessian_calculator=misleading_hessian,
    )

    opt.run(fmax=0.0, steps=2)

    assert opt.n_step_contractions >= 1
    assert opt.seed_weight < 1.0
    assert opt.n_hessian_builds == 1


def test_mace_seed_lbfgs_supports_periodic_fixed_cell_atoms():
    cell = np.eye(3) * 5.43
    target = np.array([[0.1, 0.1, 0.1], [1.45, 1.45, 1.45]])
    atoms = Atoms(
        "Si2",
        positions=[[0.25, 0.1, 0.1], [1.45, 1.65, 1.45]],
        cell=cell,
        pbc=True,
    )
    atoms.calc = MACEQuadraticHessianCalculator(target, curvature=2.0)

    opt = MACESeedLBFGS(atoms, logfile=None, maxstep=0.2, memory=5)
    converged = opt.run(fmax=1.0e-4, steps=30)

    assert converged
    assert opt.periodic_system is True
    assert opt.optimizes_cell is False
    assert opt.n_hessian_builds == 1
    assert np.allclose(atoms.get_positions(), target, atol=1.0e-3)


def test_mace_seed_lbfgs_supports_cell_filter_coordinates():
    cell = np.eye(3) * 5.43
    target = np.array([[0.1, 0.1, 0.1], [1.45, 1.45, 1.45]])
    atoms = Atoms(
        "Si2",
        positions=[[0.25, 0.1, 0.1], [1.45, 1.65, 1.45]],
        cell=cell,
        pbc=True,
    )
    atoms.calc = MACEQuadraticHessianCalculator(target, curvature=2.0)
    filtered = FrechetCellFilter(atoms)

    opt = MACESeedLBFGS(filtered, logfile=None, maxstep=0.2, memory=5)
    converged = opt.run(fmax=1.0e-4, steps=30)

    assert converged
    assert opt.optimizes_cell is True
    assert opt.n_hessian_builds == 1
    assert np.allclose(atoms.get_positions(), target, atol=1.0e-3)


def test_mace_hessian_lbfgs_accepts_periodic_fixed_cell_atoms():
    cell = np.eye(3) * 5.43
    target = np.array([[0.1, 0.1, 0.1], [1.45, 1.45, 1.45]])
    atoms = Atoms(
        "Si2",
        positions=[[0.25, 0.1, 0.1], [1.45, 1.65, 1.45]],
        cell=cell,
        pbc=True,
    )
    atoms.calc = MACEQuadraticHessianCalculator(target, curvature=2.0)

    opt = MACEHessianLBFGS(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=0.2,
        memory=5,
        rebuild_interval=1,
    )
    converged = opt.run(fmax=1.0e-4, steps=20)

    assert converged
    assert opt.periodic_system is True
    assert opt.optimizes_cell is False
    assert opt.n_hessian_builds >= 1
    assert opt.n_hessian_fallbacks == 0
    assert np.allclose(atoms.cell.array, cell)
    assert np.allclose(atoms.get_positions(), target, atol=1.0e-3)


def test_mace_hessian_lbfgs_embeds_cell_filter_degrees_of_freedom():
    cell = np.eye(3) * 5.43
    target = np.array([[0.1, 0.1, 0.1], [1.45, 1.45, 1.45]])
    atoms = Atoms(
        "Si2",
        positions=[[0.25, 0.1, 0.1], [1.45, 1.65, 1.45]],
        cell=cell,
        pbc=True,
    )
    atoms.calc = MACEQuadraticHessianCalculator(target, curvature=2.0)
    filtered = FrechetCellFilter(atoms)

    opt = MACEHessianLBFGS(
        filtered,
        logfile=None,
        trajectory=None,
        maxstep=0.2,
        memory=5,
        rebuild_interval=1,
    )
    converged = opt.run(fmax=1.0e-4, steps=20)

    assert converged
    assert opt.periodic_system is True
    assert opt.optimizes_cell is True
    assert opt.n_hessian_builds >= 1
    assert opt.n_hessian_fallbacks == 0
    assert opt.analytical_hessian_precon.asarray().shape == (15, 15)
    assert np.allclose(atoms.get_positions(), target, atol=1.0e-3)


def test_mace_hessian_lbfgs_rejects_non_mace_calculator_by_default():
    target = np.array([[0.0, 0.0, 0.0]])
    atoms = Atoms("H", positions=[[0.1, 0.0, 0.0]])
    atoms.calc = QuadraticHessianCalculator(target, curvature=1.0)

    try:
        MACEHessianLBFGS(atoms, logfile=None, trajectory=None)
    except AnalyticalHessianError as exc:
        assert "requires a MACE calculator" in str(exc)
    else:
        raise AssertionError("MACEHessianLBFGS accepted a non-MACE calculator.")
