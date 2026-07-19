import math

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import LBFGS

from optimizers.lindh import (
    LindhError,
    LindhHessianBuilder,
    LindhHessianLBFGS,
    LindhPreconditioner,
    _minimum_image_pair_vectors,
)


class HarmonicReferenceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, reference, k=2.0):
        super().__init__()
        self.reference = np.asarray(reference, dtype=float)
        self.k = float(k)
        self.energy_calls = 0
        self.force_calls = 0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        disp = atoms.get_positions() - self.reference
        self.results["energy"] = 0.5 * self.k * float(np.sum(disp * disp))
        self.results["forces"] = -self.k * disp
        self.energy_calls += 1
        self.force_calls += 1


class BondHarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, equilibrium=0.74, k=4.0):
        super().__init__()
        self.equilibrium = float(equilibrium)
        self.k = float(k)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        vector = atoms.positions[1] - atoms.positions[0]
        distance = float(np.linalg.norm(vector))
        direction = vector / distance
        extension = distance - self.equilibrium
        force = self.k * extension * direction
        self.results["energy"] = 0.5 * self.k * extension**2
        self.results["forces"] = np.array([force, -force])


class PeriodicBondHarmonicCalculator(BondHarmonicCalculator):
    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        vector = atoms.get_distance(0, 1, mic=True, vector=True)
        distance = float(np.linalg.norm(vector))
        direction = vector / distance
        extension = distance - self.equilibrium
        force = self.k * extension * direction
        self.results["energy"] = 0.5 * self.k * extension**2
        self.results["forces"] = np.array([force, -force])


def water(distortion=0.0):
    atoms = Atoms(
        "OH2",
        positions=[
            [0.0000, 0.0000, 0.0000],
            [0.9584, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        pbc=False,
    )
    atoms.positions += distortion
    return atoms


def rotation_matrix_z(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms("He", positions=[[0.0, 0.0, 0.0]], pbc=False),
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]], pbc=False),
        Atoms("CO2", positions=[[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]], pbc=False),
        water(),
        Atoms("HeNe", positions=[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], pbc=False),
        Atoms("HOH", positions=[[-1.0, 0.0, 0.0], [0.0, 1.0e-6, 0.0], [1.0, 0.0, 0.0]], pbc=False),
    ],
)
def test_hessian_shape_symmetry_and_finiteness(atoms):
    hessian = LindhHessianBuilder().build(atoms)
    assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
    assert np.all(np.isfinite(hessian))
    assert np.allclose(hessian, hessian.T)


def test_regularized_active_matrix_is_positive_definite():
    atoms = water()
    precon = LindhPreconditioner(eigenvalue_floor=0.02)
    precon.make_precon(atoms)
    assert np.min(precon._eigenvalues) >= 0.02


def test_periodic_preconditioner_uses_cholesky_diagonal_shift():
    atoms = water()
    atoms.set_cell([8.0, 8.0, 8.0])
    atoms.set_pbc(True)
    precon = LindhPreconditioner(eigenvalue_floor=0.02)
    matrix = precon.make_precon(atoms)
    vector = np.arange(1, 3 * len(atoms) + 1, dtype=float)
    solved = precon.solve(vector)

    assert precon.regularization_method == "diagonal shift with CPU Cholesky solve"
    assert precon._cholesky_factor is not None
    assert np.allclose(matrix @ solved, vector, atol=1.0e-8)


def test_orthogonal_periodic_fast_mic_matches_ase():
    atoms = Atoms(
        "H4",
        positions=[
            [0.2, 0.3, 0.4],
            [4.9, 0.1, 0.2],
            [2.1, 4.8, 0.7],
            [3.8, 3.9, 4.7],
        ],
        cell=[[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=[True, True, False],
    )
    expected = -atoms.get_all_distances(mic=True, vector=True)
    assert np.allclose(_minimum_image_pair_vectors(atoms), expected, atol=1.0e-12)


def test_periodic_analytic_torsion_matches_python_finite_difference():
    atoms = Atoms(
        "CCCC",
        positions=[
            [4.8, 2.1, 2.0],
            [0.4, 2.4, 2.2],
            [1.7, 2.0, 2.8],
            [2.8, 2.7, 3.1],
        ],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    numba_hessian = LindhHessianBuilder(use_numba=True).build(atoms)
    python_hessian = LindhHessianBuilder(use_numba=False).build(atoms)
    assert np.allclose(numba_hessian, python_hessian, atol=2.0e-4)


def test_molecular_analytic_torsion_matches_python_finite_difference():
    atoms = Atoms(
        "CCCC",
        positions=[
            [0.0, 0.0, 0.0],
            [1.4, 0.1, 0.2],
            [2.6, 0.8, 0.5],
            [3.8, 0.2, 1.1],
        ],
        pbc=False,
    )
    numba_hessian = LindhHessianBuilder(use_numba=True).build(atoms)
    python_hessian = LindhHessianBuilder(use_numba=False).build(atoms)
    assert np.allclose(numba_hessian, python_hessian, atol=2.0e-4)


def test_translation_invariance():
    atoms = water()
    shifted = atoms.copy()
    shifted.positions += np.array([4.0, -3.0, 2.0])
    builder = LindhHessianBuilder()
    assert np.allclose(builder.build(atoms), builder.build(shifted), atol=1.0e-10)


def test_rotation_covariance():
    atoms = water()
    rot = rotation_matrix_z(0.37)
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rot.T
    hessian = LindhHessianBuilder().build(atoms)
    rotated_hessian = LindhHessianBuilder().build(rotated)
    transform = np.kron(np.eye(len(atoms)), rot)
    expected = transform @ hessian @ transform.T
    assert np.allclose(rotated_hessian, expected, atol=2.0e-4)


def test_molecular_numba_and_python_builders_agree():
    atoms = water()
    numba_hessian = LindhHessianBuilder(use_numba=True).build(atoms)
    python_hessian = LindhHessianBuilder(use_numba=False).build(atoms)
    assert np.allclose(numba_hessian, python_hessian, atol=2.0e-4)


@pytest.mark.parametrize("use_numba", [False, True])
def test_periodic_minimum_image_matches_equivalent_unwrapped_dimer(use_numba):
    periodic = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    unwrapped = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        pbc=False,
    )
    periodic_hessian = LindhHessianBuilder(use_numba=use_numba).build(periodic)
    molecular_hessian = LindhHessianBuilder(use_numba=use_numba).build(unwrapped)
    assert np.allclose(periodic_hessian, molecular_hessian, atol=1.0e-10)


@pytest.mark.parametrize("use_numba", [False, True])
def test_periodic_hessian_is_invariant_to_coordinate_wrapping(use_numba):
    atoms = Atoms(
        "OH2",
        positions=[
            [4.80, 2.50, 2.50],
            [5.7584, 2.50, 2.50],
            [4.5610, 3.4270, 2.50],
        ],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    wrapped = atoms.copy()
    wrapped.wrap()
    original_hessian = LindhHessianBuilder(use_numba=use_numba).build(atoms)
    wrapped_hessian = LindhHessianBuilder(use_numba=use_numba).build(wrapped)
    assert np.allclose(original_hessian, wrapped_hessian, atol=2.0e-4)


def test_fixed_atoms_receive_zero_displacement():
    atoms = water(distortion=0.2)
    atoms.set_constraint(FixAtoms(indices=[0]))
    precon = LindhPreconditioner(eigenvalue_floor=0.02)
    precon.make_precon(atoms)
    step = precon.solve(np.ones(3 * len(atoms))).reshape((-1, 3))
    assert np.allclose(step[0], 0.0)


def test_soft_modes_do_not_create_enormous_steps():
    atoms = water()
    precon = LindhPreconditioner(eigenvalue_floor=0.05)
    precon.make_precon(atoms)
    step = precon.solve(np.ones(3 * len(atoms)))
    assert np.linalg.norm(step.reshape((-1, 3)), axis=1).max() < 100.0


def test_lindh_hessian_lbfgs_decreases_and_converges(tmp_path):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    atoms.calc = BondHarmonicCalculator(equilibrium=0.74, k=4.0)
    initial_energy = atoms.get_potential_energy()
    traj = tmp_path / "lindh.traj"
    callback_steps = []
    opt = LindhHessianLBFGS(atoms, trajectory=str(traj), logfile=None, maxstep=0.2, memory=5)
    opt.attach(lambda: callback_steps.append(opt.get_number_of_steps()), interval=1)
    converged = opt.run(fmax=0.02, steps=80)
    assert converged
    assert atoms.get_potential_energy() < initial_energy
    assert np.max(np.linalg.norm(atoms.get_forces(), axis=1)) < 0.02
    assert abs(atoms.get_distance(0, 1) - 0.74) < 1.0e-3
    assert len(read(str(traj), index=":")) > 0
    assert callback_steps


def test_lindh_hessian_lbfgs_converges_without_line_search():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    atoms.calc = BondHarmonicCalculator(equilibrium=0.74, k=4.0)

    opt = LindhHessianLBFGS(atoms, logfile=None, trajectory=None, maxstep=0.2, memory=5)
    converged = opt.run(fmax=0.02, steps=80)

    assert converged
    assert np.max(np.linalg.norm(atoms.get_forces(), axis=1)) < 0.02
    assert abs(atoms.get_distance(0, 1) - 0.74) < 1.0e-3
    assert opt.get_lindh_metadata()["Line search"] == "None"
    assert opt.n_fixed_step_fallbacks == 0


def test_lindh_agrees_with_lbfgs_on_harmonic_problem(tmp_path):
    target = water().positions
    displaced = water()
    displaced.positions += np.array([[0.12, 0.05, 0.0], [-0.10, 0.02, 0.03], [0.04, -0.06, -0.02]])

    lindh_atoms = displaced.copy()
    lindh_atoms.calc = HarmonicReferenceCalculator(target, k=3.0)
    LindhHessianLBFGS(lindh_atoms, logfile=None, trajectory=str(tmp_path / "lindh.traj")).run(fmax=0.02, steps=80)

    lbfgs_atoms = displaced.copy()
    lbfgs_atoms.calc = HarmonicReferenceCalculator(target, k=3.0)
    LBFGS(lbfgs_atoms, logfile=None, trajectory=str(tmp_path / "lbfgs.traj")).run(fmax=0.02, steps=80)

    assert abs(lindh_atoms.get_potential_energy() - lbfgs_atoms.get_potential_energy()) < 1.0e-4
    assert np.linalg.norm(lindh_atoms.positions - lbfgs_atoms.positions) < 2.0e-2


def test_curvature_update_failure_is_skipped():
    atoms = water()
    atoms.calc = HarmonicReferenceCalculator(atoms.positions, k=1.0)
    opt = LindhHessianLBFGS(atoms, logfile=None)
    opt._just_reset_hessian = False
    opt.update(np.zeros((3, 3)), np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3)))
    assert len(opt.s) == 0


def test_periodic_lindh_hessian_lbfgs_converges_across_cell_boundary():
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [4.95, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.calc = PeriodicBondHarmonicCalculator(equilibrium=0.74, k=4.0)
    opt = LindhHessianLBFGS(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=0.2,
        memory=5,
    )
    converged = opt.run(fmax=0.02, steps=80)

    assert converged
    assert abs(atoms.get_distance(0, 1, mic=True) - 0.74) < 1.0e-3
    assert opt.periodic_system is True
    metadata = opt.get_lindh_metadata()
    assert metadata["Line search"] == "None"
    assert metadata["Periodic displacement convention"] == "ASE minimum image"
    assert opt.n_lindh_fallbacks == 0
