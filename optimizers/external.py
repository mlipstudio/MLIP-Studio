"""Optional third-party optimizers: Sella and geomeTRIC.

Neither package is a dependency of MLIP Studio. Availability is detected without
importing them, and each package is imported only when its optimizer is used,
so a missing package only removes that option.
"""

from __future__ import annotations

import logging
import tempfile
from importlib.util import find_spec

import numpy as np

SELLA_AVAILABLE = find_spec("sella") is not None
GEOMETRIC_AVAILABLE = find_spec("geometric") is not None


def available_external_optimizers() -> list[str]:
    """Names of the optional optimizers whose packages are installed."""
    names = []
    if SELLA_AVAILABLE:
        names.append("Sella")
    if GEOMETRIC_AVAILABLE:
        names.append("geomeTRIC")
    return names


def make_sella(atoms, trajectory=None, **kwargs):
    """Sella with its default minimization settings.

    ``order=0`` selects minimization (Sella's default, ``order=1``, is a
    saddle-point search). Molecules use Sella's redundant internal coordinates,
    its recommended setup; periodic systems use its default Cartesian mode.
    """
    from sella import Sella

    kwargs.setdefault("order", 0)
    kwargs.setdefault("internal", not np.any(atoms.pbc))
    optimizer = Sella(atoms, **kwargs)
    if trajectory is not None:
        # Sella's own trajectory also stores the dummy atoms it adds for linear
        # angles; write only the real atoms, as ASE optimizers do.
        from ase.io import Trajectory

        optimizer.attach(optimizer.closelater(Trajectory(trajectory, "w", atoms)))
    return optimizer


class GeomeTRICOptimizer:
    """geomeTRIC with default TRIC settings behind a minimal ASE-optimizer interface.

    Provides ``run(fmax, steps)``, ``attach``, ``get_number_of_steps`` and an
    ASE trajectory file. geomeTRIC's default five-criterion convergence test is
    replaced by the ASE criterion (largest atomic force below ``fmax``) so that
    ``fmax`` means the same as for the other optimizers. Molecules only.
    """

    def __init__(self, atoms, trajectory=None, logfile=None, **params):
        if np.any(getattr(atoms, "pbc", True)):
            raise ValueError("geomeTRIC supports non-periodic molecules only.")
        if len(atoms) < 2:
            raise ValueError("geomeTRIC needs at least two atoms.")
        self.atoms = atoms
        self.trajectory = trajectory
        self.params = params
        self.nsteps = 0
        self.observers = []

    def attach(self, function, interval=1, *args, **kwargs):
        if not callable(function):
            function = function.write
        self.observers.append((function, max(1, int(interval)), args, kwargs))

    def get_number_of_steps(self):
        return self.nsteps

    def _call_observers(self):
        for function, interval, args, kwargs in self.observers:
            if self.nsteps % interval == 0:
                function(*args, **kwargs)

    def _max_force(self):
        forces = self.atoms.get_forces()
        return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0

    def run(self, fmax=0.05, steps=100_000_000):
        from ase.io import Trajectory
        from ase.units import Bohr, Hartree
        from geometric.engine import Engine
        from geometric.errors import GeomOptNotConvergedError
        from geometric.internal import DelocalizedInternalCoordinates
        from geometric.molecule import Molecule
        from geometric.nifty import ang2bohr, bohr2ang
        from geometric.optimize import Optimizer
        from geometric.params import OptParams

        if steps <= 0:
            self._call_observers()
            return self._max_force() < fmax

        adapter = self
        traj = Trajectory(self.trajectory, "w", self.atoms) if self.trajectory else None

        class AtomsEngine(Engine):
            def calc_new(self, coords, dirname):
                positions = coords.reshape(-1, 3) * bohr2ang
                if np.max(np.abs(positions - adapter.atoms.positions)) > 1e-10:
                    adapter.atoms.set_positions(positions)
                energy = adapter.atoms.get_potential_energy()
                forces = adapter.atoms.get_forces()
                adapter.nsteps = optimizer.Iteration
                if traj is not None:
                    traj.write(adapter.atoms)
                adapter._call_observers()
                return {"energy": energy / Hartree, "gradient": -forces.ravel() * (Bohr / Hartree)}

        class MaxForceOptimizer(Optimizer):
            def evaluate_OPT_step(self, params, step_state, criteria_met, *args):
                _, max_gradient = self.calcGradNorm()
                converged = bool(max_gradient < fmax * Bohr / Hartree)
                return super().evaluate_OPT_step(params, step_state, converged, *args)

        molecule = Molecule(radii={"Na": 0.0, "K": 0.0}, fragment=False)
        molecule.elem = list(self.atoms.get_chemical_symbols())
        molecule.xyzs = [self.atoms.get_positions().copy()]
        molecule.comms = [self.atoms.get_chemical_formula()]
        molecule.build_topology()
        ic = DelocalizedInternalCoordinates(molecule, build=True, connect=False, addcart=False)
        engine = AtomsEngine(molecule)
        # No geomeTRIC trajectory or log files; the dirname is never written to.
        optimizer = MaxForceOptimizer(
            molecule.xyzs[0].flatten() * ang2bohr,
            molecule,
            ic,
            engine,
            tempfile.gettempdir(),
            OptParams(maxiter=int(steps), **self.params),
            print_info=False,
        )
        geometric_logger = logging.getLogger("geometric.nifty")
        previous_level = geometric_logger.level
        geometric_logger.setLevel(logging.WARNING)
        try:
            optimizer.optimizeGeometry()
            converged = True
        except GeomOptNotConvergedError:
            converged = False
        finally:
            geometric_logger.setLevel(previous_level)
            if traj is not None:
                traj.close()
        self.nsteps = int(optimizer.Iteration)
        # After a rejected step geomeTRIC's current point is the previous geometry.
        final = optimizer.X.reshape(-1, 3) * bohr2ang
        if np.max(np.abs(final - self.atoms.positions)) > 1e-10:
            self.atoms.set_positions(final)
        return converged
