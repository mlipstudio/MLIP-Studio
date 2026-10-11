"""AIMNet2 support for MLIP Studio through the legacy `aimnet==0.0.1` package.

Install with ``pip install --no-deps aimnet==0.0.1``. Everything it needs at runtime is already
part of the MLIP Studio stack, and unlike newer releases (Python >= 3.11, torch >= 2.8) it runs on
Python 3.10. It ships the AIMNet2 wB97M-D3 and B97-3c models.

The ASE wrapper inside aimnet 0.0.1 (AIMNet2ASE) drops the total charge passed to its constructor
and again whenever it is attached to an Atoms object (isayevlab/AIMNet2#52), so this module calls
the core AIMNet2Calculator directly and passes the charge on every call.
"""
import importlib.util
import logging

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import chemical_symbols

AIMNET2_AVAILABLE = importlib.util.find_spec("aimnet") is not None

# Display name -> aimnet 0.0.1 model registry alias
AIMNET2_MODELS = {
    "AIMNet2 (wB97M-D3)": "aimnet2",
    "AIMNet2 (B97-3c)": "aimnet2_b973c",
}
_AIMNET2_PAPER = (
    "Anstine, Dylan M., Roman Zubatyuk, and Olexandr Isayev. *AIMNet2: a neural network potential to meet "
    "your neutral, charged, organic, and elemental-organic needs.* Chemical Science 16 (2025): 10228-10244."
)
AIMNET2_CITATIONS = {name: _AIMNET2_PAPER for name in AIMNET2_MODELS}
AIMNET2_LICENSE = (
    "AIMNet2 models are licensed under the "
    "[MIT License](https://github.com/isayevlab/aimnetcentral/blob/main/LICENSE)."
)

# H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
AIMNET2_ELEMENTS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53})

# Tasks that need open-shell isolated atoms, periodic cells, spin states or other dedicated models
AIMNET2_UNSUPPORTED_TASKS = frozenset({
    "Atomization/Cohesive Energy",
    "Batch Atomization/Cohesive Energy",
    "Cell + Geometry Optimization",
    "Equation of State",
    "Band Gap and Density of States",
    "Spin Determination",
    "HOMO-LUMO Gap",
    "Batch HOMO-LUMO Gap Prediction",
})


def aimnet2_input_error(atoms, charge):
    """Why the legacy AIMNet2 models cannot handle this structure, or None if they can."""
    if atoms.pbc.any():
        return "AIMNet2 is available for non-periodic molecules only."
    unsupported = sorted(set(atoms.numbers.tolist()) - AIMNET2_ELEMENTS)
    if unsupported:
        symbols = ", ".join(chemical_symbols[z] for z in unsupported)
        supported = ", ".join(chemical_symbols[z] for z in sorted(AIMNET2_ELEMENTS))
        return f"AIMNet2 does not support {symbols}. Supported elements: {supported}."
    n_electrons = int(atoms.numbers.sum()) - round(charge)
    if n_electrons % 2:
        return (
            f"{n_electrons} electrons at total charge {charge:+g}: the AIMNet2 models are closed-shell only, "
            "so the total charge must give an even number of electrons."
        )
    return None


def short_hh_contacts(atoms, cutoff=0.6):
    """H-H pairs closer than `cutoff` Angstrom, as (i, j, distance).

    AIMNet2 has a spurious minimum where two hydrogens on the same atom collapse together
    (isayevlab/aimnetcentral#139), so optimized structures should be checked for it.
    """
    hydrogens = np.flatnonzero(atoms.numbers == 1)
    positions = atoms.positions[hydrogens]
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    rows, cols = np.nonzero(np.triu(distances < cutoff, k=1))
    return [(int(hydrogens[a]), int(hydrogens[b]), float(distances[a, b])) for a, b in zip(rows, cols)]


class AIMNet2LegacyCalculator(Calculator):
    """ASE calculator for the legacy AIMNet2 models.

    The total charge is read from ``atoms.info["charge"]`` when present, otherwise from ``charge``.
    """

    implemented_properties = ["energy", "free_energy", "forces", "charges", "dipole"]

    def __init__(self, model="aimnet2", charge=0, base_calc=None, **kwargs):
        super().__init__(**kwargs)
        if base_calc is None:
            # aimnet 0.0.1 calls logging.basicConfig(level=INFO) on import, which would switch the
            # caller's root logger to INFO; a temporary handler makes that call a no-op
            root_logger = logging.getLogger()
            placeholder = logging.NullHandler()
            root_logger.addHandler(placeholder)
            try:
                from aimnet.calculators import AIMNet2Calculator
            finally:
                root_logger.removeHandler(placeholder)
            base_calc = AIMNet2Calculator(model)
        self.base_calc = base_calc
        self.charge = charge

    def check_state(self, atoms, tol=1e-15):
        # ASE does not compare atoms.info, so a charge change alone must invalidate the cache
        changes = super().check_state(atoms, tol=tol)
        if self.atoms is not None and atoms.info.get("charge") != self.atoms.info.get("charge"):
            changes.append("charge")
        return changes

    def _model_input(self, atoms):
        charge = float(atoms.info.get("charge", self.charge))
        error = aimnet2_input_error(atoms, charge)
        if error:
            raise ValueError(error)
        # Flat (N, 3) input always takes the neighbor-list path. Batched (1, N, 3) input does too
        # below 320 atoms, but above that it switches to a dense all-pairs path on GPU that was
        # 2x slower with 1.6x the memory on ubiquitin (1231 atoms).
        return {"coord": atoms.positions, "numbers": atoms.numbers, "charge": [charge]}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        result = self.base_calc(self._model_input(self.atoms), forces=True)
        energy = float(result["energy"].item())
        charges = result["charges"].detach().cpu().numpy().astype(float)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": result["forces"].detach().cpu().numpy().astype(float),
            "charges": charges,
            "dipole": charges @ self.atoms.positions,  # e*Angstrom, origin-dependent for ions
        }

    def get_hessian(self, atoms=None):
        """Analytic Hessian in eV/Angstrom^2, in MACE's (3N, N, 3) layout."""
        atoms = atoms if atoms is not None else self.atoms
        result = self.base_calc(self._model_input(atoms), forces=True, hessian=True)
        n_atoms = len(atoms)
        return result["hessian"].detach().cpu().numpy().reshape(3 * n_atoms, n_atoms, 3)
