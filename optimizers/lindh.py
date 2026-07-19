"""Lindh-model-Hessian-preconditioned L-BFGS optimizer.

This module implements a Cartesian model Hessian based on the distance
dependent force constants proposed by Lindh, Bernhardsson, Karlstrom, and
Malmqvist, "On the use of a Hessian model function in molecular geometry
optimizations", Chemical Physics Letters 241, 423-428 (1995).

The paper defines, in atomic units,

    rho_ij = exp(alpha_ij * (r_ref,ij**2 - r_ij**2))
    k_ij = k_r * rho_ij
    k_ijk = k_phi * rho_ij * rho_jk
    k_ijkl = k_tau * rho_ij * rho_jk * rho_kl

for all chemically meaningful stretch, bend, and torsional primitive
coordinates. The Hessian used here is the local quadratic internal-coordinate
approximation B.T @ K @ B transformed to Cartesian coordinates and converted
to eV / Angstrom**2. Ordinary ASE LBFGS starts its two-loop recursion from a
scalar/identity-like inverse curvature estimate; ``LindhHessianLBFGS`` replaces
that central inverse action with a solve against this regularized Lindh model
while retaining the L-BFGS history corrections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import math
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian
from ase.optimize.precon import PreconLBFGS
from ase.optimize.precon.precon import Precon
from ase.units import Bohr, Hartree
from scipy.linalg import cho_factor, cho_solve

from .fixed_step import FixedStepPreconLBFGSMixin

try:  # Numba is optional; the pure NumPy/Python builder remains the fallback.
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent.
    njit = None
    NUMBA_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class LindhError(ValueError):
    """Raised when Lindh Hessian LBFGS is requested outside its supported scope."""


@dataclass(frozen=True)
class LindhParameters:
    """Published Lindh model parameters from Table 1 of the 1995 paper.

    Parameters are in atomic units. The original parameterization covers atoms
    in the first three periodic-table rows. Heavier elements are conservatively
    mapped to row 3 so the optimizer can still regularize the geometry without
    inventing additional element-specific constants.
    """

    k_stretch: float = 0.45
    k_bend: float = 0.15
    k_torsion: float = 0.005
    alpha_by_row: np.ndarray = field(
        default_factory=lambda: np.array(
            [[1.0000, 0.3949, 0.3949],
             [0.3949, 0.2800, 0.2800],
             [0.3949, 0.2800, 0.2800]],
            dtype=float,
        )
    )
    r_ref_by_row: np.ndarray = field(
        default_factory=lambda: np.array(
            [[1.35, 2.10, 2.53],
             [2.10, 2.87, 3.40],
             [2.53, 3.40, 3.40]],
            dtype=float,
        )
    )

    @staticmethod
    def row_for_atomic_number(z: int) -> int:
        """Return the Lindh row index 0, 1, or 2 for an atomic number."""
        if z <= 2:
            return 0
        if z <= 10:
            return 1
        return 2

    def pair_values(self, zi: int, zj: int) -> tuple[float, float, bool]:
        """Return ``(alpha_ij, r_ref_ij_bohr, used_fallback)``."""
        ri = self.row_for_atomic_number(int(zi))
        rj = self.row_for_atomic_number(int(zj))
        fallback = zi > 18 or zj > 18
        return (
            float(self.alpha_by_row[ri, rj]),
            float(self.r_ref_by_row[ri, rj]),
            fallback,
        )


@dataclass
class PrimitiveCoordinate:
    """Diagnostic description of one model-Hessian primitive."""

    kind: str
    atoms: tuple[int, ...]
    force_constant: float


def _as_constraint_list(atoms: Atoms) -> list[object]:
    constraints = atoms.constraints
    if constraints is None:
        return []
    if isinstance(constraints, (list, tuple)):
        return list(constraints)
    return [constraints]


def active_cartesian_mask(atoms: Atoms) -> np.ndarray:
    """Return a flat boolean mask of Cartesian components free to move."""
    mask = np.ones((len(atoms), 3), dtype=bool)
    for constraint in _as_constraint_list(atoms):
        if isinstance(constraint, FixAtoms):
            mask[np.asarray(constraint.index, dtype=int), :] = False
        elif isinstance(constraint, FixCartesian):
            mask[np.asarray(constraint.index, dtype=int), :] &= ~constraint.mask
    return mask.reshape(-1)


def validate_lindh_atoms(atoms: Atoms) -> None:
    """Validate molecular or fixed-cell periodic Cartesian coordinates."""
    if not isinstance(atoms, Atoms):
        raise LindhError(
            "Lindh Hessian LBFGS supports molecular and fixed-cell periodic geometry "
            "optimization, but not variable-cell optimization."
        )
    periodic_dimensions = int(np.count_nonzero(atoms.pbc))
    if periodic_dimensions and atoms.cell.rank < periodic_dimensions:
        raise LindhError(
            "Lindh Hessian LBFGS requires valid cell vectors for every periodic dimension."
        )


def validate_molecular_atoms(atoms: Atoms) -> None:
    """Backward-compatible alias for :func:`validate_lindh_atoms`."""
    validate_lindh_atoms(atoms)


def _minimum_image_pair_vectors(atoms: Atoms) -> np.ndarray:
    """Return ``r_i - r_j`` vectors using ASE's minimum-image convention."""
    cell = np.asarray(atoms.cell, dtype=float)
    gram = cell @ cell.T
    diagonal_scale = max(1.0, float(np.max(np.diag(gram))))
    orthogonal = (
        atoms.cell.rank == 3
        and np.allclose(
            gram - np.diag(np.diag(gram)),
            0.0,
            atol=1.0e-12 * diagonal_scale,
            rtol=0.0,
        )
    )
    if orthogonal and NUMBA_AVAILABLE:
        return _orthogonal_mic_pair_vectors_numba(
            np.asarray(atoms.get_positions(), dtype=float),
            np.linalg.inv(cell),
            cell,
            np.asarray(atoms.pbc, dtype=np.bool_),
        )
    vectors_j_minus_i = np.asarray(
        atoms.get_all_distances(mic=True, vector=True),
        dtype=float,
    )
    return -vectors_j_minus_i


def _angle_gradient_from_vectors(
    n_atoms: int,
    i: int,
    j: int,
    k: int,
    rij: np.ndarray,
    rkj: np.ndarray,
) -> np.ndarray | None:
    dij = np.linalg.norm(rij)
    dkj = np.linalg.norm(rkj)
    if dij < 1.0e-10 or dkj < 1.0e-10:
        return None
    cos_theta = float(np.dot(rij, rkj) / (dij * dkj))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    if sin_theta < 1.0e-7:
        return None

    gi = (cos_theta * rij / (dij * dij) - rkj / (dij * dkj)) / sin_theta
    gk = (cos_theta * rkj / (dkj * dkj) - rij / (dij * dkj)) / sin_theta
    grad = np.zeros((n_atoms, 3), dtype=float)
    grad[i] = gi
    grad[j] = -gi - gk
    grad[k] = gk
    return grad.reshape(-1)


def _angle_gradient(positions: np.ndarray, i: int, j: int, k: int) -> np.ndarray | None:
    rij = positions[i] - positions[j]
    rkj = positions[k] - positions[j]
    dij = np.linalg.norm(rij)
    dkj = np.linalg.norm(rkj)
    if dij < 1.0e-10 or dkj < 1.0e-10:
        return None
    cos_theta = float(np.dot(rij, rkj) / (dij * dkj))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    if sin_theta < 1.0e-7:
        return None

    gi = (cos_theta * rij / (dij * dij) - rkj / (dij * dkj)) / sin_theta
    gk = (cos_theta * rkj / (dkj * dkj) - rij / (dij * dkj)) / sin_theta
    gj = -gi - gk
    grad = np.zeros_like(positions)
    grad[i] = gi
    grad[j] = gj
    grad[k] = gk
    return grad.reshape(-1)


def _dihedral_value(points: np.ndarray) -> float | None:
    p0, p1, p2, p3 = points
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1.0e-10:
        return None
    n0 = np.cross(b0, b1)
    n1 = np.cross(b1, b2)
    n0_norm = np.linalg.norm(n0)
    n1_norm = np.linalg.norm(n1)
    if n0_norm < 1.0e-8 or n1_norm < 1.0e-8:
        return None
    b1_unit = b1 / b1_norm
    m0 = np.cross(n0, b1_unit)
    return float(np.arctan2(np.dot(m0, n1), np.dot(n0, n1)))


def _dihedral_gradient(
    positions: np.ndarray,
    i: int,
    j: int,
    k: int,
    l: int,
    eps: float = 1.0e-5,
) -> np.ndarray | None:
    indices = (i, j, k, l)
    points = positions[list(indices)].copy()
    if _dihedral_value(points) is None:
        return None
    grad = np.zeros_like(positions)
    for local_atom, atom_index in enumerate(indices):
        for axis in range(3):
            plus = points.copy()
            minus = points.copy()
            plus[local_atom, axis] += eps
            minus[local_atom, axis] -= eps
            phi_plus = _dihedral_value(plus)
            phi_minus = _dihedral_value(minus)
            if phi_plus is None or phi_minus is None:
                return None
            delta = math.atan2(
                math.sin(phi_plus - phi_minus),
                math.cos(phi_plus - phi_minus),
            )
            grad[atom_index, axis] = delta / (2.0 * eps)
    if not np.all(np.isfinite(grad)):
        return None
    return grad.reshape(-1)


def _dihedral_gradient_from_points(
    n_atoms: int,
    indices: tuple[int, int, int, int],
    points: np.ndarray,
    eps: float = 1.0e-5,
) -> np.ndarray | None:
    if _dihedral_value(points) is None:
        return None
    grad = np.zeros((n_atoms, 3), dtype=float)
    for local_atom, atom_index in enumerate(indices):
        for axis in range(3):
            plus = points.copy()
            minus = points.copy()
            plus[local_atom, axis] += eps
            minus[local_atom, axis] -= eps
            phi_plus = _dihedral_value(plus)
            phi_minus = _dihedral_value(minus)
            if phi_plus is None or phi_minus is None:
                return None
            delta = math.atan2(
                math.sin(phi_plus - phi_minus),
                math.cos(phi_plus - phi_minus),
            )
            grad[atom_index, axis] = delta / (2.0 * eps)
    if not np.all(np.isfinite(grad)):
        return None
    return grad.reshape(-1)


if NUMBA_AVAILABLE:

    @njit(fastmath=True, cache=True)
    def _row_for_z_numba(z: int) -> int:
        if z <= 2:
            return 0
        if z <= 10:
            return 1
        return 2

    @njit(fastmath=True, cache=True)
    def _norm3_numba(x: float, y: float, z: float) -> float:
        return math.sqrt(x * x + y * y + z * z)

    @njit(fastmath=True, cache=True)
    def _orthogonal_mic_pair_vectors_numba(
        positions: np.ndarray,
        inverse_cell: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
    ) -> np.ndarray:
        n_atoms = positions.shape[0]
        vectors = np.zeros((n_atoms, n_atoms, 3), dtype=np.float64)
        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dz = positions[i, 2] - positions[j, 2]
                sx = dx * inverse_cell[0, 0] + dy * inverse_cell[1, 0] + dz * inverse_cell[2, 0]
                sy = dx * inverse_cell[0, 1] + dy * inverse_cell[1, 1] + dz * inverse_cell[2, 1]
                sz = dx * inverse_cell[0, 2] + dy * inverse_cell[1, 2] + dz * inverse_cell[2, 2]
                if pbc[0]:
                    sx -= np.rint(sx)
                if pbc[1]:
                    sy -= np.rint(sy)
                if pbc[2]:
                    sz -= np.rint(sz)
                vx = sx * cell[0, 0] + sy * cell[1, 0] + sz * cell[2, 0]
                vy = sx * cell[0, 1] + sy * cell[1, 1] + sz * cell[2, 1]
                vz = sx * cell[0, 2] + sy * cell[1, 2] + sz * cell[2, 2]
                vectors[i, j, 0] = vx
                vectors[i, j, 1] = vy
                vectors[i, j, 2] = vz
                vectors[j, i, 0] = -vx
                vectors[j, i, 1] = -vy
                vectors[j, i, 2] = -vz
        return vectors

    @njit(fastmath=True, cache=True)
    def _dihedral_numba(points: np.ndarray) -> float:
        b0x = points[1, 0] - points[0, 0]
        b0y = points[1, 1] - points[0, 1]
        b0z = points[1, 2] - points[0, 2]
        b1x = points[2, 0] - points[1, 0]
        b1y = points[2, 1] - points[1, 1]
        b1z = points[2, 2] - points[1, 2]
        b2x = points[3, 0] - points[2, 0]
        b2y = points[3, 1] - points[2, 1]
        b2z = points[3, 2] - points[2, 2]

        b1n = _norm3_numba(b1x, b1y, b1z)
        if b1n < 1.0e-10:
            return math.nan

        n0x = b0y * b1z - b0z * b1y
        n0y = b0z * b1x - b0x * b1z
        n0z = b0x * b1y - b0y * b1x
        n1x = b1y * b2z - b1z * b2y
        n1y = b1z * b2x - b1x * b2z
        n1z = b1x * b2y - b1y * b2x
        n0n = _norm3_numba(n0x, n0y, n0z)
        n1n = _norm3_numba(n1x, n1y, n1z)
        if n0n < 1.0e-8 or n1n < 1.0e-8:
            return math.nan

        u1x = b1x / b1n
        u1y = b1y / b1n
        u1z = b1z / b1n
        m0x = n0y * u1z - n0z * u1y
        m0y = n0z * u1x - n0x * u1z
        m0z = n0x * u1y - n0y * u1x
        y = m0x * n1x + m0y * n1y + m0z * n1z
        x = n0x * n1x + n0y * n1y + n0z * n1z
        return math.atan2(y, x)

    @njit(fastmath=True, cache=True)
    def _dihedral_gradient_analytic_numba(
        points: np.ndarray,
        values: np.ndarray,
    ) -> bool:
        """Fill the 12 Cartesian derivatives of a periodic torsion."""
        v0x = points[1, 0] - points[0, 0]
        v0y = points[1, 1] - points[0, 1]
        v0z = points[1, 2] - points[0, 2]
        v1x = points[2, 0] - points[1, 0]
        v1y = points[2, 1] - points[1, 1]
        v1z = points[2, 2] - points[1, 2]
        v2x = points[3, 0] - points[2, 0]
        v2y = points[3, 1] - points[2, 1]
        v2z = points[3, 2] - points[2, 2]

        nv0 = _norm3_numba(v0x, v0y, v0z)
        nv1 = _norm3_numba(v1x, v1y, v1z)
        nv2 = _norm3_numba(v2x, v2y, v2z)
        if nv0 < 1.0e-10 or nv1 < 1.0e-10 or nv2 < 1.0e-10:
            return False

        u0x, u0y, u0z = v0x / nv0, v0y / nv0, v0z / nv0
        u1x, u1y, u1z = v1x / nv1, v1y / nv1, v1z / nv1
        u2x, u2y, u2z = v2x / nv2, v2y / nv2, v2z / nv2
        n01x = u0y * u1z - u0z * u1y
        n01y = u0z * u1x - u0x * u1z
        n01z = u0x * u1y - u0y * u1x
        n12x = u1y * u2z - u1z * u2y
        n12y = u1z * u2x - u1x * u2z
        n12z = u1x * u2y - u1y * u2x
        cos01 = u0x * u1x + u0y * u1y + u0z * u1z
        cos12 = u1x * u2x + u1y * u2y + u1z * u2z
        cos01 = min(1.0, max(-1.0, cos01))
        cos12 = min(1.0, max(-1.0, cos12))
        sin01_sq = max(0.0, 1.0 - cos01 * cos01)
        sin12_sq = max(0.0, 1.0 - cos12 * cos12)
        if sin01_sq < 1.0e-14 or sin12_sq < 1.0e-14:
            return False

        scale0 = -1.0 / (nv0 * sin01_sq)
        scale3 = 1.0 / (nv2 * sin12_sq)
        d0x, d0y, d0z = scale0 * n01x, scale0 * n01y, scale0 * n01z
        d3x, d3y, d3z = scale3 * n12x, scale3 * n12y, scale3 * n12z
        scale10 = (nv1 + nv0 * cos01) / nv1
        scale13 = cos12 * nv2 / nv1
        scale23 = (nv1 + nv2 * cos12) / nv1
        scale20 = cos01 * nv0 / nv1

        values[0], values[1], values[2] = d0x, d0y, d0z
        values[3] = -scale10 * d0x + scale13 * d3x
        values[4] = -scale10 * d0y + scale13 * d3y
        values[5] = -scale10 * d0z + scale13 * d3z
        values[6] = -scale23 * d3x + scale20 * d0x
        values[7] = -scale23 * d3y + scale20 * d0y
        values[8] = -scale23 * d3z + scale20 * d0z
        values[9], values[10], values[11] = d3x, d3y, d3z
        return True

    @njit(fastmath=True, cache=True)
    def _accumulate_sparse_numba(
        hessian: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        count: int,
        force_constant: float,
    ) -> None:
        """Accumulate one primitive outer product over its 6-12 nonzero entries."""
        for local_a in range(count):
            a = indices[local_a]
            ga = values[local_a]
            for local_b in range(count):
                b = indices[local_b]
                hessian[a, b] += force_constant * ga * values[local_b]

    @njit(fastmath=True, cache=True)
    def _lindh_hessian_numba(
        positions: np.ndarray,
        pair_vectors: np.ndarray,
        periodic: bool,
        numbers: np.ndarray,
        alpha_by_row: np.ndarray,
        r_ref_by_row: np.ndarray,
        k_stretch: float,
        k_bend: float,
        k_torsion: float,
        rho_cutoff: float,
        min_distance: float,
        hartree: float,
        bohr: float,
    ) -> tuple[np.ndarray, int, int, int, int, int]:
        n_atoms = positions.shape[0]
        size = 3 * n_atoms
        hessian = np.zeros((size, size), dtype=np.float64)
        pair_rho = np.zeros((n_atoms, n_atoms), dtype=np.float64)
        edges = np.zeros((n_atoms, n_atoms), dtype=np.bool_)
        grad_indices = np.empty(12, dtype=np.int64)
        grad_values = np.empty(12, dtype=np.float64)
        primitive_count = 0
        skipped_close = 0
        skipped_angles = 0
        skipped_torsions = 0
        fallback_pairs = 0

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                if periodic:
                    vx = pair_vectors[i, j, 0]
                    vy = pair_vectors[i, j, 1]
                    vz = pair_vectors[i, j, 2]
                else:
                    vx = positions[i, 0] - positions[j, 0]
                    vy = positions[i, 1] - positions[j, 1]
                    vz = positions[i, 2] - positions[j, 2]
                distance = _norm3_numba(vx, vy, vz)
                if distance < min_distance:
                    skipped_close += 1
                    continue
                ri = _row_for_z_numba(numbers[i])
                rj = _row_for_z_numba(numbers[j])
                if numbers[i] > 18 or numbers[j] > 18:
                    fallback_pairs += 1
                alpha = alpha_by_row[ri, rj]
                r_ref = r_ref_by_row[ri, rj]
                r_bohr = distance / bohr
                rho = math.exp(alpha * (r_ref * r_ref - r_bohr * r_bohr))
                pair_rho[i, j] = rho
                pair_rho[j, i] = rho
                if rho < rho_cutoff:
                    continue
                edges[i, j] = True
                edges[j, i] = True

                inv_distance = 1.0 / distance
                for axis in range(3):
                    grad_indices[axis] = 3 * i + axis
                    grad_indices[3 + axis] = 3 * j + axis
                grad_values[0] = vx * inv_distance
                grad_values[1] = vy * inv_distance
                grad_values[2] = vz * inv_distance
                grad_values[3] = -grad_values[0]
                grad_values[4] = -grad_values[1]
                grad_values[5] = -grad_values[2]
                _accumulate_sparse_numba(
                    hessian,
                    grad_indices,
                    grad_values,
                    6,
                    k_stretch * rho * hartree / (bohr * bohr),
                )
                primitive_count += 1

        for j in range(n_atoms):
            for i in range(n_atoms - 1):
                if i == j or not edges[i, j]:
                    continue
                for k in range(i + 1, n_atoms):
                    if k == j or not edges[j, k]:
                        continue
                    if periodic:
                        ax = pair_vectors[i, j, 0]
                        ay = pair_vectors[i, j, 1]
                        az = pair_vectors[i, j, 2]
                        bx = pair_vectors[k, j, 0]
                        by = pair_vectors[k, j, 1]
                        bz = pair_vectors[k, j, 2]
                    else:
                        ax = positions[i, 0] - positions[j, 0]
                        ay = positions[i, 1] - positions[j, 1]
                        az = positions[i, 2] - positions[j, 2]
                        bx = positions[k, 0] - positions[j, 0]
                        by = positions[k, 1] - positions[j, 1]
                        bz = positions[k, 2] - positions[j, 2]
                    da = _norm3_numba(ax, ay, az)
                    db = _norm3_numba(bx, by, bz)
                    if da < 1.0e-10 or db < 1.0e-10:
                        skipped_angles += 1
                        continue
                    cos_theta = (ax * bx + ay * by + az * bz) / (da * db)
                    if cos_theta > 1.0:
                        cos_theta = 1.0
                    elif cos_theta < -1.0:
                        cos_theta = -1.0
                    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
                    if sin_theta < 1.0e-7:
                        skipped_angles += 1
                        continue

                    gix = (cos_theta * ax / (da * da) - bx / (da * db)) / sin_theta
                    giy = (cos_theta * ay / (da * da) - by / (da * db)) / sin_theta
                    giz = (cos_theta * az / (da * da) - bz / (da * db)) / sin_theta
                    gkx = (cos_theta * bx / (db * db) - ax / (da * db)) / sin_theta
                    gky = (cos_theta * by / (db * db) - ay / (da * db)) / sin_theta
                    gkz = (cos_theta * bz / (db * db) - az / (da * db)) / sin_theta

                    for axis in range(3):
                        grad_indices[axis] = 3 * i + axis
                        grad_indices[3 + axis] = 3 * j + axis
                        grad_indices[6 + axis] = 3 * k + axis
                    grad_values[0] = gix
                    grad_values[1] = giy
                    grad_values[2] = giz
                    grad_values[3] = -gix - gkx
                    grad_values[4] = -giy - gky
                    grad_values[5] = -giz - gkz
                    grad_values[6] = gkx
                    grad_values[7] = gky
                    grad_values[8] = gkz
                    force_constant = k_bend * pair_rho[i, j] * pair_rho[j, k] * hartree
                    _accumulate_sparse_numba(
                        hessian,
                        grad_indices,
                        grad_values,
                        9,
                        force_constant,
                    )
                    primitive_count += 1

        points = np.empty((4, 3), dtype=np.float64)
        for j in range(n_atoms - 1):
            for k in range(j + 1, n_atoms):
                if not edges[j, k]:
                    continue
                for i in range(n_atoms):
                    if i == j or i == k or not edges[i, j]:
                        continue
                    for l in range(n_atoms):
                        if l == i or l == j or l == k or not edges[k, l]:
                            continue
                        idx0 = i
                        idx1 = j
                        idx2 = k
                        idx3 = l
                        if periodic:
                            for axis in range(3):
                                points[0, axis] = pair_vectors[idx0, idx1, axis]
                                points[1, axis] = 0.0
                                points[2, axis] = pair_vectors[idx2, idx1, axis]
                                points[3, axis] = (
                                    points[2, axis]
                                    + pair_vectors[idx3, idx2, axis]
                                )
                        else:
                            for axis in range(3):
                                points[0, axis] = positions[idx0, axis]
                                points[1, axis] = positions[idx1, axis]
                                points[2, axis] = positions[idx2, axis]
                                points[3, axis] = positions[idx3, axis]
                        ok = True
                        for local_atom in range(4):
                            atom_index = idx0
                            if local_atom == 1:
                                atom_index = idx1
                            elif local_atom == 2:
                                atom_index = idx2
                            elif local_atom == 3:
                                atom_index = idx3
                            for axis in range(3):
                                grad_indices[3 * local_atom + axis] = 3 * atom_index + axis
                        ok = _dihedral_gradient_analytic_numba(points, grad_values)
                        if not ok:
                            skipped_torsions += 1
                            continue
                        force_constant = (
                            k_torsion
                            * pair_rho[i, j]
                            * pair_rho[j, k]
                            * pair_rho[k, l]
                            * hartree
                        )
                        _accumulate_sparse_numba(
                            hessian,
                            grad_indices,
                            grad_values,
                            12,
                            force_constant,
                        )
                        primitive_count += 1

        for a in range(size):
            for b in range(a + 1, size):
                value = 0.5 * (hessian[a, b] + hessian[b, a])
                hessian[a, b] = value
                hessian[b, a] = value
        return hessian, primitive_count, skipped_close, skipped_angles, skipped_torsions, fallback_pairs

else:
    _orthogonal_mic_pair_vectors_numba = None
    _lindh_hessian_numba = None


class LindhHessianBuilder:
    """Build a Cartesian Lindh model Hessian in eV / Angstrom**2."""

    def __init__(
        self,
        parameters: LindhParameters | None = None,
        rho_cutoff: float = 1.0e-4,
        min_distance: float = 1.0e-6,
        use_numba: bool | None = None,
    ) -> None:
        self.parameters = parameters or LindhParameters()
        self.rho_cutoff = float(rho_cutoff)
        self.min_distance = float(min_distance)
        self.use_numba = NUMBA_AVAILABLE if use_numba is None else bool(use_numba)
        self.diagnostics: list[str] = []
        self.primitives: list[PrimitiveCoordinate] = []
        self.primitive_count = 0

    def build(self, atoms: Atoms) -> np.ndarray:
        """Return the symmetric Cartesian Lindh model Hessian.

        Parameters
        ----------
        atoms
            Molecular or fixed-cell periodic ASE atoms object. Periodic pair
            vectors use ASE's minimum-image convention. No energies or forces
            are requested.

        Returns
        -------
        numpy.ndarray
            Dense Hessian with shape ``(3 * len(atoms), 3 * len(atoms))`` in
            eV / Angstrom**2.
        """
        validate_lindh_atoms(atoms)
        self.diagnostics = []
        self.primitives = []
        n_atoms = len(atoms)
        size = 3 * n_atoms
        hessian = np.zeros((size, size), dtype=float)
        if n_atoms == 0:
            return hessian

        positions = np.asarray(atoms.get_positions(), dtype=float)
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        periodic = bool(np.any(atoms.pbc))
        pair_vectors = (
            _minimum_image_pair_vectors(atoms)
            if periodic
            else np.empty((0, 0, 3), dtype=float)
        )
        if self.use_numba and _lindh_hessian_numba is not None:
            hessian, primitive_count, skipped_close, skipped_angles, skipped_torsions, fallback_pairs = (
                _lindh_hessian_numba(
                    positions,
                    pair_vectors,
                    periodic,
                    numbers,
                    self.parameters.alpha_by_row,
                    self.parameters.r_ref_by_row,
                    self.parameters.k_stretch,
                    self.parameters.k_bend,
                    self.parameters.k_torsion,
                    self.rho_cutoff,
                    self.min_distance,
                    Hartree,
                    Bohr,
                )
            )
            self.primitive_count = int(primitive_count)
            self.primitives = []
            if skipped_close:
                self.diagnostics.append(
                    f"Skipped {skipped_close} near-coincident stretch coordinate(s)."
                )
            if skipped_angles:
                self.diagnostics.append(
                    f"Skipped {skipped_angles} singular or nearly linear angle coordinate(s)."
                )
            if skipped_torsions:
                self.diagnostics.append(
                    f"Skipped {skipped_torsions} undefined torsion coordinate(s)."
                )
            if fallback_pairs:
                self.diagnostics.append(
                    f"Used row-3 Lindh fallback parameters for {fallback_pairs} heavy-element pair(s)."
                )
            if not np.all(np.isfinite(hessian)):
                raise FloatingPointError("Lindh model Hessian contains non-finite values.")
            return hessian

        pair_rho: dict[tuple[int, int], float] = {}
        pair_distance: dict[tuple[int, int], float] = {}

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                vec = pair_vectors[i, j] if periodic else positions[i] - positions[j]
                distance = float(np.linalg.norm(vec))
                if distance < self.min_distance:
                    self.diagnostics.append(
                        f"Skipped near-coincident stretch ({i}, {j}) at {distance:.3e} Angstrom."
                    )
                    continue
                alpha, r_ref_bohr, fallback = self.parameters.pair_values(numbers[i], numbers[j])
                if fallback:
                    self.diagnostics.append(
                        f"Using row-3 Lindh fallback parameters for pair ({i}, {j})."
                    )
                r_bohr = distance / Bohr
                rho = math.exp(alpha * (r_ref_bohr * r_ref_bohr - r_bohr * r_bohr))
                pair_rho[(i, j)] = rho
                pair_distance[(i, j)] = distance
                if rho < self.rho_cutoff:
                    continue

                grad = np.zeros((n_atoms, 3), dtype=float)
                unit = vec / distance
                grad[i] = unit
                grad[j] = -unit
                k_ev_per_a2 = self.parameters.k_stretch * rho * Hartree / (Bohr * Bohr)
                self._accumulate(hessian, grad.reshape(-1), k_ev_per_a2)
                self.primitives.append(
                    PrimitiveCoordinate("stretch", (i, j), k_ev_per_a2)
                )

        edges = {
            key for key, rho in pair_rho.items()
            if rho >= self.rho_cutoff and pair_distance[key] >= self.min_distance
        }
        neighbors: dict[int, set[int]] = {i: set() for i in range(n_atoms)}
        for i, j in edges:
            neighbors[i].add(j)
            neighbors[j].add(i)

        for j in range(n_atoms):
            nbrs = sorted(neighbors[j])
            for a, i in enumerate(nbrs[:-1]):
                for k in nbrs[a + 1:]:
                    rho_ij = self._rho(pair_rho, i, j)
                    rho_jk = self._rho(pair_rho, j, k)
                    force_constant = self.parameters.k_bend * rho_ij * rho_jk * Hartree
                    if force_constant <= 0.0:
                        continue
                    grad = (
                        _angle_gradient_from_vectors(
                            n_atoms,
                            i,
                            j,
                            k,
                            pair_vectors[i, j],
                            pair_vectors[k, j],
                        )
                        if periodic
                        else _angle_gradient(positions, i, j, k)
                    )
                    if grad is None:
                        self.diagnostics.append(
                            f"Skipped singular or nearly linear angle ({i}, {j}, {k})."
                        )
                        continue
                    self._accumulate(hessian, grad, force_constant)
                    self.primitives.append(
                        PrimitiveCoordinate("bend", (i, j, k), force_constant)
                    )

        seen_torsions: set[tuple[int, int, int, int]] = set()
        for j, k in sorted(edges):
            for i in sorted(neighbors[j] - {k}):
                for l in sorted(neighbors[k] - {j, i}):
                    torsion = (i, j, k, l)
                    reverse = tuple(reversed(torsion))
                    canonical = min(torsion, reverse)
                    if canonical in seen_torsions:
                        continue
                    seen_torsions.add(canonical)
                    rho_ij = self._rho(pair_rho, i, j)
                    rho_jk = self._rho(pair_rho, j, k)
                    rho_kl = self._rho(pair_rho, k, l)
                    force_constant = (
                        self.parameters.k_torsion * rho_ij * rho_jk * rho_kl * Hartree
                    )
                    if force_constant <= 0.0:
                        continue
                    if periodic:
                        points = np.empty((4, 3), dtype=float)
                        points[0] = pair_vectors[i, j]
                        points[1] = 0.0
                        points[2] = pair_vectors[k, j]
                        points[3] = points[2] + pair_vectors[l, k]
                        grad = _dihedral_gradient_from_points(
                            n_atoms,
                            torsion,
                            points,
                        )
                    else:
                        grad = _dihedral_gradient(positions, i, j, k, l)
                    if grad is None:
                        self.diagnostics.append(
                            f"Skipped undefined torsion ({i}, {j}, {k}, {l})."
                        )
                        continue
                    self._accumulate(hessian, grad, force_constant)
                    self.primitives.append(
                        PrimitiveCoordinate("torsion", torsion, force_constant)
                    )

        hessian = 0.5 * (hessian + hessian.T)
        if not np.all(np.isfinite(hessian)):
            raise FloatingPointError("Lindh model Hessian contains non-finite values.")
        self.primitive_count = len(self.primitives)
        return hessian

    @staticmethod
    def _rho(pair_rho: dict[tuple[int, int], float], i: int, j: int) -> float:
        return pair_rho[(i, j) if i < j else (j, i)]

    @staticmethod
    def _accumulate(hessian: np.ndarray, gradient: np.ndarray, force_constant: float) -> None:
        if not np.all(np.isfinite(gradient)):
            return
        hessian += force_constant * np.outer(gradient, gradient)


class LindhPreconditioner(Precon):
    """ASE preconditioner that solves against a regularized Lindh Hessian."""

    def __init__(
        self,
        eigenvalue_floor: float = 0.02,
        rebuild_interval: int = 1,
        builder: LindhHessianBuilder | None = None,
        fallback_alpha: float = 70.0,
        diagnostic_logging: bool = False,
        use_cuda_factorization: bool = False,
    ) -> None:
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.rebuild_interval = max(1, int(rebuild_interval))
        self.builder = builder or LindhHessianBuilder()
        self.fallback_alpha = float(fallback_alpha)
        self.diagnostic_logging = bool(diagnostic_logging)
        self.cuda_factorization_requested = bool(use_cuda_factorization)
        self.use_cuda_factorization = False
        self._torch = None
        if self.cuda_factorization_requested:
            try:
                import torch

                if torch.cuda.is_available():
                    self._torch = torch
                    self.use_cuda_factorization = True
            except Exception:  # pragma: no cover - optional acceleration only.
                self._torch = None
        self.P: np.ndarray | None = None
        self._active_mask: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None
        self._cholesky_factor: tuple[np.ndarray, bool] | None = None
        self._cuda_cholesky_factor = None
        self.regularization_method: str | None = None
        self._calls = 0
        self.n_builds = 0
        self.n_fallbacks = 0
        self.last_fallback_reason: str | None = None
        self.diagnostics: list[str] = []

    def make_precon(self, atoms: Atoms, reinitialize: bool | None = None) -> np.ndarray:
        self._calls += 1
        should_rebuild = (
            reinitialize is True
            or self.P is None
            or (self._calls - 1) % self.rebuild_interval == 0
        )
        if not should_rebuild:
            return self.asarray()

        try:
            validate_lindh_atoms(atoms)
            hessian = self.builder.build(atoms)
            active = active_cartesian_mask(atoms)
            self._set_from_hessian(
                hessian,
                active,
                use_diagonal_shift=bool(np.any(atoms.pbc)),
            )
            self.diagnostics = list(self.builder.diagnostics)
            self.n_builds += 1
            self.last_fallback_reason = None
            if self.diagnostic_logging and self.diagnostics:
                LOGGER.info("Lindh diagnostics: %s", "; ".join(self.diagnostics))
        except Exception as exc:  # noqa: BLE001 - fallback must be robust here.
            self._set_fallback(atoms, str(exc))
        return self.asarray()

    def _set_from_hessian(
        self,
        hessian: np.ndarray,
        active: np.ndarray,
        use_diagonal_shift: bool = False,
    ) -> None:
        self._cholesky_factor = None
        self._cuda_cholesky_factor = None
        if not np.any(active):
            self._active_mask = active
            self._eigenvalues = np.empty(0)
            self._eigenvectors = np.empty((0, 0))
            self.P = np.eye(hessian.shape[0], dtype=float) * self.fallback_alpha
            self.regularization_method = "no active coordinates"
            return
        all_active = bool(np.all(active))
        active_hessian = (
            np.asarray(hessian, dtype=float)
            if all_active
            else hessian[np.ix_(active, active)]
        )
        if not all_active:
            active_hessian = 0.5 * (active_hessian + active_hessian.T)
        if not np.all(np.isfinite(active_hessian)):
            raise FloatingPointError("Active Lindh Hessian contains non-finite values.")

        if use_diagonal_shift:
            regularized_active = active_hessian.copy()
            regularized_active.flat[:: regularized_active.shape[0] + 1] += (
                self.eigenvalue_floor
            )
            if self.use_cuda_factorization and self._torch is not None:
                try:
                    matrix = self._torch.as_tensor(
                        regularized_active,
                        device="cuda",
                        dtype=self._torch.float64,
                    )
                    self._cuda_cholesky_factor = self._torch.linalg.cholesky(matrix)
                except Exception as exc:  # noqa: BLE001 - CPU remains the robust path.
                    LOGGER.warning(
                        "CUDA Lindh factorization failed; using CPU Cholesky: %s",
                        exc,
                    )
                    self.use_cuda_factorization = False
                    self._cuda_cholesky_factor = None
            if self._cuda_cholesky_factor is None:
                self._cholesky_factor = cho_factor(
                    regularized_active,
                    lower=True,
                    overwrite_a=False,
                    check_finite=False,
                )
            self._active_mask = active
            self._eigenvalues = None
            self._eigenvectors = None
            if all_active:
                self.P = regularized_active
            else:
                regularized = np.eye(hessian.shape[0], dtype=float) * self.fallback_alpha
                regularized[np.ix_(active, active)] = regularized_active
                self.P = 0.5 * (regularized + regularized.T)
            self.regularization_method = (
                "diagonal shift with CUDA Cholesky solve"
                if self._cuda_cholesky_factor is not None
                else "diagonal shift with CPU Cholesky solve"
            )
            return

        eigenvalues, eigenvectors = np.linalg.eigh(active_hessian)
        eigenvalues = np.maximum(eigenvalues, self.eigenvalue_floor)
        self._active_mask = active
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        regularized = np.eye(hessian.shape[0], dtype=float) * self.fallback_alpha
        regularized[np.ix_(active, active)] = (
            eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        )
        self.P = 0.5 * (regularized + regularized.T)
        self.regularization_method = "spectral eigenvalue floor"

    def _set_fallback(self, atoms: Atoms, reason: str) -> None:
        size = 3 * len(atoms)
        active = active_cartesian_mask(atoms)
        self.P = np.eye(size, dtype=float) * self.fallback_alpha
        self._active_mask = active
        self._cholesky_factor = None
        self._cuda_cholesky_factor = None
        n_active = int(np.count_nonzero(active))
        self._eigenvalues = np.full(n_active, self.fallback_alpha, dtype=float)
        self._eigenvectors = np.eye(n_active, dtype=float)
        self.n_fallbacks += 1
        self.last_fallback_reason = reason
        self.regularization_method = "scaled identity fallback"
        self.diagnostics = [f"Fell back to scaled identity preconditioner: {reason}"]
        LOGGER.warning("Lindh preconditioner fallback: %s", reason)

    def solve(self, x: np.ndarray) -> np.ndarray:
        vector = np.asarray(x, dtype=float).reshape(-1)
        if self._active_mask is None:
            raise RuntimeError("Lindh preconditioner has not been built.")
        result = np.zeros_like(vector)
        if not np.any(self._active_mask):
            return result
        active_vector = vector[self._active_mask]
        if self._cuda_cholesky_factor is not None and self._torch is not None:
            right_hand_side = self._torch.as_tensor(
                active_vector,
                device="cuda",
                dtype=self._torch.float64,
            )
            solved = (
                self._torch.cholesky_solve(
                    right_hand_side[:, None],
                    self._cuda_cholesky_factor,
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )
        elif self._cholesky_factor is not None:
            solved = cho_solve(
                self._cholesky_factor,
                active_vector,
                overwrite_b=False,
                check_finite=False,
            )
        else:
            if self._eigenvalues is None or self._eigenvectors is None:
                raise RuntimeError("Lindh preconditioner factorization is unavailable.")
            projected = self._eigenvectors.T @ active_vector
            solved = self._eigenvectors @ (projected / self._eigenvalues)
        result[self._active_mask] = solved
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Lindh preconditioner solve produced non-finite values.")
        return result

    def Pdot(self, x: np.ndarray) -> np.ndarray:
        return self.asarray().dot(np.asarray(x).reshape(-1))

    def asarray(self) -> np.ndarray:
        if self.P is None:
            raise RuntimeError("Lindh preconditioner has not been built.")
        return np.asarray(self.P)

    def copy(self) -> "LindhPreconditioner":
        return LindhPreconditioner(
            eigenvalue_floor=self.eigenvalue_floor,
            rebuild_interval=self.rebuild_interval,
            builder=self.builder,
            fallback_alpha=self.fallback_alpha,
            diagnostic_logging=self.diagnostic_logging,
            use_cuda_factorization=self.cuda_factorization_requested,
        )


class _LindhPreconditionedLBFGS(PreconLBFGS, ABC):
    """Abstract L-BFGS state and Lindh preconditioner integration."""

    def __init__(
        self,
        atoms: Atoms,
        restart: str | None = None,
        logfile: str | object = "-",
        trajectory: str | None = None,
        maxstep: float = 0.20,
        memory: int = 20,
        eigenvalue_floor: float = 0.02,
        rebuild_interval: int = 1,
        diagnostic_logging: bool = False,
        use_cuda_factorization: bool | None = None,
        **kwargs: object,
    ) -> None:
        kwargs.pop("use_armijo", None)
        kwargs["use_armijo"] = False
        validate_lindh_atoms(atoms)
        self.periodic_system = bool(np.any(atoms.pbc))
        request_cuda_factorization = (
            self.periodic_system
            if use_cuda_factorization is None
            else bool(use_cuda_factorization)
        )
        self.lindh_precon = LindhPreconditioner(
            eigenvalue_floor=eigenvalue_floor,
            rebuild_interval=rebuild_interval,
            diagnostic_logging=diagnostic_logging,
            use_cuda_factorization=request_cuda_factorization,
        )
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            memory=memory,
            precon=self.lindh_precon,
            variable_cell=False,
            **kwargs,
        )
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.rebuild_interval = int(rebuild_interval)

    @abstractmethod
    def step(self, f=None) -> None:
        """Implemented by the released line-search-free stepping mixin."""
        raise NotImplementedError

    @property
    def n_lindh_builds(self) -> int:
        return self.lindh_precon.n_builds

    @property
    def n_lindh_fallbacks(self) -> int:
        return self.lindh_precon.n_fallbacks

    def update(self, r: np.ndarray, f: np.ndarray, r0: np.ndarray | None, f0: np.ndarray | None) -> None:
        """Update L-BFGS history, skipping unstable curvature pairs."""
        if not self._just_reset_hessian and r0 is not None and f0 is not None:
            s0 = r.reshape(-1) - r0.reshape(-1)
            y0 = f0.reshape(-1) - f.reshape(-1)
            curvature = float(np.dot(y0, s0))
            scale = max(1.0, float(np.linalg.norm(s0) * np.linalg.norm(y0)))
            if curvature > 1.0e-10 * scale and np.isfinite(curvature):
                self.s.append(s0)
                self.y.append(y0)
                self.rho.append(1.0 / curvature)
            else:
                LOGGER.info(
                    "Skipped Lindh Hessian LBFGS curvature pair with s.y=%g.", curvature
                )
        self._just_reset_hessian = False

        while len(self.y) > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def get_lindh_metadata(self) -> dict[str, int | float | str | None]:
        """Return user-facing optimizer metadata."""
        return {
            "Optimizer": "Lindh Hessian LBFGS",
            "L-BFGS memory": self.memory,
            "Lindh Hessian rebuild interval": self.rebuild_interval,
            "Hessian regularization floor": self.eigenvalue_floor,
            "Hessian regularization method": self.lindh_precon.regularization_method,
            "CUDA Cholesky acceleration": (
                "Enabled" if self.lindh_precon.use_cuda_factorization else "Disabled"
            ),
            "Line search": "None",
            "Periodic system": self.periodic_system,
            "Periodic displacement convention": (
                "ASE minimum image" if self.periodic_system else "Not applicable"
            ),
            "Cell optimization": "Not supported",
            "Number of Lindh Hessian builds": self.n_lindh_builds,
            "Number of fallback preconditioner uses": self.n_lindh_fallbacks,
            "Last fallback reason": self.lindh_precon.last_fallback_reason,
        }


class LindhHessianLBFGS(FixedStepPreconLBFGSMixin, _LindhPreconditionedLBFGS):
    """Line-search-free Lindh L-BFGS for molecular or fixed-cell periodic geometry."""

    def get_lindh_metadata(self) -> dict[str, int | float | str | None]:
        metadata = super().get_lindh_metadata()
        metadata.update(self.get_fixed_step_metadata())
        metadata["Optimizer"] = "Lindh Hessian LBFGS"
        return metadata
