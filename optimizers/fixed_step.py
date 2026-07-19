"""ASE-style fixed-step updates for preconditioned L-BFGS optimizers."""

from __future__ import annotations

import numpy as np


class FixedStepPreconLBFGSMixin:
    """Replace ``PreconLBFGS`` line search with maxstep-limited updates."""

    def __init__(self, *args, **kwargs) -> None:
        self.n_fixed_step_fallbacks = 0
        self.last_fixed_step_fallback_reason: str | None = None
        super().__init__(*args, **kwargs)

    def _fallback_direction(self, forces: np.ndarray, reason: str) -> np.ndarray:
        self.n_fixed_step_fallbacks += 1
        self.last_fixed_step_fallback_reason = reason
        self.reset_hessian()
        direction = self.H0 * np.asarray(forces, dtype=float)
        if not np.all(np.isfinite(direction)):
            raise RuntimeError("Fixed-step scalar L-BFGS fallback produced a non-finite direction.")
        return direction

    def _clip_step(self, direction: np.ndarray) -> np.ndarray:
        step = np.asarray(direction, dtype=float).copy()
        lengths = np.linalg.norm(step, axis=1)
        longest = float(np.max(lengths)) if len(lengths) else 0.0
        if longest > self.maxstep:
            step *= self.maxstep / longest
        return step * self.damping

    def step(self, f=None) -> None:
        positions = np.asarray(self._actual_atoms.get_positions(), dtype=float)
        forces = (
            np.asarray(self._actual_atoms.get_forces(), dtype=float)
            if f is None
            else np.asarray(f, dtype=float)
        )
        self.update(positions, forces, self.r0, self.f0)

        loopmax = min(self.memory, len(self.y))
        a = np.empty(loopmax, dtype=np.float64)
        q = -forces.reshape(-1)
        for index in range(loopmax - 1, -1, -1):
            a[index] = self.rho[index] * np.dot(self.s[index], q)
            q -= a[index] * self.y[index]

        if self.precon is None:
            z = np.dot(self.Hinv, q) if self.Hinv is not None else self.H0 * q
        else:
            self.precon.make_precon(self._actual_atoms)
            z = self.precon.solve(q)

        for index in range(loopmax):
            b = self.rho[index] * np.dot(self.y[index], z)
            z += self.s[index] * (a[index] - b)

        direction = -np.asarray(z, dtype=float).reshape(positions.shape)
        gradient = -forces.reshape(-1)
        slope = float(np.dot(gradient, direction.reshape(-1)))
        if not np.all(np.isfinite(direction)):
            direction = self._fallback_direction(forces, "non-finite preconditioned direction")
        elif not np.isfinite(slope) or slope >= 0.0:
            direction = self._fallback_direction(forces, "non-descent preconditioned direction")

        displacement = self._clip_step(direction)
        self.p = direction
        self.alpha_k = 1.0
        self._actual_atoms.set_positions(positions + displacement)

        self.iteration += 1
        self.r0 = positions
        self.f0 = forces.copy()
        self.dump(
            (
                self.iteration,
                self.s,
                self.y,
                self.rho,
                self.r0,
                self.f0,
                self.e0,
                self.task,
            )
        )

    def get_fixed_step_metadata(self) -> dict[str, int | str | None]:
        return {
            "Step strategy": "ASE-style maxstep clipping",
            "Line search": "None",
            "Fixed-step direction fallbacks": self.n_fixed_step_fallbacks,
            "Last fixed-step fallback reason": self.last_fixed_step_fallback_reason,
        }


class AdaptiveStepPreconLBFGSMixin:
    """Line-search-free L-BFGS with a response-controlled step radius.

    One new geometry is evaluated per optimizer cycle.  The energy and force
    response at that geometry controls the *next* step radius, so no additional
    trial evaluations are required.  When the preconditioned seed gives a poor
    response, its inverse action is blended toward ASE's conservative scalar
    L-BFGS initialization.  Valid positive-curvature history is retained.
    """

    def __init__(
        self,
        *args,
        initial_step_radius: float = 0.10,
        minimum_step_radius: float = 0.01,
        radius_shrink: float = 0.50,
        radius_growth: float = 1.35,
        seed_shrink: float = 0.50,
        force_growth_limit: float = 1.35,
        poor_reduction_ratio: float = 0.05,
        good_reduction_ratio: float = 0.60,
        energy_tolerance: float = 1.0e-7,
        **kwargs,
    ) -> None:
        self._adaptive_initial_step_radius = float(initial_step_radius)
        self.minimum_step_radius = float(minimum_step_radius)
        self.radius_shrink = float(radius_shrink)
        self.radius_growth = float(radius_growth)
        self.seed_shrink = float(seed_shrink)
        self.force_growth_limit = float(force_growth_limit)
        self.poor_reduction_ratio = float(poor_reduction_ratio)
        self.good_reduction_ratio = float(good_reduction_ratio)
        self.energy_tolerance = float(energy_tolerance)
        self.n_step_contractions = 0
        self.n_step_expansions = 0
        self.n_poor_responses = 0
        self.n_adaptive_direction_fallbacks = 0
        self.last_reduction_ratio: float | None = None
        self.last_adjustment_reason: str | None = None
        self.seed_weight = 1.0
        self._previous_energy: float | None = None
        self._previous_fmax: float | None = None
        self._previous_predicted_reduction: float | None = None
        self._previous_step_hit_radius = False
        super().__init__(*args, **kwargs)

        if not 0.0 < self.minimum_step_radius <= self.maxstep:
            raise ValueError("minimum_step_radius must be in (0, maxstep].")
        if not 0.0 < self.radius_shrink < 1.0:
            raise ValueError("radius_shrink must be in (0, 1).")
        if self.radius_growth <= 1.0:
            raise ValueError("radius_growth must be greater than 1.")
        if not 0.0 <= self.seed_shrink < 1.0:
            raise ValueError("seed_shrink must be in [0, 1).")
        self.step_radius = float(
            np.clip(
                self._adaptive_initial_step_radius,
                self.minimum_step_radius,
                self.maxstep,
            )
        )

    @staticmethod
    def _maximum_force(forces: np.ndarray) -> float:
        lengths = np.linalg.norm(np.asarray(forces, dtype=float), axis=1)
        return float(np.max(lengths)) if len(lengths) else 0.0

    def _current_energy(self) -> float | None:
        try:
            energy = float(self._actual_atoms.get_potential_energy())
        except Exception:  # noqa: BLE001 - force-only calculators remain supported.
            return None
        return energy if np.isfinite(energy) else None

    def _contract(self, reason: str) -> None:
        previous_radius = self.step_radius
        self.step_radius = max(
            self.minimum_step_radius,
            self.step_radius * self.radius_shrink,
        )
        if self.step_radius < previous_radius:
            self.n_step_contractions += 1
        self.seed_weight *= self.seed_shrink
        self.n_poor_responses += 1
        self.last_adjustment_reason = reason

    def _adapt_from_response(self, energy: float | None, fmax: float) -> None:
        if self._previous_fmax is None:
            return

        force_ratio = fmax / max(self._previous_fmax, 1.0e-14)
        reduction_ratio = None
        energy_increased = False
        if energy is not None and self._previous_energy is not None:
            actual_reduction = self._previous_energy - energy
            tolerance = self.energy_tolerance * max(
                1.0,
                abs(self._previous_energy),
                abs(energy),
            )
            energy_increased = actual_reduction < -tolerance
            predicted = self._previous_predicted_reduction
            if predicted is not None and predicted > tolerance and abs(actual_reduction) > tolerance:
                reduction_ratio = actual_reduction / predicted
        self.last_reduction_ratio = reduction_ratio

        poor_ratio = (
            reduction_ratio is not None
            and reduction_ratio < self.poor_reduction_ratio
        )
        if energy_increased or force_ratio > self.force_growth_limit or poor_ratio:
            reason = (
                "energy increased"
                if energy_increased
                else "force norm increased"
                if force_ratio > self.force_growth_limit
                else "poor predicted/actual reduction agreement"
            )
            self._contract(reason)
            return

        good_ratio = (
            reduction_ratio is not None
            and reduction_ratio >= self.good_reduction_ratio
        )
        strong_force_reduction = force_ratio <= 0.70
        if self._previous_step_hit_radius and (good_ratio or strong_force_reduction):
            previous_radius = self.step_radius
            self.step_radius = min(self.maxstep, self.step_radius * self.radius_growth)
            if self.step_radius > previous_radius:
                self.n_step_expansions += 1

    def _initial_inverse_action(self, q: np.ndarray) -> np.ndarray:
        scalar = self.H0 * q
        if self.precon is None or self.seed_weight <= 0.0:
            return scalar
        self.precon.make_precon(self._actual_atoms)
        seeded = np.asarray(self.precon.solve(q), dtype=float)
        return self.seed_weight * seeded + (1.0 - self.seed_weight) * scalar

    def _fallback_direction(self, forces: np.ndarray, reason: str) -> np.ndarray:
        self.n_adaptive_direction_fallbacks += 1
        self.last_adjustment_reason = reason
        self.seed_weight = 0.0
        self.reset_hessian()
        direction = self.H0 * np.asarray(forces, dtype=float)
        if not np.all(np.isfinite(direction)):
            raise RuntimeError("Adaptive scalar L-BFGS fallback produced a non-finite direction.")
        return direction

    def _clip_adaptive_step(self, direction: np.ndarray) -> tuple[np.ndarray, bool]:
        step = np.asarray(direction, dtype=float).copy()
        lengths = np.linalg.norm(step, axis=1)
        longest = float(np.max(lengths)) if len(lengths) else 0.0
        hit_radius = longest > self.step_radius
        if hit_radius:
            step *= self.step_radius / longest
        return step * self.damping, hit_radius

    def step(self, f=None) -> None:
        positions = np.asarray(self._actual_atoms.get_positions(), dtype=float)
        forces = (
            np.asarray(self._actual_atoms.get_forces(), dtype=float)
            if f is None
            else np.asarray(f, dtype=float)
        )
        energy = self._current_energy()
        fmax = self._maximum_force(forces)
        self._adapt_from_response(energy, fmax)
        self.update(positions, forces, self.r0, self.f0)

        loopmax = min(self.memory, len(self.y))
        coefficients = np.empty(loopmax, dtype=np.float64)
        q = -forces.reshape(-1)
        for index in range(loopmax - 1, -1, -1):
            coefficients[index] = self.rho[index] * np.dot(self.s[index], q)
            q -= coefficients[index] * self.y[index]

        z = self._initial_inverse_action(q)
        for index in range(loopmax):
            beta = self.rho[index] * np.dot(self.y[index], z)
            z += self.s[index] * (coefficients[index] - beta)

        direction = -np.asarray(z, dtype=float).reshape(positions.shape)
        gradient = -forces.reshape(-1)
        slope = float(np.dot(gradient, direction.reshape(-1)))
        if not np.all(np.isfinite(direction)):
            direction = self._fallback_direction(forces, "non-finite seeded direction")
        elif not np.isfinite(slope) or slope >= 0.0:
            direction = self._fallback_direction(forces, "non-descent seeded direction")

        displacement, hit_radius = self._clip_adaptive_step(direction)
        predicted_reduction = float(np.dot(forces.reshape(-1), displacement.reshape(-1)))
        self.p = direction
        self.alpha_k = 1.0
        self._actual_atoms.set_positions(positions + displacement)

        self._previous_energy = energy
        self._previous_fmax = fmax
        self._previous_predicted_reduction = max(0.0, predicted_reduction)
        self._previous_step_hit_radius = hit_radius
        self.iteration += 1
        self.r0 = positions
        self.f0 = forces.copy()
        self.dump(
            (
                self.iteration,
                self.s,
                self.y,
                self.rho,
                self.r0,
                self.f0,
                self.e0,
                self.task,
            )
        )

    def get_adaptive_step_metadata(self) -> dict[str, int | float | str | None]:
        return {
            "Step strategy": "response-controlled radius; one trial per cycle",
            "Line search": "None",
            "Initial step radius": self._adaptive_initial_step_radius,
            "Final step radius": self.step_radius,
            "Minimum step radius": self.minimum_step_radius,
            "Final MACE seed weight": self.seed_weight,
            "Step-radius contractions": self.n_step_contractions,
            "Step-radius expansions": self.n_step_expansions,
            "Poor-response adjustments": self.n_poor_responses,
            "Adaptive direction fallbacks": self.n_adaptive_direction_fallbacks,
            "Last reduction ratio": self.last_reduction_ratio,
            "Last adaptive adjustment reason": self.last_adjustment_reason,
        }
