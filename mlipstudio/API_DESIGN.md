# MLIP Studio Python API: design and implementation plan

## Goal

Expose the model catalog and calculation workflows without importing or running
Streamlit. A typical call should operate directly on an ASE `Atoms` object and
return raw numerical data rather than UI-formatted strings, tables, or plots.

The initial API added with this document is:

```python
from ase.build import molecule
import mlipstudio

atoms = molecule("H2O")
calculator = mlipstudio.create_calculator(
    "MACE MPA Medium",
    device="cuda",
    dispersion=False,
)

single_point = mlipstudio.SinglePointTask(
    properties=("energy", "forces"),
)
prediction = single_point.calculate(atoms, calculator)

relaxation = mlipstudio.OptimizationTask(
    optimizer="LBFGS",
    fmax=0.01,
    steps=200,
)
optimized = relaxation.calculate(atoms, calculator)
```

`create_calculator()` returns the family-specific ASE calculator or a dedicated
property provider for models such as PET-MAD-DOS and QM9-Gap. Model libraries
are imported lazily. `CalculatorFactory` stores a reusable recipe and is
intended for workflows that need more than one independent calculator.

The implementation also includes `BandGapDOSTask` with PET-MAD-DOS,
`HOMOLUMOGapTask` with the bundled QM9 checkpoint,
`SpinDeterminationTask` for charge/multiplicity energy scans, analytical MACE
Hessians, finite-difference vibrations, EOS fitting, model consensus, and the
three GUI batch workflows.

## Current repository findings

1. `Home.py` is both application entry point and computational implementation.
   Importing it runs Streamlit configuration and widgets, attempts Hugging Face
   login, imports every heavyweight model stack, and can mutate session state.
   It cannot be used as a library module.
2. Calculator construction is duplicated between `Home.py` and
   `model_consensus.py`. The consensus module has the better pattern: explicit
   model metadata, family-specific lazy imports, validation, and numerical
   result dataclasses.
3. Most task code mixes four concerns: numerical calculation, Streamlit state,
   progress rendering, and conversion into display strings/dataframes/plots.
   The numerical parts must move into `mlipstudio.tasks`; the GUI should later
   call those tasks and render their results.
4. The released `optimizers` package is reusable. Standard ASE optimizers and
   the Lindh Hessian, MACE Hessian, and MACE-Seed LBFGS methods now share the
   API optimizer registry. `FASTMSO`, however, is still embedded in `Home.py`.
5. A minimal dependency-free `setup.py` now supports local API installation,
   but there is no `pyproject.toml`, dependency-extras matrix, or documentation
   build. The current requirements file installs all model families and several
   Git `main` branches together.

## Public API boundaries

The public surface should remain small:

- `ModelSpec`, `list_models()`, and `get_model_spec()` for discovery.
- `create_calculator()` for one loaded ASE calculator.
- `CalculatorFactory` for a serializable model/device/parameter recipe.
- One task class and one result dataclass per workflow.
- A single exception hierarchy rooted at `MLIPStudioError`.

Tasks own their configuration and expose `calculate(atoms, calculator)`. They
copy the input structure by default, return calculator-free structures, use ASE
units, and do not print, plot, download files, or depend on Streamlit. Long tasks
should later accept a callback/event sink and cancellation token without making
those concepts mandatory for simple synchronous calls.

Result objects should keep raw values:

- energy: eV
- force: eV/Angstrom
- stress: ASE Voigt order in eV/Angstrom^3
- Hessian: eV/Angstrom^2
- frequencies: both complex ASE values and explicitly derived real/imaginary
  representations
- time: seconds; MD time should additionally expose ASE time units and fs

Presentation adapters in the GUI may convert these values into strings,
dataframes, CSV, figures, and download artifacts.

## Phased implementation

### Phase 1: API foundation (initial slice implemented)

- Add an importable `mlipstudio` package with no Streamlit dependency.
- Add the 62-universal-model catalog and lazy factories for MACE, FairChem,
  ORB, MatterSim, UPET/PET, and SevenNet, plus the in-house QM9 model.
- Require explicit UMA `task_name` and multimodal SevenNet `modal` values.
- Add single-point energy/force/stress and standard ASE geometry/cell
  optimization.
- Add typed results, input validation, copy-by-default behavior, and readable
  exceptions.
- Add dedicated band-gap/DOS, HOMO-LUMO gap, and spin-determination tasks;
  task-specific providers are cataloged without presenting them as general
  energy/force calculators.
- Make ORB entries in `model_config.py` identifiers rather than imported
  callables so catalog inspection does not import ORB.

Before calling this phase stable, replace or complement the basic `setup.py`
with `pyproject.toml`, define a supported Python/ASE version policy, add unit
tests for every factory using mocked upstream modules, and add one opt-in
integration test per model family.

### Phase 2: make the API the single computational backend

- Move the canonical model catalog into `mlipstudio.models.catalog`; leave
  `model_config.py` as a temporary compatibility re-export.
- Replace the calculator branches in `Home.py` and the private consensus
  builders with `create_calculator()`/`CalculatorFactory`.
- Move UFF and xTB calculators out of `Home.py`. Treat D3 as a composable
  correction, not as interchangeable with a complete potential.
- Move `FASTMSO` into `optimizers`. The three Hessian-guided optimizers are now
  registered and `OptimizationTask` accepts a separate Hessian calculator;
  accepting a calculator factory for that provider remains future work.
- Introduce an explicit capability/compatibility registry. Model name
  substring checks must not decide whether charge, spin, stress, Hessian,
  dipole, uncertainty, or a task is supported.
- Add GUI adapters that render API result objects. Delete the old numerical
  branches only after parity tests pass.

### Phase 3: existing task parity (API extraction implemented)

Recommended extraction order:

1. Batch single-point calculation, implemented as orchestration over the same
   `SinglePointTask` rather than a separate numerical implementation.
2. Equation of state with a result containing sampled structures, volumes,
   energies, covariance, fitted parameters, residuals, and fit diagnostics.
3. Vibrational analysis with isolated working directories and guaranteed
   cleanup; do not discard complex-frequency information.
4. Analytical Hessian as a MACE capability, separated from Hessian-guided
   optimization.
5. Migrate the implemented API spin scan into the GUI and replace its current
   inline loop with a model-specific metadata adapter.
6. Migrate the implemented atomization/cohesive task and its strict,
   provenance-bearing reference-energy provider into the GUI.
7. Migrate the implemented DOS/band-gap, in-house QM9, and MACE-POLAR
   dipole/partial-charge tasks into the GUI.
8. Model consensus is now UI-independent and accepts calculators or factories.
   Perturbation scans and migration of the GUI to the new API remain pending.

Each task needs pure unit tests, input immutability tests, unit/shape tests,
failure-path tests, and GUI parity fixtures based on small deterministic
structures.

The extracted task APIs have deterministic unit coverage. GUI parity fixtures
and replacing the remaining inline GUI implementations are still required.

### Phase 4: molecular dynamics

Add an `MolecularDynamicsTask` only after defining:

- ensemble and integrator (`NVE`, Langevin/NVT, and later NPT);
- timestep units, total steps/time, temperature initialization, center-of-mass
  handling, random seed, and constraints;
- streaming observations and bounded in-memory trajectory storage;
- checkpoint/restart state, including velocities, integrator/thermostat state,
  RNG state, and model identity;
- output cadence independent of integration cadence;
- safety checks for non-conservative/direct-force models, energy drift, NaNs,
  exploding temperature/forces, and unsupported stress for NPT;
- calculator reuse rules and explicit device synchronization for trustworthy
  timing.

The result should expose thermodynamic time series and trajectory frames while
allowing file-backed trajectories for long runs.

### Phase 5: NEB

NEB changes the task input from one `Atoms` object to an ordered sequence of
images. Design it around `CalculatorFactory`, not one shared calculator:

- validate atom ordering, composition, PBC, cells, constraints, endpoints, and
  image count;
- support interpolation policy, climbing image, spring constants, optimizer,
  force threshold, and parallel mode;
- create independent calculators per image unless an upstream calculator is
  explicitly documented as safe to share;
- avoid loading all large models on one GPU when sequential image evaluation is
  required; make memory policy explicit;
- return the converged path, energies relative to the initial state, tangent
  forces, barrier in both directions, convergence history, and failed image
  diagnostics;
- add restartable trajectory/checkpoint files and deterministic tests using a
  cheap analytical potential.

## Caveats and problems to resolve

### Dependency and packaging

- The current install is not suitable for a small PyPI base package. Split
  extras by family, for example `mlipstudio[mace]`, `[fairchem]`, `[orb]`,
  `[sevennet]`, `[upet]`, `[mattersim]`, `[gui]`, and `[all]`.
- Several dependencies come from mutable Git branches. Releases need immutable
  tags/commits and a tested constraints/lock strategy.
- The documented FairChem installation bypasses both dependencies and its
  Python requirement. This should not be encoded into PyPI metadata; define a
  supported compatibility matrix instead.
- Model downloads, Hugging Face authentication, license acceptance, offline
  behavior, cache locations, checksums, and retry/error messages need a common
  policy. Importing `mlipstudio` must never log in or download anything.
- The bundled QM9 checkpoint is now package data and has been validated through
  a wheel-style local installation. Before PyPI release, decide whether a 4.9 MB
  bundled checkpoint or a checksummed, versioned download cache is preferable.
- The repository license must be checked for PyPI redistribution, and each
  model's license/citation should remain discoverable from `ModelSpec`.

### Model correctness

- “Supported model” is not the same as “supports every task.” Add explicit
  domains, elements, dtype/device constraints, conservative/direct force mode,
  stress support, periodicity, charge/spin schema, and task capabilities.
- Charge/spin metadata currently uses `charge`, `total_charge`, `spin`, and
  `total_spin`, while comments disagree about spin versus multiplicity. Define
  one public convention (recommended: total charge and multiplicity) and adapt
  it per family.
- `PET-MAD-S-V1.1.0` currently maps to version `1.5.0` in
  `UPET_MODELS_VERSIONS`; verify whether that is intentional before release.
- The GUI sets `external_field` only in one energy branch, so MACE-POLAR behavior
  may differ between tasks. Model-specific defaults belong in one adapter.

### Numerical workflow caveats

- `VibrationalModeTask` uses finite-difference forces and therefore needs up to
  six force evaluations per selected atom with `nfree=2`. It intentionally
  preserves imaginary modes. Rigid-body modes are separated by count, not
  projected out: the n smallest |frequencies| are translations/rotations, with
  n detected from the system type or set through `rigid_body_modes`
  (`mlipstudio/vibrational_analysis.py`, shared with the GUI). A direct
  analytical-Hessian normal-mode path is not implemented yet.
- `EquationOfStateTask` is a static isotropic volume scan. It does not relax
  internal coordinates or cell shape at each volume. EOS parameters are only
  trustworthy when the sampled range brackets a well-resolved minimum; users
  must inspect residuals, covariance, and whether the result is model/domain
  appropriate.
- Consensus disagreement can reveal model spread but cannot establish
  accuracy: models may have correlated errors or overlapping training data.
  The API currently evaluates a single structure; the GUI perturbation scan is
  still separate.
- Batch tasks are deliberately sequential to avoid multiplying GPU memory use.
  They report per-item errors but do not yet stream inputs/results, checkpoint,
  parallelize, or cache calculated isolated-atom references across different
  batch items.
- Lindh Hessian LBFGS does not support cell degrees of freedom. MACE Hessian and
  MACE-Seed can precondition a cell filter, but the analytical Hessian covers
  Cartesian atoms; the cell block is regularized diagonal.
- A calculator instance may cache atoms/results and is not assumed thread-safe.
  Reuse it sequentially; use factories for concurrent workers or NEB images.

### Numerical and workflow behavior

- Never return formatted numerical strings from the API. They lose units,
  precision, uncertainty, and machine readability.
- Decide and document input mutation. The initial API copies by default;
  in-place operation is opt-in and should remain so.
- Stress shape/sign/Voigt order and per-atom versus total energy must be
  explicit. Do not silently skip unsupported stress.
- Cell relaxation should require physically valid 3D periodic cells unless a
  specialized lower-dimensional filter is intentionally supported.
- The EOS code labels the fitted bulk modulus dimension as eV/Angstrom^2 in a
  comment; it is eV/Angstrom^3. Add fit bounds, minimum-point checks, residual
  diagnostics, and robust handling when the sampled minimum is an endpoint.
- Vibrational calculations create many displaced evaluations and files. They
  need cleanup on interruption, selected indices/constraints, linear-molecule
  handling, and careful complex-frequency treatment.
- Reference isolated-atom energies are model/task/version specific. The API now
  validates coverage and reports provenance, but the bundled table still needs
  explicit versioning alongside future model-catalog releases.
- Long calculations need progress, cancellation, logging, checkpointing, and
  partial failure reporting that do not import a UI framework.
- GPU memory cleanup via global `gc`/CUDA cache calls is not a concurrency
  strategy. Define ownership and lifetime for loaded calculators.

### Compatibility and testing

- Keep names stable but introduce machine-readable IDs before release; display
  labels contain spaces, punctuation, and version-like text.
- Version the public API independently from the model catalog. Catalog additions
  should not require breaking code changes.
- Add deprecation warnings and a compatibility window before renaming public
  objects or parameters.
- CI should cover the lightweight base install without any model family, plus
  a matrix of optional extras. Network/model-download integration tests should
  be opt-in and cached, not part of ordinary unit tests.
