"""Model catalog and lazy ASE-calculator construction.

Importing this module does not import any model-family runtime.  MACE,
FairChem, ORB, MatterSim, SevenNet, UPET, and AIMNet2 are imported only when a
caller asks to create a calculator from that family.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Mapping

from model_config import (
    FAIRCHEM_CITATIONS,
    FAIRCHEM_MODELS,
    MACE_CITATIONS,
    MACE_MODELS,
    MATTERSIM_CITATIONS,
    MATTERSIM_MODELS,
    ORB_CITATIONS,
    ORB_MODELS,
    SEVEN_NET_CITATIONS,
    SEVEN_NET_MODELS,
    UPET_CITATIONS,
    UPET_MODELS,
    UPET_MODELS_VERSIONS,
)
from aimnet2_legacy import AIMNET2_CITATIONS, AIMNET2_MODELS

from .exceptions import (
    ModelConfigurationError,
    ModelDependencyError,
    ModelLoadError,
    UnknownModelError,
)


@dataclass(frozen=True)
class ModelSpec:
    """Static catalog information for one MLIP Studio model."""

    name: str
    family: str
    identifier: str
    citation: str | None = None
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    required_parameters: tuple[str, ...] = ()
    allowed_parameters: tuple[str, ...] = ()
    calculator_kind: str = "ase"


_SEVENNET_MODALS = {
    "7net-mf-ompa": ("omat24", "mpa"),
    "7net-omni": (
        "matpes_r2scan",
        "mpa",
        "omat24",
        "matpes_pbe",
        "oc20",
        "oc22",
        "odac23",
        "omol25_low",
        "omol25_high",
        "spice",
        "qcml",
        "pet_mad",
        "mp_r2scan",
    ),
}

_PACKAGE_ROOT = Path(__file__).resolve().parent


def _build_catalog() -> dict[str, ModelSpec]:
    catalog: dict[str, ModelSpec] = {}

    for name, identifier in MACE_MODELS.items():
        catalog[name] = ModelSpec(
            name=name,
            family="MACE",
            identifier=identifier,
            citation=MACE_CITATIONS.get(name),
            default_parameters={"dispersion": False, "default_dtype": "float32"},
            allowed_parameters=("dispersion", "default_dtype"),
        )

    for name, identifier in FAIRCHEM_MODELS.items():
        is_uma = name.startswith("UMA ")
        catalog[name] = ModelSpec(
            name=name,
            family="FairChem",
            identifier=identifier,
            citation=FAIRCHEM_CITATIONS.get(name),
            default_parameters={} if is_uma else {"task_name": "omol"},
            required_parameters=("task_name",) if is_uma else (),
            allowed_parameters=("task_name", "inference_settings"),
        )

    for name, identifier in ORB_MODELS.items():
        catalog[name] = ModelSpec(
            name=name,
            family="ORB",
            identifier=identifier,
            citation=ORB_CITATIONS.get(name),
            default_parameters={"precision": "float32-high"},
            allowed_parameters=("precision",),
        )

    for name, identifier in MATTERSIM_MODELS.items():
        catalog[name] = ModelSpec(
            name=name,
            family="MatterSim",
            identifier=identifier,
            citation=MATTERSIM_CITATIONS.get(name),
        )

    for name, identifier in UPET_MODELS.items():
        is_dos = identifier == "pet-mad-dos"
        catalog[name] = ModelSpec(
            name=name,
            family="UPET",
            identifier=identifier,
            citation=UPET_CITATIONS.get(name),
            default_parameters=(
                {"version": "latest"}
                if is_dos
                else {
                    "version": UPET_MODELS_VERSIONS[name],
                    "non_conservative": True,
                }
            ),
            allowed_parameters=("version",) if is_dos else ("version", "non_conservative"),
            calculator_kind="electronic_structure" if is_dos else "ase",
        )

    for name, identifier in SEVEN_NET_MODELS.items():
        requires_modal = identifier in _SEVENNET_MODALS
        catalog[name] = ModelSpec(
            name=name,
            family="SevenNet",
            identifier=identifier,
            citation=SEVEN_NET_CITATIONS.get(name),
            required_parameters=("modal",) if requires_modal else (),
            allowed_parameters=("modal",),
        )

    for name, identifier in AIMNET2_MODELS.items():
        catalog[name] = ModelSpec(
            name=name,
            family="AIMNet2",
            identifier=identifier,
            citation=AIMNET2_CITATIONS.get(name),
            default_parameters={"charge": 0},
            allowed_parameters=("charge",),
        )

    catalog["QM9-Gap"] = ModelSpec(
        name="QM9-Gap",
        family="In-House",
        identifier=str(_PACKAGE_ROOT / "mlip-studio-qm9-gap.pt"),
        default_parameters={"cutoff": 10.0},
        allowed_parameters=("model_path", "cutoff"),
        calculator_kind="property_predictor",
    )

    return catalog


_MODEL_CATALOG = _build_catalog()


def list_models(family: str | None = None) -> tuple[ModelSpec, ...]:
    """Return supported model specifications in stable display order."""

    models = tuple(_MODEL_CATALOG.values())
    if family is None:
        return models
    family_key = family.casefold()
    return tuple(model for model in models if model.family.casefold() == family_key)


def get_model_spec(model_name: str) -> ModelSpec:
    """Look up a model and provide a useful suggestion for misspellings."""

    try:
        return _MODEL_CATALOG[model_name]
    except KeyError as exc:
        suggestions = get_close_matches(model_name, _MODEL_CATALOG, n=3, cutoff=0.5)
        suffix = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise UnknownModelError(f"Unknown MLIP Studio model '{model_name}'.{suffix}") from exc


def _resolved_parameters(spec: ModelSpec, supplied: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(supplied).difference(spec.allowed_parameters)
    if unknown:
        names = ", ".join(sorted(unknown))
        allowed = ", ".join(spec.allowed_parameters) or "none"
        raise ModelConfigurationError(
            f"Unsupported parameter(s) for {spec.name}: {names}. Allowed: {allowed}."
        )
    parameters = dict(spec.default_parameters)
    parameters.update(supplied)
    missing = [name for name in spec.required_parameters if parameters.get(name) is None]
    if missing:
        raise ModelConfigurationError(
            f"{spec.name} requires parameter(s): {', '.join(missing)}."
        )
    return parameters


def _create_mace(spec: ModelSpec, device: str, parameters: Mapping[str, Any]) -> Any:
    from mace.calculators import mace_mp

    return mace_mp(model=spec.identifier, device=device, **parameters)


def _create_fairchem(spec: ModelSpec, device: str, parameters: Mapping[str, Any]) -> Any:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    settings = dict(parameters)
    task_name = settings.pop("task_name")
    inference_settings = settings.pop("inference_settings", "default")
    predictor = pretrained_mlip.get_predict_unit(
        spec.identifier,
        inference_settings=inference_settings,
        device=device,
    )
    return FAIRChemCalculator(predictor, task_name=task_name)


def _create_orb(spec: ModelSpec, device: str, parameters: Mapping[str, Any]) -> Any:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    try:
        factory = getattr(pretrained, spec.identifier)
    except AttributeError as exc:
        raise ModelLoadError(f"ORB factory '{spec.identifier}' is unavailable.") from exc
    model = factory(device=device, **parameters)
    return ORBCalculator(model, device=device)


def _create_mattersim(spec: ModelSpec, device: str, parameters: Mapping[str, Any]) -> Any:
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path=spec.identifier, device=device, **parameters)


def _create_upet(spec: ModelSpec, device: str, parameters: Mapping[str, Any]) -> Any:
    if spec.identifier == "pet-mad-dos":
        if device != "cpu":
            raise ModelConfigurationError(
                "PET-MAD-DOS currently supports only device='cpu'."
            )
        from upet.calculator import PETMADDOSCalculator

        return PETMADDOSCalculator(
            version=str(parameters.get("version", "latest")),
            device="cpu",
        )

    from upet.calculator import UPETCalculator

    return UPETCalculator(model=spec.identifier, device=device, **parameters)


def _create_sevennet(spec: ModelSpec, device: str, parameters: Mapping[str, Any]) -> Any:
    modal = parameters.get("modal")
    allowed_modals = _SEVENNET_MODALS.get(spec.identifier)
    if allowed_modals is not None and modal not in allowed_modals:
        raise ModelConfigurationError(
            f"Invalid modal '{modal}' for {spec.name}. Allowed: {', '.join(allowed_modals)}."
        )
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model=spec.identifier, device=device, **parameters)


def _create_aimnet2(spec: ModelSpec, device: str, parameters: Mapping[str, Any]) -> Any:
    charge = parameters["charge"]
    if isinstance(charge, bool) or not isinstance(charge, numbers.Integral):
        raise ModelConfigurationError(f"charge must be an integer, got {charge!r}.")
    import aimnet  # noqa: F401  # optional: pip install --no-deps aimnet==0.0.1
    import torch

    # aimnet 0.0.1 chooses its own device and always uses CUDA when it is available
    aimnet_device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != aimnet_device:
        raise ModelConfigurationError(
            f"{spec.name} runs on '{aimnet_device}' on this machine (aimnet 0.0.1 always "
            f"uses CUDA when it is available); pass device='{aimnet_device}'."
        )
    from aimnet2_legacy import AIMNet2LegacyCalculator

    return AIMNet2LegacyCalculator(model=spec.identifier, charge=int(charge))


def _create_inhouse(spec: ModelSpec, device: str, parameters: Mapping[str, Any]) -> Any:
    from .predictors import QM9GapCalculator

    settings = dict(parameters)
    model_path = settings.pop("model_path", spec.identifier)
    return QM9GapCalculator(
        model_path=model_path,
        device=device,
        **settings,
    )


_BUILDERS = {
    "MACE": _create_mace,
    "FairChem": _create_fairchem,
    "ORB": _create_orb,
    "MatterSim": _create_mattersim,
    "UPET": _create_upet,
    "SevenNet": _create_sevennet,
    "AIMNet2": _create_aimnet2,
    "In-House": _create_inhouse,
}


def create_calculator(model_name: str, *, device: str = "cpu", **parameters: Any) -> Any:
    """Create a general ASE calculator or task-specific model provider.

    Model-family libraries and checkpoints are loaded only by this call.  A
    fresh calculator is returned each time; callers should reuse it for
    sequential calculations when doing so is supported by the upstream model.
    """

    spec = get_model_spec(model_name)
    if device not in {"cpu", "cuda"}:
        raise ModelConfigurationError("device must be either 'cpu' or 'cuda'.")
    resolved = _resolved_parameters(spec, parameters)
    try:
        calculator = _BUILDERS[spec.family](spec, device, resolved)
        try:
            setattr(calculator, "_mlipstudio_model_name", spec.name)
            setattr(calculator, "_mlipstudio_model_family", spec.family)
            setattr(calculator, "_mlipstudio_calculator_kind", spec.calculator_kind)
            setattr(calculator, "_mlipstudio_parameters", dict(resolved))
        except Exception:
            pass
        return calculator
    except (ModelConfigurationError, ModelLoadError):
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        dependency = getattr(exc, "name", None) or spec.family
        raise ModelDependencyError(
            f"{model_name} requires the optional dependency '{dependency}'."
        ) from exc
    except Exception as exc:
        raise ModelLoadError(f"Failed to initialize {model_name}: {exc}") from exc


@dataclass(frozen=True)
class CalculatorFactory:
    """Serializable recipe that can create one or more equivalent calculators.

    The recipe is useful for workflows such as NEB where separate calculator
    instances may be required for multiple images.
    """

    model_name: str
    device: str = "cpu"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> ModelSpec:
        return get_model_spec(self.model_name)

    def create(self) -> Any:
        return create_calculator(
            self.model_name,
            device=self.device,
            **dict(self.parameters),
        )

    __call__ = create
