"""Minimal setuptools configuration for the MLIP Studio Python API.

Dependencies are intentionally not declared here.  MLIP model stacks have
family-specific installation constraints; install them using README.md, then
install this API with ``python -m pip install . --no-deps``.  Editable mode is
intended only for local development.
"""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="mlipstudio",
    version="0.1.0",
    description="Programmatic API for MLIP Studio",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Manas Sharma",
    license="Academic Software License",
    license_files=("LICENSE.md",),
    python_requires=">=3.10",
    packages=find_packages(
        include=("mlipstudio", "mlipstudio.*", "optimizers", "optimizers.*")
    ),
    py_modules=("model_config", "aimnet2_legacy"),
    package_data={
        "mlipstudio": (
            "mlip-studio-qm9-gap.pt",
            "reference_energies.yaml",
        )
    },
    include_package_data=True,
    install_requires=[],
    zip_safe=False,
)
