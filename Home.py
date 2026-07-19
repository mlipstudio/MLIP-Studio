import streamlit as st
import time
import os
import uuid
import io
import base64
import tempfile
from pathlib import Path
import platform
import psutil
import random
import traceback
import time
import gc
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
# FOR CPU only mode
torch._dynamo.config.suppress_errors = True
# Or disable compilation entirely
# torch.backends.cudnn.enabled = False
import plotly.express as px
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.optimize import Optimizer
from ase.optimize import BFGS, LBFGS, FIRE, LBFGSLineSearch, BFGSLineSearch, GPMin, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.optimize.basin import BasinHopping
from ase.optimize.minimahopping import MinimaHopping
from optimizers import LindhHessianLBFGS, MACEHessianLBFGS, MACESeedLBFGS
from optimizers.analytical_hessian import AnalyticalHessianError
from optimizers.lindh import LindhError
from ase.units import kB
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.visualize import view
import py3Dmol
from mace.calculators import mace_mp
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from sevenn.calculator import SevenNetCalculator
import pandas as pd
import yaml # Added for FairChem reference energies
import subprocess
import sys
import pkg_resources
from ase.vibrations import Vibrations
from mp_api.client import MPRester
import pubchempy as pcp
from io import StringIO
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Molecule
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
mattersim_available = True
if mattersim_available:
    from mattersim.forcefield import MatterSimCalculator
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from ase.units import Hartree, Bohr
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from upet.calculator import UPETCalculator
from upet.calculator import PETMADDOSCalculator
from model_config import (
    MACE_MODELS, MACE_CITATIONS, FAIRCHEM_MODELS, ORB_MODELS,
    MATTERSIM_MODELS, UPET_MODELS, UPET_MODELS_VERSIONS,
    SEVEN_NET_MODELS, SAMPLE_STRUCTURES, FAIRCHEM_CITATIONS, UPET_CITATIONS, ORB_CITATIONS, MATTERSIM_CITATIONS, SEVEN_NET_CITATIONS
)
from data import atoms_to_graph
from model import MPNN
from torch_geometric.data import DataLoader
from predict import load_model


from huggingface_hub import login

try:
    hf_token = os.getenv("YOUR SECRET KEY") # Replace with your actual Hugging Face token or manage secrets appropriately
    if hf_token:
        login(token = hf_token)
    else:
        print("Hugging Face token not found. Some models might not be accessible.")
except Exception as e:
     print(f"hf login error: {e}")


os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Set page configuration
st.set_page_config(
    page_title="MLIP Studio - Run, Test and Benchmark MLIPs",
    page_icon="🧪",
    layout="wide"
)

# === Background video styling ===
def set_css():
    st.markdown("""
        <style>
            #myVideo {
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100%;
                min-height: 100%;
                opacity: 0.08; /* adjust opacity */
                pointer-events: none;
            }
            .content {
                position: fixed;
                bottom: 0;
                background: rgba(1, 1, 1, 1.0);
                color: #f1f1f1;
                width: 100%;
                padding: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

# === Embed background video OR remove based on choice ===
def embed_video(video_choice):
    if video_choice == "Off":
        # Remove the video element by injecting empty HTML
        st.sidebar.markdown(
            """<style>#myVideo { display: none !important; }</style>""",
            unsafe_allow_html=True,
        )
        return

    video_links = {
        "Video 1": "https://raw.githubusercontent.com/manassharma07/MLIP-Playground/main/video1.mp4",
        "Video 2": "https://raw.githubusercontent.com/manassharma07/MLIP-Playground/main/video2.mp4",
        "Video 3": "https://raw.githubusercontent.com/manassharma07/MLIP-Playground/main/video3.mp4",
        "Video 4": "https://raw.githubusercontent.com/manassharma07/MLIP-Playground/main/video4.mp4",
    }

    selected_src = video_links.get(video_choice)
    st.sidebar.markdown(f"""
        <video autoplay muted loop id="myVideo">
            <source src="{selected_src}">
            Your browser does not support HTML5 video.
        </video>
    """, unsafe_allow_html=True)




# === UI Control  ===
with st.sidebar:
    with st.expander("Background"):
        # st.markdown("<p style='font-size:12px; opacity:0.7;'>Background Video</p>", unsafe_allow_html=True)
        # video_off = st.checkbox("Turn off background video", value=False)
        video_on = st.toggle("Background video", value=True)
        video_off = not video_on

# Randomly choose one of 4 videos (only if not turned off)
video_files = ["Video 1", "Video 2", "Video 3", "Video 4"]
video_choice = "Off" if video_off else random.choice(video_files)

# Apply CSS + video
set_css()
embed_video(video_choice)


def _find_value(mapping, keywords):
    """
    Find the first value in a dict-like object whose key matches
    any of the keywords (case-insensitive, substring match).
    """
    if mapping is None:
        return None

    for key, value in mapping.items():
        key_l = key.lower()
        for kw in keywords:
            if kw.lower() in key_l:  # lowercase kw too
                return value
    return None


# Unit conversions
KCAL_PER_MOL_TO_EV = 0.04336411530877085  # 1 kcal/mol = 0.043364115... eV


class UFFCalculator(Calculator):
    """
    ASE Calculator using RDKit UFF.

    Energy: eV
    Forces: eV/Angstrom

    Notes:
    - Non-periodic systems only.
    - Best for molecular systems with normal covalent bonding.
    - Bond connectivity is determined once and then reused.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, charge=0, cov_factor=1.3, **kwargs):
        super().__init__(**kwargs)
        self.charge = charge
        self.cov_factor = cov_factor

        self._mol = None
        self._symbols = None
        self._natoms = None

    def _atoms_to_xyz_block(self, atoms):
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        lines = [str(len(atoms)), "ASE structure for RDKit UFF"]
        for sym, pos in zip(symbols, positions):
            lines.append(
                f"{sym:2s} {pos[0]: .12f} {pos[1]: .12f} {pos[2]: .12f}"
            )

        return "\n".join(lines)

    def _build_rdkit_mol(self, atoms):
        xyz_block = self._atoms_to_xyz_block(atoms)

        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            raise RuntimeError("RDKit could not read XYZ block.")

        try:
            rdDetermineBonds.DetermineBonds(
                mol,
                charge=self.charge,
                covFactor=self.cov_factor,
            )
        except Exception as err:
            print("Warning: RDKit bond-order perception failed.")
            print("Falling back to connectivity-only bond perception.")
            print(f"RDKit error was: {err}")

            rdDetermineBonds.DetermineConnectivity(
                mol,
                covFactor=self.cov_factor,
            )

        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
            print("Warning: RDKit sanitization was not fully successful.")

        if not AllChem.UFFHasAllMoleculeParams(mol):
            raise RuntimeError(
                "UFF parameters are not available for all atoms in this structure."
            )

        return mol

    def _update_rdkit_positions(self, mol, atoms):
        conf = mol.GetConformer()
        positions = atoms.get_positions()

        for i, pos in enumerate(positions):
            conf.SetAtomPosition(
                i,
                Point3D(float(pos[0]), float(pos[1]), float(pos[2])),
            )

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        symbols = tuple(atoms.get_chemical_symbols())
        natoms = len(atoms)

        rebuild = (
            self._mol is None
            or self._natoms != natoms
            or self._symbols != symbols
        )

        if rebuild:
            self._mol = self._build_rdkit_mol(atoms)
            self._symbols = symbols
            self._natoms = natoms
        else:
            self._update_rdkit_positions(self._mol, atoms)

        ff = AllChem.UFFGetMoleculeForceField(self._mol, confId=0)

        if ff is None:
            raise RuntimeError(
                "Failed to initialize RDKit UFF force field."
            )

        ff.Initialize()

        energy_kcal = ff.CalcEnergy()
        grad_kcal_per_mol_A = np.array(ff.CalcGrad(), dtype=float).reshape(natoms, 3)

        energy_ev = energy_kcal * KCAL_PER_MOL_TO_EV

        # RDKit gives gradient dE/dx.
        # ASE wants force = -dE/dx.
        forces_ev_A = -grad_kcal_per_mol_A * KCAL_PER_MOL_TO_EV

        self.results = {
            "energy": energy_ev,
            "forces": forces_ev_A,
        }


class XTBCalculator(Calculator):
    r"""ASE Calculator interface for xTB via command line execution.
    
    Parameters
    ----------
    xtb_command : str or Path, optional
        Path to xTB executable. If not provided, tries to find 'xtb' in PATH.
        Examples:
            - Windows: 'D:\Downloads\xtb-6.7.1\bin\xtb.exe'
            - Linux: '/usr/local/bin/xtb' or just 'xtb'
    method : str, optional
        xTB method to use. Default is 'GFN2-xTB' (--gfn 2).
        Options: 'GFN2-xTB', 'GFN1-xTB', 'GFN0-xTB'
    solvent : str, optional
        Solvent model (e.g., 'water', 'dmso'). Default is None (gas phase).
    accuracy : float, optional
        Numerical accuracy (--acc). Default is 1.0.
    electronic_temperature : float, optional
        Electronic temperature in K (--etemp). Default is 300.0.
    max_iterations : int, optional
        Maximum SCF iterations (--iterations). Default is 250.
    charge : int, optional
        Molecular charge (--chrg). Default is 0.
    uhf : int, optional
        Number of unpaired electrons (--uhf). Default is 0.
    extra_args : list of str, optional
        Additional command line arguments to pass to xTB.
    debug : bool, optional
        If True, print xTB output and save files. Default is False.
    keep_files : bool, optional
        If True, keep temporary files in a specified directory. Default is False.
    work_dir : str or Path, optional
        Directory to save files when keep_files=True. Default is './xtb_calc'.
    """
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, 
                 xtb_command=None,
                 method='GFN2-xTB',
                 solvent=None,
                 accuracy=1.0,
                 electronic_temperature=300.0,
                 max_iterations=250,
                 charge=0,
                 uhf=0,
                 extra_args=None,
                 debug=False,
                 keep_files=False,
                 work_dir='./',
                 **kwargs):
        
        Calculator.__init__(self, **kwargs)
        
        # Find xTB executable
        if xtb_command is None:
            # Try to find xtb in PATH
            import shutil
            xtb_path = shutil.which('xtb')
            if xtb_path is None:
                raise ValueError(
                    "xTB executable not found in PATH. "
                    "Please provide xtb_command parameter."
                )
            self.xtb_command = xtb_path
        else:
            xtb_cmd_str = str(xtb_command)
            # If it's just 'xtb', try to find it in PATH
            if xtb_cmd_str == 'xtb':
                import shutil
                xtb_path = shutil.which('xtb')
                if xtb_path:
                    self.xtb_command = 'xtb'  # Keep as 'xtb' to use PATH
                else:
                    raise ValueError("xTB executable not found in PATH.")
            else:
                self.xtb_command = xtb_cmd_str
            
        # Check if executable exists (skip check if using PATH)
        if self.xtb_command != 'xtb' and not os.path.isfile(self.xtb_command):
            raise FileNotFoundError(f"xTB executable not found: {self.xtb_command}")
        
        # Store parameters
        self.method = method
        self.solvent = solvent
        self.accuracy = accuracy
        self.electronic_temperature = electronic_temperature
        self.max_iterations = max_iterations
        self.charge = charge
        self.uhf = uhf
        self.extra_args = extra_args or []
        self.debug = debug
        self.keep_files = keep_files
        self.work_dir = Path(work_dir) if keep_files else None
        
        # Create work directory if needed
        if self.keep_files and self.work_dir:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def write_coord_file(self, atoms, filename):
        """Write coordinates in Turbomole format.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object to write
        filename : str or Path
            Output file path
        """
        with open(filename, 'w') as f:
            # Check for periodic boundary conditions
            if any(atoms.pbc):
                # Write cell parameters
                cell = atoms.cell
                lengths = cell.lengths()  # in Angstrom
                angles = cell.angles()    # in degrees
                
                f.write("$cell angs\n")
                f.write(f"  {lengths[0]:.8f}   {lengths[1]:.8f}   {lengths[2]:.8f}   "
                       f"{angles[0]:.14f}   {angles[1]:.14f}   {angles[2]:.14f}\n")
                
                # Determine periodicity (1D, 2D, or 3D)
                periodicity = sum(atoms.pbc)
                f.write(f"$periodic {periodicity}\n")
            
            # Write coordinates in Bohr
            f.write("$coord\n")
            positions_bohr = atoms.positions / Bohr  # Convert Angstrom to Bohr
            
            for pos, symbol in zip(positions_bohr, atoms.get_chemical_symbols()):
                f.write(f"  {pos[0]:18.14f}  {pos[1]:18.14f}  {pos[2]:18.14f} {symbol.lower()}\n")
            
            f.write("$end\n")
    
    def build_command(self, coord_file):
        """Build xTB command line.
        
        Parameters
        ----------
        coord_file : str or Path
            Path to coordinate file
            
        Returns
        -------
        list of str
            Command line arguments
        """
        cmd = [self.xtb_command, str(coord_file)]
        # cmd = [self.xtb_command, 'coord']
        
        # Add method flag
        if self.method == 'GFN2-xTB':
            cmd.extend(['--gfn', '2'])
        elif self.method == 'GFN1-xTB':
            cmd.extend(['--gfn', '1'])
        elif self.method == 'GFN0-xTB':
            cmd.extend(['--gfn', '0'])
        
        # Add other parameters
        if self.solvent:
            cmd.extend(['--gbsa', self.solvent])
        
        cmd.extend(['--acc', str(self.accuracy)])
        cmd.extend(['--etemp', str(self.electronic_temperature)])
        cmd.extend(['--iterations', str(self.max_iterations)])
        cmd.extend(['--chrg', str(self.charge)])
        cmd.extend(['--uhf', str(self.uhf)])
        
        # Request gradient calculation
        cmd.append('--grad')
        
        # Add any extra arguments
        cmd.extend(self.extra_args)
        
        return cmd
    
    def parse_xtb_output(self, output_file):
        """Parse xTB output file for energy.
        
        Parameters
        ----------
        output_file : str or Path
            Path to output file
            
        Returns
        -------
        energy : float
            Total energy in eV
        """
        with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Look for the final total energy
        import re
        # Pattern: | TOTAL ENERGY              -15.878299743742 Eh   |
        match = re.search(r'\|\s+TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh', content)
        
        if match is None:
            raise RuntimeError("Could not parse TOTAL ENERGY from xTB output")
        
        energy_hartree = float(match.group(1))
        energy = energy_hartree * Hartree  # Convert to eV
        
        return energy

    def parse_gradient_file(self, gradient_file):
        """Parse xTB gradient file for forces.
        
        Parameters
        ----------
        gradient_file : str or Path
            Path to gradient file
            
        Returns
        -------
        forces : np.ndarray
            Atomic forces in eV/Angstrom, shape (natoms, 3)
        """
        with open(gradient_file, 'r') as f:
            lines = f.readlines()
        
        # Find gradient section
        grad_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('$grad'):
                grad_start = i + 2  # Skip $grad and cycle line
                break
        
        if grad_start is None:
            raise RuntimeError("Could not find gradient section in file")
        
        # Read coordinates and gradients
        gradients = []
        i = grad_start
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('$end'):
                break
            
            # Check if this is a coordinate line (ends with an element symbol)
            # Coordinate lines have 4 fields: x y z element
            parts = line.split()
            if len(parts) == 4 and parts[3].isalpha():
                # This is a coordinate line, skip it
                i += 1
                continue
            
            # Parse gradient line (should have 3 numeric values)
            if len(parts) >= 3:
                try:
                    grad = [float(x.replace('D', 'E')) for x in parts[:3]]
                    gradients.append(grad)
                except ValueError:
                    # Skip lines that can't be parsed as numbers
                    pass
            
            i += 1
        
        gradients = np.array(gradients)
        
        # Convert gradients to forces
        # xTB gives gradients in Hartree/Bohr
        # Forces = -gradient, convert to eV/Angstrom
        forces = -gradients * (Hartree / Bohr)
        
        return forces

    def calculate(self, atoms=None, properties=['energy', 'forces'], 
                system_changes=all_changes):
        """Run xTB calculation.
        
        Parameters
        ----------
        atoms : ase.Atoms, optional
            Atoms object to calculate
        properties : list of str, optional
            Properties to calculate
        system_changes : list of str, optional
            List of changes since last calculation
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Determine working directory
        if self.keep_files:
            tmpdir = self.work_dir
            cleanup = False
        else:
            tmpdir = Path(tempfile.mkdtemp())
            cleanup = True
            
        try:
            coord_file = tmpdir / 'coord'
            gradient_file = tmpdir / 'gradient'
            output_file = tmpdir / 'xtb_output.log'
            
            # Write coordinate file
            self.write_coord_file(atoms, coord_file)
            
            if self.debug:
                print(f"\n{'='*60}")
                print("XTB CALCULATION DEBUG INFO")
                print(f"{'='*60}")
                print(f"Working directory: {tmpdir}")
                print(f"\nCoordinate file content:")
                with open(coord_file, 'r') as f:
                    print(f.read())
            
            # Build and run command
            cmd = self.build_command(coord_file)
            
            if self.debug:
                print(f"\nCommand: {' '.join(cmd)}")
                print(f"{'='*60}\n")
            
            try:
                # Use shell=True on Windows if needed for PATH resolution
                use_shell = platform.system() == 'Windows' and self.xtb_command == 'xtb'
                
                result = subprocess.run(
                    cmd,
                    cwd=str(tmpdir),
                    capture_output=True,
                    text=True,
                    check=True,
                    shell=use_shell,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # Save output to file
                stdout_text = result.stdout if result.stdout else "(no stdout)"
                stderr_text = result.stderr if result.stderr else "(no stderr)"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("STDOUT:\n")
                    f.write(stdout_text)
                    f.write("\n\nSTDERR:\n")
                    f.write(stderr_text)
                
                if self.debug:
                    print("XTB OUTPUT:")
                    print(stdout_text)
                    if result.stderr:
                        print("\nXTB STDERR:")
                        print(stderr_text)
                    print(f"\n{'='*60}\n")
                    
            except subprocess.CalledProcessError as e:
                stdout_text = e.stdout if e.stdout else "(no stdout)"
                stderr_text = e.stderr if e.stderr else "(no stderr)"
                error_msg = (
                    f"xTB calculation failed:\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Working dir: {tmpdir}\n"
                    f"Return code: {e.returncode}\n"
                    f"Output: {stdout_text}\n"
                    f"Error: {stderr_text}"
                )
                if self.debug:
                    print(f"\nERROR: {error_msg}")
                raise RuntimeError(error_msg)
            except FileNotFoundError as e:
                error_msg = (
                    f"xTB executable not found:\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Path: {self.xtb_command}\n"
                    f"Error: {str(e)}"
                )
                if self.debug:
                    print(f"\nERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Parse results
            if not gradient_file.exists():
                error_msg = f"Gradient file not found. xTB output:\n{result.stdout if result.stdout else '(no output)'}"
                if self.debug:
                    print(f"\nERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            if self.debug:
                print("Gradient file content:")
                with open(gradient_file, 'r') as f:
                    print(f.read())
                print(f"{'='*60}\n")
            
            # Parse energy from output and forces from gradient file
            energy = self.parse_xtb_output(output_file)
            forces = self.parse_gradient_file(gradient_file)
            
            if self.debug:
                print(f"Parsed energy: {energy:.6f} eV")
                print(f"Parsed forces shape: {forces.shape}")
                print(f"Max force magnitude: {np.abs(forces).max():.6f} eV/Å")
                print(f"{'='*60}\n")
            
            # Store results
            self.results = {
                'energy': energy,
                'forces': forces,
            }
            
        finally:
            # Cleanup temporary directory if needed
            if cleanup:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)


class FASTMSO(Optimizer):
    """
    FAST-MSO: Deterministic multi-stage optimizer

    Stage 1: FIRE   (robust for large forces)
    Stage 2: MDMin  (fast downhill relaxation)
    Stage 3: LBFGS  (rapid final convergence)

    Stage order is monotonic:
        FIRE → MDMin → LBFGS
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile='-',
        trajectory=None,
        f_fire=0.8,
        f_md=0.25,
        fire_kwargs=None,
        md_kwargs=None,
        lbfgs_kwargs=None,
    ):
        super().__init__(atoms, restart, logfile, trajectory)

        self.f_fire = f_fire
        self.f_md = f_md

        self.fire_kwargs = fire_kwargs or {}
        self.md_kwargs = md_kwargs or {}
        self.lbfgs_kwargs = lbfgs_kwargs or {}

        # ---- Create optimizers ONCE (important) ----
        np.random.seed(0)
        self._fire = FIRE(
            atoms,
            logfile=logfile,
            trajectory=trajectory,
            **self.fire_kwargs,
        )
        np.random.seed(0)
        self._md = MDMin(
            atoms,
            logfile=logfile,
            trajectory=trajectory,
            **self.md_kwargs,
        )
        np.random.seed(0)
        self._lbfgs = LBFGS(
            atoms,
            logfile=logfile,
            trajectory=trajectory,
            **self.lbfgs_kwargs,
        )

        self._stage = "FIRE"  # start deterministically

    def step(self):
        forces = self.atoms.get_forces()
        fmax = np.max(np.linalg.norm(forces, axis=1))

        old_stage = self._stage
        
        # ---- Monotonic stage switching ----
        if self._stage == "FIRE" and fmax < self.f_fire:
            self._stage = "MDMin"

        elif self._stage == "MDMin" and fmax < self.f_md:
            self._stage = "LBFGS"

        # ---- Reset optimizer on transition ----
        if old_stage != self._stage:
            if self._stage == "MDMin":
                np.random.seed(0)
                self._md = MDMin(
                    self.atoms,
                    logfile=self.logfile,
                    trajectory=self.trajectory,
                    **self.md_kwargs,
                )
            elif self._stage == "LBFGS":
                np.random.seed(0)
                self._lbfgs = LBFGS(
                    self.atoms,
                    logfile=self.logfile,
                    trajectory=self.trajectory,
                    **self.lbfgs_kwargs,
                )

        # ---- Execute one step ----
        if self._stage == "FIRE":
            self._fire.step()
        elif self._stage == "MDMin":
            self._md.step()
        else:
            self._lbfgs.step()


# Equation of State functions
def murnaghan(V, Emin, Vmin, B, Bprime):
    return Emin + B * Vmin * (1 / (Bprime * (Bprime - 1)) * pow((V / Vmin), 1 - Bprime) + 
                               1 / Bprime * (V / Vmin) - 1 / (Bprime - 1))

def birchMurnaghan(V, Emin, Vmin, B, Bprime):
    return Emin + 9.0 / 16.0 * B * Vmin * (pow(pow((Vmin / V), 2.0 / 3.0) - 1, 3.0) * Bprime + 
                                            pow(pow(Vmin / V, 2.0 / 3.0) - 1, 2.0) * 
                                            (6 - 4.0 * pow(Vmin / V, 2.0 / 3.0)))

def vinet(V, Emin, Vmin, B, Bprime):
    x = pow(V / Vmin, 1.0 / 3.0)
    return Emin + 2.0 / pow(Bprime - 1, 2.0) * B * Vmin * \
           (2.0 - (5.0 + 3.0 * x * (Bprime - 1) - 3.0 * Bprime) * 
            np.exp(-3.0 / 2.0 * (Bprime - 1.0) * (x - 1.0)))

def calculate_bulk_modulus(calc_atoms, calc, num_points, volume_range, eos_type, results):
    """
    Calculate bulk modulus by fitting equation of state to energy-volume data.
    
    Parameters:
    -----------
    calc_atoms : ASE Atoms object
        The atomic structure with calculator assigned
    calc : Calculator object
        The calculator (MACE or FairChem)
    results : dict
        Dictionary to store results
    """

    # Check if structure is periodic
    if not any(calc_atoms.pbc):
        st.error("❌ Bulk modulus calculation requires a periodic structure (at least one periodic dimension).")
        results["Error"] = "Non-periodic structure"
        return
    
    # Get original cell and volume
    original_cell = calc_atoms.get_cell()
    original_volume = calc_atoms.get_volume()
    original_positions_scaled = calc_atoms.get_scaled_positions()
    
    st.write(f"**Original cell volume:** {original_volume:.4f} Å³")
    st.write(f"**Number of atoms:** {len(calc_atoms)}")
    
    # Generate volume range
    volume_factor = volume_range / 100.0
    volumes = np.linspace(original_volume * (1 - volume_factor), 
                            original_volume * (1 + volume_factor), 
                            num_points)
    
    # Calculate energies for each volume
    energies = []
    cell_params_list = []
    
    progress_text = "Calculating energies at different volumes: 0% complete"
    progress_bar = st.progress(0, text=progress_text)
    
    for i, vol in enumerate(volumes):
        # Scale cell uniformly to achieve target volume
        scale_factor = (vol / original_volume) ** (1.0 / 3.0)
        new_cell = original_cell * scale_factor
        
        # Create new atoms object with scaled cell but same fractional coordinates
        temp_atoms = calc_atoms.copy()
        temp_atoms.set_cell(new_cell, scale_atoms=False)
        temp_atoms.set_scaled_positions(original_positions_scaled)
        temp_atoms.calc = calc
        
        # Calculate energy
        try:
            energy = temp_atoms.get_potential_energy()
            energies.append(energy)
            
            # Store cell parameters
            cell_lengths = temp_atoms.cell.cellpar()[:3]  # a, b, c
            cell_angles = temp_atoms.cell.cellpar()[3:]   # alpha, beta, gamma
            cell_params_list.append({
                'Volume': vol,
                'a': cell_lengths[0],
                'b': cell_lengths[1],
                'c': cell_lengths[2],
                'α': cell_angles[0],
                'β': cell_angles[1],
                'γ': cell_angles[2],
                'Energy': energy
            })
        except Exception as e:
            st.error(f"Error calculating energy at volume {vol:.4f} Å³: {str(e)}")
            progress_bar.empty()
            return
        
        progress_val = (i + 1) / len(volumes)
        progress_bar.progress(progress_val, 
                            text=f"Calculating energies: {int(progress_val * 100)}% complete")
    
    progress_bar.empty()
    
    # Convert to numpy arrays
    volumes = np.array(volumes)
    energies = np.array(energies)
    
    # Find minimum energy point for initial guess
    min_idx = np.argmin(energies)
    V0_guess = volumes[min_idx]
    E0_guess = energies[min_idx]
    
    # Estimate bulk modulus from curvature (initial guess)
    # B ≈ V * d²E/dV² at minimum
    if len(volumes) >= 3:
        # Use finite differences for second derivative
        dV = volumes[1] - volumes[0]
        d2E_dV2 = (energies[min_idx + 1] - 2 * energies[min_idx] + energies[min_idx - 1]) / (dV ** 2) if min_idx > 0 and min_idx < len(energies) - 1 else 0.1
        B_guess = max(V0_guess * d2E_dV2, 1.0)  # Ensure positive
    else:
        B_guess = 100.0  # Default guess in eV/Ų
    
    Bprime_guess = 4.0  # Typical value
    
    # Select EOS function
    eos_functions = {
        "Birch-Murnaghan": birchMurnaghan,
        "Murnaghan": murnaghan,
        "Vinet": vinet
    }
    eos_func = eos_functions[eos_type]
    
    # Fit equation of state
    try:
        popt, pcov = curve_fit(eos_func, volumes, energies, 
                                p0=[E0_guess, V0_guess, B_guess, Bprime_guess],
                                maxfev=10000)
        
        E_fit, V_fit, B_fit, Bprime_fit = popt
        
        # Convert bulk modulus from eV/Ų to GPa
        # 1 eV/Ų = 160.21766208 GPa
        B_GPa = B_fit * 160.21766208
        
        # Calculate uncertainties
        perr = np.sqrt(np.diag(pcov))
        B_err_GPa = perr[2] * 160.21766208
        
    except Exception as e:
        st.error(f"❌ Failed to fit {eos_type} equation of state: {str(e)}")
        st.info("Try adjusting the volume range or number of points.")
        results["Error"] = f"EOS fit failed: {str(e)}"
        return
    
    # Store results
    results["Bulk Modulus (B₀)"] = f"{B_GPa:.2f} ± {B_err_GPa:.2f} GPa"
    results["B₀'"] = f"{Bprime_fit:.3f} ± {perr[3]:.3f}"
    results["Equilibrium Volume (V₀)"] = f"{V_fit:.4f} Å³"
    results["Equilibrium Energy (E₀)"] = f"{E_fit:.6f} eV"
    results["EOS Type"] = eos_type
    
    # Display results
    st.success("✅ Bulk modulus calculation completed!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bulk Modulus (B₀)", f"{B_GPa:.2f} GPa", 
                    delta=f"± {B_err_GPa:.2f} GPa")
        st.metric("Equilibrium Volume (V₀)", f"{V_fit:.4f} Å³")
    with col2:
        st.metric("B₀' (pressure derivative)", f"{Bprime_fit:.3f}",
                    delta=f"± {perr[3]:.3f}")
        st.metric("Equilibrium Energy (E₀)", f"{E_fit:.6f} eV")
    
    # Create data table
    st.subheader("Energy vs Volume Data")
    df = pd.DataFrame(cell_params_list)
    df = df[['Volume', 'Energy', 'a', 'b', 'c', 'α', 'β', 'γ']]
    df['Volume'] = df['Volume'].round(4)
    df['Energy'] = df['Energy'].round(6)
    df['a'] = df['a'].round(4)
    df['b'] = df['b'].round(4)
    df['c'] = df['c'].round(4)
    df['α'] = df['α'].round(2)
    df['β'] = df['β'].round(2)
    df['γ'] = df['γ'].round(2)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Plot equation of state
    st.subheader("Equation of State")
    
    # Generate smooth curve for fitted EOS
    V_smooth = np.linspace(volumes.min(), volumes.max(), 200)
    E_smooth = eos_func(V_smooth, *popt)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add calculated points
    fig.add_trace(go.Scatter(
        x=volumes, y=energies,
        mode='markers',
        name='Calculated',
        marker=dict(size=10, color='blue', symbol='circle'),
        hovertemplate='Volume: %{x:.4f} Ų<br>Energy: %{y:.6f} eV<extra></extra>'
    ))
    
    # Add fitted curve
    fig.add_trace(go.Scatter(
        x=V_smooth, y=E_smooth,
        mode='lines',
        name=f'{eos_type} Fit',
        line=dict(color='red', width=2),
        hovertemplate='Volume: %{x:.4f} Ų<br>Energy: %{y:.6f} eV<extra></extra>'
    ))
    
    # Add equilibrium point
    fig.add_trace(go.Scatter(
        x=[V_fit], y=[E_fit],
        mode='markers',
        name='Equilibrium',
        marker=dict(size=15, color='green', symbol='star'),
        hovertemplate=f'V₀: {V_fit:.4f} Ų<br>E₀: {E_fit:.6f} eV<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{eos_type} Equation of State<br><sub>B₀ = {B_GPa:.2f} GPa, B₀\' = {Bprime_fit:.3f}</sub>',
        xaxis_title='Volume (Å³)',
        yaxis_title='Energy (eV)',
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional info
    with st.expander("ℹ️ Interpretation Guide"):
        st.markdown(f"""
        **Bulk Modulus (B₀):** {B_GPa:.2f} GPa
        - Measures resistance to compression
        - Higher values indicate harder/less compressible materials
        - Typical ranges: Soft materials (~10 GPa), Metals (~100-200 GPa), Hard materials (>300 GPa)
        
        **B₀' (Pressure Derivative):** {Bprime_fit:.3f}
        - Describes how bulk modulus changes with pressure
        - Typical values range from 3-5 for most materials
        - Values outside 2-7 may indicate poor fit or unusual material behavior
        
        **Equilibrium Volume (V₀):** {V_fit:.4f} Ų
        - Volume at minimum energy (most stable configuration)
        - Compare with input structure to check relaxation quality
        
        **Note:** For publication-quality results, ensure:
        1. Structure is fully relaxed/optimized
        2. Sufficient volume range is sampled
        3. Adequate number of data points (11+ recommended)
        4. Forces on atoms are minimized (<0.01 eV/Å)
        """)

@st.cache_data
def load_reference_energies():
    with open("reference_energies.yaml", "r") as f:
        return yaml.safe_load(f)

ELEMENT_REF_ENERGIES = load_reference_energies()






# Title and description
st.markdown('## MLIP Studio', unsafe_allow_html=True)
st.write('#### Run, test and compare 62 state-of-the-art universal machine learning interatomic potentials (MLIPs) for atomistic simulations of molecules and materials')
st.markdown('Upload molecular structure files or select from predefined examples, then compute energies and forces using foundation models such as those from MACE or FairChem (Meta).', unsafe_allow_html=True)

# Create a directory for sample structures if it doesn't exist
SAMPLE_DIR = "sample_structures"
os.makedirs(SAMPLE_DIR, exist_ok=True)


def get_trajectory_viz(trajectory, style='stick', show_unit_cell=True, width=400, height=400, 
                      show_path=True, path_color='red', path_radius=0.02):
    """
    Visualize optimization trajectory with multiple frames
    
    Args:
        trajectory: List of ASE atoms objects representing the optimization steps
        style: Visualization style ('stick', 'ball', 'ball-stick')
        show_unit_cell: Whether to show unit cell
        show_path: Whether to show trajectory paths for each atom
        path_color: Color of trajectory paths
        path_radius: Radius of trajectory path cylinders
    """
    if not trajectory:
        return None
    
    view = py3Dmol.view(width=width, height=height)
    
    # Add all frames to the viewer
    for frame_idx, atoms_obj in enumerate(trajectory):
        xyz_str = ""
        xyz_str += f"{len(atoms_obj)}\n"
        xyz_str += f"Frame {frame_idx}\n"
        for atom in atoms_obj:
            xyz_str += f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}\n"
        
        view.addModel(xyz_str, "xyz")
    
    # Set style for all models
    if style.lower() == 'ball-stick':
        view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
    elif style.lower() == 'stick':
        view.setStyle({'stick': {}})
    elif style.lower() == 'ball':
        view.setStyle({'sphere': {'scale': 0.4}})
    else:
        view.setStyle({'stick': {'radius': 0.15}})
    
    # Add trajectory paths
    if show_path and len(trajectory) > 1:
        for atom_idx in range(len(trajectory[0])):
            for frame_idx in range(len(trajectory) - 1):
                start_pos = trajectory[frame_idx][atom_idx].position
                end_pos = trajectory[frame_idx + 1][atom_idx].position
                
                view.addCylinder({
                    'start': {'x': start_pos[0], 'y': start_pos[1], 'z': start_pos[2]},
                    'end': {'x': end_pos[0], 'y': end_pos[1], 'z': end_pos[2]},
                    'radius': path_radius,
                    'color': path_color,
                    'alpha': 0.5
                })
    
    # Add unit cell for the last frame
    if show_unit_cell and trajectory[-1].pbc.any():
        cell = trajectory[-1].get_cell()
        origin = np.array([0.0, 0.0, 0.0])
        if cell is not None and cell.any():
            edges = [
                (origin, cell[0]), (origin, cell[1]), (cell[0], cell[0] + cell[1]), (cell[1], cell[0] + cell[1]),
                (cell[2], cell[2] + cell[0]), (cell[2], cell[2] + cell[1]),
                (cell[2] + cell[0], cell[2] + cell[0] + cell[1]), (cell[2] + cell[1], cell[2] + cell[0] + cell[1]),
                (origin, cell[2]), (cell[0], cell[0] + cell[2]), (cell[1], cell[1] + cell[2]),
                (cell[0] + cell[1], cell[0] + cell[1] + cell[2])
            ]
            for start, end in edges:
                view.addCylinder({
                    'start': {'x': start[0], 'y': start[1], 'z': start[2]},
                    'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                    'radius': 0.05, 'color': 'black', 'alpha': 0.7
                })
    
    view.zoomTo()
    view.setBackgroundColor('white')
    return view


def get_animated_trajectory_viz(trajectory, style='stick', show_unit_cell=True, width=400, height=400):
    """
    Create an animated trajectory visualization
    """
    if not trajectory:
        return None
    
    view = py3Dmol.view(width=width, height=height)
    
    # Add all frames
    for frame_idx, atoms_obj in enumerate(trajectory):
        xyz_str = ""
        xyz_str += f"{len(atoms_obj)}\n"
        xyz_str += f"Frame {frame_idx}\n"
        for atom in atoms_obj:
            xyz_str += f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}\n"
        
        view.addModel(xyz_str, "xyz")
    
    # Set style
    if style.lower() == 'ball-stick':
        view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
    elif style.lower() == 'stick':
        view.setStyle({'stick': {}})
    elif style.lower() == 'ball':
        view.setStyle({'sphere': {'scale': 0.4}})
    else:
        view.setStyle({'stick': {'radius': 0.15}})
    
    # Add unit cell for last frame
    if show_unit_cell and trajectory[-1].pbc.any():
        cell = trajectory[-1].get_cell()
        origin = np.array([0.0, 0.0, 0.0])
        if cell is not None and cell.any():
            edges = [
                (origin, cell[0]), (origin, cell[1]), (cell[0], cell[0] + cell[1]), (cell[1], cell[0] + cell[1]),
                (origin, cell[2]), (cell[0], cell[0] + cell[2]), (cell[1], cell[1] + cell[2]),
                (cell[0] + cell[1], cell[0] + cell[1] + cell[2]),
                (cell[2], cell[2] + cell[0]), (cell[2], cell[2] + cell[1]),
                (cell[2] + cell[0], cell[2] + cell[0] + cell[1]), (cell[2] + cell[1], cell[2] + cell[0] + cell[1])
            ]
            for start, end in edges:
                view.addCylinder({
                    'start': {'x': start[0], 'y': start[1], 'z': start[2]},
                    'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                    'radius': 0.05, 'color': 'black', 'alpha': 0.7
                })
    
    view.zoomTo()
    view.setBackgroundColor('white')
    
    # Enable animation
    view.animate({'loop': 'forward', 'reps': 0, 'interval': 500})
    
    return view


# Streamlit implementation example
def display_optimization_trajectory(trajectory, viz_style='ball-stick'):
    """
    Display optimization trajectory in Streamlit with controls
    """
    if not trajectory:
        st.error("No trajectory data available")
        return
    
    st.subheader(f"Optimization Trajectory ({len(trajectory)} steps)")
    
    # Trajectory options
    col1, col2 = st.columns(2)
    
    with col1:
        viz_mode = st.selectbox(
            "Visualization Mode",
            ["Animation", "Static with paths", "Step-by-step"],
            key="viz_mode"
        )
    
    with col2:
        if viz_mode == "Static with paths":
            show_paths = st.checkbox("Show trajectory paths", value=True)
            path_color = st.selectbox("Path color", ["red", "blue", "green", "orange"], index=0)
        elif viz_mode == "Step-by-step":
            frame_idx = st.slider("Frame", 0, len(trajectory)-1, 0, key="frame_slider")
    
    # Display visualization based on mode
    if viz_mode == "Static with paths":
        opt_view = get_trajectory_viz(
            trajectory, 
            style=viz_style, 
            show_unit_cell=True, 
            width=400, 
            height=400,
            show_path=show_paths,
            path_color=path_color
        )
        st.components.v1.html(opt_view._make_html(), width=400, height=400)
        
    elif viz_mode == "Animation":
        opt_view = get_animated_trajectory_viz(
            trajectory, 
            style=viz_style, 
            show_unit_cell=True, 
            width=400, 
            height=400
        )
        st.components.v1.html(opt_view._make_html(), width=400, height=400)
        
    elif viz_mode == "Step-by-step":
        opt_view = get_structure_viz2(
            trajectory[frame_idx], 
            style=viz_style, 
            show_unit_cell=True, 
            width=400, 
            height=400
        )
        st.components.v1.html(opt_view._make_html(), width=400, height=400)
        st.write(f"Step {frame_idx + 1} of {len(trajectory)}")

def get_structure_viz2(atoms_obj, style='stick', show_unit_cell=True, width=400, height=400):
    xyz_str = ""
    xyz_str += f"{len(atoms_obj)}\n"
    xyz_str += "Structure\n"
    for atom in atoms_obj:
        xyz_str += f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}\n"
    
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_str, "xyz")
    
    if style.lower() == 'ball-stick':
        view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
    elif style.lower() == 'stick':
        view.setStyle({'stick': {}})
    elif style.lower() == 'ball':
        view.setStyle({'sphere': {'scale': 0.4}})
    else:
        view.setStyle({'stick': {'radius': 0.15}})
    
    if show_unit_cell and atoms_obj.pbc.any(): # Check pbc.any()
        cell = atoms_obj.get_cell()
        origin = np.array([0.0, 0.0, 0.0])
        if cell is not None and cell.any(): # Ensure cell is not None and not all zeros
            edges = [
                (origin, cell[0]), (origin, cell[1]), (cell[0], cell[0] + cell[1]), (cell[1], cell[0] + cell[1]),
                (cell[2], cell[2] + cell[0]), (cell[2], cell[2] + cell[1]),
                (cell[2] + cell[0], cell[2] + cell[0] + cell[1]), (cell[2] + cell[1], cell[2] + cell[0] + cell[1]),
                (origin, cell[2]), (cell[0], cell[0] + cell[2]), (cell[1], cell[1] + cell[2]),
                (cell[0] + cell[1], cell[0] + cell[1] + cell[2])
            ]
            for start, end in edges:
                view.addCylinder({
                    'start': {'x': start[0], 'y': start[1], 'z': start[2]},
                    'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                    'radius': 0.05, 'color': 'black', 'alpha': 0.7
                })
    view.zoomTo()
    view.setBackgroundColor('white')
    return view

opt_log = [] # Define globally or pass around if necessary
table_placeholder = st.empty() # Define globally if updated from callback

def write_single_frame_extxyz(atoms):
    buf = io.StringIO()
    write(buf, atoms, format="extxyz")   # <-- ASE writes this frame alone
    return buf.getvalue()

def get_wrapped_atoms(optimizable):
    return getattr(optimizable, "atoms", optimizable)

def force_fmax(forces):
    forces = np.asarray(forces)
    return float(np.max(np.linalg.norm(forces, axis=1))) if forces.shape[0] > 0 else 0.0

def optimizer_fmax(optimizable):
    return force_fmax(optimizable.get_forces())

def atomic_fmax(optimizable):
    return force_fmax(get_wrapped_atoms(optimizable).get_forces())

def streamlit_log(opt):
    global opt_log, table_placeholder
    try:
        optimizable = opt.atoms
        energy = get_wrapped_atoms(optimizable).get_potential_energy()
        opt_fmax_step = optimizer_fmax(optimizable)
        atom_fmax_step = atomic_fmax(optimizable)
        step = opt.get_number_of_steps() if hasattr(opt, "get_number_of_steps") else opt.nsteps
        row = {
            "Step": step,
            "Energy (eV)": round(energy, 6),
            "Optimizer Fmax": round(opt_fmax_step, 6),
            "Atomic Fmax (eV/A)": round(atom_fmax_step, 6),
        }
        if opt_log and opt_log[-1].get("Step") == step:
            opt_log[-1] = row
        else:
            opt_log.append(row)
        df = pd.DataFrame(opt_log)
        table_placeholder.dataframe(df)
    except Exception as e:
        st.warning(f"Error in optimization logger: {e}")



@torch.no_grad()
def predict_atoms_HOMO_LUMO_QM9_inhouse(model, atoms, cutoff=10.0, device=torch.device("cuda")):
    graph = atoms_to_graph(atoms, cutoff=cutoff, target_key=None)
    loader = DataLoader([graph], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    pred = model(batch)
    return float(pred.cpu().numpy())


@st.cache_resource
def get_mace_model(model_path, dispersion, device, selected_default_dtype):
    return mace_mp(model=model_path, dispersion=dispersion, device=device, default_dtype=selected_default_dtype)

@st.cache_resource
def get_fairchem_model(selected_model_name, model_path_or_name, device, selected_task_type_fc): # Renamed args to avoid conflict
    predictor = pretrained_mlip.get_predict_unit(model_path_or_name, inference_settings="default", device=device)
    if "UMA Small" in selected_model_name:
        calc = FAIRChemCalculator(predictor, task_name=selected_task_type_fc)
    else:
        calc = FAIRChemCalculator(predictor, task_name="omol")
    return calc

# --- INITIALIZATION (Must be run first) ---
if "atoms" not in st.session_state:
    st.session_state.atoms = None
if "atoms_list" not in st.session_state:
    st.session_state.atoms_list = []

# Reset atoms state if input method changes, to prevent using old data
# Use a key to track the currently active input method
if 'current_input_method' not in st.session_state:
    st.session_state.current_input_method = "Select Example"

st.sidebar.markdown("## Input Options")
input_method = st.sidebar.radio("Choose Input Method:", 
                                ["Select Example", "Upload File", "Paste Content", "Materials Project ID", "PubChem", "Batch Upload", "extXYZ Trajectory Upload"])

# If the input method changes, clear the loaded structure
if input_method != st.session_state.current_input_method:
    st.session_state.atoms = None
    st.session_state.current_input_method = input_method

# --- UPLOAD FILE ---
if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload structure file", type=["xyz", "cif", "POSCAR", "mol", "tmol", "vasp", "sdf", "CONTCAR"])
    
    # Load immediately upon file upload/change (no button needed)
    if uploaded_file:
        try:
            # Check if this file content has already been loaded to prevent redundant temp file operations
            if 'uploaded_file_hash' not in st.session_state or st.session_state.uploaded_file_hash != uploaded_file.name: 
                
                # Use tempfile to handle the uploaded file content
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_filepath = tmp_file.name
                    
                atoms_to_store = read(tmp_filepath)
                st.session_state.atoms = atoms_to_store
                st.session_state.uploaded_file_hash = uploaded_file.name # Track the loaded file
                st.sidebar.success(f"Successfully loaded structure with {len(atoms_to_store)} atoms!")
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            st.session_state.atoms = None
            st.session_state.uploaded_file_hash = None # Clear hash on failure
        finally:
            # Clean up the temporary file
            if 'tmp_filepath' in locals() and os.path.exists(tmp_filepath):
                 os.unlink(tmp_filepath)
    else:
        # Clear structure if file uploader is empty
        st.session_state.atoms = None

# --- SELECT EXAMPLE ---
elif input_method == "Select Example":
    # Load immediately upon selection change (no button needed)
    example_name = st.sidebar.selectbox("Select Example Structure:", list(SAMPLE_STRUCTURES.keys()))
    
    # Only load if a valid example is selected and it's different from the current state
    if example_name and (st.session_state.atoms is None or st.session_state.atoms.info.get('source_name') != example_name):
        file_path = os.path.join(SAMPLE_DIR, SAMPLE_STRUCTURES[example_name])
        try:
            atoms_to_store = read(file_path)
            atoms_to_store.info['source_name'] = example_name # Add a tag for tracking
            st.session_state.atoms = atoms_to_store
            st.sidebar.success(f"Loaded {example_name} with {len(atoms_to_store)} atoms!")
        except Exception as e:
            st.sidebar.error(f"Error loading example: {str(e)}")
            st.session_state.atoms = None

# --- PASTE CONTENT ---
elif input_method == "Paste Content":
    file_format = st.sidebar.selectbox("File Format:", ["XYZ", "CIF", "extXYZ", "POSCAR (VASP)", "Turbomole", "MOL"])
    content = st.sidebar.text_area("Paste file content here:", height=200, key="paste_content_input")
    
    # Load immediately upon content change (no button needed)
    # Check if content is present and is different from the last successfully parsed content
    if content:
        # Simple check to avoid parsing on every single character change
        if 'last_parsed_content' not in st.session_state or st.session_state.last_parsed_content != content:
            
            try:
                suffix_map = {"XYZ": ".xyz", "CIF": ".cif", "extXYZ": ".extxyz", "POSCAR (VASP)": ".vasp", "Turbomole": ".tmol", "MOL": ".mol"}
                suffix = suffix_map.get(file_format, ".xyz")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(content.encode())
                    tmp_filepath = tmp_file.name
                    
                atoms_to_store = read(tmp_filepath)
                st.session_state.atoms = atoms_to_store
                st.session_state.last_parsed_content = content # Track the parsed content
                st.sidebar.success(f"Successfully parsed structure with {len(atoms_to_store)} atoms!")
            except Exception as e:
                st.sidebar.error(f"Error parsing content: {str(e)}")
                st.session_state.atoms = None
                st.session_state.last_parsed_content = None
            finally:
                if 'tmp_filepath' in locals() and os.path.exists(tmp_filepath):
                    os.unlink(tmp_filepath)
    else:
        # Clear structure if text area is empty
        st.session_state.atoms = None

# --- PUBCHEM SEARCH MODE ---
elif input_method == "PubChem":
    

    st.sidebar.markdown("### Search PubChem")

    query = st.sidebar.text_input("Enter name or formula (e.g., H2O, water, methane):", 
                                  key="pubchem_query", value="water")

    # Reset atoms if no query
    if query.strip() == "":
        st.session_state.atoms = None

    # Step 1: Search PubChem
    if query and query.strip():
        # Avoid re-searching if query is unchanged
        if "pubchem_last_query" not in st.session_state or st.session_state.pubchem_last_query != query:
            try:
                with st.spinner("Searching PubChem..."):
                    results = pcp.get_compounds(query, "name")  # name OR formula works
                st.session_state.pubchem_results = results
                st.session_state.pubchem_last_query = query
            except Exception as e:
                st.sidebar.error(f"Error searching PubChem: {str(e)}")
                st.session_state.pubchem_results = None

        results = st.session_state.get("pubchem_results", [])
        if results:
            # Convert to displayable table
            df = pd.DataFrame(
                [(c.cid, c.iupac_name, c.molecular_formula, c.molecular_weight, c.isomeric_smiles)
                 for c in results],
                columns=["CID", "Name", "Formula", "Weight", "SMILES"]
            )
            st.sidebar.success(f"Found {len(df)} result(s).")
            st.sidebar.dataframe(df)

            # Choose a CID
            cid = st.sidebar.selectbox("Select CID", df["CID"], key="pubchem_cid")

            # Step 2: Retrieve 3D structure for selected CID 
            if cid:
                if "pubchem_last_cid" not in st.session_state or st.session_state.pubchem_last_cid != cid:
                    try:
                        with st.spinner("Fetching 3D coordinates..."):
                            # Function to format floating-point numbers with alignment
                            def format_number(num, width=10, precision=5):
                                # Handles positive/negative numbers while maintaining alignment
                                return f"{num: {width}.{precision}f}"
                            # CID to XYZ
                            def generate_xyz_coordinates(cid):
                                compound = pcp.Compound.from_cid(cid, record_type='3d')
                                atoms = compound.atoms
                                coords = [(atom.x, atom.y, atom.z) for atom in atoms]

                                num_atoms = len(atoms)
                                xyz_text = f"{num_atoms}\n{compound.cid}\n"

                                for atom, coord in zip(atoms, coords):
                                    atom_symbol = atom.element
                                    x, y, z = coord
                                    xyz_text += f"{atom_symbol} {format_number(x, precision=8)} {format_number(y, precision=8)} {format_number(z, precision=8)}\n"

                                return xyz_text
                            def get_molecule(cid):
                                xyz_str = generate_xyz_coordinates(cid)
                                return Molecule.from_str(xyz_str, fmt='xyz'), xyz_str
                            # Fetch SDF with 3D conformer
                            # sdf_str = pcp.Compound.from_cid(int(cid)).to_sdf()
                            selected_molecule, xyz_str = get_molecule(cid)

                        # Convert SDF → ASE Atoms using temporary memory buffer
                        atoms_to_store = read(StringIO(xyz_str), format="xyz")

                        atoms_to_store.info["source_name"] = f"PubChem CID {cid}"
                        st.session_state.atoms = atoms_to_store
                        st.session_state.pubchem_last_cid = cid

                        st.sidebar.success(f"Loaded PubChem structure with {len(atoms_to_store)} atoms!")

                    except Exception as e:
                        st.sidebar.error(f"Unable to retrieve 3D structure: {str(e)}")
                        st.session_state.atoms = None
                        st.session_state.pubchem_last_cid = None
        else:
            st.sidebar.info("No PubChem results found.")

# --- MATERIALS PROJECT ID ---
elif input_method == "Materials Project ID":
    mp_api_key = os.getenv("MP_API_KEY")
    material_id = st.sidebar.text_input("Enter Material ID:", value="mp-149", key="mp_id_input")
    cell_type = st.sidebar.radio("Unit Cell Type:", ['Primitive Cell', 'Conventional Unit Cell'], key="cell_type_radio")
    
    # Reactive Loading (No button needed)
    # Check for valid inputs and if the current material_id/cell_type is different from the loaded one
    if mp_api_key and material_id: 
        
        # Simple tracking to avoid API call if nothing has changed
        current_mp_key = f"{material_id}_{cell_type}"
        if 'last_fetched_mp_key' not in st.session_state or st.session_state.last_fetched_mp_key != current_mp_key:
            
            try:
                with st.spinner(f"Fetching {material_id}..."):
                    with MPRester(mp_api_key) as mpr:
                        pmg_structure = mpr.get_structure_by_material_id(material_id)
                        analyzer = SpacegroupAnalyzer(pmg_structure)
                        
                        if cell_type == 'Conventional Unit Cell':
                            final_structure = analyzer.get_conventional_standard_structure()
                        else:
                            final_structure = analyzer.get_primitive_standard_structure()
                            
                        atoms_to_store = AseAtomsAdaptor.get_atoms(final_structure)
                        st.session_state.atoms = atoms_to_store
                        st.session_state.last_fetched_mp_key = current_mp_key # Update tracking key
                        st.sidebar.success(f"Loaded {material_id} ({cell_type}) with {len(st.session_state.atoms)} atoms.")
                        
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {str(e)}")
                st.session_state.atoms = None
                st.session_state.last_fetched_mp_key = None # Clear key on failure
    
    # Handle error messages when inputs are missing
    elif not mp_api_key:
        st.sidebar.error("Please set your Materials Project API Key (MP_API_KEY environment variable).")
    elif not material_id:
        st.sidebar.error("Please enter a Material ID.")

# --- BATCH UPLOAD MULTIPLE FILES ---
elif input_method == "Batch Upload":

    uploaded_files = st.sidebar.file_uploader(
        "Upload multiple structure files",
        type=["xyz", "cif", "POSCAR", "vasp", "CONTCAR", "mol", "sdf", "tmol", "extxyz"],
        accept_multiple_files=True
    )

    # Clear state if no files present
    if not uploaded_files:
        st.session_state.atoms_list = []
        st.session_state.atoms = None

    else:
        atoms_list = []
        errors = []

        for file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                atoms_obj = read(tmp_path)
                atoms_obj.info["source_name"] = file.name
                atoms_list.append(atoms_obj)

            except Exception as e:
                errors.append(f"{file.name}: {str(e)}")

            finally:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Store everything only if at least one success
        if atoms_list:
            st.session_state.atoms_list = atoms_list
            st.session_state.atoms = atoms_list[0]  # default: first item
            st.sidebar.success(f"Loaded {len(atoms_list)} structures successfully!")

            if len(atoms_list) > 1:
                st.sidebar.info("You can now process them as a batch.")
                st.sidebar.warning("The visualizer will only display the first structure uploaded by you.")

        if errors:
            st.sidebar.error("Some files could not be loaded:\n" + "\n".join(errors))
elif input_method == "extXYZ Trajectory Upload":
    uploaded_traj = st.sidebar.file_uploader(
        "Upload extxyz trajectory (multi-frame)",
        type=["extxyz", "xyz"]  # extxyz is the key one; xyz sometimes is extxyz content too
    )

    if uploaded_traj:
        try:
            # Avoid re-loading same file repeatedly
            file_id = f"{uploaded_traj.name}_{uploaded_traj.size}"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_traj.name)[1] or ".extxyz"
            ) as tmp:
                tmp.write(uploaded_traj.getvalue())
                tmp_path = tmp.name

            # Read all frames in extxyz
            atoms_list = read(tmp_path, index=":")  # list[Atoms] [web:4]
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]

            st.session_state.atoms_list = atoms_list
            print(len(atoms_list))
            st.session_state.atoms = atoms_list[0]
            st.session_state.uploaded_traj_hash = file_id

            st.sidebar.success(f"Loaded extxyz trajectory with {len(atoms_list)} frame(s).")

            # ---- Property discovery / reporting ----
            # Collect per-config info keys, per-atom arrays keys, and calc.results keys
            info_keys = set()
            array_keys = set()
            calc_keys = set()

            for a in atoms_list:
                # per-frame metadata
                if hasattr(a, "info") and isinstance(a.info, dict):
                    info_keys.update(a.info.keys())

                # per-atom properties (forces may show up here in some cases)
                if hasattr(a, "arrays") and isinstance(a.arrays, dict):
                    array_keys.update(a.arrays.keys())

                # calculator results (energy/forces often land here in newer ASE behavior)
                if getattr(a, "calc", None) is not None and hasattr(a.calc, "results"):
                    if isinstance(a.calc.results, dict):
                        calc_keys.update(a.calc.results.keys())
                # print(a.calc.results)

            st.sidebar.markdown("### extxyz properties detected (ASE)")
            st.sidebar.write("Per-frame (atoms.info) keys:", sorted(info_keys))
            st.sidebar.write("Per-atom (atoms.arrays) keys:", sorted(array_keys))
            st.sidebar.write("Calculator (atoms.calc.results) keys:", sorted(calc_keys))

            # Optional: show quick sanity checks for first frame if present
            a0 = atoms_list[0]
            if getattr(a0, "calc", None) is not None:
                # These will work if ASE mapped them into calculator results
                try:
                    e0 = a0.get_potential_energy()
                    st.sidebar.write("First-frame potential energy:", float(e0))
                except Exception:
                    pass
                try:
                    f0 = a0.get_forces()
                    st.sidebar.write("First-frame forces shape:", getattr(f0, "shape", None))
                except Exception:
                    pass

        except Exception as e:
            st.sidebar.error(f"Error loading extxyz trajectory: {str(e)}")
            st.session_state.atoms = None
            st.session_state.atoms_list = []
            st.session_state.uploaded_traj_hash = None
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        st.session_state.atoms = None
        st.session_state.atoms_list = []

# ----------------------------------------------------
# --- FINAL STRUCTURE RETRIEVAL (The persistent structure) ---
# ----------------------------------------------------
atoms = st.session_state.atoms

if atoms is not None:
    if not hasattr(atoms, 'info'):
        atoms.info = {}
    atoms.info["charge"] = atoms.info.get("charge", 0) # Default charge
    atoms.info["spin"] = atoms.info.get("spin", 1) # Default spin (usually 2S for ASE, model might want 2S+1)
    
    # Display confirmation in the main area (optional, helps the user confirm what's loaded)
    # st.markdown(f"**Loaded Structure:** {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

st.sidebar.markdown("## Model Selection")
if mattersim_available:
    model_type = st.sidebar.radio("Select Model Type:", ["MACE", "FairChem", "ORB", "SEVEN_NET", "MatterSim", "UPET", "UFF", "D3 dispersion", "xTB", "In-House"])
else:
    model_type = st.sidebar.radio("Select Model Type:", ["MACE", "FairChem", "ORB", "SEVEN_NET", "UPET", "UFF", "D3 dispersion", "xTB", "In-House"])

is_omol_model = False
selected_task_type = None # For FairChem UMA

if model_type == "MACE":
    # Add option to choose between predefined models, upload, or URL
    model_source = st.sidebar.radio(
        "Model Source:",
        ["Predefined Models", "Upload Model", "URL"]
    )
    
    if model_source == "Predefined Models":
        selected_model = st.sidebar.selectbox("Select MACE Model:", list(MACE_MODELS.keys()))
        model_path = MACE_MODELS[selected_model]
        
        if selected_model in ["MACE OMAT Medium", " MACE OMAT Small", "MACE MATPES r2SCAN Medium", "MACE MATPES r2SCAN Medium", 
                            "MACE OMOL-0 XL 4M", "MACE OFF 24 Medium", 
                            "MACE OFF 23 Large", "MACE OFF 23 Medium", "MACE OFF 24 Small", "MACE POLAR 1 S", "MACE POLAR 1 M", "MACE POLAR 1 L"]:
            st.sidebar.info("Using model under [Academic Software License (ASL)](https://github.com/gabor1/ASL/blob/main/ASL.md).")
        # Display Citation
        if selected_model in MACE_CITATIONS:
            st.sidebar.info(MACE_CITATIONS[selected_model])
        else:
            st.sidebar.warning("Citation not available for this model.")
    elif model_source == "Upload Model":
        uploaded_file = st.sidebar.file_uploader(
            "Upload .model file",
            type=['model'],
            help="Upload your custom MACE model file"
        )

        if uploaded_file is not None:
            temp_dir = tempfile.gettempdir()

            unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            model_path = os.path.join(temp_dir, unique_name)

            with open(model_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.sidebar.success(f"Loaded: {uploaded_file.name}")
            selected_model = "Custom (Uploaded)"
        else:
            st.sidebar.info("Please upload a .model file")
            model_path = None
            selected_model = None
    
    else:  # URL
        model_url = st.sidebar.text_input(
            "Model URL:",
            placeholder="https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
            help="Provide a direct link to a .model file"
        )
        
        if model_url:
            if model_url.endswith('.model'):
                model_path = model_url
                selected_model = "Custom (URL)"
                st.sidebar.success("URL provided")
            else:
                st.sidebar.error("URL must point to a .model file")
                model_path = None
                selected_model = None
        else:
            st.sidebar.info("Please enter a model URL")
            model_path = None
            selected_model = None
    
    # Only show these options if a model is selected/loaded
    if model_path is not None:
        selected_default_dtype = 'float32'
        dispersion = st.sidebar.toggle("Dispersion correction?", value=False)
        

        if model_source == "Upload Model" or model_source == "URL":
            is_omol_model = st.sidebar.checkbox("This is an OMOL-like model (requires charge/spin)", value=False)
        if model_source == "Predefined Models":
            if "OMOL" in selected_model.upper() or "POLAR" in selected_model.upper():
                is_omol_model = True
        
        if is_omol_model:
            charge = st.sidebar.number_input(
                "Total Charge", 
                min_value=-10, 
                max_value=10, 
                value=0
            )
            spin_multiplicity = st.sidebar.number_input(
                "Spin Multiplicity (2S + 1)", 
                min_value=1, 
                max_value=20, 
                step=1, 
                value=1
            )
            atoms.info["total_charge"] = charge
            atoms.info["total_spin"] = spin_multiplicity
            atoms.info["charge"] = charge
            atoms.info["spin"] = spin_multiplicity
if model_type == "FairChem":
    selected_model = st.sidebar.selectbox("Select FairChem Model:", list(FAIRCHEM_MODELS.keys()))
    model_path = FAIRCHEM_MODELS[selected_model]
    # Display Citation
    if selected_model in FAIRCHEM_CITATIONS:
        st.sidebar.info(FAIRCHEM_CITATIONS[selected_model])
    else:
        st.sidebar.warning("Citation not available for this model.")
    if "UMA Small" in selected_model:
        st.sidebar.info("Meta FAIR [Acceptable Use Policy](https://huggingface.co/facebook/UMA/blob/main/LICENSE) applies.")
        selected_task_type = st.sidebar.selectbox("Select UMA Model Task Type:", ["omol", "omat", "omc", "odac", "oc20"])
        if selected_task_type == "omol" and atoms is not None:
            is_omol_model = True
            if atoms is not None:
                charge = st.sidebar.number_input("Total Charge", min_value=-10, max_value=10, value=0)
                spin_multiplicity = st.sidebar.number_input("Spin Multiplicity (2S + 1)", min_value=1, max_value=20, step=1, value=1) # Assuming spin in atoms.info is S
                atoms.info["charge"] = charge
                atoms.info["spin"] = spin_multiplicity # FairChem expects multiplicity
        else:
            if atoms is not None:
                atoms.info["charge"] = 0
                atoms.info["spin"] = 1 # FairChem expects multiplicity
if model_type == "ORB":
    selected_model = st.sidebar.selectbox("Select ORB Model:", list(ORB_MODELS.keys()))
    model_path = ORB_MODELS[selected_model]
    # Display Citation
    if selected_model in ORB_CITATIONS:
        st.sidebar.info(ORB_CITATIONS[selected_model])
    else:
        st.sidebar.warning("Citation not available for this model.")
    st.sidebar.info("ORB models are licensed under the [Apache License, Version 2.0.](https://github.com/orbital-materials/orb-models/blob/main/LICENSE)")
    # selected_default_dtype = st.sidebar.selectbox("Select Precision (default_dtype):", ['float32-high', 'float32-highest', 'float64'])
    selected_default_dtype = st.sidebar.selectbox("Select Precision (default_dtype):", ['float32-high', 'float32-highest'])
    if "OMOL" in selected_model and atoms is not None:
        is_omol_model = True
        if atoms is not None:
            charge = st.sidebar.number_input("Total Charge", min_value=-10, max_value=10, value=0)
            spin_multiplicity = st.sidebar.number_input("Spin Multiplicity (2S + 1)", min_value=1, max_value=20, step=1, value=1) # Assuming spin in atoms.info is S
            atoms.info["charge"] = charge
            atoms.info["spin"] = spin_multiplicity # Orb expects multiplicity
    else:
        if atoms is not None:
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1 # Orb expects multiplicity
if model_type == "MatterSim":
    selected_model = st.sidebar.selectbox("Select MatterSim Model:", list(MATTERSIM_MODELS.keys()))
    model_path = MATTERSIM_MODELS[selected_model]
    # Display Citation
    if selected_model in MATTERSIM_CITATIONS:
        st.sidebar.info(MATTERSIM_CITATIONS[selected_model])
    else:
        st.sidebar.warning("Citation not available for this model.")
if model_type == "SEVEN_NET":
    selected_model = st.sidebar.selectbox("Select SEVENNET Model:", list(SEVEN_NET_MODELS.keys()))
    if selected_model == '7net-mf-ompa':
        selected_modal_7net = st.sidebar.selectbox("Select Modal (multi fidelity model):", ['omat24', 'mpa'])
    # if selected_model == '7net-omni-i8':
    #     selected_modal_7net = st.sidebar.selectbox("Select Modal (multi fidelity model):", ['matpes_r2scan', 'mpa', 'omol25_low'])
    # if selected_model == '7net-omni-i12':
    #     selected_modal_7net = st.sidebar.selectbox("Select Modal (multi fidelity model):", ['matpes_r2scan', 'mpa', 'omol25_low'])
    if selected_model == '7net-omni':
        selected_modal_7net = st.sidebar.selectbox("Select Modal (multi fidelity model):", ['matpes_r2scan', 'mpa', 'omat24', 'matpes_pbe', 'oc20', 'oc22', 'odac23', 'omol25_low', 'omol25_high', 'spice', 'qcml', 'pet_mad', 'mp_r2scan'])
    model_path = SEVEN_NET_MODELS[selected_model]
    # Display Citation
    if selected_model in SEVEN_NET_CITATIONS:
        st.sidebar.info(SEVEN_NET_CITATIONS[selected_model])
    else:
        st.sidebar.warning("Citation not available for this model.")
if model_type == "UPET":
    selected_model = st.sidebar.selectbox("Select UPET Model:", list(UPET_MODELS.keys()))
    model_path = UPET_MODELS[selected_model]
    # Display Citation
    if selected_model in UPET_CITATIONS:
        st.sidebar.info(UPET_CITATIONS[selected_model])
    else:
        st.sidebar.warning("Citation not available for this model.")
    non_conservative = st.sidebar.toggle("Direct (non-conservative forces)?", value=True)
    uncertainty = st.sidebar.toggle("Uncertainty Prediction?", value=False)
if model_type=="UFF":
    selected_model = "N/A"
    # st.sidebar.warning('The currently implemented UFF calculator is found to be somewhat unstable in internal tests. Its usage is only recommended for energy value evaluations and not for geometry optimizations.')
if model_type=="xTB":
    selected_model = "N/A"
if model_type=="D3 dispersion":
    selected_model = "N/A"
    # Exchange-correlation functional
    xc_dsip = st.sidebar.text_input("XC Functional", value="PBE")
    st.sidebar.info('You can get the codes of supported XC functionals from this [link]([https://github.com/pfnet-research/torch-dftd/blob/master/torch_dftd/dftd3_xc_params.py).')

    # D2 or D3 selection
    method_disp = st.sidebar.radio(
        "Dispersion Method",
        ("DFTD2", "DFTD3"),
        index=1  # default DFTD3
    )
    old_disp = (method_disp == "DFTD2")  # D2 → old=True

    # Damping method
    damping_disp = st.sidebar.selectbox(
        "Damping Method",
        ["zero", "bj", "zerom", "bjm"],
        index=1
    )
if model_type == "In-House":
    selected_model = st.sidebar.selectbox("Select In-House Model:", ['QM9-Gap'])
    model_path = 'mlip-studio-qm9-gap.pt'
if atoms is not None and selected_model is not None:

    if atoms.pbc.any() and model_type=="UFF":
        st.error("UFF Calculator does not support PBC!")
        st.stop()
    if atoms.pbc.any() and model_type=="xTB":
        st.sidebar.warning("xTB Calculator sometimes fails for some dense periodic solids such as Silicon!")



device = st.sidebar.radio("Computation Device:", ["CPU", "CUDA (GPU)"], index=0 if not torch.cuda.is_available() else 1)
device = "cuda" if device == "CUDA (GPU)" and torch.cuda.is_available() else "cpu"

if device == "cpu" and torch.cuda.is_available():
    st.sidebar.info("GPU is available but CPU was selected.")
elif device == "cpu" and not torch.cuda.is_available():
    st.sidebar.info("No GPU detected. Using CPU.")

st.sidebar.markdown("## Task Selection")
if input_method=="Batch Upload" or input_method=="extXYZ Trajectory Upload":
    task = st.sidebar.selectbox("Select Calculation Task:", 
                           ["Batch Energy + Forces + Stress Calculation", 
                            "Batch Atomization/Cohesive Energy", 
                            "Batch HOMO-LUMO Gap Prediction"
                            ])
else:
    task = st.sidebar.selectbox("Select Calculation Task:", 
                            ["Energy Calculation", 
                            "Energy + Forces + Stress Calculation", 
                            "Hessian Calculation", 
                            "Atomization/Cohesive Energy", 
                            "Geometry Optimization", 
                            "Cell + Geometry Optimization",
                            #"Global Optimization",
                            "Vibrational Mode Analysis",
                            #"Phonons",
                            "Band Gap and Density of States",
                            "Dipole Moment and Partial Charges",
                            "Equation of State",
                            "Spin Determination",
                            "HOMO-LUMO Gap"
                            ])

if "Optimization" in task:
    # st.sidebar.markdown("### Optimization Parameters")
    # max_steps = st.sidebar.slider("Maximum Steps:", min_value=10, max_value=200, value=50, step=1) # Increased max_steps
    # fmax = st.sidebar.slider("Convergence Threshold (eV/Å):", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f") # Adjusted default fmax
    # optimizer_type = st.sidebar.selectbox("Optimizer:", ["BFGS", "LBFGS", "FIRE"], index=1) # Renamed to optimizer_type
    st.sidebar.markdown("### Optimization Parameters")
    
    # 1. Configuration for GLOBAL Optimization
    if task == "Global Optimization":
        global_method = st.sidebar.selectbox("Method:", ["Basin Hopping", "Minima Hopping"])
        
        # Common parameters
        temperature_K = st.sidebar.number_input("Temperature (K):", min_value=10.0, max_value=2000.0, value=300.0, step=10.0)
        global_steps = st.sidebar.number_input("Search Steps:", min_value=10, max_value=500, value=50, step=10)
        # Basin Hopping specific
        if global_method == "Basin Hopping":
            dr_amp = st.sidebar.number_input("Displacement Amplitude (Å):", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
            fmax_local = st.sidebar.number_input("Local Relaxation Threshold (eV/Å):", value=0.05, format="%.3f")
            
        # Minima Hopping specific
        elif global_method == "Minima Hopping":
            st.sidebar.caption("Minima Hopping automates threshold adjustments to escape local minima.")
            fmax_local = st.sidebar.number_input("Local Relaxation Threshold (eV/Å):", value=0.05, format="%.3f")

    # 2. Configuration for LOCAL/CELL Optimization
    else:
        max_steps = st.sidebar.slider("Maximum Steps:", min_value=1, max_value=1000, value=50, step=1)
        fmax = st.sidebar.slider("Convergence Threshold (eV/Å):", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
        # optimizer_type = st.sidebar.selectbox("Optimizer:", ["BFGS", "LBFGS", "FIRE"], index=1)
        optimizer_type = st.sidebar.selectbox(
            "Optimizer:",
            [
                "BFGS",
                "BFGSLineSearch",
                "LBFGS",
                "LBFGSLineSearch",
                "FIRE",
                "GPMin",
                "MDMin",
                "FASTMSO",
                "Lindh Hessian LBFGS",
                "MACE Hessian LBFGS",
                "MACE-Seed LBFGS",
            ],
            index=2,
        )
        if optimizer_type == "FASTMSO":
            st.sidebar.markdown(
                        """
                        **FASTMSO (Fast Multi-Stage Optimizer)**  
                        An adaptive optimizer that automatically switches between FIRE, MDMin,
                        and LBFGS based on the current force magnitude.  
                        Designed for fast and robust geometry optimization, especially with
                        machine-learning interatomic potentials.
                        """
                    )
            f_fire = st.sidebar.number_input(
                "FIRE → MDMin force threshold (eV/Å)",
                value=0.8
            )
            f_md = st.sidebar.number_input(
                "MDMin → LBFGS force threshold (eV/Å)",
                value=0.25
            )

        elif optimizer_type == "Lindh Hessian LBFGS":
            st.sidebar.caption(
                "Lindh model-Hessian preconditioning with no line search and "
                "one target force evaluation per cycle."
            )
            with st.sidebar.expander("Advanced Lindh Hessian LBFGS settings"):
                lindh_maxstep = st.number_input(
                    "Maximum atomic step (A)", min_value=0.001, max_value=1.0,
                    value=0.20, step=0.01, format="%.3f",
                )
                lindh_memory = st.number_input(
                    "L-BFGS memory", min_value=1, max_value=200, value=20, step=1,
                )
                lindh_rebuild_interval = st.number_input(
                    "Hessian rebuild interval", min_value=1, max_value=50,
                    value=1, step=1,
                )
                lindh_eigenvalue_floor = st.number_input(
                    "Regularization floor/shift (eV/A^2)", min_value=1.0e-5,
                    max_value=10.0, value=0.10, step=0.01, format="%.5f",
                )
                lindh_diagnostic_logging = st.checkbox(
                    "Enable diagnostic logging", value=False,
                )
        elif optimizer_type == "MACE Hessian LBFGS":
            mace_hessian_provider_model = None
            if model_type != "MACE":
                st.sidebar.info(
                    "The selected target model still provides energies and forces. "
                    "A MACE model is used only for analytical-Hessian construction."
                )
                mace_hessian_provider_options = list(MACE_MODELS.keys())
                default_provider = "MACE OMAT Small"
                mace_hessian_provider_model = st.sidebar.selectbox(
                    "MACE Hessian provider:",
                    mace_hessian_provider_options,
                    index=(
                        mace_hessian_provider_options.index(default_provider)
                        if default_provider in mace_hessian_provider_options else 0
                    ),
                )
            st.sidebar.caption(
                "Analytical MACE-Hessian preconditioning with no line search and "
                "one new target geometry per cycle."
            )
            with st.sidebar.expander("Advanced MACE Hessian LBFGS settings"):
                mace_hessian_maxstep = st.number_input(
                    "Maximum atomic step (A)", min_value=0.001, max_value=1.0,
                    value=0.20, step=0.01, format="%.3f",
                )
                mace_hessian_memory = st.number_input(
                    "L-BFGS memory", min_value=1, max_value=200, value=20, step=1,
                )
                mace_hessian_rebuild_interval = st.number_input(
                    "Analytical Hessian rebuild interval", min_value=1,
                    max_value=50, value=1, step=1,
                )
                mace_hessian_eigenvalue_floor = st.number_input(
                    "Eigenvalue floor (eV/A^2)", min_value=1.0e-5,
                    max_value=10.0, value=0.10, step=0.01, format="%.5f",
                )
                mace_hessian_diagnostic_logging = st.checkbox(
                    "Enable analytical Hessian diagnostic logging", value=False,
                )
        elif optimizer_type == "MACE-Seed LBFGS":
            st.sidebar.caption(
                "One initial MACE OMAT Small Hessian, no line search, and one "
                "target force evaluation per cycle."
            )
            with st.sidebar.expander("Advanced MACE-Seed LBFGS settings"):
                mace_seed_maxstep = st.number_input(
                    "Maximum atomic step (A)", min_value=0.001, max_value=1.0,
                    value=0.20, step=0.01, format="%.3f",
                )
                mace_seed_initial_radius = st.number_input(
                    "Initial adaptive step radius (A)", min_value=0.001,
                    max_value=1.0, value=0.10, step=0.01, format="%.3f",
                )
                mace_seed_minimum_radius = st.number_input(
                    "Minimum adaptive step radius (A)", min_value=0.001,
                    max_value=1.0, value=0.01, step=0.005, format="%.3f",
                )
                mace_seed_memory = st.number_input(
                    "L-BFGS memory", min_value=1, max_value=200, value=20, step=1,
                )
                mace_seed_eigenvalue_floor = st.number_input(
                    "Initial Hessian eigenvalue floor (eV/A^2)", min_value=1.0e-5,
                    max_value=10.0, value=0.10, step=0.01, format="%.5f",
                )
                mace_seed_diagnostic_logging = st.checkbox(
                    "Enable initial Hessian diagnostic logging", value=False,
                )

if "Equation of State" in task:
    st.sidebar.info("⚠️ **Note:** For accurate bulk modulus calculations, please use an optimized/relaxed structure. "
            "This calculation uses the same fractional coordinates for all volumes and does not optimize atomic positions.")
    
    # Configuration options
    num_points = st.sidebar.number_input("Number of volume points", min_value=5, max_value=25, value=11, 
                                     help="Number of volumes to calculate (odd number recommended)")
    # with col2:
    volume_range = st.sidebar.slider("Volume range (%)", min_value=5, max_value=30, value=10,
                                help="Percentage deviation from original volume (±%)")
    # with col3:
    eos_type = st.sidebar.selectbox("Equation of State", 
                            ["Birch-Murnaghan", "Murnaghan", "Vinet"],
                            help="Choose the EOS to fit")

if "Vibration" in task:
    st.write("### Thermodynamic Quantities (Molecule Only)")
    T = st.sidebar.number_input("Temperature (K)", value=298.15)


if task == "Spin Determination":
    if is_omol_model:
        
        # Get charge
        charge = st.sidebar.number_input("Charge", value=0, step=1, key="spin_opt_charge")
        
        # Determine reasonable spin range based on number of electrons
        n_electrons = sum([atom.number for atom in atoms]) - charge
        max_unpaired = min(n_electrons, 10)  # Limit to reasonable range
        
        # Spin multiplicity range (2S+1)
        min_mult = 1
        max_mult = max_unpaired + 1
        
        st.sidebar.write(f"**System info:** {n_electrons} electrons (after accounting for charge)")
        st.sidebar.write(f"Testing spin multiplicities from {min_mult} to {max_mult}")
        
        spin_range = st.sidebar.slider(
            "Spin multiplicity range to test (2S+1)", 
            min_value=1, 
            max_value=max_mult, 
            value=(1, min(5, max_mult)),
            help="Spin multiplicity = 2S + 1, where S is total spin"
        )

if atoms is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### Structure Visualization', unsafe_allow_html=True)
        viz_style = st.selectbox("Select Visualization Style:", 
                           ["ball-stick", 
                            "stick", 
                            "ball"])
        view_3d = get_structure_viz2(atoms, style=viz_style, show_unit_cell=True, width=400, height=400)
        st.components.v1.html(view_3d._make_html(), width=400, height=400)
        
        st.markdown("### Structure Information")
        atoms_info = {
            "Number of Atoms": len(atoms),
            "Chemical Formula": atoms.get_chemical_formula(),
            "Periodic Boundary Conditions (PBC)": atoms.pbc.tolist(),
            "Cell Dimensions": np.round(atoms.cell.cellpar(),3).tolist() if atoms.pbc.any() and atoms.cell is not None and atoms.cell.any() else "No cell / Non-periodic",
            "Atom Types": ", ".join(sorted(list(set(atoms.get_chemical_symbols()))))
        }
        for key, value in atoms_info.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.markdown('## Calculation Setup', unsafe_allow_html=True)
        st.markdown("### Selected Model")
        st.write(f"**Model Type:** {model_type}")
        st.write(f"**Model:** {selected_model}")
        if model_type == "FairChem" and "UMA Small" in selected_model:
            st.write(f"**UMA Task Type:** {selected_task_type}")
        if model_type == "MACE":
            st.write(f"**Dispersion:** {dispersion}")
        st.write(f"**Device:** {device}")
        
        st.markdown("### Selected Task")
        st.write(f"**Task:** {task}")
        
        if "Geometry Optimization" in task:
            st.write(f"**Max Steps:** {max_steps}")
            st.write(f"**Convergence Threshold:** {fmax} eV/Å")
            st.write(f"**Optimizer:** {optimizer_type}")
        
        run_calculation = st.button("Run Calculation", type="primary")
        
        if run_calculation:            
            results = {}
            #global table_placeholder # Ensure they are accessible
            opt_log = [] # Reset log for each run
            if "Optimization" in task:
                 table_placeholder = st.empty() # Recreate placeholder for table

            try:
                torch.set_default_dtype(torch.float32)
                with st.spinner("Running calculation... Please wait."):
                    calc_atoms = atoms.copy()
                    calc = None
                    if model_type == "MACE":

                        calc = get_mace_model(model_path, dispersion, device, 'float32')
                    elif model_type == "FairChem":  # FairChem

                        calc = get_fairchem_model(selected_model, model_path, device, selected_task_type)
                    elif model_type == "ORB":

                        orbff = model_path(device=device, precision=selected_default_dtype)
                        calc = ORBCalculator(orbff, device=device)
                    elif model_type == "MatterSim":

                        # NOTE: Running mattersim on windows requires changing source code file
                        # https://github.com/microsoft/mattersim/issues/112
                        # mattersim/datasets/utils/convertor.py: 117

                        calc = MatterSimCalculator(load_path=model_path, device=device)
                    elif model_type == "SEVEN_NET":
                        # st.write("Setting up SEVENNET calculator...")
                        if model_path=='7net-mf-ompa' or model_path=='7net-omni' or model_path=='7net-omni-i8' or model_path=='7net-omni-i12':
                            calc = SevenNetCalculator(model=model_path, modal=selected_modal_7net, device=device)
                        else:
                            calc = SevenNetCalculator(model=model_path, device=device)
                    elif model_type == "UPET":
                        if model_path!='pet-mad-dos':
                            calc = UPETCalculator(
                                                    model=model_path,
                                                    version=UPET_MODELS_VERSIONS[selected_model],
                                                    non_conservative=non_conservative,
                                                    device=device
                                                )
                        else:
                            calc = PETMADDOSCalculator(version="latest", device='cpu')
                            st.sidebar.warning('NOTE: The PET-MAD-DOS Calculator only supports Band Gap and DOS inference.')
                            st.sidebar.info('Currently, the PET-MAD-DOS Calculator only seems to work with CPU as the device. \nSo the calculation will use CPU regardless of what the user selected.')
                    elif model_type == "UFF":
                        calc = UFFCalculator(charge=0)
                    elif model_type == "xTB":
                        if atoms.pbc.any():
                            xtb_method = 'GFN1-xTB'
                            print(xtb_method)
                        else:
                            xtb_method = 'GFN2-xTB'
                        calc = XTBCalculator(xtb_command='xtb', method=xtb_method, debug=True, keep_files=True)
                    elif model_type == "D3 dispersion":
                        calc = TorchDFTD3Calculator(atoms=atoms, device=device, old=old_disp, damping=damping_disp, dtype=torch.float32)
                    elif model_type == "In-House":   
                        if "HOMO-LUMO" not in task:
                            st.warning('NOTE: The In-House `mlip-studio-qm9-gap.model` only supports HOMO-LUMO gap inference.')

                    if calc is not None:
                        calc_atoms.calc = calc
                    
                    if task == "Energy Calculation":
                        t0 = time.perf_counter()
                        calc_atoms.info["external_field"] = [0.0, 0.0, 0.0]
                        energy = calc_atoms.get_potential_energy()
                        
                        if model_type=="UPET" and uncertainty:
                            energy_uncertainty = calc.get_energy_uncertainty(atoms, per_atom=True)
                            energy_ensemble = calc.get_energy_ensemble(atoms, per_atom=True)
                        t1 = time.perf_counter()
                        results["Energy"] = f"{energy:.6f} eV"
                        results["Time Taken"] = f"{t1 - t0:.4f} seconds"
                        if model_type=="UPET" and uncertainty:
                            results["Energy Uncertainty"] = f"{energy_uncertainty[0][0]:.6f} eV/atom"
                            results["Energy Ensemble"] = energy_ensemble
                        st.success("Calculation completed successfully!")
                        st.markdown("### Results")
                        for key, value in results.items():
                            st.write(f"**{key}:** {value}")
                        # =========================
                        # Ensemble Analysis
                        # =========================
                        if model_type == "UPET" and uncertainty:


                            ensemble = np.array(energy_ensemble).flatten()
                            mean_energy = ensemble.mean()
                            std_energy = ensemble.std()

                            st.markdown("---")
                            st.markdown("### 🔬 Ensemble Statistics")

                            st.metric("Uncertainty (eV/atom)", f"{energy_uncertainty[0][0]:.6f}")
                            col1, col2 = st.columns(2)
                            col1.metric("Ensemble Mean (eV/atom)", f"{mean_energy:.6f}")
                            col2.metric("Std Dev (eV/atom)", f"{std_energy:.6f}")
                            

                            # =========================
                            # Interactive Distribution Plot
                            # =========================
                            fig = px.histogram(
                                ensemble,
                                nbins=30,
                                marginal="box",
                                title="Energy Ensemble Distribution",
                            )

                            fig.add_vline(
                                x=mean_energy,
                                line_dash="dash",
                                annotation_text="Mean",
                                annotation_position="top right"
                            )

                            fig.update_layout(
                                template="plotly_white",
                                xaxis_title="Energy (eV)",
                                yaxis_title="Count",
                                height=500,
                            )

                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif task == "Energy + Forces + Stress Calculation":
                        t0 = time.perf_counter()
                        energy = calc_atoms.get_potential_energy()
                        forces = calc_atoms.get_forces()
                        max_force = np.max(np.linalg.norm(forces, axis=1)) if len(forces) > 0 else 0.0
                        t1 = time.perf_counter()
                        # Store results
                        results["Energy"] = f"{energy:.6f} eV"
                        results["Maximum Force"] = f"{max_force:.6f} eV/Å"
                        results["Time Taken"] = f"{t1 - t0:.4f} seconds"
                        st.success("Calculation completed successfully!")
                        st.markdown("### Results")

                        # Display energy & max force
                        for key, value in results.items():
                            st.write(f"**{key}:** {value}")

                        # --- Atomic Forces Table ---
                        st.markdown("### Atomic Forces (eV/Å)")
                        force_df = pd.DataFrame(
                            forces,
                            columns=["Fx (eV/Å)", "Fy (eV/Å)", "Fz (eV/Å)"]
                        )
                        force_df["Atom Index"] = force_df.index
                        force_df = force_df[["Atom Index", "Fx (eV/Å)", "Fy (eV/Å)", "Fz (eV/Å)"]]
                        st.dataframe(force_df, use_container_width=True)

                        # --- Stress Tensor (if applicable) ---
                        if calc_atoms.get_cell().volume > 1e-6:  # has a real cell
                            try:
                                stress = calc_atoms.get_stress()  # ASE returns Voigt: 6 components
                                # Convert to a nicer 3×3 tensor
                                stress_tensor = np.array([
                                    [stress[0], stress[5], stress[4]],
                                    [stress[5], stress[1], stress[3]],
                                    [stress[4], stress[3], stress[2]],
                                ])
                                st.markdown("### Stress Tensor (eV/Å³)")
                                st.write(pd.DataFrame(
                                    stress_tensor,
                                    columns=["σxx", "σxy", "σxz"],
                                    index=["σxx", "σyy", "σzz"]
                                ))
                            except Exception as e:
                                st.warning(f"Stress could not be computed: {e}")
                    elif task == "Band Gap and Density of States":

                        st.markdown("### Electronic Structure: Band Gap & Density of States")

                        if model_path == "pet-mad-dos":

                            st.info(
                                "This task calculates **band gap**, **Fermi level**, and "
                                "**density of states (DOS)** using the `pet-mad-dos` model."
                            )

                            with st.spinner("Computing DOS and band gap..."):

                                calc_atoms = calc_atoms.copy()
                                energies, dos = calc.calculate_dos(calc_atoms)
                                bandgap = calc.calculate_bandgap(calc_atoms, dos=dos)
                                fermi_level = calc.calculate_efermi(calc_atoms, dos=dos)

                            # Convert to numpy
                            energies = energies.squeeze().detach().cpu().numpy()
                            dos = dos.squeeze().detach().cpu().numpy()
                            fermi_level = fermi_level.item()
                            bandgap = bandgap.item()

                            # Shift energies relative to Fermi level
                            energies_shifted = energies - fermi_level

                            # --- Clean metrics layout ---
                            col1, col2 = st.columns(2)
                            col1.metric("Band Gap (eV)", f"{bandgap:.4f}")
                            col2.metric("Fermi Level (eV)", f"{fermi_level:.4f}")

                            st.markdown("---")

                            # --- Interactive DOS Plot ---
                            import plotly.graph_objects as go

                            fig = go.Figure()

                            fig.add_trace(
                                go.Scatter(
                                    x=energies_shifted,
                                    y=dos,
                                    mode="lines",
                                    name="DOS",
                                    line=dict(width=2),
                                    hovertemplate="Energy (E - Eₓ): %{x:.3f} eV<br>DOS: %{y:.4f}<extra></extra>",
                                )
                            )

                            # Fermi level vertical line (at 0 eV)
                            fig.add_vline(
                                x=0,
                                line_width=2,
                                line_dash="dash",
                                annotation_text="Fermi Level",
                                annotation_position="top right"
                            )

                            fig.update_layout(
                                template="plotly_white",
                                title="Density of States",
                                xaxis_title="Energy (E - Eₓ) [eV]",
                                yaxis_title="Density of States",
                                hovermode="x unified",
                                height=550,
                            )

                            st.plotly_chart(fig, use_container_width=True)

                        else:
                            st.error(
                                "Band Gap and DOS prediction is only supported by the "
                                "`pet-mad-dos` UPET model. Please select a compatible model."
                            )
                    elif task=="HOMO-LUMO Gap":
                        
                        if "qm9" in model_path:
                            st.info(
                                "This task calculates the **HOMO-LUMO gap** of molecules containing C, H, N,O and F atoms using our in-house `mlip-studio-qm9-gap` model."
                            )
                            if any(atoms.pbc):
                                st.error("❌ HOMO-LUMO Gap calculation can only be done for molecular structures.")
                                results["Error"] = "Periodic structure"
                            else:
                                # QM9 allowed elements
                                qm9_elements = {"H", "C", "N", "O", "F"}

                                # Extract elements from atoms object
                                atom_symbols = set(atoms.get_chemical_symbols())

                                # Check for invalid elements
                                invalid_elements = atom_symbols - qm9_elements
                                if invalid_elements:
                                    st.error(
                                        f"Unsupported element(s) found: {', '.join(sorted(invalid_elements))}. "
                                        f"QM9 models only support: {', '.join(sorted(qm9_elements))}."
                                    )
                                    st.stop()
                                # Usage
                                ##########
                                # QM 9 HOMO LUMO in house model
                                ##########
                                t0 = time.perf_counter()
                                model_mlip_studio_qm9_gap, train_args = load_model(model_path, device=torch.device(device))
                                gap = predict_atoms_HOMO_LUMO_QM9_inhouse(model_mlip_studio_qm9_gap, atoms)
                                t1 = time.perf_counter()
                                # Store results
                                st.metric(f"HOMO-LUMO Gap (eV)", f"{gap:.4f}")
                                results["HOMO-LUMO Gap"] = f"{gap:.6f} eV"
                                results["Time Taken"] = f"{t1 - t0:.4f} seconds"
                                st.success("Calculation completed successfully!")
                            st.markdown("### Results")
                            for key, value in results.items():
                                st.write(f"**{key}:** {value}")
                        else:
                            st.error(
                                "HOMO-LUMO gap prediction is only supported by the "
                                "in-house `mlip-studio-qm9-gap` model. Please select the right model."
                            )
                    elif task == "Batch HOMO-LUMO Gap Prediction":
                        t0 = time.perf_counter()
                        if len(atoms_list) == 0:
                            st.warning("Please upload multiple structures using 'Batch Upload' mode.")
                        else:
                            st.subheader("Batch HOMO-LUMO Gap Prediction")
                            st.write(f"Processing {len(atoms_list)} structures...")

                            # Load model once
                            model_mlip_studio_qm9_gap, train_args = load_model(model_path, device=torch.device(device))

                            # Results collectors
                            batch_results = []
                            ref_gaps, calc_gaps = [], []

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for idx, atoms_obj in enumerate(atoms_list):
                                status_text.text(f"Calculating structure {idx+1}/{len(atoms_list)}...")

                                try:
                                    # -----------------------------------------------
                                    # 1. Look for reference gap in calc results
                                    # -----------------------------------------------
                                    ref_gap = None

                                    if atoms_obj.calc is not None and hasattr(atoms_obj.calc, "results"):
                                        ref_gap = _find_value(
                                            atoms_obj.calc.results,
                                            keywords=["gap", "homo_lumo_gap", "HOMO_LUMO_gap",
                                                    "homo-lumo_gap", "HL_gap", "hl_gap",
                                                    "GAP", "HOMO_LUMO", "homo_lumo"]
                                        )

                                    # -----------------------------------------------
                                    # 2. Fallback: atoms.info
                                    # -----------------------------------------------
                                    if ref_gap is None:
                                        ref_gap = _find_value(
                                            atoms_obj.info,
                                            keywords=["gap", "homo_lumo_gap", "HOMO_LUMO_gap",
                                                    "homo-lumo_gap", "HL_gap", "hl_gap",
                                                    "GAP", "HOMO_LUMO", "homo_lumo"]
                                        )

                                    has_ref_gap = ref_gap is not None

                                    # -----------------------------------------------
                                    # 3. Predict gap
                                    # -----------------------------------------------
                                    gap = predict_atoms_HOMO_LUMO_QM9_inhouse(model_mlip_studio_qm9_gap, atoms_obj)

                                    # Collect parity data
                                    if has_ref_gap:
                                        ref_gaps.append(float(ref_gap))
                                        calc_gaps.append(float(gap))

                                    # Metadata
                                    filename = atoms_obj.info.get("source_name", f"structure_{idx+1}")
                                    formula  = atoms_obj.get_chemical_formula()
                                    natoms   = len(atoms_obj)
                                    filetype = os.path.splitext(filename)[1].lstrip('.')

                                    result_dict = {
                                        "Filename":            filename,
                                        "Formula":             formula,
                                        "N_atoms":             natoms,
                                        "Filetype":            filetype,
                                        "HOMO-LUMO Gap (eV)":  f"{gap:.6f}",
                                    }

                                    if has_ref_gap:
                                        error = gap - float(ref_gap)
                                        result_dict["Ref Gap (eV)"]        = f"{float(ref_gap):.6f}"
                                        result_dict["Gap Error (eV)"]      = f"{error:.6f}"
                                        result_dict["Gap Abs Error (eV)"]  = f"{abs(error):.6f}"

                                    batch_results.append(result_dict)

                                except Exception as e:
                                    batch_results.append({
                                        "Filename":           atoms_obj.info.get("source_name", f"structure_{idx+1}"),
                                        "Formula":            "Error",
                                        "N_atoms":            "-",
                                        "Filetype":           "-",
                                        "HOMO-LUMO Gap (eV)": "Failed",
                                        "Gap Error (eV)":     str(e),
                                    })

                                progress_bar.progress((idx + 1) / len(atoms_list))

                            t1 = time.perf_counter()
                            status_text.text("Calculation complete!")
                            st.write(f"Time Taken = {t1 - t0:.4f} seconds")
                            st.success("Calculation completed successfully!")

                            # ===============================================
                            # PARITY PLOT
                            # ===============================================
                            if len(ref_gaps) > 0:
                                st.markdown("## 📊 Parity Plot (Reference vs Calculated)")
                                st.markdown("*Points closer to the diagonal indicate better agreement.*")

                                st.markdown("### ⚛️ HOMO-LUMO Gap Parity Plot")

                                ref_arr  = np.array(ref_gaps,  dtype=np.float64)
                                calc_arr = np.array(calc_gaps, dtype=np.float64)

                                valid_mask = ~(np.isnan(ref_arr) | np.isnan(calc_arr))
                                ref_arr  = ref_arr[valid_mask]
                                calc_arr = calc_arr[valid_mask]

                                if len(ref_arr) > 0:
                                    fig_parity, ax_parity = plt.subplots(figsize=(7, 6))

                                    ax_parity.scatter(ref_arr, calc_arr, alpha=0.6, s=50,
                                                    edgecolors='black', linewidth=0.5, color='steelblue')

                                    min_val = min(ref_arr.min(), calc_arr.min())
                                    max_val = max(ref_arr.max(), calc_arr.max())
                                    ax_parity.plot([min_val, max_val], [min_val, max_val],
                                                'r--', lw=2, label='Perfect agreement')

                                    mae  = np.mean(np.abs(ref_arr - calc_arr))
                                    rmse = np.sqrt(np.mean((ref_arr - calc_arr) ** 2))
                                    ss_res = np.sum((ref_arr - calc_arr) ** 2)
                                    ss_tot = np.sum((ref_arr - np.mean(ref_arr)) ** 2)
                                    r2   = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

                                    textstr = (f'MAE  = {mae:.4f} eV\n'
                                            f'RMSE = {rmse:.4f} eV\n'
                                            f'R²   = {r2:.4f}\n'
                                            f'N    = {len(ref_arr)}')
                                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                                    ax_parity.text(0.05, 0.95, textstr, transform=ax_parity.transAxes,
                                                fontsize=10, verticalalignment='top', bbox=props)

                                    ax_parity.set_xlabel("Reference HOMO-LUMO Gap (eV)", fontsize=12, fontweight='bold')
                                    ax_parity.set_ylabel("Predicted HOMO-LUMO Gap (eV)", fontsize=12, fontweight='bold')
                                    ax_parity.set_title("HOMO-LUMO Gap Parity Plot", fontsize=13, fontweight='bold')
                                    ax_parity.legend(loc='lower right')
                                    ax_parity.grid(True, alpha=0.3)
                                    ax_parity.set_aspect('equal', adjustable='box')

                                    st.pyplot(fig_parity)
                                    plt.close(fig_parity)
                            else:
                                st.info("ℹ️ No reference gap data found. Parity plot requires structures "
                                        "with a reference gap stored in calc.results or atoms.info.")

                            # ===============================================
                            # RESULTS TABLE
                            # ===============================================
                            st.markdown("---")
                            st.markdown("## 📋 Calculation Results")

                            df_results = pd.DataFrame(batch_results)
                            st.dataframe(df_results, use_container_width=True)

                            # ===============================================
                            # DISTRIBUTION PLOTS
                            # ===============================================
                            st.markdown("---")
                            st.markdown("## 📈 Statistical Analysis")

                            df_results["Gap_float"] = pd.to_numeric(df_results["HOMO-LUMO Gap (eV)"], errors="coerce")
                            gaps_clean = df_results["Gap_float"].dropna()

                            if len(gaps_clean) > 0:
                                # --- Histogram ---
                                st.markdown("### Gap Distribution (Histogram)")
                                fig_hist, ax_hist = plt.subplots()
                                ax_hist.hist(gaps_clean, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
                                ax_hist.axvline(gaps_clean.mean(), color='red',   linestyle='--',
                                                linewidth=1.5, label=f'Mean = {gaps_clean.mean():.3f} eV')
                                ax_hist.axvline(gaps_clean.median(), color='orange', linestyle='--',
                                                linewidth=1.5, label=f'Median = {gaps_clean.median():.3f} eV')
                                ax_hist.set_xlabel("HOMO-LUMO Gap (eV)")
                                ax_hist.set_ylabel("Count")
                                ax_hist.set_title("HOMO-LUMO Gap Distribution Across Structures")
                                ax_hist.legend()
                                ax_hist.grid(True, alpha=0.3)
                                st.pyplot(fig_hist)
                                plt.close(fig_hist)

                                # --- Gap vs Structure Index ---
                                st.markdown("### Gap vs Structure Index")
                                fig_trend, ax_trend = plt.subplots()
                                ax_trend.plot(range(len(gaps_clean)), gaps_clean.values,
                                            marker='o', linestyle='-', linewidth=1.5, color='steelblue')
                                ax_trend.set_xlabel("Structure Index")
                                ax_trend.set_ylabel("HOMO-LUMO Gap (eV)")
                                ax_trend.set_title("HOMO-LUMO Gap Trend Across Batch")
                                ax_trend.xaxis.set_major_locator(MaxNLocator(integer=True))
                                ax_trend.grid(True, alpha=0.3)
                                st.pyplot(fig_trend)
                                plt.close(fig_trend)

                                # --- Error distribution (only if ref data available) ---
                                if "Gap Error (eV)" in df_results.columns:
                                    df_results["GapError_float"] = pd.to_numeric(
                                        df_results["Gap Error (eV)"], errors="coerce")
                                    errors_clean = df_results["GapError_float"].dropna()

                                    if len(errors_clean) > 0:
                                        st.markdown("### Gap Error Distribution (Predicted − Reference)")
                                        fig_err, ax_err = plt.subplots()
                                        ax_err.hist(errors_clean, bins=20, edgecolor='black',
                                                    alpha=0.7, color='salmon')
                                        ax_err.axvline(0, color='black', linestyle='-',  linewidth=1.0)
                                        ax_err.axvline(errors_clean.mean(), color='red', linestyle='--',
                                                    linewidth=1.5,
                                                    label=f'Mean error = {errors_clean.mean():.4f} eV')
                                        ax_err.set_xlabel("Gap Error (eV)")
                                        ax_err.set_ylabel("Count")
                                        ax_err.set_title("HOMO-LUMO Gap Prediction Error Distribution")
                                        ax_err.legend()
                                        ax_err.grid(True, alpha=0.3)
                                        st.pyplot(fig_err)
                                        plt.close(fig_err)
                    elif task == "Dipole Moment and Partial Charges":

                        st.markdown("### Electrostatics: Dipole Moment and Partial Charges")

                        if "MACE POLAR" in selected_model:

                            st.info(
                                "This task calculates **dipole moment** (along x, y and z axes) and "
                                "**partial charges** for each atom using the `MACE POLAR` models."
                            )

                            with st.spinner("Computing dipole moment and charges..."):
                                _ = calc_atoms.get_potential_energy()
                                dipole_moment = calc_atoms.calc.results["dipole"]
                                # dipole_moment = calc_atoms.calc.get_dipole_moment()
                                charges = calc_atoms.calc.results["charges"]

                                # --- Dipole Moment ---
                                st.markdown("#### ⚡ Dipole Moment")
                                cols = st.columns(3)
                                labels = ["X", "Y", "Z"]
                                for i, (col, label) in enumerate(zip(cols, labels)):
                                    with col:
                                        st.metric(label=f"μ_{label}", value=f"{dipole_moment[i]:.4f}", delta=None)
                                magnitude = float((dipole_moment ** 2).sum() ** 0.5)
                                st.markdown(
                                    f"<div style='text-align:right; color:gray; font-size:0.85em;'>|μ| = <b>{magnitude:.4f}</b> e·Å</div>",
                                    unsafe_allow_html=True,
                                )

                                st.divider()

                                # --- Partial Charges ---
                                st.markdown("#### 🔬 Partial Charges")

                                charge_df = pd.DataFrame({
                                    "Atom Index": range(len(charges)),
                                    "Symbol": [calc_atoms[i].symbol for i in range(len(charges))],
                                    "Charge (e)": [round(float(q), 5) for q in charges],
                                })

                                st.dataframe(
                                    charge_df.style
                                        .background_gradient(subset=["Charge (e)"], cmap="RdBu_r", vmin=-1, vmax=1)
                                        .format({"Charge (e)": "{:+.5f}"})
                                        .set_properties(**{"text-align": "center"}),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                total_charge = float(np.sum(charges))
                                st.caption(f"Σ charges = {total_charge:+.5f} e")


                        else:
                            st.error(
                                "Dipole Moment and Partial Charges prediction is only supported by the "
                                "`MACE POLAR 1` models. Please select a compatible model."
                            )
                    elif task == "Hessian Calculation":

                        st.markdown("### Analytical Hessian")

                        if "MACE" in model_type:

                            st.info(
                                "This task calculates the **Hessian** analytically using the `MACE` models."
                            )

                            if len(calc_atoms)>50:
                                st.error("Since, Hessian calculations are expensive, only systems with less than 50 atoms are supported on the web application. For larger systems, you can download the web application and run it on your own resources.")
                            else:
                                with st.spinner("Computing Hessian..."):
                                    t0 = time.perf_counter()
                                    hessian = calc_atoms.calc.get_hessian(atoms=calc_atoms)
                                    t1 = time.perf_counter()
                                    st.write(f"Time Taken = {t1-t0} s")

                                n_atoms = len(calc_atoms)
                                H = np.array(hessian).reshape(3 * n_atoms, 3 * n_atoms)
                                n_dof = 3 * n_atoms
                                with st.expander('Explanation of output:'):
                                    # -- Hessian equation ----------------------------------------------
                                    st.latex(
                                        r"H_{\alpha i,\, \beta j} = \frac{\partial^2 E}{\partial r_{\alpha i} \, \partial r_{\beta j}}"
                                    )
                                    st.markdown(
                                        "where *E* is the potential energy, **α, β** index atoms (0 to N−1), "
                                        "**i, j** index directions x, y, z (0, 1, 2), and "
                                        "**r_{αi}** is the displacement of atom α along direction i. "
                                        "Since both axes run over all N atoms × 3 directions, the Hessian is a **(3N × 3N)** matrix. "
                                        f"For this system: 3 × {n_atoms} = **{n_dof}**, giving a **{n_dof} × {n_dof}** matrix."
                                    )

                                    st.markdown("---")

                                    # -- Shape explanation ---------------------------------------------
                                    st.markdown("#### Output Shape & Degrees of Freedom")
                                    st.markdown(
                                        f"The system has **{n_atoms} atoms** ({n_atoms} × 3 = **{n_dof} degrees of freedom**, "
                                        f"one for each atom's displacement along x, y, z)."
                                    )
                                    st.markdown(
                                        f"The calculator returned shape **{np.array(hessian).shape}**, which encodes:\n"
                                        f"- **Axis 0 ({n_dof}):** the {n_dof} DOFs being displaced — *\"what if I nudge this DOF?\"*\n"
                                        f"- **Axis 1 ({n_atoms}):** the {n_atoms} atoms whose force response is measured\n"
                                        f"- **Axis 2 (3):** the x, y, z components of that force response\n\n"
                                        f"Since axes 1 and 2 together also span {n_atoms} × 3 = {n_dof} DOFs, the array is "
                                        f"reshaped to **({n_dof} × {n_dof})** — a square matrix where both axes index DOFs (0–{n_dof-1}):"
                                    )
                                    st.code(
                                        "\n".join(
                                            f"  DOF {i*3}–{i*3+2}  →  atom {i} (x, y, z)"
                                            for i in range(n_atoms)
                                        ),
                                        language="none",
                                    )

                                    st.markdown("---")

                                    # -- Block structure explanation ------------------------------------
                                    st.markdown("#### Block Structure of the Matrix")
                                    st.markdown(
                                        "The reshaped matrix is naturally partitioned into **3×3 blocks**, "
                                        "one per pair of atoms (α, β):\n\n"
                                        "- **Diagonal blocks** (α = β): self-interaction — "
                                        "how atom α resists its own displacement. Reflects local bond stiffness.\n"
                                        "- **Off-diagonal blocks** (α ≠ β): coupling — "
                                        "how displacing atom α exerts a force on atom β. "
                                        "Large values indicate strong interactions (e.g. directly bonded pairs); "
                                        "near-zero values indicate weak or no coupling."
                                    )

                                    st.markdown("---")

                                # -- Summary metrics ------------------------------------------------
                                st.markdown("#### Matrix Properties")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Matrix Size", f"{H.shape[0]} × {H.shape[1]}")
                                col2.metric("Max |Element|", f"{np.abs(H).max():.4f}")
                                col3.metric("Symmetry Error", f"{np.max(np.abs(H - H.T)):.2e}",
                                            help="Should be ~0: since ∂²E/∂r_{αi}∂r_{βj} = ∂²E/∂r_{βj}∂r_{αi}, the Hessian must be symmetric.")

                                # -- Heatmap --------------------------------------------------------
                                st.markdown("#### Hessian Matrix Heatmap")
                                st.markdown(
                                    "Colour encodes the value of each element **H_{αi, βj}** in **eV/Å²**. "
                                    "**Blue** = positive force constant (restoring), **red** = negative (destabilising), "
                                    "**white** ≈ zero (no coupling). "
                                    "Faint grid lines separate the 3×3 per-atom blocks."
                                )

                                abs_max = float(np.abs(H).max()) or 1.0
                                fig = go.Figure(
                                    go.Heatmap(
                                        z=H,
                                        colorscale="RdBu",
                                        zmid=0,
                                        zmin=-abs_max,
                                        zmax=abs_max,
                                        colorbar=dict(title="eV/Å²"),
                                        hovertemplate="row %{y}, col %{x}<br>value: %{z:.4f}<extra></extra>",
                                    )
                                )

                                for k in range(1, n_atoms):
                                    pos = k * 3 - 0.5
                                    fig.add_shape(type="line", x0=pos, x1=pos, y0=-0.5, y1=n_dof-0.5,
                                                line=dict(color="rgba(255,255,255,0.35)", width=1))
                                    fig.add_shape(type="line", x0=-0.5, x1=n_dof-0.5, y0=pos, y1=pos,
                                                line=dict(color="rgba(255,255,255,0.35)", width=1))

                                fig.update_layout(
                                    height=520,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    xaxis=dict(title="DOF index (αi)", showgrid=False),
                                    yaxis=dict(title="DOF index (βj)", showgrid=False, autorange="reversed"),
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                with st.expander("📋 Raw Hessian tensor"):
                                    st.code(repr(hessian), language="python")

                                buf = io.BytesIO()
                                np.save(buf, H)
                                buf.seek(0)
                                st.download_button(
                                    label="⬇ Download Hessian (.npy)",
                                    data=buf,
                                    file_name="hessian.npy",
                                    mime="application/octet-stream",
                                )

                        else:
                            st.error(
                                "Analytical Hessian is only supported by the "
                                "`MACE` models. Please select a compatible model."
                            )
                    
                    elif task == "Spin Determination":
                        if is_omol_model:
                            st.markdown("### Spin Determination")
                            st.info("This task calculates energies for different spin states to find the optimal spin multiplicity.")
                            results_data = []
                            t0 = time.perf_counter()
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            spin_mults = range(spin_range[0], spin_range[1] + 1)
                            total = len(spin_mults)
                            
                            for idx, spin_mult in enumerate(spin_mults):
                                S = (spin_mult - 1) / 2
                                unpaired = spin_mult - 1
                                
                                status_text.text(f"Calculating spin state: 2S+1 = {spin_mult}, S = {S}, unpaired = {unpaired}")
                                
                                try:
                                    # Set charge and spin
                                    calc_atoms = calc_atoms.copy()
                                    calc_atoms.info["charge"] = charge
                                    calc_atoms.info["spin"] = spin_mult
                                    calc_atoms.calc = calc
                                    # Calculate energy
                                    t0 = time.perf_counter()
                                    energy = calc_atoms.get_potential_energy()
                                    t1 = time.perf_counter()
                                    calc_time = t1 - t0
                                    
                                    results_data.append({
                                        "S": S,
                                        "2S+1": spin_mult,
                                        "Unpaired Electrons": unpaired,
                                        "Energy (eV)": energy,
                                        "Time (s)": calc_time
                                    })
                                    
                                except Exception as e:
                                    st.warning(f"Failed for spin multiplicity {spin_mult}: {str(e)}")
                                
                                progress_bar.progress((idx + 1) / total)
                            
                            status_text.empty()
                            progress_bar.empty()
                            t1 = time.perf_counter()
                            results["Time Taken"] = f"{t1 - t0:.4f} seconds"
                            if results_data:
                                st.success("Spin optimization completed successfully!")
                                # Create DataFrame
                                df = pd.DataFrame(results_data)
                                
                                # Find minimum energy
                                min_idx = df["Energy (eV)"].idxmin()
                                optimal_S = df.loc[min_idx, "S"]
                                optimal_mult = df.loc[min_idx, "2S+1"]
                                optimal_unpaired = df.loc[min_idx, "Unpaired Electrons"]
                                min_energy = df.loc[min_idx, "Energy (eV)"]
                                
                                # Display optimal result
                                st.markdown("### Optimal Spin State")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("S", f"{optimal_S:.1f}")
                                col2.metric("2S+1", f"{int(optimal_mult)}")
                                col3.metric("Unpaired e⁻", f"{int(optimal_unpaired)}")
                                col4.metric("Energy", f"{min_energy:.6f} eV")
                                
                                # Display results table
                                st.markdown("### Results Table")
                                # Format the dataframe for display
                                display_df = df.copy()
                                display_df["Energy (eV)"] = display_df["Energy (eV)"].apply(lambda x: f"{x:.6f}")
                                display_df["Time (s)"] = display_df["Time (s)"].apply(lambda x: f"{x:.4f}")
                                display_df["S"] = display_df["S"].apply(lambda x: f"{x:.1f}")
                                display_df["2S+1"] = display_df["2S+1"].astype(int)
                                display_df["Unpaired Electrons"] = display_df["Unpaired Electrons"].astype(int)
                                
                                # Highlight minimum energy row
                                def highlight_min(s):
                                    is_min = s == df.loc[min_idx, s.name]
                                    return ['background-color: #90EE90' if v else '' for v in is_min]
                                
                                st.dataframe(
                                    display_df.style.apply(highlight_min, subset=["Energy (eV)"]),
                                    use_container_width=True
                                )
                                
                                # Create plots
                                st.markdown("### Energy Landscape")
                                
                                # Create three subplots
                                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                                
                                # Common styling
                                colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
                                
                                # Plot 1: Energy vs S
                                axes[0].plot(df["S"], df["Energy (eV)"], 'o-', linewidth=2.5, 
                                            markersize=10, color='#2E86AB', markeredgecolor='white', 
                                            markeredgewidth=2)
                                axes[0].scatter([optimal_S], [min_energy], s=300, c='#FF6B6B', 
                                            edgecolors='white', linewidths=2, zorder=5, marker='*')
                                axes[0].set_xlabel('Total Spin (S)', fontsize=13, fontweight='bold')
                                axes[0].set_ylabel('Energy (eV)', fontsize=13, fontweight='bold')
                                axes[0].set_title('Energy vs Total Spin', fontsize=14, fontweight='bold', pad=15)
                                axes[0].grid(True, alpha=0.3, linestyle='--')
                                axes[0].spines['top'].set_visible(False)
                                axes[0].spines['right'].set_visible(False)
                                
                                # Plot 2: Energy vs 2S+1
                                axes[1].plot(df["2S+1"], df["Energy (eV)"], 'o-', linewidth=2.5, 
                                            markersize=10, color='#A23B72', markeredgecolor='white', 
                                            markeredgewidth=2)
                                axes[1].scatter([optimal_mult], [min_energy], s=300, c='#FF6B6B', 
                                            edgecolors='white', linewidths=2, zorder=5, marker='*')
                                axes[1].set_xlabel('Spin Multiplicity (2S+1)', fontsize=13, fontweight='bold')
                                axes[1].set_ylabel('Energy (eV)', fontsize=13, fontweight='bold')
                                axes[1].set_title('Energy vs Spin Multiplicity', fontsize=14, fontweight='bold', pad=15)
                                axes[1].grid(True, alpha=0.3, linestyle='--')
                                axes[1].spines['top'].set_visible(False)
                                axes[1].spines['right'].set_visible(False)
                                
                                # Plot 3: Energy vs Unpaired Electrons
                                axes[2].plot(df["Unpaired Electrons"], df["Energy (eV)"], 'o-', linewidth=2.5, 
                                            markersize=10, color='#F18F01', markeredgecolor='white', 
                                            markeredgewidth=2)
                                axes[2].scatter([optimal_unpaired], [min_energy], s=300, c='#FF6B6B', 
                                            edgecolors='white', linewidths=2, zorder=5, marker='*')
                                axes[2].set_xlabel('Unpaired Electrons', fontsize=13, fontweight='bold')
                                axes[2].set_ylabel('Energy (eV)', fontsize=13, fontweight='bold')
                                axes[2].set_title('Energy vs Unpaired Electrons', fontsize=14, fontweight='bold', pad=15)
                                axes[2].grid(True, alpha=0.3, linestyle='--')
                                axes[2].spines['top'].set_visible(False)
                                axes[2].spines['right'].set_visible(False)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Summary statistics
                                st.markdown("### Summary Statistics")
                                energy_range = df["Energy (eV)"].max() - df["Energy (eV)"].min()
                                st.write(f"**Energy range:** {energy_range:.6f} eV")
                                st.write(f"**Total calculations:** {len(df)}")
                                st.write(f"**Total time:** {df['Time (s)'].sum():.4f} seconds")
                                
                            else:
                                st.error("No successful calculations completed. Please check your system setup.")
                        else:
                            st.error("Spin optimization can only be done using an OMOL model. Please select a model compatible with Spin.")
                    elif task == "Batch Energy Calculation":
                        t0 = time.perf_counter()
                        st.write(f"Processing {len(atoms_list)} structures...")
                        
                        # Prepare results list
                        batch_results = []
                        batch_xyz_list = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, atoms_obj in enumerate(atoms_list):
                            status_text.text(f"Calculating structure {idx+1}/{len(atoms_list)}...")
                            
                            try:
                                # Create a copy and attach calculator
                                calc_atoms = atoms_obj.copy()
                                calc_atoms.calc = calc
                                
                                # Calculate energy
                                energy = calc_atoms.get_potential_energy()

                                batch_xyz_list.append(write_single_frame_extxyz(calc_atoms))
                                
                                # Get metadata
                                filename = atoms_obj.info.get("source_name", f"structure_{idx+1}")
                                formula = calc_atoms.get_chemical_formula()
                                natoms = len(calc_atoms)
                                pbc = str(calc_atoms.pbc.tolist())
                                filetype = os.path.splitext(filename)[1].lstrip('.')
                                
                                batch_results.append({
                                    "Filename": filename,
                                    "Formula": formula,
                                    "N_atoms": natoms,
                                    "PBC": pbc,
                                    "Filetype": filetype,
                                    "Energy (eV)": f"{energy:.6f}"
                                })
                                
                            except Exception as e:
                                batch_results.append({
                                    "Filename": atoms_obj.info.get("source_name", f"structure_{idx+1}"),
                                    "Formula": "Error",
                                    "N_atoms": "-",
                                    "PBC": "-",
                                    "Filetype": "-",
                                    "Energy (eV)": f"Failed: {str(e)}"
                                })
                            
                            progress_bar.progress((idx + 1) / len(atoms_list))
                        t1 = time.perf_counter()
                        status_text.text("Calculation complete!")
                        st.success("Calculation completed successfully!")
                        st.markdown("### Results")
                        # for key, value in results.items():
                        #     st.write(f"**{key}:** {value}")
                        # Display results table
                        df_results = pd.DataFrame(batch_results)
                        st.dataframe(df_results, use_container_width=True)

                        
                        all_frames_text = "".join(batch_xyz_list)
                        

                        # Download button without reloading the app
                        

                        def make_download_link(content, filename, mimetype="chemical/x-extxyz"):
                            if isinstance(content, str):
                                b = content.encode("utf-8")
                            else:
                                b = content
                            b64 = base64.b64encode(b).decode()
                            return f'<a href="data:{mimetype};base64,{b64}" download="{filename}">📥 Download {filename}</a>'

                        st.markdown(
                                    make_download_link(all_frames_text, "batch_structures.extxyz"),
                                    unsafe_allow_html=True
                                )
                        
                        
                    
                    elif task == "Batch Energy + Forces + Stress Calculation":
                        t0 = time.perf_counter()
                        if len(atoms_list) == 0:
                            st.warning("Please upload multiple structures using 'Batch Upload' mode.")
                        else:
                            st.subheader("Batch Energy + Forces + Stress Calculation")
                            st.write(f"Processing {len(atoms_list)} structures...")
                            
                            # Prepare results lists
                            batch_results = []
                            batch_xyz_list = []
                            
                            # Parity plot data collectors
                            ref_energies, calc_energies = [], []
                            ref_forces_all, calc_forces_all = [], []
                            ref_forces_by_element, calc_forces_by_element = {}, {}
                            ref_stresses, calc_stresses = [], []
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Collect per-atom counts for structures that succeeded with ref energy
                            energy_natoms = []
                            for idx, atoms_obj in enumerate(atoms_list):
                                status_text.text(f"Calculating structure {idx+1}/{len(atoms_list)}...")
                                
                                try:
                                    # Get reference values from the original calculator if available
                                    ref_energy = None
                                    ref_forces = None
                                    ref_stress = None
                                    # ----------------------------
                                    # 1. Calculator results
                                    # ----------------------------
                                    if atoms_obj.calc is not None and hasattr(atoms_obj.calc, "results"):
                                        calc_results = atoms_obj.calc.results
                                        
                                        # Prefer free energy if present
                                        ref_energy = _find_value(
                                            calc_results,
                                            # keywords=["free_energy", "energy"]
                                            keywords=["energy", "free_energy", "ref_energy", "REF_energy", "ENERGY", "REF_ENERGY"] # prioritize energy over free energy
                                        )

                                        ref_forces = _find_value(
                                            calc_results,
                                            keywords=["forces", "ref_forces", "REF_forces", "REF_FORCES"]
                                        )

                                        ref_stress = _find_value(
                                            calc_results,
                                            keywords=["stress", "ref_stress", "REF_stress", "STRESS", "REF_STRESS"]
                                        )
                                    if atoms_obj is not None:
                                        energy_natoms.append(len(atoms_obj))

                                    # ----------------------------
                                    # 2. atoms.info fallback
                                    # ----------------------------
                                    if ref_energy is None:
                                        ref_energy = _find_value(
                                            atoms_obj.info,
                                            # keywords=["free_energy", "energy"]
                                            keywords=["energy", "free_energy", "ref_energy", "REF_energy", "ENERGY", "REF_ENERGY"] # prioritize energy over free energy
                                        )
                                    if ref_stress is None:
                                        ref_stress = _find_value(
                                            atoms_obj.info,
                                            keywords=["stress", "ref_stress", "REF_stress", "STRESS", "REF_STRESS"]
                                        )

                                    # ----------------------------
                                    # 3. atoms.arrays fallback
                                    # ----------------------------
                                    if ref_forces is None:
                                        ref_forces = _find_value(
                                            atoms_obj.arrays,
                                            keywords=["forces", "ref_forces", "REF_forces", "REF_FORCES"]
                                        )
                                    
                                    has_ref_energy = ref_energy is not None
                                    has_ref_forces = ref_forces is not None
                                    has_ref_stress = ref_stress is not None
                                    
                                    # Create a copy and attach NEW calculator
                                    calc_atoms = atoms_obj.copy()
                                    calc_atoms.calc = calc
                                    
                                    # Calculate properties
                                    energy = calc_atoms.get_potential_energy()
                                    forces = calc_atoms.get_forces()
                                    
                                    # Try to get stress if system has PBC
                                    stress = None
                                    if np.any(calc_atoms.pbc):
                                        try:
                                            stress = calc_atoms.get_stress()
                                        except:
                                            pass
                                    
                                    # Collect parity data for energy
                                    if has_ref_energy:
                                        ref_energies.append(ref_energy)
                                        calc_energies.append(energy)
                                    
                                    # Collect parity data for forces
                                    if has_ref_forces:
                                        # Ensure shapes match before collecting
                                        if ref_forces.shape == forces.shape:
                                            ref_forces_all.extend(ref_forces.flatten())
                                            calc_forces_all.extend(forces.flatten())
                                            
                                            # Collect forces by element type
                                            symbols = calc_atoms.get_chemical_symbols()
                                            for atom_idx, symbol in enumerate(symbols):
                                                if symbol not in ref_forces_by_element:
                                                    ref_forces_by_element[symbol] = []
                                                    calc_forces_by_element[symbol] = []
                                                ref_forces_by_element[symbol].extend(ref_forces[atom_idx])
                                                calc_forces_by_element[symbol].extend(forces[atom_idx])
                                    
                                    # Collect parity data for stress
                                    if has_ref_stress and stress is not None:
                                        # Ensure shapes match before collecting
                                        if len(ref_stress) == len(stress):
                                            ref_stresses.extend(ref_stress)
                                            calc_stresses.extend(stress)
                                    
                                    # Calculate force statistics
                                    max_force = np.max(np.sqrt(np.sum(forces**2, axis=1))) if forces.shape[0] > 0 else 0.0
                                    mean_force = np.mean(np.sqrt(np.sum(forces**2, axis=1))) if forces.shape[0] > 0 else 0.0
                                    batch_xyz_list.append(write_single_frame_extxyz(calc_atoms))
                                    
                                    # Get metadata
                                    filename = atoms_obj.info.get("source_name", f"structure_{idx+1}")
                                    formula = calc_atoms.get_chemical_formula()
                                    natoms = len(calc_atoms)
                                    pbc = str(calc_atoms.pbc.tolist())
                                    filetype = os.path.splitext(filename)[1].lstrip('.')
                                    
                                    result_dict = {
                                        "Filename": filename,
                                        "Formula": formula,
                                        "N_atoms": natoms,
                                        "PBC": pbc,
                                        "Filetype": filetype,
                                        "Energy (eV)": f"{energy:.6f}",
                                    }
                                    
                                    # Add ref energy & error if available
                                    if has_ref_energy:
                                        result_dict["Ref Energy (eV)"] = f"{ref_energy:.6f}"
                                        result_dict["Energy Error (eV)"] = f"{energy - ref_energy:.6f}"
                                        result_dict["Energy Error/atom (eV)"] = f"{(energy - ref_energy) / natoms:.6f}"
                                    
                                    result_dict["Max Force (eV/Å)"] = f"{max_force:.6f}"
                                    result_dict["Mean Force (eV/Å)"] = f"{mean_force:.6f}"
                                    
                                    # Add ref forces & error if available
                                    if has_ref_forces and ref_forces.shape == forces.shape:
                                        ref_max_force = np.max(np.sqrt(np.sum(ref_forces**2, axis=1))) if ref_forces.shape[0] > 0 else 0.0
                                        ref_mean_force = np.mean(np.sqrt(np.sum(ref_forces**2, axis=1))) if ref_forces.shape[0] > 0 else 0.0
                                        force_component_mae = np.mean(np.abs(forces - ref_forces))
                                        force_component_rmse = np.sqrt(np.mean((forces - ref_forces)**2))
                                        result_dict["Ref Max Force (eV/Å)"] = f"{ref_max_force:.6f}"
                                        result_dict["Ref Mean Force (eV/Å)"] = f"{ref_mean_force:.6f}"
                                        result_dict["Force MAE (eV/Å)"] = f"{force_component_mae:.6f}"
                                        result_dict["Force RMSE (eV/Å)"] = f"{force_component_rmse:.6f}"
                                    
                                    if stress is not None:
                                        result_dict["Max Stress (eV/Å³)"] = f"{np.max(np.abs(stress)):.6f}"
                                    
                                    # Add ref stress & error if available
                                    if has_ref_stress and stress is not None and len(ref_stress) == len(stress):
                                        ref_stress_arr = np.array(ref_stress)
                                        stress_arr = np.array(stress)
                                        result_dict["Ref Max Stress (eV/Å³)"] = f"{np.max(np.abs(ref_stress_arr)):.6f}"
                                        result_dict["Stress MAE (eV/Å³)"] = f"{np.mean(np.abs(stress_arr - ref_stress_arr)):.6f}"
                                        result_dict["Stress RMSE (eV/Å³)"] = f"{np.sqrt(np.mean((stress_arr - ref_stress_arr)**2)):.6f}"
                                    
                                    batch_results.append(result_dict)
                                
                                    
                                except Exception as e:
                                    batch_results.append({
                                        "Filename": atoms_obj.info.get("source_name", f"structure_{idx+1}"),
                                        "Formula": "Error",
                                        "N_atoms": "-",
                                        "PBC": "-",
                                        "Filetype": "-",
                                        "Energy (eV)": f"Failed",
                                        "Max Force (eV/Å)": "-",
                                        "Mean Force (eV/Å)": f"{str(e)}"
                                    })
                                
                                progress_bar.progress((idx + 1) / len(atoms_list))
                            
                            t1 = time.perf_counter()
                            status_text.text("Calculation complete!")
                            st.write(f"Time Taken = {t1 - t0:.4f} seconds")
                            st.success("Calculation completed successfully!")
                            
                            # ===============================================
                            # PARITY PLOTS SECTION
                            # ===============================================
                            st.markdown("## 📊 Parity Plots (Reference vs Calculated)")
                            st.markdown("*Parity plots show how well the calculator reproduces reference values. Points closer to the diagonal line indicate better agreement.*")
                            
                            def calculate_metrics(ref, calc):
                                """Calculate MAE, RMSE, and R²"""
                                ref_arr = np.array(ref)
                                calc_arr = np.array(calc)
                                mae = np.mean(np.abs(ref_arr - calc_arr))
                                rmse = np.sqrt(np.mean((ref_arr - calc_arr)**2))
                                
                                # Calculate R² (coefficient of determination)
                                ss_res = np.sum((ref_arr - calc_arr)**2)
                                ss_tot = np.sum((ref_arr - np.mean(ref_arr))**2)
                                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                                
                                return mae, rmse, r2
                            
                            def plot_parity(ref, calc, xlabel, ylabel, title, ax=None):
                                """Create a beautiful parity plot"""
                                if ax is None:
                                    fig, ax = plt.subplots(figsize=(7, 6))
                                
                                # Convert to arrays and filter out None values
                                ref_arr = np.array(ref, dtype=float)
                                calc_arr = np.array(calc, dtype=float)
                                
                                # Ensure arrays have same length
                                if len(ref_arr) != len(calc_arr):
                                    min_len = min(len(ref_arr), len(calc_arr))
                                    ref_arr = ref_arr[:min_len]
                                    calc_arr = calc_arr[:min_len]
                                
                                # Remove any NaN or None values
                                valid_mask = ~(np.isnan(ref_arr) | np.isnan(calc_arr))
                                ref_arr = ref_arr[valid_mask]
                                calc_arr = calc_arr[valid_mask]
                                
                                if len(ref_arr) == 0:
                                    return None
                                
                                # Plot data points
                                ax.scatter(ref_arr, calc_arr, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                                
                                # Plot diagonal line
                                min_val = min(ref_arr.min(), calc_arr.min())
                                max_val = max(ref_arr.max(), calc_arr.max())
                                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect agreement')
                                
                                # Calculate and display metrics
                                mae, rmse, r2 = calculate_metrics(ref_arr, calc_arr)
                                
                                # Add metrics text box
                                textstr = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}\nN = {len(ref_arr)}'
                                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                                    verticalalignment='top', bbox=props)
                                
                                ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
                                ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
                                ax.set_title(title, fontsize=13, fontweight='bold')
                                ax.legend(loc='lower right')
                                ax.grid(True, alpha=0.3)
                                ax.set_aspect('equal', adjustable='box')
                                
                                return ax.figure if ax.figure else fig
                            
                            # Energy Parity Plot
                            if len(ref_energies) > 0:
                                st.markdown("### 🔋 Energy Parity Plot")

                                
                                use_mev  = False 

                                # Fallback if lengths don't align
                                if len(energy_natoms) != len(ref_energies):
                                    energy_natoms = [1] * len(ref_energies)

                                ref_e_arr   = np.array(ref_energies,  dtype=np.float64)
                                calc_e_arr  = np.array(calc_energies, dtype=np.float64)
                                natoms_arr  = np.array(energy_natoms, dtype=int)

                                
                                ref_vals  = ref_e_arr
                                calc_vals = calc_e_arr
                                unit_label = "eV"
                                title_suffix = "Total"

                                # --- Convert to meV if needed ---
                                if use_mev:
                                    ref_vals  *= 1000.0
                                    calc_vals *= 1000.0
                                    unit_label = unit_label.replace("eV", "meV")

                                # --- Relative energies ---
                                ref_zero = 0
                                calc_zero = 0
                                ref_rel  = ref_vals  - ref_zero
                                calc_rel = calc_vals - calc_zero

                                valid_mask = ~(np.isnan(ref_rel) | np.isnan(calc_rel))
                                ref_plot   = ref_rel[valid_mask]
                                calc_plot  = calc_rel[valid_mask]

                                # Parity plot with absolute energies
                                with st.expander('Total Energy Parity Plot', False):
                                    if len(ref_plot) > 0:
                                        fig_energy, ax_energy = plt.subplots(figsize=(7, 6))

                                        ax_energy.scatter(ref_plot, calc_plot, alpha=0.6, s=50,
                                                        edgecolors='black', linewidth=0.5)

                                        min_val = min(ref_plot.min(), calc_plot.min())
                                        max_val = max(ref_plot.max(), calc_plot.max())
                                        ax_energy.plot([min_val, max_val], [min_val, max_val],
                                                    'r--', lw=2, label='Perfect agreement')

                                        mae, rmse, r2 = calculate_metrics(ref_plot, calc_plot)

                                        textstr = (f'MAE  = {mae:.4f} {unit_label}\n'
                                                f'RMSE = {rmse:.4f} {unit_label}\n'
                                                f'R²   = {r2:.4f}\n'
                                                f'N    = {len(ref_plot)}')

                                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                                        ax_energy.text(0.05, 0.95, textstr, transform=ax_energy.transAxes,
                                                    fontsize=10, verticalalignment='top', bbox=props)

                                        ax_energy.set_xlabel(f"Reference Energy ({unit_label})",
                                                            fontsize=12, fontweight='bold')
                                        ax_energy.set_ylabel(f"Calculated Energy ({unit_label})",
                                                            fontsize=12, fontweight='bold')
                                        ax_energy.set_title(f"Energy Parity Plot ({title_suffix})",
                                                            fontsize=13, fontweight='bold')

                                        ax_energy.legend(loc='lower right')
                                        ax_energy.grid(True, alpha=0.3)
                                        ax_energy.set_aspect('equal', adjustable='box')

                                        st.pyplot(fig_energy)
                                        plt.close(fig_energy)
                            
                                # Parity plot with per atom absolute energies
                                ref_vals  = ref_e_arr  / natoms_arr
                                calc_vals = calc_e_arr / natoms_arr
                                unit_label = "eV/atom"
                                title_suffix = "Per Atom"
                                # --- Relative energies ---
                                ref_zero = 0
                                calc_zero = 0
                                ref_rel  = ref_vals  - ref_zero
                                calc_rel = calc_vals - calc_zero
                                valid_mask = ~(np.isnan(ref_rel) | np.isnan(calc_rel))
                                ref_plot   = ref_rel[valid_mask]
                                calc_plot  = calc_rel[valid_mask]
                                with st.expander('Per Atom Parity Plot', False):
                                    if len(ref_plot) > 0:
                                        fig_energy, ax_energy = plt.subplots(figsize=(7, 6))

                                        ax_energy.scatter(ref_plot, calc_plot, alpha=0.6, s=50,
                                                        edgecolors='black', linewidth=0.5)

                                        min_val = min(ref_plot.min(), calc_plot.min())
                                        max_val = max(ref_plot.max(), calc_plot.max())
                                        ax_energy.plot([min_val, max_val], [min_val, max_val],
                                                    'r--', lw=2, label='Perfect agreement')

                                        mae, rmse, r2 = calculate_metrics(ref_plot, calc_plot)

                                        textstr = (f'MAE  = {mae:.4f} {unit_label}\n'
                                                f'RMSE = {rmse:.4f} {unit_label}\n'
                                                f'R²   = {r2:.4f}\n'
                                                f'N    = {len(ref_plot)}')

                                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                                        ax_energy.text(0.05, 0.95, textstr, transform=ax_energy.transAxes,
                                                    fontsize=10, verticalalignment='top', bbox=props)

                                        ax_energy.set_xlabel(f"Reference Energy ({unit_label})",
                                                            fontsize=12, fontweight='bold')
                                        ax_energy.set_ylabel(f"Calculated Energy ({unit_label})",
                                                            fontsize=12, fontweight='bold')
                                        ax_energy.set_title(f"Energy Parity Plot ({title_suffix})",
                                                            fontsize=13, fontweight='bold')

                                        ax_energy.legend(loc='lower right')
                                        ax_energy.grid(True, alpha=0.3)
                                        ax_energy.set_aspect('equal', adjustable='box')
                                        st.pyplot(fig_energy)
                                        plt.close(fig_energy)    

                                # Parity plot with per atom relative energies
                                ref_vals  = ref_e_arr  / natoms_arr
                                calc_vals = calc_e_arr / natoms_arr
                                unit_label = "eV/atom"
                                title_suffix = "Per Atom, wrt first frame"
                                # --- Relative energies ---
                                ref_zero = ref_vals[0] 
                                calc_zero = calc_vals[0] 
                                ref_rel  = ref_vals  - ref_zero
                                calc_rel = calc_vals - calc_zero
                                valid_mask = ~(np.isnan(ref_rel) | np.isnan(calc_rel))
                                ref_plot   = ref_rel[valid_mask]
                                calc_plot  = calc_rel[valid_mask]
                                if len(ref_plot) > 0:
                                    fig_energy, ax_energy = plt.subplots(figsize=(7, 6))

                                    ax_energy.scatter(ref_plot, calc_plot, alpha=0.6, s=50,
                                                    edgecolors='black', linewidth=0.5)

                                    min_val = min(ref_plot.min(), calc_plot.min())
                                    max_val = max(ref_plot.max(), calc_plot.max())
                                    ax_energy.plot([min_val, max_val], [min_val, max_val],
                                                'r--', lw=2, label='Perfect agreement')

                                    mae, rmse, r2 = calculate_metrics(ref_plot, calc_plot)

                                    textstr = (f'MAE  = {mae:.4f} {unit_label}\n'
                                            f'RMSE = {rmse:.4f} {unit_label}\n'
                                            f'R²   = {r2:.4f}\n'
                                            f'N    = {len(ref_plot)}')

                                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                                    ax_energy.text(0.05, 0.95, textstr, transform=ax_energy.transAxes,
                                                fontsize=10, verticalalignment='top', bbox=props)

                                    ax_energy.set_xlabel(f"Reference Energy ({unit_label})",
                                                        fontsize=12, fontweight='bold')
                                    ax_energy.set_ylabel(f"Calculated Energy ({unit_label})",
                                                        fontsize=12, fontweight='bold')
                                    ax_energy.set_title(f"Relative Energy Parity Plot ({title_suffix})",
                                                        fontsize=13, fontweight='bold')

                                    ax_energy.legend(loc='lower right')
                                    ax_energy.grid(True, alpha=0.3)
                                    ax_energy.set_aspect('equal', adjustable='box')

                                    st.pyplot(fig_energy)
                                    plt.close(fig_energy) 
                            # Forces Parity Plot (All forces combined)
                            if len(ref_forces_all) > 0:
                                st.markdown("### ⚡ Force Parity Plot (All Components)")
                                fig_forces = plot_parity(ref_forces_all, calc_forces_all,
                                                        "Reference Force (eV/Å)",
                                                        "Calculated Force (eV/Å)",
                                                        "Force Component Parity Plot")
                                if fig_forces is not None:
                                    st.pyplot(fig_forces)
                                    plt.close(fig_forces)
                                
                                # Forces Parity Plot by Element
                                if len(ref_forces_by_element) > 1:
                                    st.markdown("### ⚡ Force Parity Plot (By Element)")
                                    
                                    fig_forces_elem, ax_forces_elem = plt.subplots(figsize=(8, 7))
                                    
                                    # Color palette
                                    colors = plt.cm.tab10(np.linspace(0, 1, len(ref_forces_by_element)))
                                    
                                    for idx, (element, ref_f) in enumerate(ref_forces_by_element.items()):
                                        calc_f = calc_forces_by_element[element]
                                        # Convert to arrays and filter
                                        ref_arr = np.array(ref_f, dtype=float)
                                        calc_arr = np.array(calc_f, dtype=float)
                                        valid_mask = ~(np.isnan(ref_arr) | np.isnan(calc_arr))
                                        ref_arr = ref_arr[valid_mask]
                                        calc_arr = calc_arr[valid_mask]
                                        
                                        if len(ref_arr) > 0:
                                            ax_forces_elem.scatter(ref_arr, calc_arr, alpha=0.6, s=50, 
                                                                label=element, color=colors[idx],
                                                                edgecolors='black', linewidth=0.5)
                                    
                                    # Plot diagonal - collect all valid data
                                    all_ref_list = []
                                    all_calc_list = []
                                    for element in ref_forces_by_element:
                                        ref_arr = np.array(ref_forces_by_element[element], dtype=float)
                                        calc_arr = np.array(calc_forces_by_element[element], dtype=float)
                                        valid_mask = ~(np.isnan(ref_arr) | np.isnan(calc_arr))
                                        all_ref_list.append(ref_arr[valid_mask])
                                        all_calc_list.append(calc_arr[valid_mask])
                                    
                                    all_ref = np.concatenate(all_ref_list) if all_ref_list else np.array([])
                                    all_calc = np.concatenate(all_calc_list) if all_calc_list else np.array([])
                                    
                                    if len(all_ref) > 0:
                                        min_val = min(all_ref.min(), all_calc.min())
                                        max_val = max(all_ref.max(), all_calc.max())
                                    if len(all_ref) > 0:
                                        min_val = min(all_ref.min(), all_calc.min())
                                        max_val = max(all_ref.max(), all_calc.max())
                                        ax_forces_elem.plot([min_val, max_val], [min_val, max_val], 
                                                        'r--', lw=2, label='Perfect agreement')
                                        
                                        # Calculate overall metrics
                                        mae, rmse, r2 = calculate_metrics(all_ref, all_calc)
                                        textstr = f'Overall MAE = {mae:.4f}\nOverall RMSE = {rmse:.4f}\nR² = {r2:.4f}\nN = {len(all_ref)}'
                                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                                        ax_forces_elem.text(0.05, 0.95, textstr, transform=ax_forces_elem.transAxes, 
                                                        fontsize=10, verticalalignment='top', bbox=props)
                                        
                                        ax_forces_elem.set_xlabel("Reference Force (eV/Å)", fontsize=12, fontweight='bold')
                                        ax_forces_elem.set_ylabel("Calculated Force (eV/Å)", fontsize=12, fontweight='bold')
                                        ax_forces_elem.set_title("Force Component Parity Plot (Colored by Element)", 
                                                                fontsize=13, fontweight='bold')
                                        ax_forces_elem.legend(loc='lower right', framealpha=0.9)
                                        ax_forces_elem.grid(True, alpha=0.3)
                                        ax_forces_elem.set_aspect('equal', adjustable='box')
                                        
                                        st.pyplot(fig_forces_elem)
                                    plt.close(fig_forces_elem)
                            
                            # Stress Parity Plot
                            if len(ref_stresses) > 0:
                                st.markdown("### 💎 Stress Parity Plot")
                                fig_stress = plot_parity(ref_stresses, calc_stresses,
                                                        "Reference Stress (eV/Å³)",
                                                        "Calculated Stress (eV/Å³)",
                                                        "Stress Component Parity Plot")
                                if fig_stress is not None:
                                    st.pyplot(fig_stress)
                                    plt.close(fig_stress)
                            
                            if len(ref_energies) == 0 and len(ref_forces_all) == 0 and len(ref_stresses) == 0:
                                st.info("ℹ️ No reference data found in uploaded structures. Parity plots require structures with reference energies, forces, or stresses.")
                            
                            # ===============================================
                            # RESULTS TABLE AND DOWNLOAD
                            # ===============================================
                            st.markdown("---")
                            st.markdown("## 📋 Calculation Results")
                            
                            df_results = pd.DataFrame(batch_results)
                            st.dataframe(df_results, use_container_width=True)
                            
                            all_frames_text = "".join(batch_xyz_list)
                            
                            def make_download_link(content, filename, mimetype="chemical/x-extxyz"):
                                if isinstance(content, str):
                                    b = content.encode("utf-8")
                                else:
                                    b = content
                                b64 = base64.b64encode(b).decode()
                                return f'<a href="data:{mimetype};base64,{b64}" download="{filename}">📥 Download {filename}</a>'
                            
                            st.markdown(
                                make_download_link(all_frames_text, "batch_structures.extxyz"),
                                unsafe_allow_html=True
                            )
                            
                            # ===============================================
                            # STATISTICAL ANALYSIS (Original Code)
                            # ===============================================
                            st.markdown("---")
                            st.markdown("## 📈 Statistical Analysis")
                            
                            # Convert values to float for plotting
                            df_results["Energy_float"] = pd.to_numeric(df_results["Energy (eV)"], errors="coerce")
                            df_results["MaxForce_float"] = pd.to_numeric(df_results["Max Force (eV/Å)"], errors="coerce")
                            df_results["MeanForce_float"] = pd.to_numeric(df_results["Mean Force (eV/Å)"], errors="coerce")
                            
                            energies = df_results["Energy_float"].dropna()
                            max_forces = df_results["MaxForce_float"].dropna()
                            mean_forces = df_results["MeanForce_float"].dropna()
                            
                            # Energy Histogram
                            st.markdown("### Energy Distribution (Histogram)")
                            fig1, ax1 = plt.subplots()
                            ax1.hist(energies, bins=20, edgecolor='black', alpha=0.7)
                            ax1.set_xlabel("Energy (eV)")
                            ax1.set_ylabel("Count")
                            ax1.set_title("Energy Distribution Across Structures")
                            ax1.grid(True, alpha=0.3)
                            st.pyplot(fig1)
                            plt.close(fig1)
                            
                            # Energy vs Structure Index
                            st.markdown("### Energy vs Structure Index")
                            fig2, ax2 = plt.subplots()
                            ax2.plot(range(len(energies)), energies, marker="o", linestyle='-', linewidth=1.5)
                            ax2.set_xlabel("Structure Index")
                            ax2.set_ylabel("Energy (eV)")
                            ax2.set_title("Energy Trend Across Batch")
                            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
                            ax2.grid(True, alpha=0.3)
                            st.pyplot(fig2)
                            plt.close(fig2)
                            
                            # Max Force Histogram
                            st.markdown("### Max Force Distribution (Histogram)")
                            fig3, ax3 = plt.subplots()
                            ax3.hist(max_forces, bins=20, edgecolor='black', alpha=0.7)
                            ax3.set_xlabel("Max Force (eV/Å)")
                            ax3.set_ylabel("Count")
                            ax3.set_title("Max Force Distribution Across Structures")
                            ax3.grid(True, alpha=0.3)
                            st.pyplot(fig3)
                            plt.close(fig3)
                            
                            # Mean Force Histogram
                            st.markdown("### Mean Force Distribution (Histogram)")
                            fig4, ax4 = plt.subplots()
                            ax4.hist(mean_forces, bins=20, edgecolor='black', alpha=0.7)
                            ax4.set_xlabel("Mean Force (eV/Å)")
                            ax4.set_ylabel("Count")
                            ax4.set_title("Mean Force Distribution Across Structures")
                            ax4.grid(True, alpha=0.3)
                            st.pyplot(fig4)
                            plt.close(fig4)
                    elif task == "Equation of State":
                        t0 = time.perf_counter()
                        calculate_bulk_modulus(calc_atoms, calc, num_points, volume_range, eos_type, results)
                        t1 = time.perf_counter()
                        results["Time Taken"] = f"{t1 - t0:.4f} seconds"
                        st.success("Calculation completed successfully!")
                        st.markdown("### Results")
                        for key, value in results.items():
                            st.write(f"**{key}:** {value}")

                    elif task == "Atomization/Cohesive Energy":
                        st.write("Calculating system energy...")
                        t0 = time.perf_counter()
                        E_system = calc_atoms.get_potential_energy()
                        num_atoms = len(calc_atoms)

                        if num_atoms == 0:
                            st.error("Cannot calculate atomization/cohesive energy for a system with zero atoms.")
                            results["Error"] = "System has no atoms."
                        else:
                            atomic_numbers = calc_atoms.get_atomic_numbers()
                            E_isolated_atoms_total = 0.0
                            calculation_possible = True

                            if model_type == "FairChem":
                                st.write("Fetching FairChem reference energies for isolated atoms...")
                                ref_key_suffix = "_elem_refs"
                                chosen_ref_list_name = None
                                if "UMA Small" in selected_model:
                                    if selected_task_type:
                                        chosen_ref_list_name = selected_task_type + ref_key_suffix
                                elif "ESEN" in selected_model:
                                    chosen_ref_list_name = "omol" + ref_key_suffix
                                
                                if chosen_ref_list_name and chosen_ref_list_name in ELEMENT_REF_ENERGIES:
                                    ref_energies = ELEMENT_REF_ENERGIES[chosen_ref_list_name]
                                    missing_Z_refs = []
                                    for Z_val in atomic_numbers:
                                        if Z_val > 0 and Z_val < len(ref_energies):
                                            E_isolated_atoms_total += ref_energies[Z_val]
                                        else:
                                            if Z_val not in missing_Z_refs: missing_Z_refs.append(Z_val)
                                    if missing_Z_refs:
                                        st.warning(f"Reference energy for atomic number(s) {sorted(list(set(missing_Z_refs)))} "
                                                   f"not found in '{chosen_ref_list_name}' list (max Z defined: {len(ref_energies)-1}). "
                                                   "These atoms are treated as having 0 reference energy.")
                                else:
                                    st.error(f"Could not find or determine reference energy list for FairChem model: '{selected_model}' "
                                             f"and UMA task type: '{selected_task_type}'. Cannot calculate atomization/cohesive energy.")
                                    results["Error"] = "Missing FairChem reference energies."
                                    calculation_possible = False
                            
                            else:
                                st.write("Calculating isolated atom energies with MACE...")
                                unique_atomic_numbers = sorted(list(set(atomic_numbers)))
                                atom_counts = {Z_unique: np.count_nonzero(atomic_numbers == Z_unique) for Z_unique in unique_atomic_numbers}
                                
                                progress_text = "Calculating isolated atom energies: 0% complete"
                                mace_progress_bar = st.progress(0, text=progress_text)
                                
                                for i, Z_unique in enumerate(unique_atomic_numbers):
                                    isolated_atom = Atoms(numbers=[Z_unique], cell=[20, 20, 20], pbc=False)
                                    if not hasattr(isolated_atom, 'info'): isolated_atom.info = {}
                                    isolated_atom.info["charge"] = 0 
                                    isolated_atom.info["spin"] = 0 
                                    isolated_atom.calc = calc # Use the same MACE calculator
                                    
                                    E_isolated_atom_type = isolated_atom.get_potential_energy()
                                    E_isolated_atoms_total += E_isolated_atom_type * atom_counts[Z_unique]
                                    
                                    progress_val = (i + 1) / len(unique_atomic_numbers)
                                    mace_progress_bar.progress(progress_val, text=f"Calculating isolated atom energies for Z={Z_unique}: {int(progress_val*100)}% complete")
                                mace_progress_bar.empty()

                            if calculation_possible:
                                is_periodic = any(calc_atoms.pbc)
                                if is_periodic:
                                    cohesive_E = (E_isolated_atoms_total - E_system) / num_atoms
                                    results["Cohesive Energy"] = f"{cohesive_E:.6f} eV/atom"
                                else: 
                                    atomization_E = E_isolated_atoms_total - E_system
                                    results["Atomization Energy"] = f"{atomization_E:.6f} eV"
                                
                                results["System Energy ($E_{system}$)"] = f"{E_system:.6f} eV"
                                results["Total Isolated Atom Energy ($\sum E_{atoms}$)"] = f"{E_isolated_atoms_total:.6f} eV"
                        st.success("Calculation completed successfully!")
                        t1 = time.perf_counter()
                        results["Time Taken"] = f"{t1 - t0:.4f} seconds"
                        st.markdown("### Results")
                        for key, value in results.items():
                            st.write(f"**{key}:** {value}")

                    elif task == "Batch Atomization/Cohesive Energy":
                        if len(atoms_list) == 0:
                            st.warning("Please upload multiple structures using 'Batch Upload' mode.")
                        else:
                            st.subheader("Batch Atomization/Cohesive Energy Calculation")
                            st.write(f"Processing {len(atoms_list)} structures...")
                            
                            # Prepare results list
                            batch_results = []
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Pre-calculate MACE isolated atom energies if needed (to avoid redundant calculations)
                            mace_isolated_energies = {}
                            if model_type == "MACE":
                                st.write("Pre-calculating MACE isolated atom reference energies...")
                                all_atomic_numbers = set()
                                for atoms_obj in atoms_list:
                                    all_atomic_numbers.update(atoms_obj.get_atomic_numbers())
                                
                                unique_Z_all = sorted(list(all_atomic_numbers))
                                mace_ref_progress = st.progress(0)
                                
                                for i, Z_unique in enumerate(unique_Z_all):
                                    isolated_atom = Atoms(numbers=[Z_unique], cell=[20, 20, 20], pbc=False)
                                    if not hasattr(isolated_atom, 'info'): 
                                        isolated_atom.info = {}
                                    isolated_atom.info["charge"] = 0 
                                    isolated_atom.info["spin"] = 0 
                                    isolated_atom.calc = calc
                                    
                                    mace_isolated_energies[Z_unique] = isolated_atom.get_potential_energy()
                                    mace_ref_progress.progress((i + 1) / len(unique_Z_all))
                                
                                mace_ref_progress.empty()
                                st.success(f"Pre-calculated reference energies for {len(unique_Z_all)} unique elements.")
                            
                            # Get FairChem reference energies if needed
                            ref_energies = None
                            if model_type == "FairChem":
                                ref_key_suffix = "_elem_refs"
                                chosen_ref_list_name = None
                                if "UMA Small" in selected_model:
                                    if selected_task_type:
                                        chosen_ref_list_name = selected_task_type + ref_key_suffix
                                elif "ESEN" in selected_model:
                                    chosen_ref_list_name = "omol" + ref_key_suffix
                                
                                if chosen_ref_list_name and chosen_ref_list_name in ELEMENT_REF_ENERGIES:
                                    ref_energies = ELEMENT_REF_ENERGIES[chosen_ref_list_name]
                                    st.success(f"Using FairChem reference energies from '{chosen_ref_list_name}'")
                                else:
                                    st.error(f"Could not find reference energy list for FairChem model: '{selected_model}'")
                            
                            # Process each structure
                            for idx, atoms_obj in enumerate(atoms_list):
                                status_text.text(f"Calculating structure {idx+1}/{len(atoms_list)}...")
                                
                                try:
                                    # Create a copy and attach calculator
                                    calc_atoms = atoms_obj.copy()
                                    calc_atoms.calc = calc
                                    
                                    # Calculate system energy
                                    E_system = calc_atoms.get_potential_energy()
                                    num_atoms = len(calc_atoms)
                                    
                                    if num_atoms == 0:
                                        raise ValueError("System has no atoms")
                                    
                                    atomic_numbers = calc_atoms.get_atomic_numbers()
                                    E_isolated_atoms_total = 0.0
                                    calculation_possible = True
                                    
                                    # Calculate isolated atom energies
                                    if model_type == "FairChem":
                                        if ref_energies:
                                            for Z_val in atomic_numbers:
                                                if Z_val > 0 and Z_val < len(ref_energies):
                                                    E_isolated_atoms_total += ref_energies[Z_val]
                                                # Missing refs treated as 0
                                        else:
                                            calculation_possible = False
                                    
                                    else:  # MACE
                                        for Z_val in atomic_numbers:
                                            E_isolated_atoms_total += mace_isolated_energies.get(Z_val, 0.0)
                                    
                                    if calculation_possible:
                                        # Get metadata
                                        filename = atoms_obj.info.get("source_name", f"structure_{idx+1}")
                                        formula = calc_atoms.get_chemical_formula()
                                        pbc = str(calc_atoms.pbc.tolist())
                                        filetype = os.path.splitext(filename)[1].lstrip('.')
                                        is_periodic = any(calc_atoms.pbc)
                                        
                                        if is_periodic:
                                            cohesive_E = (E_isolated_atoms_total - E_system) / num_atoms
                                            batch_results.append({
                                                "Filename": filename,
                                                "Formula": formula,
                                                "N_atoms": num_atoms,
                                                "PBC": pbc,
                                                "Filetype": filetype,
                                                "Type": "Cohesive",
                                                "System Energy (eV)": f"{E_system:.6f}",
                                                "Isolated Atoms Energy (eV)": f"{E_isolated_atoms_total:.6f}",
                                                "Cohesive Energy (eV/atom)": f"{cohesive_E:.6f}",
                                                "Atomization Energy (eV)": "-"
                                            })
                                        else:
                                            atomization_E = E_isolated_atoms_total - E_system
                                            batch_results.append({
                                                "Filename": filename,
                                                "Formula": formula,
                                                "N_atoms": num_atoms,
                                                "PBC": pbc,
                                                "Filetype": filetype,
                                                "Type": "Atomization",
                                                "System Energy (eV)": f"{E_system:.6f}",
                                                "Isolated Atoms Energy (eV)": f"{E_isolated_atoms_total:.6f}",
                                                "Cohesive Energy (eV/atom)": "-",
                                                "Atomization Energy (eV)": f"{atomization_E:.6f}"
                                            })
                                    else:
                                        raise ValueError("Missing reference energies")
                                        
                                except Exception as e:
                                    batch_results.append({
                                        "Filename": atoms_obj.info.get("source_name", f"structure_{idx+1}"),
                                        "Formula": "Error",
                                        "N_atoms": "-",
                                        "PBC": "-",
                                        "Filetype": "-",
                                        "Type": "-",
                                        "System Energy (eV)": "-",
                                        "Isolated Atoms Energy (eV)": "-",
                                        "Cohesive Energy (eV/atom)": "-",
                                        "Atomization Energy (eV)": f"Failed: {str(e)}"
                                    })
                                
                                progress_bar.progress((idx + 1) / len(atoms_list))
                            
                            status_text.text("Calculation complete!")
                            
                            # Display results table
                            df_results = pd.DataFrame(batch_results)
                            st.dataframe(df_results, use_container_width=True)
                    elif "Geometry Optimization" in task: # Handles both Geometry and Cell+Geometry Opt
                        t0 = time.perf_counter()
                        is_periodic = any(calc_atoms.pbc)
                        if task == "Cell + Geometry Optimization" and not is_periodic:
                            st.warning(
                                "Cell + Geometry Optimization requires a periodic structure "
                                "with a valid unit cell. For isolated molecules, use "
                                "Geometry Optimization."
                            )
                            st.stop()
                        if (
                            optimizer_type == "Lindh Hessian LBFGS"
                            and task == "Cell + Geometry Optimization"
                        ):
                            st.warning(
                                "Lindh Hessian LBFGS supports molecular and fixed-cell "
                                "periodic optimization, but not variable-cell optimization."
                            )
                            st.stop()
                        opt_atoms_obj = FrechetCellFilter(calc_atoms) if task == "Cell + Geometry Optimization" else calc_atoms
                        # Create temporary trajectory file
                        traj_filename = tempfile.NamedTemporaryFile(delete=False, suffix=".traj").name
                        if optimizer_type == "BFGS":
                            opt = BFGS(opt_atoms_obj, trajectory=traj_filename)

                        elif optimizer_type == "BFGSLineSearch":
                            opt = BFGSLineSearch(opt_atoms_obj, trajectory=traj_filename)

                        elif optimizer_type == "LBFGS":
                            opt = LBFGS(opt_atoms_obj, trajectory=traj_filename)

                        elif optimizer_type == "LBFGSLineSearch":
                            opt = LBFGSLineSearch(opt_atoms_obj, trajectory=traj_filename)

                        elif optimizer_type == "FIRE":
                            np.random.seed(0)
                            opt = FIRE(opt_atoms_obj, trajectory=traj_filename)

                        elif optimizer_type == "GPMin":
                            np.random.seed(0)
                            opt = GPMin(opt_atoms_obj, trajectory=traj_filename)

                        elif optimizer_type == "MDMin":
                            np.random.seed(0)
                            opt = MDMin(opt_atoms_obj, trajectory=traj_filename)

                        elif optimizer_type == "Custom1":
                            opt = create_hybrid_optimizer(opt_atoms_obj, trajectory=traj_filename)
                        
                        elif optimizer_type == "FASTMSO":
                            np.random.seed(1)
                            # opt = FASTMSO(
                            #     opt_atoms_obj,
                            #     trajectory=traj_filename,
                            #     maxstep=0.2
                            # )
                            opt = FASTMSO(
                                opt_atoms_obj,
                                trajectory=traj_filename,
                                f_fire=f_fire,
                                f_md=f_md,
                                fire_kwargs={"dt": 0.1, "maxstep": 0.3},
                                md_kwargs={"dt": 0.15},
                                lbfgs_kwargs={"maxstep": 0.2},
                                )

                        elif optimizer_type == "Lindh Hessian LBFGS":
                            try:
                                opt = LindhHessianLBFGS(
                                    opt_atoms_obj,
                                    trajectory=traj_filename,
                                    maxstep=float(lindh_maxstep),
                                    memory=int(lindh_memory),
                                    eigenvalue_floor=float(lindh_eigenvalue_floor),
                                    rebuild_interval=int(lindh_rebuild_interval),
                                    diagnostic_logging=bool(lindh_diagnostic_logging),
                                )
                            except LindhError as exc:
                                st.warning(str(exc))
                                st.stop()

                        elif optimizer_type == "MACE Hessian LBFGS":
                            hessian_calc = None
                            hessian_calc_label = None
                            if model_type != "MACE":
                                try:
                                    hessian_calc_label = mace_hessian_provider_model
                                    hessian_calc = get_mace_model(
                                        MACE_MODELS[hessian_calc_label],
                                        False,
                                        device,
                                        "float32",
                                    )
                                except Exception as exc:
                                    st.error(f"Failed to initialize MACE Hessian provider: {exc}")
                                    st.stop()
                            try:
                                opt = MACEHessianLBFGS(
                                    opt_atoms_obj,
                                    trajectory=traj_filename,
                                    maxstep=float(mace_hessian_maxstep),
                                    memory=int(mace_hessian_memory),
                                    eigenvalue_floor=float(mace_hessian_eigenvalue_floor),
                                    rebuild_interval=int(mace_hessian_rebuild_interval),
                                    diagnostic_logging=bool(mace_hessian_diagnostic_logging),
                                    hessian_calculator=hessian_calc,
                                    hessian_calculator_label=hessian_calc_label,
                                )
                            except AnalyticalHessianError as exc:
                                st.error(str(exc))
                                st.stop()

                        elif optimizer_type == "MACE-Seed LBFGS":
                            hessian_calc = None
                            hessian_calc_label = "MACE OMAT Small"
                            if model_type != "MACE" or selected_model != hessian_calc_label:
                                try:
                                    hessian_calc = get_mace_model(
                                        MACE_MODELS[hessian_calc_label],
                                        False,
                                        device,
                                        "float32",
                                    )
                                except Exception as exc:
                                    st.error(
                                        f"Failed to initialize MACE OMAT Small Hessian provider: {exc}"
                                    )
                                    st.stop()
                            try:
                                opt = MACESeedLBFGS(
                                    opt_atoms_obj,
                                    trajectory=traj_filename,
                                    maxstep=float(mace_seed_maxstep),
                                    initial_step_radius=float(mace_seed_initial_radius),
                                    minimum_step_radius=float(mace_seed_minimum_radius),
                                    memory=int(mace_seed_memory),
                                    eigenvalue_floor=float(mace_seed_eigenvalue_floor),
                                    diagnostic_logging=bool(mace_seed_diagnostic_logging),
                                    hessian_calculator=hessian_calc,
                                    hessian_calculator_label=hessian_calc_label,
                                )
                            except (AnalyticalHessianError, ValueError) as exc:
                                st.error(str(exc))
                                st.stop()

                        opt.attach(lambda: streamlit_log(opt), interval=1)
                        st.write(f"Running {task.lower()}...")
                        is_converged = opt.run(fmax=fmax, steps=max_steps)
                        
                        energy = calc_atoms.get_potential_energy()
                        max_force = optimizer_fmax(opt_atoms_obj)
                        max_atomic_force = atomic_fmax(opt_atoms_obj)
                        
                        results["Final Energy"] = f"{energy:.6f} eV"
                        results["Final Optimizer Fmax"] = f"{max_force:.6f}"
                        results["Final Atomic Maximum Force"] = f"{max_atomic_force:.6f} eV/A"
                        results["Steps Taken"] = opt.get_number_of_steps()
                        results["Converged"] = "Yes" if is_converged else "No"
                        if (
                            optimizer_type == "Lindh Hessian LBFGS"
                            and hasattr(opt, "get_lindh_metadata")
                        ):
                            for key, value in opt.get_lindh_metadata().items():
                                if value is not None:
                                    results[key] = value
                        if (
                            optimizer_type in ("MACE Hessian LBFGS", "MACE-Seed LBFGS")
                            and hasattr(opt, "get_hessian_metadata")
                        ):
                            for key, value in opt.get_hessian_metadata().items():
                                if value is not None:
                                    results[key] = value
                        if task == "Cell + Geometry Optimization":
                            results["Final Cell Parameters"] = np.round(calc_atoms.cell.cellpar(), 4).tolist()
                        t1 = time.perf_counter()
                        results["Time Taken"] = f"{t1 - t0:.4f} seconds"
                        st.success("Calculation completed successfully!")
                        st.markdown("### Results")
                        for key, value in results.items():
                            st.write(f"**{key}:** {value}")
                    
                    if "Optimization" in task and "Final Energy" in results: # Check if opt was successful
                        st.markdown("### Optimized Structure")

                        opt_view = get_structure_viz2(calc_atoms, style=viz_style, show_unit_cell=True, width=400, height=400)
                        st.components.v1.html(opt_view._make_html(), width=400, height=400)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xyz", mode="w+") as tmp_file_opt:
                            if is_periodic:
                                write(tmp_file_opt.name, calc_atoms, format="extxyz")
                            else:
                                write(tmp_file_opt.name, calc_atoms, format="xyz")
                            tmp_filepath_opt = tmp_file_opt.name
                        
                        with open(tmp_filepath_opt, 'r') as file_opt:
                            xyz_content_opt = file_opt.read()
                        @st.fragment
                        def show_optimized_structure_download_button():
                            # st.button("Release the balloons", help="Fragment rerun")
                            # st.balloons()
                            st.download_button(
                                label="Download Optimized Structure (XYZ)",
                                data=xyz_content_opt,
                                file_name="optimized_structure.xyz",
                                mime="chemical/x-xyz"
                            )
                        show_optimized_structure_download_button()
                        # --- Energy vs. Optimization Cycles Plot ---
                        @st.fragment
                        def show_energy_plot(traj_filename):
                            

                            if os.path.exists(traj_filename):
                                try:
                                    trajectory = read(traj_filename, index=":")
                                    
                                    # Extract energy and step number
                                    energies = [atoms.get_potential_energy() for atoms in trajectory]
                                    steps = list(range(len(energies)))
                                    
                                    # Create a DataFrame for Plotly
                                    data = {
                                        "Optimization Cycle": steps,
                                        "Energy (eV)": energies
                                    }
                                    
                                    df = pd.DataFrame(data)
                                    
                                    st.markdown("### Energy Profile During Optimization")

                                    # -- Plotly chart (always visible) ------------------------------------------
                                    fig = px.line(
                                        df,
                                        x="Optimization Cycle",
                                        y="Energy (eV)",
                                        markers=True,
                                        title="Energy Convergence vs. Optimization Cycle",
                                    )

                                    fig.update_layout(
                                        xaxis_title="Optimization Cycle",
                                        yaxis_title="Energy (eV)",
                                        hovermode="x unified",
                                        template="plotly_white",
                                        font=dict(size=12),
                                        title=dict(x=0.5),
                                    )


                                    fig.add_hline(
                                        y=energies[-1],
                                        line_dash="dot",
                                        line_color="red",
                                        annotation_text=f"Final Energy: {energies[-1]:.4f} eV",
                                        annotation_position="bottom right",
                                        annotation_font=dict(color="black"),
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # -- Matplotlib chart (inside expander, no rerun triggered) -----------------
                                    with st.expander("🔍 View Matplotlib version (publication-ready)"):
                                        mpl_fig, ax = plt.subplots(figsize=(8, 4))

                                        ax.plot(
                                            df["Optimization Cycle"],
                                            df["Energy (eV)"],
                                            marker="o",
                                            linewidth=1.5,
                                            markersize=4,
                                            color="steelblue",
                                            label="Energy",
                                        )

                                        # Converged energy line
                                        ax.axhline(
                                            y=energies[-1],
                                            color="red",
                                            linestyle="dotted",
                                            linewidth=1.5,
                                            label=f"Final Energy: {energies[-1]:.4f} eV",
                                        )

                                        ax.set_xlabel("Optimization Cycle", fontsize=12, color="black")
                                        ax.set_ylabel("Energy (eV)", fontsize=12, color="black")
                                        ax.set_title("Energy Convergence vs. Optimization Cycle", fontsize=13, color="black")
                                        ax.tick_params(colors="black")
                                        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                                        ax.legend(fontsize=10)
                                        ax.grid(True, linestyle="--", alpha=0.5)

                                        for spine in ax.spines.values():
                                            spine.set_edgecolor("black")

                                        plt.tight_layout()
                                        st.pyplot(mpl_fig)
                                        plt.close(mpl_fig)  # prevent memory leak

                                except Exception as e:
                                    st.error(f"Error generating energy plot: {e}")
                            else:
                                st.warning("Cannot generate energy plot: Trajectory file not found.")

                        show_energy_plot(traj_filename)
                        # --- End of Energy Plot Code ---
                        os.unlink(tmp_filepath_opt)

                        @st.fragment
                        def show_trajectory_and_controls():
                            from ase.io import read
                            import py3Dmol
                        
                            if "traj_frames" not in st.session_state:
                                if os.path.exists(traj_filename):
                                    try:
                                        trajectory = read(traj_filename, index=":")
                                        st.session_state.traj_frames = trajectory
                                        st.session_state.traj_index = 0
                                    except Exception as e:
                                        st.error(f"Error reading trajectory: {e}")
                                        return
                                    # finally:
                                    #     os.unlink(traj_filename)
                                else:
                                    st.warning("Trajectory file not found.")
                                    return
                        
                            trajectory = st.session_state.traj_frames
                            index = st.session_state.traj_index
                        
                            st.markdown("### Optimization Trajectory")
                            st.write(f"Captured {len(trajectory)} optimization steps")
                        
                            # Navigation Buttons
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                if st.button("⏮ First"):
                                    st.session_state.traj_index = 0
                            with col2:
                                if st.button("◀ Previous") and index > 0:
                                    st.session_state.traj_index -= 1
                            with col3:
                                if st.button("Next ▶") and index < len(trajectory) - 1:
                                    st.session_state.traj_index += 1
                            with col4:
                                if st.button("Last ⏭"):
                                    st.session_state.traj_index = len(trajectory) - 1
                        
                            # Show current frame
                            current_atoms = trajectory[st.session_state.traj_index]
                            st.write(f"Frame {st.session_state.traj_index + 1}/{len(trajectory)}")
                        
                            def atoms_to_xyz_string(atoms, step_idx=None):
                                xyz_str = f"{len(atoms)}\n"
                                if step_idx is not None:
                                    xyz_str += f"Step {step_idx}, Energy = {atoms.get_potential_energy():.6f} eV\n"
                                else:
                                    xyz_str += f"Energy = {atoms.get_potential_energy():.6f} eV\n"
                                for atom in atoms:
                                    xyz_str += f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}\n"
                                return xyz_str
                        
                            traj_view = get_structure_viz2(current_atoms, style=viz_style, show_unit_cell=True, width=400, height=400)
                            st.components.v1.html(traj_view._make_html(), width=400, height=400)
                        
                            # Download button for entire trajectory
                            trajectory_xyz = ""
                            for i, atoms in enumerate(trajectory):
                                trajectory_xyz += atoms_to_xyz_string(atoms, i)
                            st.download_button(
                                label="Download Optimization Trajectory (XYZ)",
                                data=trajectory_xyz,
                                file_name="optimization_trajectory.xyz",
                                mime="chemical/x-xyz"
                            )
                        
                        show_trajectory_and_controls()
                    
                    elif task == "Vibrational Mode Analysis":
                        # Conversion factors
                        from ase.units import kB as kB_eVK, _Nav, J  # ASE's constants
                        from scipy.constants import physical_constants
                        kB_JK = physical_constants["Boltzmann constant"][0]  # J/K
                        is_periodic = any(calc_atoms.pbc)
                        st.write("Running vibrational mode analysis using finite differences...")

                        natoms = len(calc_atoms)
                        is_linear = False  # Set manually or auto-detect
                        nmodes_expected = 3 * natoms - (5 if is_linear else 6)

                        # Create temporary directory to store .vib files
                        with tempfile.TemporaryDirectory() as tmpdir:
                            vib = Vibrations(calc_atoms, name=os.path.join(tmpdir, 'vib'))

                            with st.spinner("Calculating vibrational modes... This may take a few minutes."):
                                vib.run()
                                freqs = vib.get_frequencies()
                                energies = vib.get_energies()
                                
                                print('\n\n\n\n\n\n\n\n')
                                # vib.get_hessian_2d()
                                # st.write(vib.summary())
                                # print('\n')
                                # vib.tabulate()
                            
                            freqs_cm = freqs
                            freqs_eV = energies

                            # Classify frequencies
                            mode_data = []
                            for i, freq in enumerate(freqs_cm):
                                if freq < 0:
                                    label = "Imaginary"
                                elif abs(freq) < 500:
                                    label = "Low"
                                else:
                                    label = "Physical"
                                mode_data.append({
                                    "Mode": i + 1,
                                    "Frequency (cm⁻¹)": round(freq, 2),
                                    "Type": label
                                })

                            df_modes = pd.DataFrame(mode_data)

                            # Display summary and mode count
                            st.success("Vibrational analysis completed.")
                            st.write(f"Number of atoms: {natoms}")
                            st.write(f"Expected vibrational modes: {nmodes_expected}")
                            st.write(f"Found {len(freqs_cm)} modes (including translational/rotational modes).")

                            # Show table of modes
                            st.write("### Vibrational Mode Summary")
                            st.dataframe(df_modes, use_container_width=True)

                            # Store in results dictionary
                            results["Vibrational Modes"] = df_modes.to_dict(orient="records")

                            # Histogram plot of vibrational frequencies
                            st.write("### Frequency Distribution Histogram")
                            fig, ax = plt.subplots()
                            ax.hist(freqs_cm, bins=30, color='skyblue', edgecolor='black')
                            ax.set_xlabel("Frequency (cm⁻¹)")
                            ax.set_ylabel("Number of Modes")
                            ax.set_title("Distribution of Vibrational Frequencies")
                            st.pyplot(fig)

                            # CSV download
                            csv_buffer = io.StringIO()
                            df_modes.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="Download Vibrational Frequencies (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name="vibrational_modes.csv",
                                mime="text/csv"
                            )
                            # -------- Thermodynamic Analysis for Molecules --------
                        if not is_periodic:

                            # Filter physical frequencies > 1 cm⁻¹ (to avoid numerical issues)
                            physical_freqs_eV = np.array([f for f in freqs_eV if f > 1e-5])

                            # Zero-point vibrational energy (ZPE)
                            ZPE = 0.5 * np.sum(physical_freqs_eV)  # in eV

                            # Vibrational entropy (in eV/K)
                            vib_entropy = 0.0
                            for f in physical_freqs_eV:
                                x = f / (kB_eVK * T)
                                vib_entropy += (x / (np.exp(x) - 1) - np.log(1 - np.exp(-x)))

                            S_vib_eVK = kB_eVK * vib_entropy  # eV/K
                            S_vib_JmolK = S_vib_eVK * J * _Nav  # J/mol·K

                            results["ZPE (eV)"] = ZPE.real
                            results["Vibrational Entropy (eV/K)"] = S_vib_eVK
                            results["Vibrational Entropy (J/mol·K)"] = S_vib_JmolK

                            st.write(f"**Zero-point vibrational energy (ZPE)**: {ZPE.real:.6f} eV")
                            st.write(f"**Vibrational entropy**: {S_vib_eVK:.6f} eV/K")

                        else:
                            st.info("Thermodynamic properties like ZPE and entropy are currently only meaningful for isolated molecules (non-periodic systems).")
                    
            except Exception as e:
                st.error(f"🔴 Calculation error: {str(e)}")
                # st.error("Please check the structure, model compatibility, and parameters. For FairChem UMA, ensure the task type (omol, omat etc.) is appropriate for your system (e.g. omol for molecules, omat for materials).")
                st.error(f"Traceback: {traceback.format_exc()}")

else:
    st.info("👋 Welcome! Please select or upload a structure using the sidebar options to begin.")

st.markdown("---")
with st.expander('ℹ️ About This App & Foundational MLIPs'):
    st.write("""
    **Test, compare, and benchmark universal machine learning interatomic potentials (MLIPs).**
    This application allows you to perform atomistic simulations using pre-trained foundational MLIPs 
    from the MACE, MatterSim (Microsoft), SevenNet, Orb (Orbital Materials) and FairChem (Meta AI) developers and researchers.
    
    **Features:**
    - Upload/Paste structure files (XYZ, CIF, POSCAR, etc.), import from Materials Project/PubChem or use built-in examples.
    - Select from various MACE, ORB, SevenNet, MatterSim and FairChem models.
    - Calculate energies, forces, cohesive/atomization energy, vibrational modes and perform geometry/cell optimizations.
    - Visualize atomic structures in 3D and download results, optimized structures and optimization trajectories.
    
    **Quick Start:**
    1.  **Input**: Choose an input method in the sidebar (e.g., "Select Example").
    2.  **Model**: Pick a model type (MACE/FairChem/MatterSim/ORB/SevenNet) and specific model. For FairChem UMA, select the appropriate task type (e.g., `omol` for molecules, `omat` for materials). 
    For models trained on OMOL25 dataset (whenever the model name contains `omol`) then the user also needs to provide a charge and spin multiplicity (`2S+1`) value. By default the charge is set to zero and spin multiplicity to 1 (S=0).
    3.  **Task**: Select a calculation task (e.g., "Energy Calculation", "Atomization/Cohesive Energy", "Geometry Optimization").
    4.  **Run**: Click "Run Calculation" and view the results.

    **Atomization/Cohesive Energy Notes:**
    - **Atomization Energy** ($E_{\\text{atomization}} = \sum E_{\\text{isolated atoms}} - E_{\\text{molecule}}$) is typically for non-periodic systems (molecules).
    - **Cohesive Energy** ($E_{\\text{cohesive}} = (\sum E_{\\text{isolated atoms}} - E_{\\text{bulk system}}) / N_{\\text{atoms}}$) is for periodic systems.
    - For **MACE models**, isolated atom energies are computed on-the-fly.
    - For **FairChem models**, isolated atom energies are based on pre-tabulated reference values (provided in a YAML-like structure within the app). Ensure the selected FairChem task type (`omol`, `omat`, etc. for UMA models) or model type (ESEN models use `omol` references) aligns with the system and has the necessary elemental references.
    """)


with st.expander('🔧 Tech Stack & System Information'):

    
    st.markdown("### System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Operating System:**")
        st.write(f"- OS: {platform.system()} {platform.release()}")
        st.write(f"- Version: {platform.version()}")
        st.write(f"- Architecture: {platform.machine()}")
        st.write(f"- Processor: {platform.processor()}")
        
        st.write("\n**Python Environment:**")
        st.write(f"- Python Version: {platform.python_version()}")
        st.write(f"- Python Implementation: {platform.python_implementation()}")
    
    with col2:
        st.write("**Hardware Resources:**")
        st.write(f"- CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        st.write(f"- CPU Usage: {psutil.cpu_percent(interval=1)}%")
        
        memory = psutil.virtual_memory()
        st.write(f"- Total RAM: {memory.total / (1024**3):.2f} GB")
        st.write(f"- Available RAM: {memory.available / (1024**3):.2f} GB")
        st.write(f"- RAM Usage: {memory.percent}%")
        
        disk = psutil.disk_usage('/')
        st.write(f"- Total Disk Space: {disk.total / (1024**3):.2f} GB")
        st.write(f"- Free Disk Space: {disk.free / (1024**3):.2f} GB")
        st.write(f"- Disk Usage: {disk.percent}%")
    
    st.markdown("### Package Versions")
    
    packages_to_check = [
        'streamlit', 'torch', 'numpy', 'ase', 'py3Dmol', 
        'mace-torch', 'fairchem-core', 'orb-models', 'sevenn',
        'pandas', 'matplotlib', 'scipy', 'yaml', 'huggingface-hub', 
        'upet', 'metatrain', 'metatensor', 'rdkit', 'metatomic-torch',
        'hydra-core', 'ray', 'torchtnt', 'pubchempy', 'warp-lang', 
        'hf-xet', 'mp_api', 'pymatgen', 'rdkit',
    ]
    
    if mattersim_available:
        packages_to_check.append('mattersim')
    
    package_versions = {}
    for package in packages_to_check:
        try:
            version = pkg_resources.get_distribution(package).version
            package_versions[package] = version
        except pkg_resources.DistributionNotFound:
            package_versions[package] = "Not installed"
    
    # Display in two columns
    col1, col2 = st.columns(2)
    items = list(package_versions.items())
    mid_point = len(items) // 2
    
    with col1:
        for package, version in items[:mid_point]:
            st.write(f"**{package}:** {version}")
    
    with col2:
        for package, version in items[mid_point:]:
            st.write(f"**{package}:** {version}")
    
    # PyTorch specific information
    st.markdown("### PyTorch Configuration")
    st.write(f"**PyTorch Version:** {torch.__version__}")
    st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.write(f"**CUDA Version:** {torch.version.cuda}")
        st.write(f"**cuDNN Version:** {torch.backends.cudnn.version()}")
        st.write(f"**Number of GPUs:** {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            st.write(f"**GPU {i}:** {torch.cuda.get_device_name(i)}")
    else:
        st.write("Running on CPU only")
torch.cuda.empty_cache()

gc.collect()

st.markdown("---")
st.markdown("Universal MLIP Studio App | Created with Streamlit, ASE, MACE, FairChem, SevenNet, ORB, MatterSim, UPET, Py3DMol, Pymatgen and ❤️")
st.markdown("Developed by [Dr. Manas Sharma](https://manas.bragitoff.com/) in the groups of [Prof. Ananth Govind Rajan Group](https://www.agrgroup.org/) and [Prof. Sudeep Punnathanam](https://chemeng.iisc.ac.in/sudeep/) at  [IISc Bangalore](https://iisc.ac.in/)")
