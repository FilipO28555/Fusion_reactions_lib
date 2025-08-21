#%%
import re
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import constants as const
from torch_interpolations import RegularGridInterpolator as interpolate

# --- Physical Constants ---
c = const.c
k_b = const.k  # Boltzmann constant in J/K
e = const.e    # Elementary charge in C

# --- Particle Species Definitions ---
PARTICLE_SPECIES = {
    # Fundamental Particles
    'ELECTRON': 0, 'PROTON': 1, 'NEUTRON': 6, 'GAMMA': 7,
    # Hydrogen Isotopes
    'DEUTERON': 2, 'TRITON': 3,
    # Helium Isotopes
    'HELIUM3': 4, 'HELIUM4': 5,
    # Lithium Isotopes
    'LITHIUM6': 8, 'LITHIUM7': 9,
    # Beryllium Isotopes
    'BERYLLIUM7': 10, 'BERYLLIUM9': 11,
    # Boron Isotopes
    'BORON10': 12, 'BORON11': 13,
    # Carbon Isotopes
    'CARBON12': 14, 'CARBON13': 15,
    'ANY': 16,  # Placeholder for any particle species
}

# --- Translation Map for EXFOR Codes ---
TRANSLATION_MAP = {
    # Full codes to common names
    '1-H-1': 'p', 
    '1-H-2': 'd', 
    '1-H-3': 't',
    '2-HE-3': 'He-3', 
    '2-HE-4': 'alpha', 
    '0-N-1': 'n',
    # Abbreviations to common names
    'P': 'p', 
    'D': 'd', 
    'T': 't', 
    'A': 'alpha', 
    'N': 'n', 
    'G': 'gamma'
}

# --- Common Name Mapping ---
COMMON_NAME_TO_ID = {
    # Standard abbreviations
    'e': PARTICLE_SPECIES['ELECTRON'], 'p': PARTICLE_SPECIES['PROTON'],
    'P': PARTICLE_SPECIES['PROTON'], 'd': PARTICLE_SPECIES['DEUTERON'],
    't': PARTICLE_SPECIES['TRITON'], 'T': PARTICLE_SPECIES['TRITON'],
    'n': PARTICLE_SPECIES['NEUTRON'], 'N': PARTICLE_SPECIES['NEUTRON'],
    'A': PARTICLE_SPECIES['HELIUM4'], 'alpha': PARTICLE_SPECIES['HELIUM4'],
    'gamma': None, 'TOT': PARTICLE_SPECIES['ANY'],
    
    # Common names for Helium isotopes
    'He-3': PARTICLE_SPECIES['HELIUM3'], 'HE3': PARTICLE_SPECIES['HELIUM3'],
    
    # Full nuclide names
    'H-1': PARTICLE_SPECIES['PROTON'], 'H-2': PARTICLE_SPECIES['DEUTERON'],
    'H-3': PARTICLE_SPECIES['TRITON'], 'HE-3': PARTICLE_SPECIES['HELIUM3'],
    'HE-4': PARTICLE_SPECIES['HELIUM4'], 'LI-6': PARTICLE_SPECIES['LITHIUM6'],
    'LI-7': PARTICLE_SPECIES['LITHIUM7'], 'BE-7': PARTICLE_SPECIES['BERYLLIUM7'],
    'BE-9': PARTICLE_SPECIES['BERYLLIUM9'], 'B-10': PARTICLE_SPECIES['BORON10'],
    'B-11': PARTICLE_SPECIES['BORON11'], 'C-12': PARTICLE_SPECIES['CARBON12'],
    'C-13': PARTICLE_SPECIES['CARBON13'],
}

# --- Particle Physical Properties ---
PARTICLE_PROPERTIES = {
    # Fundamental Particles
    PARTICLE_SPECIES['ELECTRON']: {'mass': const.m_e, 'charge': -const.e, 'name': 'e'},
    PARTICLE_SPECIES['PROTON']: {'mass': const.m_p, 'charge': const.e, 'name': 'p'},
    PARTICLE_SPECIES['NEUTRON']: {'mass': const.m_n, 'charge': 0, 'name': 'n'},
    PARTICLE_SPECIES['GAMMA']: {'mass': 0, 'charge': 0, 'name': 'gamma'},
    # Hydrogen Isotopes
    PARTICLE_SPECIES['DEUTERON']: {'mass': const.physical_constants['deuteron mass'][0], 'charge': const.e, 'name': 'd'},
    PARTICLE_SPECIES['TRITON']: {'mass': const.physical_constants['triton mass'][0], 'charge': const.e, 'name': 't'},
    # Helium Isotopes
    PARTICLE_SPECIES['HELIUM3']: {'mass': const.physical_constants['helion mass'][0], 'charge': 2 * const.e, 'name': 'He-3'},
    PARTICLE_SPECIES['HELIUM4']: {'mass': const.physical_constants['alpha particle mass'][0], 'charge': 2 * const.e, 'name': 'alpha'},
    # Lithium Isotopes
    PARTICLE_SPECIES['LITHIUM6']: {'mass': 6.0151228874 * const.u, 'charge': 3 * const.e, 'name': 'Li-6'},
    PARTICLE_SPECIES['LITHIUM7']: {'mass': 7.0160034366 * const.u, 'charge': 3 * const.e, 'name': 'Li-7'},
    # Beryllium Isotopes
    PARTICLE_SPECIES['BERYLLIUM7']: {'mass': 7.016928716 * const.u, 'charge': 4 * const.e, 'name': 'Be-7'},
    PARTICLE_SPECIES['BERYLLIUM9']: {'mass': 9.012183066 * const.u, 'charge': 4 * const.e, 'name': 'Be-9'},
    # Boron Isotopes
    PARTICLE_SPECIES['BORON10']: {'mass': 10.012936992 * const.u, 'charge': 5 * const.e, 'name': 'B-10'},
    PARTICLE_SPECIES['BORON11']: {'mass': 11.009305167 * const.u, 'charge': 5 * const.e, 'name': 'B-11'},
    # Carbon Isotopes
    PARTICLE_SPECIES['CARBON12']: {'mass': 12.0000000000 * const.u, 'charge': 6 * const.e, 'name': 'C-12'},
    PARTICLE_SPECIES['CARBON13']: {'mass': 13.003354835 * const.u, 'charge': 6 * const.e, 'name': 'C-13'},
}


class Reaction:
    """
    Represents a single nuclear reaction with its data and properties.
    """
    
    def __init__(self, reaction_code, energies=None, sigmas=None, interpolator=None):
        self.reaction_code = reaction_code
        self.energies = energies  # These will be CM energies after transformation
        self.energies_lab = None  # Store original lab energies
        self.sigmas = sigmas
        self.interpolator = interpolator
        
        # Parse reaction components
        self._parse_reaction()
        self._calculate_q_value()
    
    def _get_particle_mass(self, particle_id):
        """Get particle mass in kg."""
        if particle_id is None or particle_id not in PARTICLE_PROPERTIES:
            return None
        return PARTICLE_PROPERTIES[particle_id]['mass']
    
    def _relativistic_energy(self, momentum, mass):
        """Calculate relativistic energy: E = sqrt((pc)^2 + (mc^2)^2)"""
        if mass == 0:  # Photon
            return momentum * c
        return torch.sqrt((momentum * c)**2 + (mass * c**2)**2)
    
    def _relativistic_momentum(self, kinetic_energy, mass):
        """Calculate relativistic momentum from kinetic energy."""
        if mass == 0:  # Photon
            return kinetic_energy / c
        
        total_energy = kinetic_energy + mass * c**2
        momentum_squared = (total_energy**2 - (mass * c**2)**2) / c**2
        return torch.sqrt(torch.clamp(momentum_squared, min=0))
    
    def _lab_to_cm_energy_relativistic(self, energy_lab_ev):
        """
        Transform kinetic energy from lab frame to center of mass frame.
        Fully relativistic treatment.
        
        Args:
            energy_lab_ev: Lab frame kinetic energy in eV (tensor)
            
        Returns:
            CM frame kinetic energy in eV (tensor)
        """
        if self.target_id is None or self.projectile_id is None:
            raise ValueError("Target and projectile IDs must be set before transformation.")
            return energy_lab_ev  # No transformation possible
        
        # Get masses
        m1 = self._get_particle_mass(self.projectile_id)  # Projectile
        m2 = self._get_particle_mass(self.target_id)      # Target (at rest in lab)
        
        if m1 is None or m2 is None:
            return energy_lab_ev  # No transformation possible
        
        # Convert eV to Joules
        energy_lab_j = energy_lab_ev * e
        
        # Lab frame: projectile has kinetic energy, target at rest
        # Projectile total energy and momentum
        E1_lab = energy_lab_j + m1 * c**2  # Total energy of projectile
        p1_lab = self._relativistic_momentum(energy_lab_j, m1)  # Momentum of projectile
        
        # Target at rest in lab frame
        E2_lab = m2 * c**2
        p2_lab = torch.zeros_like(p1_lab)
        
        # Total 4-momentum in lab frame
        E_total_lab = E1_lab + E2_lab
        p_total_lab = p1_lab + p2_lab  # Vector addition (1D case)
        
        # Invariant mass (center of mass energy)
        s = E_total_lab**2 - (p_total_lab * c)**2
        E_cm_total = torch.sqrt(torch.clamp(s, min=0))
        
        # In CM frame, total momentum is zero, so each particle has equal and opposite momentum
        # For two-body system in CM: E_cm = E1_cm + E2_cm where E1_cm^2 - p_cm^2*c^2 = m1^2*c^4
        
        # Energy of each particle in CM frame
        E1_cm = (E_cm_total**2 + (m1 * c**2)**2 - (m2 * c**2)**2) / (2 * E_cm_total)
        E2_cm = (E_cm_total**2 + (m2 * c**2)**2 - (m1 * c**2)**2) / (2 * E_cm_total)
        
        # CM momentum of each particle
        p_cm_squared = torch.clamp(E1_cm**2 - (m1 * c**2)**2, min=0) / c**2
        p_cm = torch.sqrt(p_cm_squared)
        
        # Kinetic energy in CM frame
        T1_cm = E1_cm - m1 * c**2
        T2_cm = E2_cm - m2 * c**2
        
        # Total kinetic energy in CM frame
        T_cm_total = T1_cm + T2_cm
        
        # Convert back to eV
        return T_cm_total / e
    
    def _cm_to_lab_energy_relativistic(self, energy_cm_ev):
        """
        Transform kinetic energy from center of mass frame to lab frame.
        Inverse of lab_to_cm transformation.
        
        Args:
            energy_cm_ev: CM frame kinetic energy in eV (tensor)
            
        Returns:
            Lab frame kinetic energy in eV (tensor)
        """
        if self.target_id is None or self.projectile_id is None:
            return energy_cm_ev
        
        # Get masses
        m1 = self._get_particle_mass(self.projectile_id)  # Projectile
        m2 = self._get_particle_mass(self.target_id)      # Target
        
        if m1 is None or m2 is None:
            return energy_cm_ev
        
        # Convert to Joules
        energy_cm_j = energy_cm_ev * e
        
        # In CM frame, we know the total kinetic energy
        # We need to find the lab frame energy that gives this CM energy
        
        # For a two-body system, if we know the CM kinetic energy,
        # we can work backwards to find the lab energy
        
        # Total energy in CM frame
        E_cm_total = energy_cm_j + (m1 + m2) * c**2
        
        # Lab frame calculation (reverse of the forward transformation)
        # From invariant mass: s = E_cm_total^2
        # In lab frame: s = (E1_lab + m2*c^2)^2 - (p1_lab*c)^2
        # where E1_lab = T1_lab + m1*c^2 and p1_lab = sqrt(E1_lab^2 - (m1*c^2)^2)/c
        
        # Solving for lab kinetic energy
        s = E_cm_total**2
        
        # Lab frame projectile total energy
        E1_lab = (s + (m1 * c**2)**2 - (m2 * c**2)**2) / (2 * m2 * c**2)
        
        # Lab frame projectile kinetic energy
        T1_lab = E1_lab - m1 * c**2
        
        # Convert back to eV
        return T1_lab / e
    
    def set_lab_energies(self, lab_energies_ev):
        """
        Set lab frame energies and automatically convert to CM frame.
        
        Args:
            lab_energies_ev: Lab frame energies in eV (tensor)
        """
        self.energies_lab = lab_energies_ev
        self.energies = self._lab_to_cm_energy_relativistic(lab_energies_ev)
    
    def get_cross_section_at_energy_lab(self, energy_lab_ev):
        """Get cross-section value at specific lab frame energy."""
        if self.interpolator is None:
            return None
        
        # Convert lab energy to CM energy
        energy_cm_ev = self._lab_to_cm_energy_relativistic(torch.tensor([energy_lab_ev], dtype=torch.float64, device=self.energies.device))
        return self.interpolator([energy_cm_ev]).item()
    
    def _format_nuclide(self, code):
        """Format a nuclide code from EXFOR format into a more readable string."""
        if code in TRANSLATION_MAP:
            return TRANSLATION_MAP[code]
        
        parts = code.split('-')
        if len(parts) == 3:
            z, symbol, a = parts
            if code in TRANSLATION_MAP:
                return TRANSLATION_MAP[code]
            return f"{symbol}-{a}"
            
        return code
    
    def _is_valid_nucleus(self, nucleus_code):
        """Check if a nucleus code represents a valid/stable nucleus."""
        # List of known unstable or invalid nuclei to filter out
        invalid_nuclei = ['LI-5', 'BE-5', 'B-5', 'C-5', 'N-5', 'O-5']
        
        # Check if the formatted nucleus is in our known invalid list
        formatted = self._format_nuclide(nucleus_code)
        return formatted not in invalid_nuclei
    
    def _parse_reaction(self):
        """Parse the reaction code to extract reactants and products."""
        code = self.reaction_code.split(',,')[0]
        
        # Enhanced regex to handle various formats
        patterns = [
            r'([^(]+)\(([^,]+),([^)]+)\)(.*)',  # Standard format
            r'([^(]+)\(([^,]+),([^)]+)$',       # Missing closing parenthesis and residual
            r'^([^(]+)\(([^,]+)\)$',            # Only target(projectile) format
        ]
        
        match = None
        for pattern in patterns:
            match = re.match(pattern, code)
            if match:
                break
        
        if not match:
            print(f"Warning: Could not parse reaction code: {self.reaction_code}")
            self.target = self.projectile = self.emitted = self.residual = None
            self.target_id = self.projectile_id = self.emitted_id = self.residual_id = None
            return
        
        groups = match.groups()
        
        if len(groups) >= 3:
            target_code = groups[0]
            projectile_code = groups[1]
            emitted_code = groups[2]
            residual_code = groups[3] if len(groups) > 3 else ''
        else:
            # Handle incomplete parsing
            target_code = groups[0] if len(groups) > 0 else ''
            projectile_code = groups[1] if len(groups) > 1 else ''
            emitted_code = 'TOT'
            residual_code = 'TOT'
        
        # Format particles
        self.target = self._format_nuclide(target_code.strip())
        self.projectile = self._format_nuclide(projectile_code.strip())
        self.emitted = self._format_nuclide(emitted_code.strip())
        self.residual = self._format_nuclide(residual_code.strip())
        
        # Validate nuclei
        if not self._is_valid_nucleus(target_code.strip()):
            print(f"Warning: Invalid target nucleus {target_code} in reaction {self.reaction_code}")
        if not self._is_valid_nucleus(residual_code.strip()) and residual_code.strip():
            print(f"Warning: Invalid residual nucleus {residual_code} in reaction {self.reaction_code}")
        
        # Handle special cases
        if self.target == 'p' or self.projectile in ['TOT', 'N'] and self.emitted == '':
            self.emitted = "TOT"
            self.residual = "TOT"
        
        # Handle neutron scattering cases like (N,TOT)
        if self.projectile == 'N' and self.emitted in ['TOT', '']:
            self.emitted = "TOT"
            self.residual = "TOT"
        
        # Get particle IDs
        self.target_id = COMMON_NAME_TO_ID.get(self.target)
        self.projectile_id = COMMON_NAME_TO_ID.get(self.projectile)
        self.emitted_id = COMMON_NAME_TO_ID.get(self.emitted)
        self.residual_id = COMMON_NAME_TO_ID.get(self.residual)
        
        # Set reactants and products
        if None not in (self.target_id, self.projectile_id, self.emitted_id, self.residual_id):
            self.reactants = (self.target_id, self.projectile_id)
            self.products = (self.residual_id, self.emitted_id)
        else:
            self.reactants = self.products = None
            if self.target_id is None:
                print(f"Warning: Unknown target particle '{self.target}' in {self.reaction_code}")
            if self.projectile_id is None:
                print(f"Warning: Unknown projectile particle '{self.projectile}' in {self.reaction_code}")
            if self.emitted_id is None:
                print(f"Warning: Unknown emitted particle '{self.emitted}' in {self.reaction_code}")
            if self.residual_id is None:
                print(f"Warning: Unknown residual particle '{self.residual}' in {self.reaction_code}")

    def _calculate_q_value(self):
        """Calculate the Q-value of the reaction in MeV."""
        if self.reactants is None or self.products is None:
            self.q_value_mev = 0.0
            return
        
        try:
            reactant_mass = sum(PARTICLE_PROPERTIES[species_id]['mass'] for species_id in self.reactants)
            product_mass = sum(PARTICLE_PROPERTIES[species_id]['mass'] for species_id in self.products)
            mass_defect = reactant_mass - product_mass
            q_value_joules = mass_defect * c**2
            self.q_value_mev = q_value_joules / (const.e * 1e6)  # Convert to MeV
        except KeyError:
            self.q_value_mev = 0.0
    
    def get_human_readable_equation(self):
        """Return the reaction in human-readable form."""
        if None in (self.target, self.projectile, self.emitted, self.residual):
            return f"Invalid reaction: {self.reaction_code}"
        return f"{self.target} + {self.projectile} -> {self.residual} + {self.emitted}"
    
    def get_cross_section_at_energy(self, energy_ev):
        """Get cross-section value at specific energy."""
        if self.interpolator is None:
            return None
        
        energy_tensor = torch.tensor([energy_ev], dtype=torch.float64, device=self.energies.device)
        return self.interpolator([energy_tensor]).item()

    def get_cross_section_at_energies(self, energy_evs):
        """Get cross-section values at specific energies."""
        if self.interpolator is None:
            return None

        energy_tensor = torch.tensor(energy_evs, dtype=torch.float64, device=self.energies.device)
        return self.interpolator([energy_tensor])
    
    def get_energy_and_cross_section(self):
        """Return energies and cross-sections as numpy arrays."""
        if self.energies is None or self.sigmas is None:
            return None, None
        
        return self.energies.cpu().numpy(), self.sigmas.cpu().numpy()
    
    def get_reduced_mass(self, kg):
        """Calculate the reduced mass of the reaction."""
        if self.reactants is None or len(self.reactants) != 2:
            return None
        
        m1 = self._get_particle_mass(self.reactants[0])
        m2 = self._get_particle_mass(self.reactants[1])
        
        if m1 is None or m2 is None:
            return None
        
        m_kg = (m1 * m2) / (m1 + m2)
        if kg:
            return m_kg
        return m_kg / const.u  # Convert kg to atomic mass units (amu)
    
    def get_Z1_Z2(self):
        """Get the atomic numbers of the reactants."""
        if self.reactants is None or len(self.reactants) != 2:
            return None, None
        
        z1 = PARTICLE_PROPERTIES[self.reactants[0]]['charge'] / e
        z2 = PARTICLE_PROPERTIES[self.reactants[1]]['charge'] / e
        # round to nearest integer
        z1 = int(round(z1))
        z2 = int(round(z2))
        return z1, z2

    def __str__(self):
        return f"Reaction: {self.get_human_readable_equation()} (Q = {self.q_value_mev:.2f} MeV)"
    
    def __repr__(self):
        return f"Reaction('{self.reaction_code}', Q={self.q_value_mev:.2f} MeV)"
    def copy(self):
        """Create a deep copy of the reaction."""
        # Create new tensors (copies, not references)
        new_energies = self.energies.clone() if self.energies is not None else None
        new_sigmas = self.sigmas.clone() if self.sigmas is not None else None
        new_energies_lab = self.energies_lab.clone() if hasattr(self, 'energies_lab') and self.energies_lab is not None else None
        
        # Create new interpolator if it exists
        new_interpolator = None
        if self.interpolator is not None and new_energies is not None and new_sigmas is not None:
            new_interpolator = interpolate([new_energies], new_sigmas)
        
        # Create new reaction instance
        new_reaction = Reaction(self.reaction_code, new_energies, new_sigmas, new_interpolator)
        
        # Copy lab energies if they exist
        if new_energies_lab is not None:
            new_reaction.energies_lab = new_energies_lab
            
        return new_reaction
def check_en_sigma_lengths(library):
    """Check if all reactions in the library have matching energy and sigma lengths."""
    for code, reaction in library.reactions.items():
        en_len = len(reaction.energies)
        sig_len = len(reaction.sigmas)
        if en_len != sig_len:
            print(f"Reaction {code}: Energy grid length = {en_len}")
            print(f"sigma length: {sig_len}")
            print(f"Warning: Energy and sigma lengths do not match for {code} ({en_len} vs {sig_len})")

class FusionReactionsLibrary:
    """
    A library for managing and analyzing fusion reactions from EXFOR data.
    """    
    def __init__(self,library=None, device='cuda' if torch.cuda.is_available() else 'cpu', reactions=None, use_cm_frame=True):
        self.device = device
        self.reactions = {}
        self.use_cm_frame = use_cm_frame  # Flag to enable/disable CM transformation
        print(f"Using device: {self.device}")
        print(f"Center of mass frame transformation: {'Enabled' if use_cm_frame else 'Disabled'}")
        if library is not None:
            self.device = library.device
            self.use_cm_frame = library.use_cm_frame
            # Deep copy reactions instead of shallow copy
            self.reactions = {k: v.copy() for k, v in library.reactions.items()}
        if reactions is not None:
            # Deep copy reactions instead of shallow copy
            self.reactions = {k: v.copy() for k, v in reactions.items()}
            
        check_en_sigma_lengths(self)
    
    def load_from_json_file(self, file_path):
        """Load reactions from a single JSON file."""
        try:
            with open(file_path, 'r') as f:
                json_data = f.read()
            
            self._parse_exfor_data(json_data)
            print(f"Loaded reactions from {file_path}")
            
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
    
    def load_from_folder(self, folder_path):
        """Load reactions from all JSON files in a folder."""
        folder = Path(folder_path)
        json_files = list(folder.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {folder_path}")
            return
        
        for json_file in json_files:
            self.load_from_json_file(json_file)
    
    def _extract_reaction_data(self, json_data_string, reaction_code):
        """Extract energy and cross-section data for a specific reaction."""
        all_energies, all_sigmas = [], []
        data = json.loads(json_data_string)
        
        for dataset in data.get('datasets', []):
            dataset_code = dataset.get('reaction', {}).get('code', '')
            
            if dataset_code != reaction_code:
                continue
            
            headers = dataset.get('headers', [])
            data_points = dataset.get('data', [])
            
            try:
                # energy can be expressed:
                #   - headers with 'x4Header' EN or EN-CM -> than we need to tell if we are in CM or not
                #   - headers with 'x4Header' MOM and 'BasicUnits' EV/C -> than we need to convert to EV by multiplying with the conversion factor AND speed of light
                # cross-section can be expressed:
                #   - headers with 'x4Header' SIG or SIG-CM
                
                en_header = next((h for h in headers if h.get('x4Header') in ['EN', 'EN-CM']), None)
                if en_header is None:
                    # Try to find momentum header as an alternative
                    en_header = next((h for h in headers if h.get('x4Header') == 'MOM' and h.get('BasicUnits') == 'EV/C'))
                    en_header['ConvFactor_C'] = c  # Speed of light in m/s
                    print(f"Using momentum header for reaction {reaction_code}")
                    print(f"Header: {en_header}")

                    sig_header = next((h for h in headers if h.get('varType') in ['Data', 'CS'] and h.get('BasicUnits') == 'B'))
                else:
                    sig_header = next(h for h in headers if h.get('x4Header') == 'DATA')    
            except StopIteration:
                # print(f"Warning: Missing headers for reaction {reaction_code}")
                # print(f"Available headers: {headers}")
                continue
            
            
            en_col_index = en_header.get('ii', -1) - 1
            sig_col_index = sig_header.get('ii', -1) - 1
            
            en_conv = float(en_header.get('ConvFactor', 1.0))
            en_conv *= float(en_header.get('ConvFactor_C', 1.0))
            sig_conv = float(sig_header.get('ConvFactor', 1.0))
            
            en_basicUnit = en_header.get('BasicUnits', 'error')
            sig_basicUnit = sig_header.get('BasicUnits', 'error')
            
            if en_basicUnit != 'EV' or sig_basicUnit != 'B':
                continue
            
            # Check if data is already in CM frame
            is_cm_data = en_header.get('x4Header') == 'EN-CM'
            
            for point in data_points:
                if len(point) > max(en_col_index, sig_col_index):
                    try:
                        energy = float(point[en_col_index]) * en_conv
                        sigma = float(point[sig_col_index]) * sig_conv
                        if energy > 0 and sigma > 0:  # Only positive values
                            all_energies.append(energy)
                            all_sigmas.append(sigma)
                    except (ValueError, TypeError):
                        continue
        
        if not all_energies:
            return None, None
        
        # Sort and convert to tensors
        sorted_pairs = sorted(zip(all_energies, all_sigmas))
        energies = torch.tensor([p[0] for p in sorted_pairs], device=self.device, dtype=torch.float64)
        sigmas = torch.tensor([p[1] for p in sorted_pairs], device=self.device, dtype=torch.float64)
        
        # Transform to CM frame if needed and enabled
        if self.use_cm_frame and not is_cm_data:
            # Create a temporary reaction to access transformation methods
            temp_reaction = Reaction(reaction_code)
            if temp_reaction.target_id is not None and temp_reaction.projectile_id is not None:
                energies_cm = temp_reaction._lab_to_cm_energy_relativistic(energies)
                print(f"Transformed {reaction_code} from lab to CM frame")
                if reaction_code == "2-HE-3(D,P)2-HE-4,,SIG":
                    print(f"Transformed energies length: {len(energies_cm)},\n sigmas length: {len(sigmas)}")
                return energies_cm, sigmas
        
        return energies, sigmas
    
    def _parse_exfor_data(self, json_data_string):
        """Parse EXFOR JSON data and extract reactions."""
        try:
            data = json.loads(json_data_string)
        except json.JSONDecodeError:
            print("Error: Could not decode JSON data.")
            return
        
        # Find all SIG reactions
        sig_reactions = set()
        rejected_reactions = []
        
        for dataset in data.get('datasets', []):
            code = dataset.get('reaction', {}).get('code', '')
            if ',,SIG' in code and ',,SFC' not in code and 'DERIV' not in code:
                # Quick validation check
                if self._is_parseable_reaction(code):
                    sig_reactions.add(code)
                else:
                    rejected_reactions.append(code)
        
        if rejected_reactions:
            print(f"Rejected {len(rejected_reactions)} reactions due to parsing issues:")
            for rejected in rejected_reactions[:10]:  # Show first 10
                print(f"  - {rejected}")
            if len(rejected_reactions) > 10:
                print(f"  ... and {len(rejected_reactions) - 10} more")
        
        # Process each reaction
        processed_count = 0
        for reaction_code in sig_reactions:
            energies, sigmas = self._extract_reaction_data(json_data_string, reaction_code)
            if reaction_code == "2-HE-3(D,P)2-HE-4,,SIG":
                print(f"Reaction {reaction_code}: Energy grid length = {len(energies)}")
                print(f"sigma length: {len(sigmas)}")
            
            if energies is not None and len(energies) >= 1:
                interpolator,_,_ = self._create_interpolator(energies, sigmas)

                if interpolator is not None:
                    reaction = Reaction(reaction_code, energies, sigmas, interpolator)
                    
                    # Only add if we have valid particle IDs
                    if reaction.reactants is not None and reaction.products is not None:
                        self.reactions[reaction_code] = reaction
                        processed_count += 1
                    else:
                        print(f"Warning: Reaction {reaction_code} has invalid reactants or products, skipping")
            else:
                print(f"Warning: No valid data for reaction {reaction_code}, skipping")
        
        print(f"Successfully processed {processed_count} reactions")
        if self.use_cm_frame:
            print("All energies have been transformed to center of mass frame")
    
    def _is_parseable_reaction(self, reaction_code):
        """Quick check if a reaction code can be parsed."""
        code = reaction_code.split(',,')[0]
        
        # Check for obvious malformations
        if code.count('(') > code.count(')'):
            return False
        if code.startswith('(') and not code.endswith(')'):
            return False
        
        # Try a simple parse test
        try:
            # Clean the code
            if code.startswith('('):
                code = code[1:]
            
            # Must contain parentheses for standard reactions
            if '(' not in code:
                return False
            
            return True
        except:
            return False
    
    def _create_interpolator(self, energies, sigmas):
        """Create interpolator from energy and cross-section data."""
 
        interpolator = interpolate([energies], sigmas)
        return interpolator, energies, sigmas
    
    def get_reaction(self, reaction_code):
        """Get a specific reaction by code."""
        return self.reactions.get(reaction_code)
    
    def list_reactions(self, sort_by='name'):
        """List all available reactions."""
        if sort_by == 'q_value':
            sorted_reactions = sorted(self.reactions.items(), 
                                    key=lambda x: x[1].q_value_mev, reverse=True)
        else:
            sorted_reactions = sorted(self.reactions.items())
        
        for code, reaction in sorted_reactions:
            print(f"{code}: {reaction.get_human_readable_equation()} (Q = {reaction.q_value_mev:.2f} MeV)")
    
    def filter_reactions(self, **criteria):
        """Filter reactions based on criteria. -> return FusionReactionsLibrary class with filtered reactions.
        Filter by reactants, products, any of the particles, Q-value or max cross-section [keV].
        Example criteria: {'reactants_exact': ['p', 'd'], 'q_value_min': 0.0, 'max_cross_section_keV': 1000.0}
        possible criteria:
        - 'particles': list of particle names that must be present in reactants or products
        - 'particles_not': list of particle names that must not be present in reactants or products
        - 'reactants': list of reactant particle names that must be present
        - 'products': list of product particle names that must be present
        - 'reactants_exact': list of reactant particle names that must be present exactly
        - 'products_exact': list of product particle names that must be present exactly
        - 'q_value_min': minimum Q-value in MeV
        - 'q_value_max': maximum Q-value in MeV
        - 'max_cross_section_barns': maximum cross-section in barns
        """
        # reaction.reactants and reaction.products are tuples of particle IDs -> to filter by particle names, convert to IDs
        filtered = {}
        # Don't copy the original dictionary - work directly with self.reactions
        
        for code, reaction in self.reactions.items():
            match = True
            
            # check particles that must be present in reactants or products
            if 'particles' in criteria:
                particle_ids = [COMMON_NAME_TO_ID.get(p) for p in criteria['particles']]
                if not any(pid in reaction.reactants or pid in reaction.products for pid in particle_ids):
                    match = False
            
            # check particles that must not be present in reactants or products
            if 'particles_not' in criteria:
                particle_ids_not = [COMMON_NAME_TO_ID.get(p) for p in criteria['particles_not']]
                if any(pid in reaction.reactants or pid in reaction.products for pid in particle_ids_not):
                    match = False
            
            # Check reactants
            if 'reactants' in criteria:
                reactant_ids = [COMMON_NAME_TO_ID.get(r) for r in criteria['reactants']]
                if not any(r in reaction.reactants for r in reactant_ids):
                    match = False
            
            # Check products
            if 'products' in criteria:
                product_ids = [COMMON_NAME_TO_ID.get(p) for p in criteria['products']]
                if not any(p in reaction.products for p in product_ids):
                    match = False
                        # Check reactants
            if 'reactants_exact' in criteria:
                reactant_ids = [COMMON_NAME_TO_ID.get(r) for r in criteria['reactants_exact']]
                if not all(r in reaction.reactants for r in reactant_ids):
                    match = False
            
            # Check products
            if 'products_exact' in criteria:
                product_ids = [COMMON_NAME_TO_ID.get(p) for p in criteria['products_exact']]
                if not all(p in reaction.products for p in product_ids):
                    match = False
            
            # Check Q-value range
            if 'q_value_min' in criteria and reaction.q_value_mev < criteria['q_value_min']:
                match = False
            if 'q_value_max' in criteria and reaction.q_value_mev > criteria['q_value_max']:
                match = False
            
            # Check max cross-section
            if 'max_cross_section_barns' in criteria:
                max_cross_section = criteria['max_cross_section_barns'] # in barns
                if reaction.sigmas is None or torch.max(reaction.sigmas).item() > max_cross_section:
                    match = False
            
            if match:
                # Create a proper deep copy of the reaction
                filtered[code] = reaction.copy()

            
        if filtered:
            print(f"Filtered {len(filtered)} reactions based on criteria: {criteria}")
        else:
            print("No reactions matched the filtering criteria")
        # Pass reactions directly instead of using library parameter
        return FusionReactionsLibrary(device=self.device, reactions=filtered, use_cm_frame=self.use_cm_frame)
    
    def unify_reactions(self):
        """Unify reactions by merging exactly the same ones based on reactants and products."""
        unified_reactions = {}
        # if a reaction has the same reactants and products, merge them
        for reaction in self.reactions.values():
                # Create a unique key based on reactants and products
            key = (tuple(sorted(reaction.reactants)), tuple(sorted(reaction.products)))
            # If this is the first time we see this key, add the reaction
            if key not in unified_reactions:
                unified_reactions[key] = reaction.copy()
            else:
                # Merge data if already exists - work with copies to avoid reference issues
                existing_reaction = unified_reactions[key]
                # Create new tensors by concatenation
                combined_energies = torch.cat((existing_reaction.energies, reaction.energies))
                combined_sigmas = torch.cat((existing_reaction.sigmas, reaction.sigmas))
                
                # Sort by energy to maintain proper ordering
                sorted_indices = torch.argsort(combined_energies)
                combined_energies = combined_energies[sorted_indices]
                combined_sigmas = combined_sigmas[sorted_indices]
                
                # Update the existing reaction with new data
                existing_reaction.energies = combined_energies
                existing_reaction.sigmas = combined_sigmas
                existing_reaction.interpolator = interpolate([combined_energies], combined_sigmas)
                
                new_length = combined_energies.shape[0]
                print(f"Merging reaction: {existing_reaction.get_human_readable_equation()} with {reaction.get_human_readable_equation()}, new length: {new_length}")

        # Replace reactions with unified ones, using proper deep copies
        self.reactions = {r.get_human_readable_equation(): r for r in unified_reactions.values()}

    def plot_reaction(self, reaction_code, fig=None, ax=None, show_errors=False, log_scale=True, save_path=None, show_data_points=True, show_interpolation=True, use_lab_frame=False):
        """Plot the cross-section data for a specific reaction."""
        if reaction_code not in self.reactions:
            print(f"Reaction {reaction_code} not found in library")
            return None, None
            
        reaction = self.reactions[reaction_code]
        
        if reaction.energies is None or reaction.interpolator is None:
            print(f"No data available for plotting reaction: {reaction_code}")
            return None, None
        
        # Create new figure if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            new_plot = True
        else:
            new_plot = False
        
        # Choose which energies to plot
        if use_lab_frame and hasattr(reaction, 'energies_lab') and reaction.energies_lab is not None:
            energies_device = reaction.energies_lab.to(self.device)
            frame_label = "Lab"
        else:
            energies_device = reaction.energies.to(self.device)
            frame_label = "CM" if self.use_cm_frame else "Lab"
        
        energies_kev = (energies_device / 1000).cpu().numpy()
        
        # Get interpolated values on device, then transfer to CPU
        sigmas_device = reaction.interpolator([energies_device])
        sigmas_data = sigmas_device.cpu().numpy()
        
        # Create label for this reaction
        reaction_label = reaction.get_human_readable_equation()
        
        # Plot original data points
        if show_data_points:
            ax.plot(energies_kev, sigmas_data, 'o', label=f'{reaction_label} (Data, {frame_label})', markersize=4, alpha=0.7)
        
        # Plot interpolated curve
        if show_interpolation:
            e_min, e_max = energies_device.min(), energies_device.max()
            interp_energies_device = torch.logspace(torch.log10(e_min), torch.log10(e_max), 500).to(self.device, dtype=torch.float64)
            interp_sigmas_device = reaction.interpolator([interp_energies_device])
            
            # Transfer to CPU for plotting
            interp_energies_kev = (interp_energies_device / 1000).cpu().numpy()
            interp_sigmas = interp_sigmas_device.cpu().numpy()
            
            line_label = f'{reaction_label} ({frame_label})' if not show_data_points else f'{reaction_label} (Fit, {frame_label})'
            ax.plot(interp_energies_kev, interp_sigmas, '-', label=line_label, linewidth=2)
        
        # Apply formatting only if this is a new plot
        if new_plot:
            ax.set_xlabel(f'Energy ({frame_label} Frame, keV)')
            ax.set_ylabel('Cross Section (barns)')
            ax.set_title(f'{reaction.get_human_readable_equation()}\nQ = {reaction.q_value_mev:.2f} MeV ({frame_label} Frame)')
            
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
            
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.7)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
        
        return fig, ax
    
    def plot_all_reactions(self, save_folder=None, **plot_kwargs):
        """Plot all reactions individually."""
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
        
        for code, reaction in self.reactions.items():
            save_path = None
            if save_folder:
                safe_filename = code.replace('/', '_').replace(',', '_')
                save_path = os.path.join(save_folder, f"{safe_filename}.png")
            
            self.plot_reaction(code, save_path=save_path, **plot_kwargs)
    
    def plot_all_reactions_on_one_plot(self, save_path=None, log_scale=True, show_data_points=True, show_interpolation=False, max_reactions=None, **plot_kwargs):
        """Plot all reactions on a single figure."""
        if not self.reactions:
            print("No reactions to plot")
            return None, None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        reaction_codes = list(self.reactions.keys())
        # sort reaction codes according to first reactant
        reaction_codes.sort(key=lambda code: self.reactions[code].reactants[0] if self.reactions[code].reactants else '')
        
        if max_reactions is not None:
            reaction_codes = reaction_codes[:max_reactions]
            print(f"Limiting to first {max_reactions} reactions")
        
        for code in reaction_codes:
            self.plot_reaction(code, fig=fig, ax=ax, 
                             show_data_points=show_data_points, 
                             show_interpolation=show_interpolation,
                             **plot_kwargs)
        
        # Apply formatting for the combined plot
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Cross Section (barns)')
        ax.set_title(f'Fusion Cross-Section Comparison ({len(reaction_codes)} reactions)')
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # Handle legend for many reactions
        if len(reaction_codes) <= 25:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            print(f"Too many reactions ({len(reaction_codes)}) for legend display")
        
        ax.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined plot saved to {save_path}")
        else:
            plt.show()
        
        return fig, ax
    
    def get_reactions(self):
        """Get all reactions in the library."""
        return self.reactions
    
    def get_reaction_codes(self):
        """Get all reaction codes in the library."""
        return list(self.reactions.keys())
    
    def list_reaction_codes(self):
        """List all reaction codes in the library."""
        for code in sorted(self.reactions.keys()):
            print(code)
    
    def get_statistics(self):
        """Get library statistics."""
        stats = {
            'total_reactions': len(self.reactions),
            'q_value_range': (0, 0),
            'particle_types': set(),
            'energy_ranges': {}
        }
        
        if self.reactions:
            q_values = [r.q_value_mev for r in self.reactions.values()]
            stats['q_value_range'] = (min(q_values), max(q_values))
            
            for reaction in self.reactions.values():
                stats['particle_types'].update([reaction.target, reaction.projectile, 
                                              reaction.emitted, reaction.residual])
                
                if reaction.energies is not None:
                    e_min = reaction.energies.min().item() / 1000  # keV
                    e_max = reaction.energies.max().item() / 1000  # keV
                    stats['energy_ranges'][reaction.reaction_code] = (e_min, e_max)
        
        return stats
    
    def print_summary(self):
        """Print a summary of the library."""
        stats = self.get_statistics()
        
        print(f"\n{'='*50}")
        print(f"FUSION REACTIONS LIBRARY SUMMARY")
        print(f"{'='*50}")
        print(f"Total reactions: {stats['total_reactions']}")
        print(f"Q-value range: {stats['q_value_range'][0]:.2f} to {stats['q_value_range'][1]:.2f} MeV")
        print(f"Particle types: {', '.join(sorted(filter(None, stats['particle_types'])))}")
        print(f"Device: {self.device}")
        print(f"{'='*50}\n")
    
    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, reaction_code):
        return self.reactions[reaction_code]
    
    def __iter__(self):
        return iter(self.reactions.items())
    
    def __contains__(self, reaction_code):
        return reaction_code in self.reactions
    
    def copy(self):
        """Create a copy of the library."""
        # Create deep copies of all reactions
        reactions_copy = {k: v.copy() for k, v in self.reactions.items()}
        return FusionReactionsLibrary(device=self.device, reactions=reactions_copy, use_cm_frame=self.use_cm_frame)
    
# Add enhanced particle name mapping for better coverage
ENHANCED_TRANSLATION_MAP = {
    **TRANSLATION_MAP,
    # Additional mappings for common variations
    'HE3': 'He-3',
    'HE-3': 'He-3', 
    'HE4': 'alpha',
    'HE-4': 'alpha',
    'TOT': 'TOT',  # Total cross section
    'EL': 'EL',    # Elastic scattering
}

# Update the format_nuclide function to use enhanced mapping
def format_nuclide_enhanced(code):
    """Enhanced version of format_nuclide with better coverage."""
    # Check enhanced map first
    if code in ENHANCED_TRANSLATION_MAP:
        return ENHANCED_TRANSLATION_MAP[code]
    
    # Original logic
    parts = code.split('-')
    if len(parts) == 3:
        z, symbol, a = parts
        return f"{symbol}-{a}"
    elif len(parts) == 2:
        symbol, a = parts
        return f"{symbol}-{a}"
        
    return code

#%%
# Example usage and testing
if __name__ == "__main__":
    # Create library instance
    library = None
    if library is None:
        library = FusionReactionsLibrary()
        library.load_from_json_file("All_reactions.json")
    check_en_sigma_lengths(library)
    library.print_summary()
    library.list_reactions(sort_by='q_value')
    
    print("Fusion Reactions Library loaded successfully!")
    #%%

    #%%
    libraryCP = library.copy()

    libraryCP.print_summary()
    libraryCP.plot_all_reactions_on_one_plot(save_path="fusion_reactions_all.png", log_scale=True, show_data_points=True, show_interpolation=True)
    libraryCP.unify_reactions()
    libraryCP.plot_all_reactions_on_one_plot(save_path="fusion_reactions_all_unified.png", log_scale=True, show_data_points=True, show_interpolation=True)
    libraryCP.print_summary()
    #%%
    he3d_library = library.filter_reactions(reactants_exact=['He-3', 'd'])
    he3d_library.plot_all_reactions_on_one_plot(show_interpolation=True)


    #%%
    def plot_reaction(**filter):
        filtered_reactions = library.filter_reactions(**filter)
        filtered_reactions.unify_reactions()
        # for code, reaction in filtered_reactions.reactions.items():
            # print(f"Plotting reaction: {code}")
            # if reaction.sigmas.max() < 1.:
                # reaction.sigmas *= 1e1  # Convert to barns if needed
        filtered_reactions.plot_all_reactions_on_one_plot(show_interpolation=True, log_scale=True, show_data_points=True)

    plot_reaction(reactants=['t'])
    #%%

    # library.list_reactions(sort_by='q_value')

    # dt_reactions = library.filter_reactions(particles=['d', 't'])
    dt_library = library.filter_reactions(reactants_exact=['d', 't'])
    dt_library.plot_all_reactions_on_one_plot()
    dt_library.list_reactions(sort_by='q_value')
    # print length of energy grid for each reaction
    for code, reaction in dt_library.reactions.items():
        print(f"Reaction {code}: Energy grid length = {len(reaction.energies)}")
    dt_library.unify_reactions()

    for code, reaction in dt_library.reactions.items():
        print(f"Reaction {code}: Energy grid length = {len(reaction.energies)}")

    dt_library.list_reactions(sort_by='q_value')
    dt_library.plot_all_reactions_on_one_plot()



# library.plot_all_reactions_on_one_plot(save_path="fusion_reactions_comparison.png", log_scale=True, show_data_points=True, show_interpolation=True)
