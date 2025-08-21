# Fusion Reactions Library

A comprehensive Python library for managing and analyzing nuclear fusion reactions from EXFOR data. This library provides tools for loading, processing, filtering, and visualizing fusion cross-section data with support for relativistic energy transformations between laboratory and center-of-mass frames.

## Features

- **Comprehensive Particle Support**: Handles various nuclear particles including hydrogen isotopes (p, d, t), helium isotopes (He-3, He-4), and heavier nuclei up to carbon
- **EXFOR Data Integration**: Parses and processes nuclear reaction data from EXFOR JSON format
- **Relativistic Transformations**: Full relativistic treatment for energy transformations between lab and center-of-mass frames
- **Advanced Filtering**: Filter reactions by reactants, products, Q-values, and cross-section magnitudes
- **Visualization Tools**: Plot individual reactions or compare multiple reactions on the same figure
- **GPU Support**: Leverages PyTorch for efficient computation with CUDA support
- **Data Interpolation**: Smooth interpolation of cross-section data for any energy value

## Installation

### Requirements
- [Interpolation Library](https://github.com/sbarratt/torch_interpolations)
- PyTorch
- NumPy
- Matplotlib
- SciPy
- JSON (standard library)

### Setup
```bash
git clone https://github.com/FilipO28555/Fusion_reactions_lib.git
cd Fusion_reactions_lib
```

## Quick Start
You can download your own EXFOR JSON data file, for example from here:
[https://www-nds.iaea.org/exfor/](https://www-nds.iaea.org/exfor/)

```python
from fusion_reactions_lib import FusionReactionsLibrary

# Create library instance
library = FusionReactionsLibrary()

# Load reactions from EXFOR JSON data
library.load_from_json_file("All_reactions.json")

# Print summary
library.print_summary()

# List all available reactions
library.list_reactions(sort_by='q_value')

# Filter reactions (e.g., deuterium-tritium fusion)
dt_reactions = library.filter_reactions(reactants_exact=['d', 't'])

# list available reaction codes
dt_reactions.list_reaction_codes()
# get all reaction codes
reaction_codes = dt_reactions.get_reaction_codes()
# Plot a specific reaction
reaction_code = reaction_codes[0]
print(f"using reaction: {reaction_code}")
dt_reactions.plot_reaction(reaction_code)
# or like this
dt_reactions.plot_reaction(dt_reactions.get_reaction_codes()[0])
# or like this
library.plot_reaction(dt_reactions.get_reaction_codes()[0])

# Get cross-section at specific energy
reaction = dt_reactions.get_reaction(reaction_code)
cross_section = reaction.get_cross_section_at_energy(6e4)  # 60 keV
print(f"Cross-section at 60 keV: {cross_section:.2e} barns")

energies = torch.linspace(0, 1e6, 100)  # Energies from 0 to 1 MeV
cross_sections = reaction.get_cross_section_at_energies(energies).cpu().numpy()
```

## Key Classes

### `FusionReactionsLibrary`
Main class for managing collections of fusion reactions.

**Key Methods:**
- `load_from_json_file(file_path)`: Load reactions from EXFOR JSON
- `filter_reactions(**criteria)`: Filter reactions by various criteria
- `plot_reaction(reaction_code)`: Visualize specific reactions
- `get_statistics()`: Get library statistics

### `Reaction`
Represents individual nuclear reactions with their properties and data.

**Key Properties:**
- `energies`: Energy grid (in eV)
- `sigmas`: Cross-section values (in barns)
- `q_value_mev`: Reaction Q-value in MeV
- `reactants`, `products`: Particle species involved

**Key Methods:**
- `get_cross_section_at_energy(energy_ev)`: Get cross-section at specific energy
- `get_human_readable_equation()`: Get reaction equation in readable format

## Filtering Criteria

The library supports sophisticated filtering:

```python
# Filter by particles involved
fusion_reactions = library.filter_reactions(
    particles=['d', 't'],  # Must contain deuterium or tritium
    q_value_min=0.0,       # Exothermic reactions only
    max_cross_section_barns=10.0  # Maximum cross-section limit
)

# Filter by exact reactants
dt_fusion = library.filter_reactions(
    reactants_exact=['d', 't']  # Exactly deuterium-tritium reactions
)

# Exclude specific particles
no_neutrons = library.filter_reactions(
    particles_not=['n']  # Exclude reactions producing neutrons
)
```

## Energy Frame Transformations

The library automatically handles relativistic transformations between laboratory and center-of-mass frames:

```python
# Enable/disable CM frame transformation
library = FusionReactionsLibrary(use_cm_frame=True)

# Plot in different frames
library.plot_reaction(reaction_code, use_lab_frame=False)  # CM frame
library.plot_reaction(reaction_code, use_lab_frame=True)   # Lab frame
```

## Supported Particles

The library includes comprehensive support for:

- **Fundamental particles**: electrons, protons, neutrons, gamma rays
- **Hydrogen isotopes**: protium (p), deuterium (d), tritium (t)
- **Helium isotopes**: He-3, He-4 (alpha)
- **Light nuclei**: Li-6, Li-7, Be-7, Be-9, B-10, B-11, C-12, C-13

## Data Visualization

### Single Reaction Plot
```python
library.plot_reaction("2-HE-3(D,P)2-HE-4,,SIG", 
                     log_scale=True, 
                     show_data_points=True, 
                     show_interpolation=True)
```

### Multiple Reactions Comparison
```python
library.plot_all_reactions_on_one_plot(
    max_reactions=10,
    show_data_points=False,
    show_interpolation=True
)
```

## Physical Constants and Properties

The library uses accurate physical constants from SciPy and includes:
- Particle masses (from CODATA values)
- Charges and other properties
- Atomic mass unit conversions
- Speed of light and fundamental constants

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source. Please check the license file for details.

## Citation

If you use this library in your research, please cite:
```
Fusion Reactions Library
FilipO28555
https://github.com/FilipO28555/Fusion_reactions_lib
```

## Contact

For questions or support, please open an issue on the GitHub repository.
