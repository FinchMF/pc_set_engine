# Requirements

## Core Requirements

- Python 3.7+
- NumPy
- Mido (for MIDI generation)
- PyYAML (for configuration files)
- tqdm (for progress bars)

## Optional Requirements

- pandas (for data analysis)
- matplotlib (for visualization)
- seaborn (for advanced visualization)
- jupyter (for interactive notebooks)
- pytest (for running tests)
- mypy (for type checking)

## Installation

### Using pip

```bash
# Install core requirements
pip install numpy mido pyyaml tqdm

# Install optional requirements
pip install pandas matplotlib seaborn jupyter pytest mypy
```

### Using Environment File

```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate pc_rules_engine
```

## Compatibility Notes

- MIDI generation requires mido and python-rtmidi for real-time playback
- Visualization features require matplotlib 3.3+ for optimal display
- Large-scale Monte Carlo simulations benefit from a multi-core processor
- Analysis notebooks work best with Jupyter Lab or VS Code Jupyter support
