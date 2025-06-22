# Repository Structure

```
pc_rules_engine/
├── configs/            # YAML configuration files
├── logs/               # Directory for log files
├── midi_files/         # Directory for generated MIDI files
├── pc_sets/
│   ├── __init__.py
│   ├── pitch_classes.py   # Core pitch class theory implementation
│   ├── engine.py          # Generation engine implementation
│   └── rhythm.py          # Rhythm generation and manipulation module
├── midi/
│   ├── __init__.py
│   └── translator.py      # MIDI translation functionality
├── utils/
│   ├── __init__.py
│   └── logging_setup.py   # Logging configuration
├── analysis/           # Data analysis tools and visualization notebooks
│   ├── weight_correlation_analysis.md         # Detailed analysis of weight correlations
│   └── weight_correlation_visualization.ipynb # Interactive visualization notebook
├── docs/               # Documentation files
├── run_engine.py       # Command-line interface
├── monte_carlo.py      # Monte Carlo simulation framework
└── README.md           # Main repository documentation
```

## Key Directories

### configs/
Contains YAML configuration files for different types of musical sequences, including:
- `melodic_basic.yaml`: Simple melodic sequence from C to G
- `chord_progression.yaml`: Chord progression from C major to G major
- `random_walk.yaml`: Random melodic sequence with high variability
- `jazz_progression.yaml`: Jazz progression with 7th chords
- `static_minor_chord.yaml`: C minor chords with subtle variations

### pc_sets/
Core implementation of pitch class theory and the generation engine.

### midi/
Translation tools for converting pitch class sequences to MIDI format.

### analysis/
Tools and notebooks for analyzing and visualizing Monte Carlo simulation results.

### docs/
Comprehensive documentation covering all aspects of the system.
