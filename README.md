# Pitch Class Rules Engine

A modular system for generating and manipulating pitch class sequences in musical composition, offering both melodic generation and chord progression capabilities with configurable randomness and variation.

## Overview

This repository contains a rule-based engine for creating musical sequences using pitch class theory. The system can generate:

- Melodic sequences (single-note progressions)
- Chord sequences (multiple pitches at once)
- Progressive transformations from one pitch class (set) to another
- Random walks with controlled variability
- Static presentations with controlled variations

The engine is highly configurable, allowing for fine-tuned control over the level of randomness, variation probability, and progression characteristics.

## Key Features

- **Configuration-Driven Workflow**: Use YAML files to define complex musical progressions
- **Melodic and Chordal Generation**: Create single-note sequences or chord progressions
- **Directed Progressions**: Smoothly transition from one pitch class (set) to another
- **Randomness Controls**: Fine-tune the balance between predictability and variation
- **MIDI Output**: Generate playable MIDI files from pitch class sequences
- **Interval Vector Weighting**: Prioritize specific intervals to influence harmonic character
- **Monte Carlo Analysis**: Study correlations between parameters and musical characteristics
- **Comprehensive Documentation**: Well-documented code with usage examples
- **Modular Design**: Easily extend with new operations and features

## Repository Structure

```
pc_rules_engine/
├── configs/            # YAML configuration files
├── logs/               # Directory for log files
├── midi_files/         # Directory for generated MIDI files
├── pc_sets/
│   ├── __init__.py
│   ├── pitch_classes.py   # Core pitch class theory implementation
│   └── engine.py          # Generation engine implementation
├── midi/
│   ├── __init__.py
│   └── translator.py      # MIDI translation functionality
├── utils/
│   ├── __init__.py
│   └── logging_setup.py   # Logging configuration
├── run_engine.py          # Command-line interface
├── monte_carlo.py         # Monte Carlo simulation framework
└── README.md              # This documentation
```

## Configuration-Driven Workflow

The engine supports a configuration-driven workflow through YAML files, enabling complex musical progressions without writing code:

### 1. Choose or Create a Configuration

Select one of the predefined configurations:
- `melodic_basic.yaml`: Simple melodic sequence from C to G
- `chord_progression.yaml`: Chord progression from C major to G major
- `random_walk.yaml`: Random melodic sequence with high variability
- `jazz_progression.yaml`: Jazz progression with 7th chords
- `static_minor_chord.yaml`: C minor chords with subtle variations

Or create your own by copying and modifying an existing one.

### 2. Run the Engine

```bash
python run_engine.py --config-file configs/chord_progression.yaml --midi output.mid
```

### 3. Customize on the Fly

Override specific configuration parameters from the command line:

```bash
python run_engine.py --config-file configs/melodic_basic.yaml --sequence-length 16 --randomness 0.7
```

## Components

### Pitch Class Module (`pitch_classes.py`)

- `PitchClass`: Represents individual pitch classes (0-11)
- `PitchClassSet`: Implements pitch class sets with various operations:
  - Normal form and prime form calculation
  - Forte number identification
  - Interval vector calculation
  - Weighted interval vectors and similarity measurements
  - Set operations (transpose, invert, complement, etc.)
- `COMMON_SETS`: Dictionary of predefined common musical pitch class sets
- `INTERVAL_WEIGHT_PROFILES`: Predefined weightings for different musical styles

### Engine Module (`engine.py`)

- `GenerationType`: Defines generation types (melodic or chordal)
- `ProgressionType`: Defines progression types (static, directed, random)
- `GenerationConfig`: Configuration data class for controlling generation parameters
- `PitchClassEngine`: Main engine class that handles generation of sequences
  - Supports directed progressions toward a target
  - Implements multiple transformation strategies
  - Controls randomness and variation

### MIDI Module (`midi/translator.py`)

- Convert pitch class sequences to MIDI files for playback and further processing
- Support for both melodic and chordal sequences
- Customizable parameters (tempo, octave, note duration, etc.)
- Functions to add rhythmic patterns to sequences
- Ability to load sequences from JSON files and convert to MIDI

### Runner Module (`run_engine.py`)

- Provides a command-line interface to the engine
- Includes preset configurations for different generation types
- Handles saving and displaying results
- Supports direct MIDI file generation
- Support for YAML configuration files

### Monte Carlo Simulator (`monte_carlo.py`)

- Run large-scale simulations with systematically varied parameters
- Generate datasets of sequences for analysis and experimentation
- Two main modes of operation:
  - General Monte Carlo: Random parameter exploration
  - Parameter Variation: Systematic study of specific parameter effects
- Support for interval weighting configurations to influence generation
- Analysis of correlations between interval weights and musical characteristics
- Support for parallel processing to speed up large simulations
- Statistical analysis of generated sequences
- Export data to JSON and CSV for further analysis

## Usage Examples

### Using YAML Configuration Files

```bash
# Generate a sequence using a configuration file
python run_engine.py --config-file configs/melodic_basic.yaml

# Override configuration parameters
python run_engine.py --config-file configs/jazz_progression.yaml --sequence-length 12

# Generate both JSON and MIDI output
python run_engine.py --config-file configs/chord_progression.yaml --output result.json --midi result.mid
```

### Using Command-Line Options

```bash
# Generate a melodic sequence with default parameters
python run_engine.py --config-type melodic

# Generate a chord progression from C major to G major
python run_engine.py --config-type chord-progression

# Generate a highly random melodic sequence
python run_engine.py --config-type melodic --randomness 0.8 --variation 0.7

# Generate a longer sequence with MIDI output
python run_engine.py --config-type random-walk --sequence-length 16 --midi output.mid
```

### Using the API Programmatically

```python
from pc_sets.engine import generate_sequence_from_config

# Define a custom configuration
config = {
    "start_pc": [0, 3, 7],  # C minor
    "generation_type": "chordal",
    "sequence_length": 8,
    "progression": True,
    "target_pc": [7, 10, 2],  # G minor
    "progression_type": "directed",
    "randomness_factor": 0.4,
    "variation_probability": 0.5
}

# Generate the sequence
sequence = generate_sequence_from_config(config)

# Convert to MIDI
from midi.translator import sequence_to_midi
sequence_to_midi(sequence, "progression.mid", is_melodic=False)
```

### Running Monte Carlo Simulations

```bash
# Run a basic Monte Carlo simulation with 100 iterations
python monte_carlo.py --num-simulations 100

# Use a specific base configuration file
python monte_carlo.py --num-simulations 50 --base-config configs/melodic_basic.yaml

# Run a parameter variation study
python monte_carlo.py --variation-mode --param-name randomness_factor --param-min 0.1 --param-max 0.9 --param-steps 9 --base-config configs/chord_progression.yaml
```

Analyzing the results:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the Monte Carlo dataset
with open('datasets/monte_carlo_dataset.json', 'r') as f:
    data = json.load(f)

# Load the statistics CSV into a pandas DataFrame
stats = pd.read_csv('datasets/monte_carlo_stats.csv')

# Plot the relationship between randomness factor and interval size
plt.scatter(stats['randomness_factor'], stats['mean_interval'])
plt.xlabel('Randomness Factor')
plt.ylabel('Mean Interval Size')
plt.title('Effect of Randomness on Interval Size')
plt.savefig('randomness_vs_intervals.png')
```

### Using Interval Weights for Character Control

```python
from pc_sets.engine import generate_sequence_from_config
from pc_sets.pitch_classes import INTERVAL_WEIGHT_PROFILES

# Using a predefined interval weight profile
config = {
    "start_pc": [0, 4, 7],  # C major
    "generation_type": "chordal",
    "sequence_length": 8,
    "progression": True,
    "progression_type": "random",
    "randomness_factor": 0.4,
    "interval_weights": "jazzy"  # Use predefined profile
}

# Or define custom interval weights
config_custom = {
    "start_pc": [0, 3, 7],  # C minor
    "generation_type": "chordal",
    "sequence_length": 8,
    "interval_weights": {
        1: 0.8,  # minor 2nds/major 7ths
        2: 1.2,  # major 2nds/minor 7ths
        3: 1.5,  # minor 3rds/major 6ths
        4: 0.9,  # major 3rds/minor 6ths
        5: 1.1,  # perfect 4ths/5ths
        6: 0.7   # tritones
    }
}

# Generate the sequence with weighted intervals
sequence = generate_sequence_from_config(config)
```

### Monte Carlo Simulations with Interval Weights

```bash
# Run a Monte Carlo simulation with a specific interval profile
python monte_carlo.py --num-simulations 50 --base-config configs/chord_progression.yaml --interval-weight-profile consonant

# Generate random interval weights for each simulation
python monte_carlo.py --num-simulations 50 --random-weights --weight-variation 0.7

# Cycle through all available interval weight profiles
python monte_carlo.py --num-simulations 50 --weight-cycling

# Generate a correlation report between weights and musical features
python monte_carlo.py --num-simulations 100 --random-weights --correlation-report
```

### Analyzing Weight Correlations

```python
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the weight correlation report
with open('datasets/weight_correlation_report.json', 'r') as f:
    correlations = json.load(f)

# Create a heatmap of interval correlations for melodic features
melodic_corr = correlations['melodic']
corr_data = []
for metric, values in melodic_corr.items():
    row = [values[f'interval_{i}'] for i in range(1, 7)]
    corr_data.append(row)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_data, 
    annot=True, 
    xticklabels=['min 2nd', 'maj 2nd', 'min 3rd', 'maj 3rd', 'P4/P5', 'tritone'],
    yticklabels=list(melodic_corr.keys()),
    cmap='coolwarm'
)
plt.title('Correlations: Interval Weights vs. Melodic Features')
plt.tight_layout()
plt.savefig('interval_correlations.png')
```

## Advanced Examples

### Create a Custom Configuration File

```yaml
# my_config.yaml
# Generation properties
generation_type: chordal
start_pc: [0, 4, 7, 11]  # Cmaj7
progression: true
progression_type: random
sequence_length: 8

# Randomness controls
randomness_factor: 0.6
variation_probability: 0.7

# Constraints
constraints:
  vary_chord: true

# MIDI properties
midi_properties:
  tempo: 90
  base_octave: 4
  note_duration: 1.0
```

```bash
python run_engine.py --config-file my_config.yaml --midi output.mid
```

### Generate a Melodic Line with Custom Rhythm

```bash
# First generate the sequence and save to JSON
python run_engine.py --config-file configs/melodic_basic.yaml --output melody.json

# Then add rhythm and convert to MIDI
python -c "
from midi.translator import load_and_convert_sequence, enhance_with_rhythm, sequence_to_midi_with_rhythm
import json

# Load the sequence
with open('melody.json', 'r') as f:
    data = json.load(f)
sequence = data['sequence']

# Define a rhythm pattern
rhythm = [1, 0.5, 0.5, 1, 0.5, 0.5, 2]

# Generate MIDI with rhythm
sequence_to_midi_with_rhythm(sequence, 'melody_rhythmic.mid', rhythm, is_melodic=True)
"
```

### Large-Scale Parameter Exploration

```python
from monte_carlo import MonteCarloSimulator

# Create a simulator with custom parameter ranges
simulator = MonteCarloSimulator(
    num_simulations=200,
    base_config_file="configs/melodic_basic.yaml",
    param_ranges={
        "randomness_factor": (0.0, 1.0),
        "variation_probability": (0.0, 1.0),
        "sequence_length": (4, 24)
    },
    output_dir="exploration_results"
)

# Run the simulation with parallel processing
dataset = simulator.run(parallel=True, max_workers=8)

# Save the dataset and export statistics
simulator.save_dataset("full_exploration.json")
simulator.export_stats_to_csv("exploration_stats.csv")
```

### Studying the Effect of a Single Parameter

```python
from monte_carlo import generate_variations_dataset

# Study how randomness affects melodic contour
generate_variations_dataset(
    base_config_file="configs/melodic_basic.yaml",
    param_name="randomness_factor",
    values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    samples_per_value=5,
    output_dir="randomness_study"
)
```

### Creating Custom Interval Weight Profiles

```yaml
# configs/my_weight_profile.yaml
# Emphasize minor thirds and perfect fourths/fifths
interval_weights:
  1: 0.6   # de-emphasize minor 2nds/major 7ths
  2: 0.8   # slightly de-emphasize major 2nds/minor 7ths
  3: 1.7   # strongly emphasize minor 3rds/major 6ths
  4: 0.9   # slightly de-emphasize major 3rds/minor 6ths
  5: 1.5   # emphasize perfect 4ths/5ths
  6: 0.4   # strongly de-emphasize tritones
```

```python
# Load custom weight profile and use it
import yaml
from pc_sets.engine import generate_sequence_from_config

with open('configs/my_weight_profile.yaml', 'r') as f:
    weight_config = yaml.safe_load(f)

config = {
    "start_pc": [0, 3, 7],  # C minor
    "generation_type": "chordal",
    "sequence_length": 8,
    "progression_type": "random",
    "interval_weights": weight_config["interval_weights"]
}

sequence = generate_sequence_from_config(config)
```

### Exploring Different Harmonic Characters with Weights

```python
from pc_sets.pitch_classes import PitchClassSet, INTERVAL_WEIGHT_PROFILES
from pc_sets.engine import generate_sequence_from_config

# Define two different chord sets
cmajor = PitchClassSet([0, 4, 7])
cminor = PitchClassSet([0, 3, 7])

# Compare their weighted similarity with different profiles
for profile_name, weights in INTERVAL_WEIGHT_PROFILES.items():
    similarity = cmajor.interval_similarity(cminor, weights)
    print(f"{profile_name} similarity: {similarity:.4f}")

# Generate sequences with different profiles to compare character
sequences = {}
for profile_name in INTERVAL_WEIGHT_PROFILES:
    config = {
        "start_pc": [0, 4, 7],  # C major
        "generation_type": "chordal",
        "sequence_length": 8,
        "progression_type": "random",
        "randomness_factor": 0.4,
        "interval_weights": profile_name
    }
    sequences[profile_name] = generate_sequence_from_config(config)
```

## Logging

The system includes a comprehensive logging mechanism to track operations and debug issues. Logs are written to both console and files in the `logs/` directory.

```bash
# Run with debug-level logging
python run_engine.py --config-file configs/melodic_basic.yaml --log-level DEBUG

# Run with warning-level logging (less verbose)
python run_engine.py --config-file configs/chord_progression.yaml --log-level WARNING
```

## Extending the System

The modular design allows for easy extension:

1. Add new transformation operations in `PitchClassEngine._apply_random_transformation()`
2. Create new preset configurations in the `configs/` directory
3. Extend `PitchClassSet` with additional music theory concepts in `pitch_classes.py`
4. Add new MIDI features in `midi/translator.py`
5. Add custom analysis metrics to `MonteCarloSimulator._analyze_sequence()`
6. Create new interval weight profiles in `INTERVAL_WEIGHT_PROFILES`

## Requirements

- Python 3.7+
- NumPy
- Mido (for MIDI generation)
- PyYAML (for configuration files)
- tqdm (for progress bars)
- pandas (optional, for data analysis)
- matplotlib (optional, for visualization)
