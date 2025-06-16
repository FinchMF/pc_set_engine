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
  - Set operations (transpose, invert, complement, etc.)
- `COMMON_SETS`: Dictionary of predefined common musical pitch class sets

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

## Requirements

- Python 3.7+
- NumPy
- Mido (for MIDI generation)
- PyYAML (for configuration files)
