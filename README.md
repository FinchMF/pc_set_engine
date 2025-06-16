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

## Repository Structure

```
pc_rules_engine/
├── pc_sets/
│   ├── __init__.py
│   ├── pitch_classes.py   # Core pitch class theory implementation
│   └── engine.py          # Generation engine implementation
├── run_engine.py          # Command-line interface
└── README.md              # This documentation
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
- `ProgressionType`: Defines progression types (static, directed, or random)
- `GenerationConfig`: Configuration data class for controlling generation parameters
- `PitchClassEngine`: Main engine class that handles generation of sequences
  - Supports directed progressions toward a target
  - Implements multiple transformation strategies
  - Controls randomness and variation

### Runner Module (`run_engine.py`)

- Provides a command-line interface to the engine
- Includes preset configurations for different generation types
- Handles saving and displaying results

## Logging

The system includes a logging mechanism to track operations and debug issues. Logs are written to a file and can be configured for verbosity.

### Basic Logging Usage

Enable logging in your script:

```python
import logging

logging.basicConfig(level=logging.INFO, filename='pc_rules_engine.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Example log message
logging.info('This is an info message')
```

## Usage Examples

### Basic Command-Line Usage

Generate a melodic sequence with default parameters:

```bash
python run_engine.py --config-type melodic
```

Generate a chord progression from C major to G major:

```bash
python run_engine.py --config-type chord-progression
```

Generate a static chord with variations:

```bash
python run_engine.py --config-type static-chord
```

### Controlling Randomness and Variation

Generate a highly random melodic sequence:

```bash
python run_engine.py --config-type melodic --randomness 0.8 --variation 0.7
```

Generate a more predictable melodic sequence:

```bash
python run_engine.py --config-type melodic --randomness 0.1 --variation 0.2
```

Generate a random walk with large interval jumps:

```bash
python run_engine.py --config-type random-walk --randomness 0.9 --sequence-length 16
```

### Saving Output

Save the generated sequence to a JSON file:

```bash
python run_engine.py --config-type chord-progression --output progression.json
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
print(sequence)
```

## Advanced Examples

### Generate a Jazz-Like Chord Progression

```python
from pc_sets.engine import generate_sequence_from_config

# Jazz progression with 7th chords
config = {
    "start_pc": [0, 4, 7, 11],  # Cmaj7
    "generation_type": "chordal",
    "sequence_length": 4,
    "progression": True,
    "progression_type": "random",
    "randomness_factor": 0.6,
    "variation_probability": 0.7,
    "constraints": {
        "vary_chord": True
    },
    "allowed_operations": ["transpose", "substitute_note"]
}

sequence = generate_sequence_from_config(config)
print(sequence)
```

### Generate a Melodic Line Moving from C to G with Medium Variation

```bash
python run_engine.py --config-type melodic --randomness 0.5 --variation 0.4 --sequence-length 12
```

## Extending the System

The modular design allows for easy extension:

1. Add new transformation operations in `PitchClassEngine._apply_random_transformation()`
2. Create new preset configurations in `run_engine.py`
3. Extend `PitchClassSet` with additional music theory concepts in `pitch_classes.py`

## Requirements

- Python 3.7+
- NumPy
