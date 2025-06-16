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
├── logs/              # Directory for log files
├── midi_files/        # Directory for generated MIDI files
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

## Logging

The system includes a comprehensive logging mechanism to track operations and debug issues. Logs are written to both console and files in the `logs/` directory.

### Logging Options

Control the logging verbosity:

```bash
# Run with debug-level logging
python run_engine.py --log-level DEBUG

# Run with warning-level logging (less verbose)
python run_engine.py --log-level WARNING
```

Log files are stored in the `logs/` directory with the naming convention `module_YYYYMMDD.log`.

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

### Generating MIDI Files

Generate a melodic sequence and save it as a MIDI file:

```bash
python run_engine.py --config-type melodic --midi melody.mid
```

Generate a chord progression with custom tempo and octave:

```bash
python run_engine.py --config-type chord-progression --midi progression.mid --tempo 100 --base-octave 5
```

Save both JSON and MIDI versions of a sequence:

```bash
python run_engine.py --config-type melodic --output sequence.json --midi sequence.mid
```

### Using the API Programmatically

#### Generating Sequences

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

#### Creating MIDI Files

```python
from midi.translator import sequence_to_midi

# Generate a melodic sequence (using the above code)
# ...

# Convert to MIDI
midi_params = {
    "tempo": 120,
    "base_octave": 4,
    "note_duration": 0.5
}

# For a melodic sequence
sequence_to_midi(sequence, "output.mid", is_melodic=True, params=midi_params)

# For a chord progression
# sequence_to_midi(sequence, "output.mid", is_melodic=False, params=midi_params)
```

#### Adding Custom Rhythms

```python
from midi.translator import sequence_to_midi_with_rhythm

# Define a rhythm pattern (in beats)
rhythm = [1, 0.5, 0.5, 2, 1, 1]  # quarter, eighth, eighth, half, quarter, quarter notes

# Apply rhythm to sequence
sequence_to_midi_with_rhythm(sequence, "rhythm.mid", rhythm, is_melodic=True)
```

## Advanced Examples

### Generate a Jazz-Like Chord Progression and Export to MIDI

```python
from pc_sets.engine import generate_sequence_from_config
from midi.translator import sequence_to_midi

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
sequence_to_midi(sequence, "jazz_progression.mid", is_melodic=False, 
                params={"tempo": 90, "base_octave": 4})
```

### Generate a Melodic Line with Custom Rhythm

```bash
# First generate the sequence and save to JSON
python run_engine.py --config-type melodic --randomness 0.5 --variation 0.4 --sequence-length 12 --output melody.json

# Then in Python, add rhythm and convert to MIDI
python -c "
from midi.translator import load_and_convert_sequence, enhance_with_rhythm, sequence_to_midi_with_rhythm
import json

# Load the sequence
with open('melody.json', 'r') as f:
    data = json.load(f)
sequence = data['sequence']

# Define a rhythm pattern
rhythm = [1, 0.5, 0.5, 1, 0.5, 0.5, 2, 1, 0.5, 0.5, 0.5, 0.5]

# Generate MIDI with rhythm
sequence_to_midi_with_rhythm(sequence, 'melody_rhythmic.mid', rhythm, is_melodic=True)
"
```

## Extending the System

The modular design allows for easy extension:

1. Add new transformation operations in `PitchClassEngine._apply_random_transformation()`
2. Create new preset configurations in `run_engine.py`
3. Extend `PitchClassSet` with additional music theory concepts in `pitch_classes.py`
4. Add new MIDI features in `midi/translator.py`

## Requirements

- Python 3.7+
- NumPy
- Mido (for MIDI generation)
