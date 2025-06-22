# Usage Examples

## Using YAML Configuration Files

```bash
# Generate a sequence using a configuration file
python run_engine.py --config-file configs/melodic_basic.yaml

# Override configuration parameters
python run_engine.py --config-file configs/jazz_progression.yaml --sequence-length 12

# Generate both JSON and MIDI output
python run_engine.py --config-file configs/chord_progression.yaml --output result.json --midi result.mid
```

## Using Command-Line Options

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

## Using the API Programmatically

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

## Using Interval Weights for Character Control

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

## Using Rhythm Patterns

```bash
# Generate a sequence with rhythm
python run_engine.py --config-file configs/melodic_basic.yaml --midi output.mid --rhythm --rhythm-type swing

# Specify time signature and subdivision
python run_engine.py --config-type melodic --rhythm --time-signature 3/4 --subdivision 8

# Use a complex rhythm pattern with specified accent type
python run_engine.py --config-type chord-progression --rhythm --rhythm-type complex --accent-type syncopated
```

## Logging

The system includes a comprehensive logging mechanism to track operations and debug issues. Logs are written to both console and files in the `logs/` directory.

```bash
# Run with debug-level logging
python run_engine.py --config-file configs/melodic_basic.yaml --log-level DEBUG

# Run with warning-level logging (less verbose)
python run_engine.py --config-file configs/chord_progression.yaml --log-level WARNING
```
