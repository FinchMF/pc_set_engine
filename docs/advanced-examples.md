# Advanced Examples

## Create a Custom Configuration File

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

## Generate a Melodic Line with Custom Rhythm

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

## Creating Custom Interval Weight Profiles

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

## Exploring Different Harmonic Characters with Weights

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

## Combining Rhythms and Pitches Programmatically

```python
from pc_sets.engine import generate_sequence_from_config
from pc_sets.rhythm import RhythmEngine, RhythmConfig
from midi.translator import sequence_to_midi_with_rhythm

# Generate pitch sequence
pitch_config = {
    "start_pc": [0, 4, 7],  # C major
    "generation_type": "chordal",
    "sequence_length": 8,
    "progression_type": "random"
}
pitch_sequence = generate_sequence_from_config(pitch_config)

# Create rhythm configuration
rhythm_config = RhythmConfig(
    time_signature=(4, 4),
    subdivision=8,  # eighth note subdivisions
    subdivision_type="swing",
    accent_type="offbeat",
    variation_probability=0.3
)

# Create rhythm engine and apply to sequence
rhythm_engine = RhythmEngine(rhythm_config)
timed_sequence = rhythm_engine.apply_rhythm_to_sequence(
    pitch_sequence, 
    is_melodic=(pitch_config["generation_type"] == "melodic")
)

# Generate MIDI with rhythm
sequence_to_midi_with_rhythm(
    pitch_sequence, 
    "output_with_rhythm.mid", 
    is_melodic=(pitch_config["generation_type"] == "melodic"),
    rhythm_config=rhythm_config
)
```

## Creating Polyrhythmic Patterns

```yaml
# configs/polyrhythm_config.yaml
# Generation properties
generation_type: melodic
start_pc: 0  # C
progression: true
progression_type: random
sequence_length: 16

# Rhythm properties
rhythm:
  time_signature: [4, 4]
  subdivision: 12
  subdivision_type: complex
  accent_pattern: [1.0, 0.5, 0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.8, 0.3, 0.6, 0.5]
  polyrhythm_ratio: [3, 4]  # 3 against 4 polyrhythm
  variation_probability: 0.2
  
# MIDI properties
midi_properties:
  tempo: 100
  base_octave: 4
```

```bash
python run_engine.py --config-file configs/polyrhythm_config.yaml --midi polyrhythm_example.mid
```

## Combining Interval Weights and Rhythm

```python
from pc_sets.engine import generate_sequence_from_config
from pc_sets.rhythm import RhythmConfig
from midi.translator import sequence_to_midi_with_rhythm

# Configuration with both interval weights and rhythm
config = {
    "start_pc": [0, 3, 7],  # C minor
    "generation_type": "chordal",
    "sequence_length": 8,
    "progression_type": "random",
    "interval_weights": "jazzy"  # Use jazz-influenced interval weights
}

# Generate the pitch sequence
sequence = generate_sequence_from_config(config)

# Create a swing rhythm configuration
rhythm_config = {
    "time_signature": (4, 4),
    "subdivision": 8,
    "subdivision_type": "swing",
    "accent_type": "syncopated",
    "tempo": 110
}

# Generate MIDI with both weighted intervals and rhythm
sequence_to_midi_with_rhythm(
    sequence, 
    "jazz_with_rhythm.mid", 
    is_melodic=False, 
    rhythm_config=rhythm_config
)
```

## Manipulating Rhythm Vectors

```python
from pc_sets.rhythm import RhythmEngine, RhythmConfig

# Create a rhythm engine
rhythm_config = RhythmConfig(time_signature=(4, 4), subdivision=4)
rhythm_engine = RhythmEngine(rhythm_config)

# Generate a basic rhythm
base_rhythm = rhythm_engine.generate(length=8)
print("Base Rhythm:", base_rhythm)

# Apply transformations
augmented = rhythm_engine.augment(base_rhythm, factor=2.0)  # Double durations
print("Augmented:", augmented)

diminished = rhythm_engine.diminish(base_rhythm, factor=2.0)  # Halve durations
print("Diminished:", diminished)

displaced = rhythm_engine.displace(base_rhythm, offset=1)  # Shift by one position
print("Displaced:", displaced)
```
