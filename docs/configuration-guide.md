# Configuration Guide

The engine supports a configuration-driven workflow through YAML files, enabling complex musical progressions without writing code.

## Configuration Workflow

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

## Configuration File Structure

```yaml
# Example configuration file
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

## Key Configuration Parameters

### Generation Parameters
- `generation_type`: Either "melodic" or "chordal"
- `start_pc`: Initial pitch class or pitch class set
- `target_pc`: Target pitch class or pitch class set (for directed progressions)
- `progression`: Boolean indicating whether to use progression logic
- `progression_type`: Type of progression ("static", "directed", or "random")
- `sequence_length`: Number of elements in the generated sequence

### Randomness Parameters
- `randomness_factor`: Controls the level of randomness in transformations (0.0 to 1.0)
- `variation_probability`: Probability of applying variations (0.0 to 1.0)

### Interval Weights
- `interval_weights`: Either a predefined profile name or a custom mapping of interval classes to weights

### MIDI Properties
- `tempo`: Playback tempo in BPM
- `base_octave`: Starting octave for melodic sequences
- `note_duration`: Duration of notes in beats

### Rhythm Properties
- `time_signature`: Array of two integers, e.g., [4, 4]
- `subdivision`: Note subdivision level (e.g., 8 for eighth notes)
- `subdivision_type`: Type of subdivision ("regular", "swing", "dotted", etc.)
- `accent_pattern`: Custom accent pattern as array of velocity values

## Example: Custom Weight Profile

```yaml
# Custom interval weight profile
interval_weights:
  1: 0.6   # de-emphasize minor 2nds/major 7ths
  2: 0.8   # slightly de-emphasize major 2nds/minor 7ths
  3: 1.7   # strongly emphasize minor 3rds/major 6ths
  4: 0.9   # slightly de-emphasize major 3rds/minor 6ths
  5: 1.5   # emphasize perfect 4ths/5ths
  6: 0.4   # strongly de-emphasize tritones
```
