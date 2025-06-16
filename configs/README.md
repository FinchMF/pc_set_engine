# Configuration Files

This directory contains YAML configuration files for the Pitch Class Rules Engine. These files can be used with the `run_engine.py` script using the `--config-file` option.

## Usage

```bash
python run_engine.py --config-file configs/melodic_basic.yaml
```

You can override specific parameters from the command line:

```bash
python run_engine.py --config-file configs/melodic_basic.yaml --sequence-length 12
```

To save the output as MIDI:

```bash
python run_engine.py --config-file configs/chord_progression.yaml --midi output.mid
```

## Available Configurations

- **melodic_basic.yaml**: Basic melodic sequence from C to G
- **chord_progression.yaml**: Chord progression from C major to G major
- **random_walk.yaml**: Random walk melodic sequence with high variability
- **jazz_progression.yaml**: Jazz-like chord progression with 7th chords
- **static_minor_chord.yaml**: Static C minor chords with variations

## File Structure

Each configuration file includes the following sections:

1. **Generation properties**: Defines the type of sequence and progression
2. **Randomness controls**: Sets the parameters for randomness and variation
3. **Constraints**: Defines constraints on the generated sequence
4. **MIDI properties**: Sets parameters for MIDI file generation

## Creating Custom Configurations

You can create your own configuration files by copying and modifying existing ones. The most important parameters are:

- `generation_type`: Either "melodic" or "chordal"
- `start_pc`: Starting pitch class (0-11) or pitch class set (list of integers)
- `progression`: Boolean indicating whether the sequence should progress/transform
- `progression_type`: "directed", "random", or "static"
- `randomness_factor`: Level of randomness (0.0-1.0)
- `variation_probability`: Probability of applying variations (0.0-1.0)

For MIDI generation, important properties are:

- `tempo`: Beats per minute
- `base_octave`: Base octave (4 = middle C)
- `note_duration`: Duration of each note in seconds
