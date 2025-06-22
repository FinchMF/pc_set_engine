# Components

## Pitch Class Module (`pitch_classes.py`)

- `PitchClass`: Represents individual pitch classes (0-11)
- `PitchClassSet`: Implements pitch class sets with various operations:
  - Normal form and prime form calculation
  - Forte number identification
  - Interval vector calculation
  - Weighted interval vectors and similarity measurements
  - Set operations (transpose, invert, complement, etc.)
- `COMMON_SETS`: Dictionary of predefined common musical pitch class sets
- `INTERVAL_WEIGHT_PROFILES`: Predefined weightings for different musical styles

## Engine Module (`engine.py`)

- `GenerationType`: Defines generation types (melodic or chordal)
- `ProgressionType`: Defines progression types (static, directed, random)
- `GenerationConfig`: Configuration data class for controlling generation parameters
- `PitchClassEngine`: Main engine class that handles generation of sequences
  - Supports directed progressions toward a target
  - Implements multiple transformation strategies
  - Controls randomness and variation

## MIDI Module (`midi/translator.py`)

- Convert pitch class sequences to MIDI files for playback and further processing
- Support for both melodic and chordal sequences
- Customizable parameters (tempo, octave, note duration, etc.)
- Functions to add rhythmic patterns to sequences
- Ability to load sequences from JSON files and convert to MIDI

## Runner Module (`run_engine.py`)

- Provides a command-line interface to the engine
- Includes preset configurations for different generation types
- Handles saving and displaying results
- Supports direct MIDI file generation
- Support for YAML configuration files

## Monte Carlo Simulator (`monte_carlo.py`)

- Run large-scale simulations with systematically varied parameters
- Generate datasets of sequences for analysis and experimentation
- Two main modes of operation:
  - General Monte Carlo: Random parameter exploration
  - Parameter Variation: Systematic study of specific parameter effects
- Support for interval weighting configurations to influence generation
- Analysis of correlations between weights and musical characteristics
- Support for parallel processing to speed up large simulations
- Statistical analysis of generated sequences
- Export data to JSON and CSV for further analysis

## Analysis Tools (`analysis/`)

- Notebooks and scripts for analyzing generated sequences and simulation results
- Visualizations of correlations between parameters and musical characteristics
- Statistical analysis of the effects of interval weighting on musical output
- Tools for interpreting Monte Carlo simulation results
- Jupyter notebooks for interactive exploration of dataset patterns

## Rhythm Module (`pc_sets/rhythm.py`)

- `RhythmEngine`: Core class for generating rhythmic patterns
- `RhythmConfig`: Configuration data class for rhythm parameters
- Supports multiple rhythm types:
  - Regular subdivisions (evenly distributed durations)
  - Swing feel (uneven subdivisions with emphasis)
  - Dotted rhythms (long-short patterns)
  - Shuffle patterns (milder swing feel)
  - Complex rhythms (mixed subdivisions)
- Time signature and subdivision-based rhythm generation
- Accent pattern application for musical expression
- Rhythm manipulation operations (augmentation, diminution, displacement)
- Polyrhythm and cross-rhythm generation
- Predefined rhythm patterns for various musical styles
- Integration with pitch class sequences for complete musical output
