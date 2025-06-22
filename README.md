# Pitch Class Rules Engine

A modular system for generating and manipulating pitch class sequences in musical composition, offering both melodic generation and chord progression capabilities with configurable randomness and variation.

## Overview

This repository contains a rule-based engine for creating musical sequences using pitch class theory. The system can generate:

- Melodic sequences (single-note progressions)
- Chord sequences (multiple pitches at once)
- Progressive transformations from one pitch class (set) to another
- Random walks with controlled variability
- Static presentations with controlled variations
- Rhythmic patterns with customizable time signatures, subdivisions, and accents

The engine is highly configurable, allowing for fine-tuned control over the level of randomness, variation probability, and progression characteristics.

## Key Features

- **Configuration-Driven Workflow**: Use YAML files to define complex musical progressions
- **Melodic and Chordal Generation**: Create single-note sequences or chord progressions
- **Directed Progressions**: Smoothly transition from one pitch class (set) to another
- **Randomness Controls**: Fine-tune the balance between predictability and variation
- **MIDI Output**: Generate playable MIDI files from pitch class sequences
- **Interval Vector Weighting**: Prioritize specific intervals to influence harmonic character
- **Rhythmic Patterns**: Apply configurable rhythmic structures with various time signatures
- **Monte Carlo Analysis**: Study correlations between parameters and musical characteristics
- **Comprehensive Documentation**: Well-documented code with usage examples
- **Modular Design**: Easily extend with new operations and features

## Documentation

Detailed documentation is available in the [docs](./docs) directory:

- [Repository Structure](./docs/repository-structure.md)
- [Configuration Guide](./docs/configuration-guide.md)
- [Components Overview](./docs/components.md)
- [Usage Examples](./docs/usage-examples.md)
- [Advanced Examples](./docs/advanced-examples.md)
- [Dataset Generation](./docs/dataset-generation.md)
- [Monte Carlo Simulations](./docs/monte-carlo-simulations.md)
- [Extending the System](./docs/extending-the-system.md)
- [Requirements](./docs/requirements.md)

## Quick Start

```bash
# Generate a melodic sequence with default parameters
python run_engine.py --config-type melodic

# Generate a chord progression from C major to G major
python run_engine.py --config-type chord-progression

# Generate a highly random melodic sequence with MIDI output
python run_engine.py --config-type melodic --randomness 0.8 --variation 0.7 --midi output.mid
```

## Requirements

- Python 3.7+
- NumPy
- Mido (for MIDI generation)
- PyYAML (for configuration files)
- tqdm (for progress bars)
- pandas (optional, for data analysis)
- matplotlib (optional, for visualization)
