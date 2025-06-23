# Pitch Class Rules Engine Simulation Analysis

## Executive Summary

This analysis examines the results of 50 simulations run with the Pitch Class Rules Engine. The simulations explored various parameter configurations for both melodic and chordal pitch class sequence generation. Key findings include:

- **Success Rate**: 92% of simulations completed successfully (46 out of 50)
- **Generation Types**: 40% melodic and 60% chordal sequences were generated
- **Progression Types**: A mix of directed (32%), random (40%), and static (28%) progressions were tested
- **Parameter Influence**: Randomness factor and variation probability showed significant effects on sequence complexity and diversity

## Overview of Simulation Results

The dataset consists of 50 simulations with varying configurations of parameters such as:
- Sequence length (ranging from 4 to 16)
- Progression types (directed, random, static)
- Randomness factor (ranging from 0.1 to 0.9)
- Variation probability (ranging from 0.2 to 0.8)
- Various rhythm configurations

### Success vs. Failure Analysis

Of the 50 simulations, 46 completed successfully while 4 failed. All failures occurred in melodic generation mode with the same error: "max() arg is an empty sequence". This suggests a potential edge case in the melodic sequence generation algorithm.

Failed simulation IDs: 11, 17, 31, 42

### Melodic vs. Chordal Generation

#### Melodic Generation Statistics

The melodic generation approach produced sequences with:
- Average unique pitch classes: 5.2 per sequence
- Average mean interval: 3.6 semitones
- Repeated notes occurrence: 33% of notes on average
- Direction changes: 1.6 per sequence on average

#### Chordal Generation Statistics

The chordal generation approach produced sequences with:
- Average chord size: 3.9 notes per chord
- Average unique chords: 5.7 per sequence
- Average dissonance ratio: 0.12 (initial) to 0.17 (final)
- Average consonance ratio: 0.48 (initial) to 0.43 (final)

## Parameter Effects Analysis

### Randomness Factor

The randomness factor parameter (ranging from 0.12 to 0.89 in successful simulations) showed strong correlation with:

1. **Sequence complexity**: Higher randomness values (>0.7) produced more varied pitch content and larger interval jumps
2. **Chord size progression**: High randomness simulations showed greater increase in chord size over time
3. **Error rate**: Simulations with high randomness were more likely to fail

### Variation Probability

Variation probability (ranging from 0.24 to 0.78) influenced:

1. **Unique chord count**: Higher variation probability led to more unique chords
2. **Direction changes**: More frequent direction changes in melodic sequences
3. **Note repetition**: Lower values resulted in more repeated notes

### Progression Types

Different progression types showed distinct characteristics:

1. **Directed progressions** consistently reached target pitch classes with:
   - Progressive movement toward target harmonies
   - Gradual transformation of dissonance/consonance ratios
   - Average of 1.8 pitch class changes per step

2. **Random progressions** exhibited:
   - Higher diversity of pitch classes
   - More abrupt changes between sequential chords
   - Larger average intervals in melodic sequences (4.2 semitones)

3. **Static progressions** resulted in:
   - Limited pitch class exploration
   - More consistent harmonic profiles
   - Often single-chord sequences

## Rhythm Configuration Analysis

The rhythm configurations varied widely, with interesting patterns:

- **Subdivision types**: Regular, swing, dotted, complex, and shuffle
- **Accent types**: Downbeat (60%), syncopated (28%), offbeat (12%)
- **Time signatures**: Predominantly 4/4 (72%), with some 3/4, 5/4, 6/8, and 7/8

Simulations with syncopated accent types produced more rhythmically complex sequences, even when the pitch material was relatively static.

## Statistical Observations

### Harmonic Statistics (Chordal Sequences)

- **Dissonance change**: 70% of sequences maintained or decreased dissonance
- **Consonance change**: 60% of sequences showed decreased consonance
- **Tritone usage**: More common in diminished chord-based sequences (starting with [0,3,6])
- **Semitone ratio**: Generally increased as chord size increased

### Melodic Statistics

- **Mean vs. median interval**: Mean interval (3.6) significantly higher than median (1.8), indicating occasional large jumps
- **Maximum intervals**: 11 semitones (octave-minus-semitone) appeared most frequently as maximum
- **Unique pitch classes**: Shorter sequences used proportionally more unique pitch classes

## Conclusions and Future Directions

1. **Algorithm improvements**: Address the "max() arg is an empty sequence" error in melodic generation
2. **Parameter optimization**: 
   - Randomness factor: Most effective range 0.3-0.7
   - Variation probability: Most effective range 0.4-0.7
3. **Musical applications**:
   - Directed progressions most suitable for goal-oriented harmonic movement
   - Random progressions create more diverse, exploratory sequences
   - Static progressions useful for establishing tonal centers
4. **Future experiments**:
   - Test wider ranges of rhythm subdivision
   - Explore weighted operations for specific musical styles
   - Implement machine learning to optimize parameter selection based on desired musical outcomes

---

*Note: This analysis represents a synthesis of 50 simulation runs. More comprehensive insights could be drawn from larger datasets with more parameter variation.*
