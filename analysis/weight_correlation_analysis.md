# Interval Weight Correlation Analysis

This document provides an interpretation of the correlations between interval weights and musical characteristics found in the Monte Carlo simulations.

## Overview

The weight correlation report shows how emphasizing different intervals (by changing their weights) affects various properties of the generated music. Correlation values range from -1.0 to 1.0:
- Positive values indicate that as the interval weight increases, the measured characteristic also tends to increase
- Negative values indicate an inverse relationship
- Values close to 0 indicate little to no correlation

## Melodic Sequence Analysis

### Pitch Class Diversity and Uniqueness
- **Interval 1 (minor 2nds)**: Strong negative correlation (-0.35) with unique pitch classes. This suggests that emphasizing semitones results in music that tends to stay within a narrower range of notes, creating more focused, less diverse melodies.
- **Interval 6 (tritones)**: Strong positive correlation (0.52) with unique pitch classes. Emphasizing tritones appears to significantly increase the diversity of notes used, creating more chromatic, wide-ranging melodies.

### Interval Size Patterns
- **Interval 3 (minor 3rds)**: Positive correlation (0.45) with mean interval size. Emphasizing minor thirds tends to create melodies with larger average interval jumps.
- **Interval 5 (perfect 4ths/5ths)**: Negative correlation (-0.24) with mean interval size. When perfect 4ths/5ths are emphasized, melodies tend to have smaller average interval jumps, suggesting more stepwise motion.
- **Interval 6 (tritones)**: Strong negative correlation (-0.62) with maximum interval size. Interestingly, while tritones increase note diversity, they seem to constrain the maximum interval size, perhaps creating music with more consistent, regular interval patterns.

### Note Repetition
- **Interval 1 (minor 2nds)**: Very strong positive correlation (0.55) with repeated notes. Emphasizing semitones significantly increases note repetition, creating more static, focused melodies.
- **Interval 6 (tritones)**: Very strong negative correlation (-0.70) with repeated notes. Emphasizing tritones dramatically reduces note repetition, creating more varied, constantly changing melodies.

### Melodic Contour
- **Interval 3 (minor 3rds)**: Strong positive correlation (0.46) with direction changes. Emphasizing minor thirds creates more complex melodic contours with frequent changes in direction.
- **Interval 6 (tritones)**: Very strong negative correlation (-0.67) with direction changes. Despite increasing pitch diversity, emphasizing tritones appears to create more directionally consistent melodic lines.

## Chordal Sequence Analysis

### Chord Structure
- **Interval 2 (major 2nds)**: Negative correlation (-0.29) with mean chord size. Emphasizing major seconds tends to produce smaller chords with fewer notes.
- **Interval 4 (major 3rds)**: Positive correlation (0.15) with mean chord size. Emphasizing major thirds tends to produce larger, fuller chords.

### Harmonic Variety
- **Interval 4 (major 3rds)**: Strong negative correlation (-0.37) with unique chords. Emphasizing major thirds reduces harmonic variety, possibly because it reinforces traditional chord structures.
- **Interval 3 (minor 3rds)**: Positive correlation (0.19) with unique chords. Emphasizing minor thirds increases harmonic variety, perhaps by introducing more modal or jazz-like harmonies.

### Harmonic Change Rate
- **Interval 4 (major 3rds)**: Strong negative correlation (-0.44) with mean pitch class changes. Emphasizing major thirds significantly reduces the rate of harmonic change, creating more stable progressions.

### Dissonance Trends
- **Interval 3 (minor 3rds)**: Positive correlation (0.37) with dissonance change. Emphasizing minor thirds tends to create progressions that become more dissonant over time.
- **Interval 4 (major 3rds)**: Negative correlation (-0.35) with dissonance change. Emphasizing major thirds tends to create progressions that become less dissonant over time.
- **Interval 2 (major 2nds)**: Positive correlation (0.38) with consonance change. Emphasizing major seconds tends to increase consonance over time.

## Practical Applications

These correlations provide valuable insights that can be used to fine-tune the generation of musical sequences:

### For More Traditional/Consonant Music
- Increase weights for intervals 4 (major 3rds) and 5 (perfect 4ths/5ths)
- Decrease weights for intervals 1 (minor 2nds) and 6 (tritones)
- This will typically result in: more stable chord progressions, fewer direction changes in melodies, and less dissonance

### For More Modern/Chromatic Music
- Increase weights for intervals 6 (tritones) and 3 (minor 3rds)
- This will typically result in: more unique pitch classes, more complex harmonies, and increasing dissonance

### For Jazz-Like Harmonies
- Increase weights for intervals 2 (major 2nds) and 3 (minor 3rds)
- Moderately increase weights for interval 6 (tritones)
- This will typically result in: more unique chords with increasing consonance but maintaining some dissonant elements

### For Minimalist/Repetitive Music
- Increase weights for interval 1 (minor 2nds)
- Decrease weights for interval 6 (tritones)
- This will typically result in: more repeated notes, fewer unique pitch classes, and more focused melodies

## Conclusion

The interval weight system provides a powerful mechanism for controlling the musical character of generated sequences. By adjusting weights based on these correlations, users can nudge the generation algorithm toward specific stylistic outcomes without having to directly program those styles.

These correlations also confirm many aspects of traditional music theory: the importance of major thirds in stable harmonies, the disruptive effect of tritones, and the role of perfect fourths and fifths in creating stepwise melodic motion.
