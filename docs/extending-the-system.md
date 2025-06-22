# Extending the System

The modular design allows for easy extension in various ways.

## Extension Points

1. Add new transformation operations in `PitchClassEngine._apply_random_transformation()`
2. Create new preset configurations in the `configs/` directory
3. Extend `PitchClassSet` with additional music theory concepts in `pitch_classes.py`
4. Add new MIDI features in `midi/translator.py` 
5. Add custom analysis metrics to `MonteCarloSimulator._analyze_sequence()`
6. Create new interval weight profiles in `INTERVAL_WEIGHT_PROFILES`
7. Develop new visualization tools in the `analysis/` directory
8. Add custom statistical tests to the analysis notebooks
9. Implement new rhythm subdivision types in `RhythmEngine._generate_base_pattern()`
10. Create additional accent patterns and time signature templates

## Adding a New Transformation

```python
def _apply_random_transformation(self, pc_set, transformation_type=None):
    """Apply a random transformation to the pitch class set."""
    if transformation_type is None:
        # Choose a random transformation
        transformation_type = random.choice([
            'transpose', 'invert', 'complement', 'rotate', 'my_new_transformation'  # Add your new transformation
        ])
    
    # ...existing code...
    
    elif transformation_type == 'my_new_transformation':
        # Implement your custom transformation logic
        result = self._apply_my_new_transformation(pc_set)
        transformation_name = "My New Transformation"
        
    # ...existing code...
    
def _apply_my_new_transformation(self, pc_set):
    """A custom transformation that does something interesting."""
    # Implementation of your transformation
    result_pcs = []
    # Your transformation logic here
    return PitchClassSet(result_pcs)
```

## Creating a New Interval Weight Profile

```python
# In pitch_classes.py, add a new profile to INTERVAL_WEIGHT_PROFILES
INTERVAL_WEIGHT_PROFILES = {
    # ...existing profiles...
    'my_custom_profile': {
        1: 0.9,  # minor 2nds/major 7ths
        2: 1.1,  # major 2nds/minor 7ths
        3: 1.5,  # minor 3rds/major 6ths
        4: 0.8,  # major 3rds/minor 6ths
        5: 1.3,  # perfect 4ths/5ths
        6: 0.5   # tritones
    }
}
```

## Adding a New Rhythm Subdivision Type

```python
def _generate_base_pattern(self, length, subdivision_type):
    """Generate a base rhythm pattern."""
    if subdivision_type == "regular":
        # Even subdivisions
        return [1.0] * length
    elif subdivision_type == "swing":
        # Swing feel (long-short pattern)
        pattern = []
        for i in range(length // 2):
            pattern.extend([1.5, 0.5])
        if length % 2 != 0:
            pattern.append(1.0)
        return pattern
    # ...existing types...
    elif subdivision_type == "my_custom_subdivision":
        # Implement your custom subdivision pattern
        pattern = []
        # Your pattern generation logic here
        return pattern
```

## Creating Custom Analysis Metrics

```python
def _analyze_sequence(self, sequence, is_melodic=True):
    """Analyze musical properties of a sequence."""
    analysis = {}
    
    # ...existing metrics...
    
    # Add your custom metric
    analysis['my_custom_metric'] = self._calculate_my_metric(sequence, is_melodic)
    
    return analysis

def _calculate_my_metric(self, sequence, is_melodic):
    """Calculate a custom musical metric."""
    # Your metric calculation logic
    result = 0
    # Implementation here
    return result
```

## Creating a Custom Configuration File

```yaml
# configs/my_custom_generation.yaml
# Custom generation configuration with specialized parameters

# Basic parameters
generation_type: chordal
start_pc: [0, 3, 7]  # C minor
sequence_length: 12

# Advanced parameters
progression: true
progression_type: directed
target_pc: [7, 11, 2]  # G dominant

# Randomness controls
randomness_factor: 0.4
variation_probability: 0.7

# Custom interval weights
interval_weights:
  1: 0.7
  2: 1.2
  3: 1.6
  4: 0.8
  5: 1.4
  6: 0.5

# Custom constraints
constraints:
  require_progression_intervals: [3, 4, 5]
  avoid_intervals: [6]

# Custom rhythm settings
rhythm:
  time_signature: [3, 4]
  subdivision: 8
  subdivision_type: "my_custom_subdivision"
  accent_pattern: [1.0, 0.7, 0.8, 0.6, 0.9, 0.7]

# MIDI settings
midi_properties:
  tempo: 88
  base_octave: 3
  note_duration: 0.8
  channel: 2
```

## Extending the Command-Line Interface

To add new command-line options, modify the `run_engine.py` file:

```python
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the pitch class generation engine")
    
    # ...existing arguments...
    
    # Add your custom argument
    parser.add_argument("--my-custom-option", type=str, help="Description of your custom option")
    
    return parser.parse_args()
```