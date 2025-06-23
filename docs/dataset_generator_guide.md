# Dataset Generator Guide

## Overview

The Dataset Generator is a powerful tool for generating large, diverse datasets by running multiple Monte Carlo simulations with different configurations. It provides a configuration-driven approach to create customized datasets that can be used for training, testing, and analysis of the PC Rules Engine.

## Features

- **Configuration-driven**: Define your dataset generation process through YAML configuration files
- **Grouped simulations**: Organize simulations into logical groups with shared characteristics
- **Parameter ranges**: Specify ranges for parameters to create diverse datasets
- **Rhythm integration**: Full support for rhythm parameters and variations
- **Consolidated output**: Automatically consolidates all results into a single dataset
- **Detailed metadata**: Comprehensive metadata about the generation process
- **Statistics**: Exports statistics for analysis

## Installation

The Dataset Generator is included in the PC Rules Engine package. No additional installation is required.

## Configuration File Structure

Dataset generation is driven by YAML configuration files. Here's the structure of a configuration file:

```yaml
dataset_name: name_of_your_dataset
description: "Optional description of your dataset"
version: 1.0  # Optional version number

simulation_groups:
  - name: group_name_1
    num_simulations: 50  # Number of simulations to run in this group
    base_config: path/to/config.yaml  # Path to a base config file
    use_rhythm: true  # Whether to use rhythm generation
    param_ranges:  # Parameter ranges for this group
      randomness_factor: [0.1, 0.9]  # Min and max values
      variation_probability: [0.2, 0.8]
      sequence_length: [4, 16]
    rhythm_param_ranges:  # Rhythm parameter ranges
      subdivision: [4, 16]
      variation_probability: [0.2, 0.7]
      
  - name: group_name_2
    num_simulations: 30
    base_config:  # Alternatively, define the base config inline
      start_pc: [0, 4, 7]
      generation_type: "chordal"
      sequence_length: 8
      progression: true
      progression_type: "random"
      allowed_operations: ["transpose", "invert", "add_note"]
      constraints:
        max_interval: 3
        vary_pc: true
      randomness_factor: 0.4
      variation_probability: 0.5
    param_ranges:
      randomness_factor: [0.2, 0.8]

# Optional global settings
settings:
  save_midi: true
  save_stats: true
  include_raw_data: true
```

### Required Fields

- **dataset_name**: Name of the dataset (used for output files)
- **simulation_groups**: List of simulation group configurations

### Simulation Group Fields

- **name**: Name of the simulation group (used for organization and output)
- **num_simulations**: Number of simulations to run in this group
- **base_config**: Either a path to a YAML file or an inline configuration dictionary
- **use_rhythm** (optional): Whether to include rhythm generation (default: true)
- **param_ranges** (optional): Parameter ranges for Monte Carlo simulation
- **rhythm_param_ranges** (optional): Rhythm parameter ranges for Monte Carlo simulation

## Command-Line Usage

The dataset generator can be invoked directly from the command line:

```bash
python dataset_generator.py path/to/config.yaml --output-dir output_directory --seed 42
```

### Arguments

- **config**: Path to the dataset configuration file (required)
- **--output-dir, -o**: Directory to save the generated dataset (default: "dataset")
- **--seed, -s**: Random seed for reproducibility (optional)

## Programmatic API Usage

You can also use the dataset generator programmatically in your Python scripts:

```python
from dataset_generator import create_dataset_from_config

# Basic usage
metadata = create_dataset_from_config(
    "path/to/config.yaml",
    "output_directory"
)

# With random seed for reproducibility
metadata = create_dataset_from_config(
    "path/to/config.yaml",
    "output_directory",
    random_seed=42
)

# Access metadata about the generated dataset
print(f"Generated {metadata['total_simulations']} simulations")
print(f"Successful: {metadata['successful_simulations']}, Failed: {metadata['failed_simulations']}")
print(f"Generation time: {metadata['generation_time']:.2f} seconds")
```

## Output Structure

The dataset generator produces the following output structure:

```
output_directory/
├── dataset_name.json       # Consolidated dataset with all simulations
├── metadata.json           # Complete generation metadata
├── group_name_1/           # Directory for first simulation group
│   ├── group_name_1_results.json  # Results for this group
│   ├── group_name_1_stats.csv     # Statistics for this group
│   └── midi/               # MIDI files for this group (if generated)
├── group_name_2/           # Directory for second simulation group
│   ├── group_name_2_results.json
│   ├── group_name_2_stats.csv
│   └── midi/
└── ...
```

## Examples

### Basic Dataset Configuration

```yaml
dataset_name: basic_dataset
description: "A simple demonstration dataset"
version: 1.0

simulation_groups:
  - name: chordal_major
    num_simulations: 20
    base_config: configs/chordal_major.yaml
    param_ranges:
      randomness_factor: [0.1, 0.9]
      variation_probability: [0.2, 0.8]
  
  - name: melodic_random
    num_simulations: 20
    base_config: configs/melodic_random.yaml
    param_ranges:
      sequence_length: [6, 12]
```

### Advanced Dataset with Rhythm Variations

```yaml
dataset_name: rhythm_exploration
description: "Dataset exploring different rhythm configurations"
version: 1.0

simulation_groups:
  - name: swing_rhythm
    num_simulations: 30
    base_config: configs/melodic_directed.yaml
    use_rhythm: true
    param_ranges:
      randomness_factor: [0.2, 0.7]
      sequence_length: [8, 16]
    rhythm_param_ranges:
      subdivision: [8, 16]
      subdivision_type: ["swing"]
      variation_probability: [0.2, 0.6]
      
  - name: complex_meters
    num_simulations: 30
    base_config: configs/chordal_major.yaml
    use_rhythm: true
    rhythm_param_ranges:
      time_signature: [[5, 4], [7, 8], [9, 8]]
      subdivision: [3, 9]
```

### Generating a Dataset with Inline Configuration

```yaml
dataset_name: custom_inline_dataset
description: "Dataset with inline configurations"

simulation_groups:
  - name: diminished_chords
    num_simulations: 15
    base_config:
      start_pc: [0, 3, 6]
      generation_type: "chordal"
      sequence_length: 8
      progression: true
      progression_type: "random"
      allowed_operations: ["transpose", "invert", "add_note"]
      constraints:
        max_interval: 3
        vary_pc: true
      randomness_factor: 0.5
      variation_probability: 0.4
    param_ranges:
      randomness_factor: [0.3, 0.7]
```

## Advanced Features

### Handling Different Time Signatures

You can specify different time signatures in the rhythm_param_ranges:

```yaml
rhythm_param_ranges:
  time_signature: [[4, 4], [3, 4], [6, 8]]
```

### Specifying Exact Values Instead of Ranges

For parameters where you want specific values rather than ranges:

```yaml
rhythm_param_ranges:
  subdivision_type: ["regular", "swing", "shuffle"]
  accent_type: ["downbeat", "syncopated"]
```

### Creating Specialized Parameter Sets

You can create specialized parameter combinations by setting up multiple simulation groups:

```yaml
simulation_groups:
  - name: sparse_chords
    # ...config for sparse chord sequences
  
  - name: dense_chords
    # ...config for dense chord sequences
    
  - name: chromatic_melodies
    # ...config for chromatic melodic sequences
```

## Best Practices

1. **Start small**: Begin with a small number of simulations to test your configuration
2. **Logical grouping**: Group related simulations together for better organization
3. **Parameter exploration**: Use parameter ranges to explore the parameter space
4. **Use random seeds**: Set a random seed for reproducible results
5. **Monitor statistics**: Check the statistics to understand the characteristics of your dataset

## Troubleshooting

### Common Issues

- **Missing base_config**: Ensure each simulation group has a valid base_config
- **Invalid parameter ranges**: Parameter ranges should be specified as [min, max]
- **File not found errors**: Check that paths to base config files are correct
- **Memory issues**: For very large datasets, consider generating in smaller batches

### Fixing Failed Simulations

If some simulations fail:

1. Check the error messages in the logs
2. Adjust the parameter ranges to avoid problematic configurations
3. Examine the successful simulations to understand what worked

## Advanced Usage: Custom Dataset Analysis

After generating a dataset, you might want to analyze it further:

```python
import json

# Load the consolidated dataset
with open("output_directory/dataset_name.json", "r") as f:
    dataset = json.load(f)

# Access all simulations
simulations = dataset["simulations"]

# Filter successful simulations
successful = [sim for sim in simulations if sim.get("success", False)]

# Group simulations by characteristics
by_group = {}
for sim in successful:
    group = sim.get("group", "unknown")
    if group not in by_group:
        by_group[group] = []
    by_group[group].append(sim)

# Analyze specific characteristics
# ... your custom analysis code here ...
```

## Conclusion

The Dataset Generator provides a powerful, flexible way to create diverse datasets for the PC Rules Engine. By properly configuring your dataset generation process, you can create datasets tailored to your specific research or creative needs.
