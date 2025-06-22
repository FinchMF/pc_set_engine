# Dataset Generation

The system can generate comprehensive datasets for music analysis and machine learning.

## Dataset Features

- **Simulation-Based Datasets**: Generate structured datasets from Monte Carlo simulations
- **Format Options**: Export in JSON, CSV, or MusicXML formats
- **Metadata Inclusion**: Store parameter configurations alongside musical output
- **Batch Processing**: Create large, varied datasets through automated batch runs
- **Feature Extraction**: Calculate musical features for each generated sequence
- **Dataset Analysis**: Built-in tools for statistical analysis of generated data
- **Data Augmentation**: Apply transformations to expand existing datasets
- **Filtering Options**: Generate targeted datasets with specific musical properties
- **Cross-Validation Sets**: Generate training/testing splits for machine learning
- **Version Control**: Track dataset provenance and generation parameters

## Basic Dataset Generation

```bash
# Generate a basic dataset with 100 samples
python monte_carlo.py --generate-dataset --samples 100 --output datasets/basic_dataset

# Create a dataset with specific musical constraints
python monte_carlo.py --generate-dataset --samples 50 --constraint "interval_content=stepwise" --output datasets/stepwise_melodies

# Generate a dataset with systematic parameter variation
python monte_carlo.py --generate-dataset --param-sweep "randomness_factor=0.1,0.9,9" --samples-per-config 10 --output datasets/randomness_study

# Create a multi-feature dataset with rich metadata
python monte_carlo.py --generate-dataset --features "interval_vector,contour,complexity" --samples 200 --include-metadata --output datasets/feature_rich
```

## Dataset Analysis

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load a generated dataset
with open('datasets/basic_dataset.json', 'r') as f:
    dataset = json.load(f)

# Extract features into a DataFrame
features = []
for item in dataset:
    features.append({
        'config': item['config'],
        'sequence_length': len(item['sequence']),
        'mean_interval': item['analysis']['mean_interval'],
        'interval_variety': item['analysis']['interval_variety'],
        'contour_direction': item['analysis']['contour_direction']
    })

df = pd.DataFrame(features)

# Plot feature distributions
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
df['mean_interval'].hist()
plt.title('Distribution of Mean Interval Sizes')

plt.subplot(2, 2, 2)
plt.scatter(df['config.randomness_factor'], df['mean_interval'])
plt.xlabel('Randomness Factor')
plt.ylabel('Mean Interval Size')
plt.title('Randomness vs. Interval Size')

plt.tight_layout()
plt.savefig('dataset_analysis.png')
```

## Data Augmentation

```python
from pc_sets.engine import augment_dataset

# Load a dataset
with open('datasets/basic_dataset.json', 'r') as f:
    original_dataset = json.load(f)

# Apply transformations to augment the dataset
augmented_dataset = augment_dataset(
    original_dataset,
    transformations=[
        'transpose',
        'invert',
        'retrograde'
    ],
    preserve_original=True
)

# Save the augmented dataset
with open('datasets/augmented_dataset.json', 'w') as f:
    json.dump(augmented_dataset, f, indent=2)
```

## Cross-Validation Sets

```bash
# Generate dataset with train/test split
python monte_carlo.py --generate-dataset --samples 200 --split-ratio 0.8 --output datasets/ml_ready

# Generate dataset with multiple folds for cross-validation
python monte_carlo.py --generate-dataset --samples 200 --cross-validation-folds 5 --output datasets/cv_folds
```

## Batch Dataset Generation

```python
from monte_carlo import batch_generate_datasets

# Generate multiple datasets with different configurations
batch_generate_datasets(
    base_configs=[
        "configs/melodic_basic.yaml",
        "configs/chord_progression.yaml",
        "configs/jazz_progression.yaml"
    ],
    samples_per_dataset=50,
    output_dir="datasets/batch"
)
```

## Dataset Version Control

```bash
# Generate dataset with version metadata
python monte_carlo.py --generate-dataset --samples 100 --version "1.0" --description "Initial dataset for melody analysis" --output datasets/melody_analysis_v1

# Generate the next version with changes
python monte_carlo.py --generate-dataset --samples 100 --version "1.1" --description "Refined dataset with improved parameter ranges" --previous-version "1.0" --output datasets/melody_analysis_v1.1
```
