# Monte Carlo Simulations

The Monte Carlo simulation framework allows you to run large-scale simulations with systematically varied parameters and analyze the results.

## Basic Monte Carlo Simulations

```bash
# Run a basic Monte Carlo simulation with 100 iterations
python monte_carlo.py --num-simulations 100

# Use a specific base configuration file
python monte_carlo.py --num-simulations 50 --base-config configs/melodic_basic.yaml

# Run a parameter variation study
python monte_carlo.py --variation-mode --param-name randomness_factor --param-min 0.1 --param-max 0.9 --param-steps 9 --base-config configs/chord_progression.yaml
```

## Analyzing Monte Carlo Results

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the Monte Carlo dataset
with open('datasets/monte_carlo_dataset.json', 'r') as f:
    data = json.load(f)

# Load the statistics CSV into a pandas DataFrame
stats = pd.read_csv('datasets/monte_carlo_stats.csv')

# Plot the relationship between randomness factor and interval size
plt.scatter(stats['randomness_factor'], stats['mean_interval'])
plt.xlabel('Randomness Factor')
plt.ylabel('Mean Interval Size')
plt.title('Effect of Randomness on Interval Size')
plt.savefig('randomness_vs_intervals.png')
```

## Monte Carlo with Interval Weights

```bash
# Run a Monte Carlo simulation with a specific interval profile
python monte_carlo.py --num-simulations 50 --base-config configs/chord_progression.yaml --interval-weight-profile consonant

# Generate random interval weights for each simulation
python monte_carlo.py --num-simulations 50 --random-weights --weight-variation 0.7

# Cycle through all available interval weight profiles
python monte_carlo.py --num-simulations 50 --weight-cycling

# Generate a correlation report between weights and musical features
python monte_carlo.py --num-simulations 100 --random-weights --correlation-report
```

## Analyzing Weight Correlations

```python
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the weight correlation report
with open('datasets/weight_correlation_report.json', 'r') as f:
    correlations = json.load(f)

# Create a heatmap of interval correlations for melodic features
melodic_corr = correlations['melodic']
corr_data = []
for metric, values in melodic_corr.items():
    row = [values[f'interval_{i}'] for i in range(1, 7)]
    corr_data.append(row)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_data, 
    annot=True, 
    xticklabels=['min 2nd', 'maj 2nd', 'min 3rd', 'maj 3rd', 'P4/P5', 'tritone'],
    yticklabels=list(melodic_corr.keys()),
    cmap='coolwarm'
)
plt.title('Correlations: Interval Weights vs. Melodic Features')
plt.tight_layout()
plt.savefig('interval_correlations.png')
```

## Large-Scale Parameter Exploration

```python
from monte_carlo import MonteCarloSimulator

# Create a simulator with custom parameter ranges
simulator = MonteCarloSimulator(
    num_simulations=200,
    base_config_file="configs/melodic_basic.yaml",
    param_ranges={
        "randomness_factor": (0.0, 1.0),
        "variation_probability": (0.0, 1.0),
        "sequence_length": (4, 24)
    },
    output_dir="exploration_results"
)

# Run the simulation with parallel processing
dataset = simulator.run(parallel=True, max_workers=8)

# Save the dataset and export statistics
simulator.save_dataset("full_exploration.json")
simulator.export_stats_to_csv("exploration_stats.csv")
```

## Studying the Effect of a Single Parameter

```python
from monte_carlo import generate_variations_dataset

# Study how randomness affects melodic contour
generate_variations_dataset(
    base_config_file="configs/melodic_basic.yaml",
    param_name="randomness_factor",
    values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    samples_per_value=5,
    output_dir="randomness_study"
)
```

## Using Analysis Tools and Notebooks

The `analysis/` directory contains tools for deeper investigation of the engine's behavior:


Example output from the notebook:
- Heatmaps showing correlations between interval weights and musical features
- Recommendations for optimal weight configurations for specific musical styles
- Statistical significance testing of parameter effects
- Generated YAML configurations based on analytical insights
