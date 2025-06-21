"""Example script for running Monte Carlo simulations with interval weights.

This script demonstrates various ways to use the Monte Carlo simulator
with different interval weight configurations to explore their effects
on generated musical sequences.
"""
import os
import sys
from pathlib import Path
import yaml

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monte_carlo import MonteCarloSimulator
from pc_sets.pitch_classes import INTERVAL_WEIGHT_PROFILES

def load_weight_configurations(config_file):
    """Load weight configurations from a YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
    

def main():

    # Create output directory
    output_dir = "weight_studies"
    os.makedirs(output_dir, exist_ok=True)

    # Load weight configurations
    config_file = Path(__file__).parent.parent / "configs" / "weight_configurations.yaml"
    weight_configs = load_weight_configurations(config_file)

    # Example 1: Compare consonant vs dissonant weights
    print("Running consonant vs dissonant comparison...")
    consonant_weights = weight_configs["weight_configurations"][1]["weights"]  # highly_consonant
    dissonant_weights = weight_configs["weight_configurations"][2]["weights"]  # highly_dissonant

    simulator = MonteCarloSimulator(
        num_simulations=20,
        base_config_file=Path(__file__).parent.parent / "configs" / "melodic_basic.yaml",
        interval_weight_configs=[consonant_weights, dissonant_weights],
        output_dir=os.path.join(output_dir, "consonant_vs_dissonant")
    )
    simulator.run()
    simulator.save_dataset()
    simulator.export_stats_to_csv()

    # Example 2: Run with random weights to explore the parameter space
    print("Running random weight exploration...")
    simulator = MonteCarloSimulator(
        num_simulations=30,
        base_config_file=Path(__file__).parent.parent / "configs" / "chord_progression.yaml",
        generate_random_weights=True,
        weight_variation_factor=0.8,  # High variation for wide exploration
        output_dir=os.path.join(output_dir, "random_weights")
    )
    simulator.run()
    simulator.save_dataset()
    simulator.export_stats_to_csv()
    simulator.generate_weight_correlation_report("weight_correlation.json")

    print("Monte Carlo weight studies completed!")
    print(f"Results saved in {output_dir}/")


if __name__ == "__main__":
    main()
