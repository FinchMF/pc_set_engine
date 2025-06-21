"""Example script for running Monte Carlo simulations with rhythm parameters.

This script demonstrates how to use the Monte Carlo simulator to explore 
different rhythm configurations and their effect on musical output.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monte_carlo import MonteCarloSimulator, generate_rhythm_variations_dataset
import numpy as np

def main():
    # Create output directory
    output_dir = "rhythm_studies"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Run a Monte Carlo simulation with rhythm
    print("\nRunning Monte Carlo simulation with rhythm variations...")
    sim = MonteCarloSimulator(
        num_simulations=20,
        base_config_file="configs/rhythm_monte_carlo.yaml",
        output_dir=os.path.join(output_dir, "general_study"),
        use_rhythm=True,
        rhythm_param_ranges={
            "subdivision": (2, 8),
            "variation_probability": (0.1, 0.8),
        }
    )
    sim.run()
    sim.save_dataset("rhythm_variations.json")
    sim.export_stats_to_csv("rhythm_stats.csv")
    
    # Example 2: Study the effect of subdivision
    print("\nStudying the effect of rhythm subdivision...")
    subdivisions = np.linspace(2, 12, 6).astype(int)
    generate_rhythm_variations_dataset(
        "configs/rhythm_monte_carlo.yaml",
        "subdivision",
        subdivisions.tolist(),
        samples_per_value=3,
        output_dir=os.path.join(output_dir, "subdivision_study")
    )
    
    # Example 3: Study the effect of different subdivision types
    print("\nStudying different subdivision types...")
    # Use values 0-4 to represent the 5 subdivision types
    subdivision_types = np.arange(5)
    generate_rhythm_variations_dataset(
        "configs/rhythm_monte_carlo.yaml",
        "subdivision_type",
        subdivision_types.tolist(),
        samples_per_value=3,
        output_dir=os.path.join(output_dir, "subdivision_type_study")
    )
    
    # Example 4: Study the effect of different time signatures
    print("\nStudying different time signatures...")
    time_signatures = np.array([3, 4, 5, 6, 7, 9, 12])
    generate_rhythm_variations_dataset(
        "configs/rhythm_monte_carlo.yaml",
        "time_signature",
        time_signatures.tolist(),
        samples_per_value=2,
        output_dir=os.path.join(output_dir, "time_signature_study")
    )
    
    print("\nRhythm studies completed!")
    print(f"Results saved in {output_dir}/")

if __name__ == "__main__":
    main()
