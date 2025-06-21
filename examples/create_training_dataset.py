"""Example script for creating and analyzing a training dataset.

This script demonstrates how to use the Monte Carlo simulator to generate
a training dataset with labeled configuration parameters for MIDI files.
The dataset is suitable for machine learning applications that learn the
relationship between configurations and musical output.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monte_carlo import MonteCarloSimulator
from dataset import DatasetManager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    # Create output directory
    output_dir = "training_dataset_example"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure the Monte Carlo simulator with diverse parameters
    simulator = MonteCarloSimulator(
        num_simulations=50,
        output_dir=output_dir,
        param_ranges={
            "randomness_factor": (0.1, 0.9),
            "variation_probability": (0.2, 0.8),
            "sequence_length": (4, 16)
        },
        rhythm_param_ranges={
            "subdivision": (2, 16),
            "variation_probability": (0.1, 0.8),
            "shift_probability": (0.0, 0.5)
        },
        generate_random_weights=True,
        weight_variation_factor=0.7,
        use_rhythm=True
    )
    
    print("Running Monte Carlo simulations...")
    dataset = simulator.run()
    
    # Save standard results
    simulator.save_dataset("simulation_results.json")
    
    # Export as a training dataset
    print("Exporting training dataset...")
    dataset_path = simulator.export_training_dataset("training_dataset")
    print(f"Training dataset saved to: {os.path.join(output_dir, 'training_dataset')}")
    
    # Load and analyze the dataset
    try:
        dataset_dir = os.path.join(output_dir, "training_dataset")
        print("\nAnalyzing the generated dataset...")
        
        dataset_manager = DatasetManager(dataset_dir)
        features_df = dataset_manager.get_features_dataframe()
        
        print(f"\nDataset contains {len(features_df)} samples with {len(features_df.columns)} features")
        print("\nFirst few samples:")
        print(features_df.head())
        
        print("\nFeature statistics:")
        # Only include numeric columns for statistics
        numeric_df = features_df.select_dtypes(include=['int64', 'float64'])
        print(numeric_df.describe())
        
        # Visualize parameter distributions
        print("\nVisualizing parameter distributions...")
        dataset_manager.analyze_parameter_distributions()
        
        print("\nVisualizing feature correlations...")
        dataset_manager.analyze_feature_correlations()
        
        # Extract MIDI features
        print("\nExtracting features from MIDI files...")
        dataset_manager.extract_midi_features()
        
        print(f"\nAnalysis results saved to {dataset_dir}")
    
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
    
    print("\nProcess complete!")

if __name__ == "__main__":
    main()
