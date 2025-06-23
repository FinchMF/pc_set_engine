"""Example script for generating datasets using the dataset generator.

This script demonstrates how to use the dataset generator to create
different types of training datasets based on configuration files.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_generator import create_dataset_from_config
import json
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    # Create base output directory
    base_output_dir = "datasets"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Example 1: Generate a small test dataset
    logger.info("Generating small test dataset...")
    small_dataset_start = time.time()
    
    small_dataset_output = os.path.join(base_output_dir, "small_test")
    small_dataset_config = "configs/dataset_configs/small_test_dataset.yaml"
    
    small_dataset_metadata = create_dataset_from_config(
        small_dataset_config,
        small_dataset_output,
        random_seed=42
    )
    
    logger.info(f"Small test dataset generation completed in {time.time() - small_dataset_start:.2f} seconds")
    logger.info(f"Generated {small_dataset_metadata['total_simulations']} simulations")
    logger.info(f"Results saved in {small_dataset_output}/")
    
    # Example 2: Generate a medium custom dataset with specific parameters
    logger.info("\nGenerating a custom specialized dataset...")
    
    # Create a custom configuration with direct dictionary instead of file
    custom_config = {
        "dataset_name": "specialized_dataset",
        "description": "Dataset focused on directed chord progressions",
        "simulation_groups": [
            {
                "name": "major_to_minor",
                "num_simulations": 10,
                "base_config": {
                    "start_pc": [0, 4, 7],
                    "generation_type": "chordal",
                    "sequence_length": 8,
                    "progression": True,
                    "target_pc": [0, 3, 7],
                    "progression_type": "directed",
                    "allowed_operations": ["transpose", "invert", "add_note"],
                    "constraints": {
                        "max_interval": 3,
                        "vary_pc": True
                    },
                    "randomness_factor": 0.4,
                    "variation_probability": 0.5
                },
                "use_rhythm": True,
                "param_ranges": {
                    "randomness_factor": [0.2, 0.6],
                    "variation_probability": [0.4, 0.7]
                }
            },
            {
                "name": "pentatonic_melodic",
                "num_simulations": 10,
                "base_config": {
                    "start_pc": 0,
                    "generation_type": "melodic",
                    "sequence_length": 16,
                    "progression": True,
                    "progression_type": "random",
                    "allowed_operations": ["transpose", "invert", "add_note"],
                    "constraints": {
                        "max_interval": 4,
                        "vary_pc": True,
                        "pitch_set": [0, 2, 4, 7, 9]  # Pentatonic scale
                    },
                    "randomness_factor": 0.5,
                    "variation_probability": 0.6
                },
                "rhythm_param_ranges": {
                    "time_signature": [[4, 4], [6, 8]],
                    "subdivision_type": ["regular", "shuffle"]
                }
            }
        ]
    }
    
    # Save the custom configuration to a file
    custom_config_path = "configs/dataset_configs/specialized_dataset.yaml"
    os.makedirs(os.path.dirname(custom_config_path), exist_ok=True)
    
    with open(custom_config_path, 'w') as file:
        import yaml
        yaml.dump(custom_config, file)
    
    custom_dataset_output = os.path.join(base_output_dir, "specialized")
    
    # Generate the custom dataset
    custom_dataset_start = time.time()
    custom_dataset_metadata = create_dataset_from_config(
        custom_config_path,
        custom_dataset_output,
        random_seed=43
    )
    
    logger.info(f"Custom dataset generation completed in {time.time() - custom_dataset_start:.2f} seconds")
    logger.info(f"Generated {custom_dataset_metadata['total_simulations']} simulations")
    logger.info(f"Results saved in {custom_dataset_output}/")
    
    # Example 3: Generate a large diverse dataset (commented out as it would take longer)
    """
    logger.info("\nGenerating large diverse dataset...")
    large_dataset_output = os.path.join(base_output_dir, "large_diverse")
    large_dataset_config = "configs/dataset_configs/large_dataset.yaml"
    
    # This would take a considerable amount of time to run
    large_dataset_metadata = create_dataset_from_config(
        large_dataset_config,
        large_dataset_output
    )
    
    logger.info(f"Large dataset generation completed")
    logger.info(f"Generated {large_dataset_metadata['total_simulations']} simulations")
    logger.info(f"Results saved in {large_dataset_output}/")
    """
    
    # Analysis example - show basic stats about the generated datasets
    logger.info("\nDataset Generation Summary:")
    logger.info("--------------------------")
    logger.info(f"Small Test Dataset: {small_dataset_metadata['successful_simulations']} successful simulations")
    logger.info(f"Specialized Dataset: {custom_dataset_metadata['successful_simulations']} successful simulations")
    
    # Total execution time
    logger.info(f"\nTotal script execution time: {time.time() - small_dataset_start:.2f} seconds")

if __name__ == "__main__":
    main()
