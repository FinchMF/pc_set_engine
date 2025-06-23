"""
Specialized dataset generation for PC Rules Engine

This module provides specialized dataset generation functions for complex cases
like multiple time signatures that may not work with the standard generator.
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import random

from dataset_generator import create_dataset_from_config
from monte_carlo import MonteCarloSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def generate_time_signatures_dataset(
    base_config_file: str,
    time_signatures: List[List[int]],
    output_dir: str,
    num_simulations_per_signature: int = 10,
    random_seed: Optional[int] = None,
    param_ranges: Optional[Dict[str, Any]] = None,
    rhythm_param_ranges: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a dataset with different time signatures.
    
    Args:
        base_config_file: Path to the base configuration file
        time_signatures: List of time signatures, e.g. [[4, 4], [3, 4], [5, 4]]
        output_dir: Directory to save the generated dataset
        num_simulations_per_signature: Number of simulations per time signature
        random_seed: Optional random seed for reproducibility
        param_ranges: Optional parameter ranges for Monte Carlo simulation
        rhythm_param_ranges: Optional rhythm parameter ranges for Monte Carlo simulation
        
    Returns:
        Dictionary containing metadata about the generated dataset
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base configuration
    with open(base_config_file, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create a mini-dataset for each time signature
    datasets = []
    total_successful = 0
    total_failed = 0
    all_simulations = []
    
    for time_sig in time_signatures:
        # Create time signature specific output directory
        time_sig_name = f"{time_sig[0]}_{time_sig[1]}"
        time_sig_dir = os.path.join(output_dir, f"time_sig_{time_sig_name}")
        os.makedirs(time_sig_dir, exist_ok=True)
        
        # Modify base config to use this time signature
        ts_config = base_config.copy()
        if "rhythm" not in ts_config:
            ts_config["rhythm"] = {}
        ts_config["rhythm"]["time_signature"] = time_sig
        
        # Save modified config
        ts_config_file = os.path.join(time_sig_dir, f"config_{time_sig_name}.yaml")
        with open(ts_config_file, 'w') as f:
            yaml.dump(ts_config, f)
        
        # Run Monte Carlo simulation
        try:
            simulator = MonteCarloSimulator(
                num_simulations=num_simulations_per_signature,
                base_config_file=ts_config_file,
                output_dir=time_sig_dir,
                param_ranges=param_ranges or {},
                use_rhythm=True,
                rhythm_param_ranges=rhythm_param_ranges or {}
            )
            
            simulator.run()
            
            # Save dataset
            dataset_file = os.path.join(time_sig_dir, f"dataset_{time_sig_name}.json")
            
            # Get simulation results
            simulation_results = []
            if hasattr(simulator, 'results'):
                simulation_results = simulator.results
            elif hasattr(simulator, 'simulations'):
                simulation_results = simulator.simulations
            
            # Add time signature info to each simulation
            for sim in simulation_results:
                sim['time_signature'] = time_sig
                sim['group'] = f"time_sig_{time_sig_name}"
                all_simulations.append(sim)
            
            # Create dataset manually
            dataset = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "num_simulations": num_simulations_per_signature,
                    "time_signature": time_sig,
                    "base_config": ts_config,
                },
                "simulations": simulation_results
            }
            
            # Save dataset
            with open(dataset_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            # Calculate success and failure counts
            successful = sum(1 for sim in simulation_results if sim.get('success', False))
            failed = num_simulations_per_signature - successful
            
            total_successful += successful
            total_failed += failed
            
            # Also save stats as CSV
            try:
                stats_file = os.path.join(time_sig_dir, f"stats_{time_sig_name}.csv")
                
                # Create stats dataframe manually
                stats_data = []
                for sim in simulation_results:
                    # Extract basic stats for each simulation
                    sim_stats = {
                        "id": sim.get("id", 0),
                        "success": sim.get("success", False),
                        "execution_time": sim.get("execution_time", 0),
                        "time_signature": f"{time_sig[0]}/{time_sig[1]}"
                    }
                    
                    # Add any statistics if available
                    if "statistics" in sim:
                        for stat_key, stat_value in sim.get("statistics", {}).items():
                            # Flatten simple values
                            if isinstance(stat_value, (int, float, str, bool)):
                                sim_stats[stat_key] = stat_value
                    
                    stats_data.append(sim_stats)
                
                if stats_data:
                    # Convert to DataFrame and save
                    import pandas as pd
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_csv(stats_file, index=False)
                    logger.info(f"Statistics saved to: {stats_file}")
            except Exception as e:
                logger.error(f"Failed to save statistics: {str(e)}")
                stats_file = None
            
            # Add to list of datasets
            datasets.append({
                "time_signature": time_sig,
                "dataset_file": dataset_file,
                "stats_file": stats_file,
                "directory": time_sig_dir,
                "successful": successful,
                "failed": failed
            })
            
            logger.info(f"Completed time signature {time_sig[0]}/{time_sig[1]} with {successful} successful simulations")
            
        except Exception as e:
            logger.error(f"Failed to process time signature {time_sig}: {str(e)}")
            total_failed += num_simulations_per_signature
    
    # Create consolidated dataset
    consolidated_dataset = {
        "metadata": {
            "generator_version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "base_config": base_config_file,
            "time_signatures": time_signatures,
            "num_simulations_per_signature": num_simulations_per_signature,
            "total_simulations": len(time_signatures) * num_simulations_per_signature,
            "successful_simulations": total_successful,
            "failed_simulations": total_failed,
            "random_seed": random_seed
        },
        "simulations": all_simulations
    }
    
    # Save consolidated dataset
    consolidated_file = os.path.join(output_dir, "time_signatures_dataset.json")
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_dataset, f, indent=2)
    
    logger.info(f"Consolidated dataset saved to: {consolidated_file}")
    
    # Create consolidated metadata
    metadata = {
        "generator_version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "base_config": base_config_file,
        "time_signatures": time_signatures,
        "num_simulations_per_signature": num_simulations_per_signature,
        "total_simulations": len(time_signatures) * num_simulations_per_signature,
        "successful_simulations": total_successful,
        "failed_simulations": total_failed,
        "datasets": datasets,
        "random_seed": random_seed,
        "consolidated_dataset": consolidated_file
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "time_signatures_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Time signatures dataset generation complete. Metadata saved to: {metadata_file}")
    logger.info(f"Total simulations: {len(time_signatures) * num_simulations_per_signature}")
    logger.info(f"Successful: {total_successful}, Failed: {total_failed}")
    
    return metadata


def fix_dataset_statistics(dataset_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Fix the statistics in a dataset file by recounting the successful and failed simulations.
    
    Args:
        dataset_file: Path to the dataset file
        output_file: Path to save the fixed dataset (if None, overwrites the original)
        
    Returns:
        Fixed dataset metadata
    """
    # Load the dataset
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    # Get simulations
    simulations = dataset.get('simulations', [])
    
    # Count successful and failed simulations
    successful = sum(1 for sim in simulations if sim.get('success', False))
    failed = len(simulations) - successful
    
    # Update metadata
    if 'metadata' in dataset:
        dataset['metadata']['successful_simulations'] = successful
        dataset['metadata']['failed_simulations'] = failed
    
    # Save the fixed dataset
    out_file = output_file or dataset_file
    with open(out_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Fixed dataset statistics: {successful} successful, {failed} failed")
    logger.info(f"Updated dataset saved to: {out_file}")
    
    return dataset.get('metadata', {})


# Add a new function to generate a complete dataset
def generate_complete_dataset(config_file: str, output_dir: str, random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a complete dataset including special cases like time signatures.
    
    Args:
        config_file: Path to the dataset configuration file
        output_dir: Directory to save the generated dataset
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Dictionary containing metadata about the generated dataset
    """
    # First, generate the main dataset
    from dataset_generator import create_dataset_from_config
    main_metadata = create_dataset_from_config(config_file, output_dir, random_seed)
    
    # Load the configuration file to check for special cases
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process complex_time_signatures group if it exists
    for group in config.get('simulation_groups', []):
        if group.get('name') == 'complex_time_signatures':
            # Extract parameters
            base_config = group.get('base_config')
            time_signatures = group.get('rhythm_param_ranges', {}).get('time_signature', [])
            num_simulations = group.get('num_simulations', 40)
            
            # Determine number of simulations per time signature
            if isinstance(time_signatures, list) and time_signatures:
                num_per_signature = num_simulations // len(time_signatures)
                if num_per_signature < 1:
                    num_per_signature = 1
                
                # Generate time signatures dataset
                ts_output_dir = os.path.join(output_dir, 'complex_time_signatures')
                time_sig_metadata = generate_time_signatures_dataset(
                    base_config_file=base_config,
                    time_signatures=time_signatures,
                    output_dir=ts_output_dir,
                    num_simulations_per_signature=num_per_signature,
                    random_seed=random_seed,
                    param_ranges=group.get('param_ranges', {}),
                    rhythm_param_ranges={k: v for k, v in group.get('rhythm_param_ranges', {}).items() 
                                       if k != 'time_signature'}
                )
                
                # Update main metadata
                main_metadata['time_signatures_metadata'] = time_sig_metadata
    
    # Fix statistics for the main dataset file
    main_dataset_file = os.path.join(output_dir, f"{config.get('dataset_name')}.json")
    if os.path.exists(main_dataset_file):
        fix_dataset_statistics(main_dataset_file)
    
    return main_metadata


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate specialized datasets for PC Rules Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Time signatures dataset command
    ts_parser = subparsers.add_parser("time_signatures", help="Generate dataset with different time signatures")
    ts_parser.add_argument("base_config", help="Base configuration file")
    ts_parser.add_argument("--output-dir", "-o", default="time_signatures_dataset", help="Output directory")
    ts_parser.add_argument("--num-per-sig", "-n", type=int, default=10, help="Number of simulations per time signature")
    ts_parser.add_argument("--seed", "-s", type=int, help="Random seed")
    ts_parser.add_argument("--time-signatures", "-t", nargs="+", default=["4/4", "3/4", "5/4", "7/8", "6/8"],
                         help="Time signatures in format '4/4 3/4 5/4'")
    
    # Fix dataset statistics command
    fix_parser = subparsers.add_parser("fix_stats", help="Fix dataset statistics")
    fix_parser.add_argument("dataset_file", help="Dataset file to fix")
    fix_parser.add_argument("--output-file", "-o", help="Output file (default: overwrite original)")
    
    # Complete dataset generation command
    complete_parser = subparsers.add_parser("complete", help="Generate complete dataset including special cases")
    complete_parser.add_argument("config_file", help="Dataset configuration file")
    complete_parser.add_argument("--output-dir", "-o", default="dataset", help="Output directory")
    complete_parser.add_argument("--seed", "-s", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    if args.command == "time_signatures":
        # Parse time signatures
        time_signatures = []
        for ts_str in args.time_signatures:
            if "/" in ts_str:
                numerator, denominator = ts_str.split("/")
                time_signatures.append([int(numerator), int(denominator)])
        
        generate_time_signatures_dataset(
            base_config_file=args.base_config,
            time_signatures=time_signatures,
            output_dir=args.output_dir,
            num_simulations_per_signature=args.num_per_sig,
            random_seed=args.seed
        )
    elif args.command == "fix_stats":
        fix_dataset_statistics(
            dataset_file=args.dataset_file,
            output_file=args.output_file
        )
    elif args.command == "complete":
        generate_complete_dataset(
            config_file=args.config_file,
            output_dir=args.output_dir,
            random_seed=args.seed
        )
    else:
        parser.print_help()
