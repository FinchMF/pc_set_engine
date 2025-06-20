"""Monte Carlo simulation for the pitch class rules engine.

This module runs multiple simulations with the pitch class rules engine,
generating a dataset of sequences with varying parameters. The results
are stored in a structured format for analysis.

Features:
- Parameter space exploration through controlled randomization
- Parallel simulation execution for efficiency
- Structured dataset generation with metadata
- Configurable output formats (JSON, CSV)
- Progress tracking and logging

Examples:
    Run a basic Monte Carlo simulation:
    ```python
    from monte_carlo import MonteCarloSimulator
    
    simulator = MonteCarloSimulator(num_simulations=100)
    dataset = simulator.run()
    simulator.save_dataset("simulation_results.json")
    ```
    
    Run with specific parameter ranges:
    ```python
    simulator = MonteCarloSimulator(
        num_simulations=50,
        param_ranges={
            "randomness_factor": (0.1, 0.9),
            "variation_probability": (0.2, 0.8)
        }
    )
    dataset = simulator.run()
    ```
"""
import os
import json
import csv
import time
import random
import argparse
import concurrent.futures
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import yaml

# Add the parent directory to sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_logger
from pc_sets.engine import EXAMPLE_CONFIG, generate_sequence_from_config
from midi import sequence_to_midi

# Initialize logger
logger = get_logger(__name__, log_file="monte_carlo.log")

class MonteCarloSimulator:
    """Monte Carlo simulator for the pitch class rules engine.
    
    This class handles running multiple simulations with varying parameters
    and collecting the results into a dataset.
    
    Attributes:
        num_simulations: Number of simulations to run
        base_config: The base configuration to modify for each simulation
        param_ranges: Dictionary of parameter ranges for randomization
        output_dir: Directory to store output files
        dataset: The generated dataset of sequences and their parameters
    """
    
    def __init__(
        self, 
        num_simulations: int = 100,
        base_config_file: Optional[str] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        output_dir: str = "datasets"
    ):
        """Initialize the Monte Carlo simulator.
        
        Args:
            num_simulations: Number of simulations to run
            base_config_file: Path to a YAML file with base configuration (optional)
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            output_dir: Directory to store output files
        """
        self.num_simulations = num_simulations
        self.output_dir = output_dir
        
        # Load base configuration
        if base_config_file:
            self.base_config = self.load_config(base_config_file)
        else:
            self.base_config = EXAMPLE_CONFIG.copy()
    
        # Set default parameter ranges if not provided
        self.param_ranges = param_ranges or {
            "randomness_factor": (0.1, 0.9),
            "variation_probability": (0.2, 0.8),
            "sequence_length": (4, 16)
        }
        
        # Initialize dataset
        self.dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_simulations": num_simulations,
                "base_config": self.base_config,
                "param_ranges": self.param_ranges
            },
            "simulations": []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized Monte Carlo simulator with {num_simulations} simulations")
    
    def load_config(self, config_file: str) -> Dict:
        """
        Load and process configuration from a file, handling special sections like midi_properties.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Processed configuration dictionary with special sections removed
        """
        try:
            with open(config_file, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # Extract and store special sections
            self.midi_properties = full_config.pop("midi_properties", None)
            
            # Return cleaned configuration
            return full_config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            raise
    
    def generate_random_config(self) -> Dict:
        """Generate a random configuration based on parameter ranges.
        
        Returns:
            A configuration dictionary with randomized parameters
        """
        config = self.base_config.copy()
        
        # Ensure midi_properties is not in the configuration
        if "midi_properties" in config:
            config.pop("midi_properties")
        
        # Randomize continuous parameters
        for param, (min_val, max_val) in self.param_ranges.items():
            if param in config and isinstance(config[param], (int, float)):
                if isinstance(config[param], int):
                    config[param] = random.randint(int(min_val), int(max_val))
                else:
                    config[param] = random.uniform(min_val, max_val)
        
        # Sometimes swap between melodic and chordal
        if random.random() < 0.5:
            if config["generation_type"] == "melodic":
                config["generation_type"] = "chordal"
                # For chordal, use a random common triad
                triads = [[0, 4, 7], [0, 3, 7], [0, 3, 6], [0, 4, 8]]
                config["start_pc"] = random.choice(triads)
                # If progression is enabled, set a target triad
                if config.get("progression", False) and config.get("progression_type") == "directed":
                    # Transpose by a random interval
                    transpose = random.randint(1, 11)
                    config["target_pc"] = [(pc + transpose) % 12 for pc in config["start_pc"]]
            else:
                config["generation_type"] = "melodic"
                # For melodic, use a random single pitch class
                config["start_pc"] = random.randint(0, 11)
                # If progression is enabled, set a target pitch class
                if config.get("progression", False) and config.get("progression_type") == "directed":
                    config["target_pc"] = random.randint(0, 11)
        
        # Randomly select progression type
        if config.get("progression", False):
            config["progression_type"] = random.choice(["directed", "random", "static"])
            
            # If not directed, remove target_pc
            if config["progression_type"] != "directed":
                config.pop("target_pc", None)
        
        return config
    
    def _run_single_simulation(self, simulation_id: int) -> Dict:
        """Run a single simulation with a randomly generated configuration.
        
        Args:
            simulation_id: Unique identifier for this simulation
            
        Returns:
            Dictionary containing simulation results
        """
        # Generate random configuration
        config = self.generate_random_config()
        
        # Record start time
        start_time = time.time()
        
        try:
            # Generate sequence
            sequence = generate_sequence_from_config(config)
            
            # Record execution time
            execution_time = time.time() - start_time
            
            # Generate statistics about the sequence
            stats = self._analyze_sequence(sequence, config["generation_type"])
            
            # Generate MIDI if output directory exists
            midi_path = None
            if self.output_dir:
                midi_filename = f"simulation_{simulation_id}.mid"
                midi_path = os.path.join(self.output_dir, midi_filename)
                try:
                    is_melodic = config["generation_type"] == "melodic"
                    
                    # Use stored midi_properties if available
                    midi_params = {
                        "tempo": 120, 
                        "base_octave": 4, 
                        "note_duration": 0.5
                    }
                    
                    if hasattr(self, 'midi_properties') and self.midi_properties:
                        midi_params.update(self.midi_properties)
                        
                    sequence_to_midi(
                        sequence, midi_path, is_melodic=is_melodic,
                        params=midi_params
                    )
                except Exception as e:
                    logger.error(f"Failed to generate MIDI for simulation {simulation_id}: {e}")
                    midi_path = None
            
            # Return simulation result
            return {
                "id": simulation_id,
                "config": config,
                "sequence": sequence,
                "midi_file": midi_path,
                "execution_time": execution_time,
                "statistics": stats,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {e}")
            return {
                "id": simulation_id,
                "config": config,
                "success": False,
                "error": str(e)
            }
    
    def _analyze_sequence(self, sequence: List, generation_type: str) -> Dict:
        """Analyze a sequence and extract statistical features.
        
        Args:
            sequence: The generated sequence
            generation_type: The type of sequence ("melodic" or "chordal")
            
        Returns:
            Dictionary of statistical features
        """
        stats = {}
        
        if generation_type == "melodic":
            # Analysis for melodic sequences
            intervals = [abs((sequence[i+1] - sequence[i]) % 12) 
                        for i in range(len(sequence)-1)]
            
            stats["unique_pitch_classes"] = len(set(sequence))
            stats["mean_interval"] = float(np.mean(intervals))
            stats["median_interval"] = float(np.median(intervals))
            stats["max_interval"] = max(intervals)
            stats["repeated_notes"] = sum(1 for i in range(len(sequence)-1) 
                                         if sequence[i] == sequence[i+1])
            
            # Direction changes (contour changes)
            if len(sequence) > 2:
                directions = [1 if sequence[i+1] > sequence[i] else 
                             (-1 if sequence[i+1] < sequence[i] else 0) 
                             for i in range(len(sequence)-1)]
                direction_changes = sum(1 for i in range(len(directions)-1) 
                                       if directions[i] != directions[i+1] 
                                       and directions[i] != 0 and directions[i+1] != 0)
                stats["direction_changes"] = direction_changes
            
        else:  # chordal
            # Analysis for chord sequences
            stats["mean_chord_size"] = float(np.mean([len(chord) for chord in sequence]))
            
            # Calculate chord variety
            unique_chords = set(tuple(sorted(chord)) for chord in sequence)
            stats["unique_chords"] = len(unique_chords)
            
            # Calculate average pitch class changes between consecutive chords
            if len(sequence) > 1:
                pc_changes = []
                for i in range(len(sequence)-1):
                    curr_set = set(sequence[i])
                    next_set = set(sequence[i+1])
                    changes = len(curr_set.symmetric_difference(next_set))
                    pc_changes.append(changes)
                
                stats["mean_pc_changes"] = float(np.mean(pc_changes))
        
        return stats
    
    def run(self, parallel: bool = True, max_workers: Optional[int] = None) -> Dict:
        """Run the Monte Carlo simulation.
        
        Args:
            parallel: Whether to run simulations in parallel
            max_workers: Maximum number of worker threads/processes
            
        Returns:
            The complete dataset with all simulation results
        """
        logger.info(f"Starting Monte Carlo simulation with {self.num_simulations} iterations")
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(total=self.num_simulations, desc="Running simulations")
        
        if parallel:
            # Parallel execution
            max_workers = max_workers or min(32, os.cpu_count() + 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_id = {executor.submit(self._run_single_simulation, i): i 
                              for i in range(self.num_simulations)}
                
                for future in concurrent.futures.as_completed(future_to_id):
                    result = future.result()
                    self.dataset["simulations"].append(result)
                    pbar.update(1)
        else:
            # Sequential execution
            for i in range(self.num_simulations):
                result = self._run_single_simulation(i)
                self.dataset["simulations"].append(result)
                pbar.update(1)
        
        pbar.close()
        
        # Record total execution time
        total_time = time.time() - start_time
        self.dataset["metadata"]["total_execution_time"] = total_time
        
        logger.info(f"Monte Carlo simulation completed in {total_time:.2f} seconds")
        return self.dataset
    
    def save_dataset(self, filename: str = "monte_carlo_dataset.json") -> str:
        """Save the dataset to a file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.dataset, f, indent=2)
            
            logger.info(f"Dataset saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def export_stats_to_csv(self, filename: str = "monte_carlo_stats.csv") -> str:
        """Export statistical data to a CSV file for analysis.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        file_path = os.path.join(self.output_dir, filename)
        
        # Extract relevant data for CSV
        rows = []
        all_fields = set()  # Track all possible field names across all simulations
        
        # First pass: collect data and gather all possible fields
        for sim in self.dataset["simulations"]:
            if sim["success"]:
                row = {
                    "id": sim["id"],
                    "generation_type": sim["config"]["generation_type"],
                    "sequence_length": sim["config"]["sequence_length"],
                    "randomness_factor": sim["config"]["randomness_factor"],
                    "variation_probability": sim["config"]["variation_probability"],
                    "progression": sim["config"]["progression"],
                    "execution_time": sim["execution_time"]
                }
                
                # Add statistics
                for key, value in sim["statistics"].items():
                    row[key] = value
                    all_fields.add(key)
                
                rows.append(row)
        
        try:
            with open(file_path, 'w', newline='') as f:
                if rows:
                    # Create a list of all field names to ensure consistent columns
                    fieldnames = ["id", "generation_type", "sequence_length", "randomness_factor", 
                                 "variation_probability", "progression", "execution_time"] + list(all_fields)
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    
                    # Ensure each row has all possible fields defined (with None for missing values)
                    for row in rows:
                        for field in all_fields:
                            if field not in row:
                                row[field] = None
                        writer.writerow(row)
                        
                    logger.info(f"Stats exported to {file_path}")
                else:
                    logger.warning("No successful simulations to export")
            
            return file_path
        except Exception as e:
            logger.error(f"Failed to export stats: {e}")
            raise


def generate_variations_dataset(
    base_config_file: str,
    param_name: str,
    values: List[float],
    samples_per_value: int = 10,
    output_dir: str = "variation_dataset"
) -> str:
    """Generate a dataset by varying a single parameter through a range of values.
    
    Args:
        base_config_file: Path to the base configuration YAML file
        param_name: Name of the parameter to vary
        values: List of values to use for the parameter
        samples_per_value: Number of samples to generate for each parameter value
        output_dir: Directory to store the results
        
    Returns:
        Path to the saved dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base configuration
    with open(base_config_file, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create dataset structure
    dataset = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "base_config": base_config,
            "varied_parameter": param_name,
            "parameter_values": values,
            "samples_per_value": samples_per_value
        },
        "variations": {}
    }
    
    # Generate sequences for each parameter value
    for value in tqdm(values, desc=f"Varying {param_name}"):
        dataset["variations"][str(value)] = []
        
        for i in range(samples_per_value):
            # Create a new configuration with this parameter value
            config = base_config.copy()
            config[param_name] = value
            
            try:
                # Generate sequence
                sequence = generate_sequence_from_config(config)
                
                # Save to dataset
                dataset["variations"][str(value)].append({
                    "sample_id": i,
                    "sequence": sequence
                })
                
                # Generate MIDI file
                midi_path = os.path.join(output_dir, f"{param_name}_{value}_{i}.mid")
                is_melodic = config["generation_type"] == "melodic"
                sequence_to_midi(
                    sequence, midi_path, is_melodic=is_melodic,
                    params={"tempo": 120, "base_octave": 4}
                )
                
            except Exception as e:
                logger.error(f"Failed to generate sample for {param_name}={value}: {e}")
    
    # Save the dataset
    output_file = os.path.join(output_dir, f"{param_name}_variations.json")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Variations dataset saved to {output_file}")
    return output_file


def main():
    """Command-line interface for the Monte Carlo simulator."""
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations with the pitch class rules engine")
    
    parser.add_argument('--num-simulations', type=int, default=100,
                      help='Number of simulations to run')
    
    parser.add_argument('--base-config', type=str,
                      help='Path to base configuration YAML file')
    
    parser.add_argument('--output-dir', type=str, default='datasets',
                      help='Directory to store output files')
    
    parser.add_argument('--no-parallel', action='store_true',
                      help='Disable parallel execution')
    
    parser.add_argument('--variation-mode', action='store_true',
                      help='Enable variation mode to study a single parameter')
    
    parser.add_argument('--param-name', type=str,
                      help='Parameter name to vary in variation mode')
    
    parser.add_argument('--param-min', type=float, default=0.0,
                      help='Minimum value for the varied parameter')
    
    parser.add_argument('--param-max', type=float, default=1.0,
                      help='Maximum value for the varied parameter')
    
    parser.add_argument('--param-steps', type=int, default=10,
                      help='Number of steps between min and max values')
    
    parser.add_argument('--samples-per-value', type=int, default=5,
                      help='Number of samples to generate for each parameter value')
    
    args = parser.parse_args()
    
    # Configuration validation
    if args.variation_mode and not args.param_name:
        parser.error("--param-name is required with --variation-mode")
    
    if args.variation_mode and not args.base_config:
        parser.error("--base-config is required with --variation-mode")
    
    try:
        if args.variation_mode:
            # Run variation mode
            values = np.linspace(args.param_min, args.param_max, args.param_steps)
            output_file = generate_variations_dataset(
                args.base_config,
                args.param_name,
                values.tolist(),
                args.samples_per_value,
                args.output_dir
            )
            print(f"Variation dataset saved to: {output_file}")
        else:
            # Run Monte Carlo simulation
            simulator = MonteCarloSimulator(
                num_simulations=args.num_simulations,
                base_config_file=args.base_config,
                output_dir=args.output_dir
            )
            
            dataset = simulator.run(parallel=not args.no_parallel)
            json_file = simulator.save_dataset()
            csv_file = simulator.export_stats_to_csv()
            
            print(f"Monte Carlo simulation completed:")
            print(f"- JSON dataset: {json_file}")
            print(f"- CSV statistics: {csv_file}")
            print(f"- Generated {args.num_simulations} samples")
    
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
