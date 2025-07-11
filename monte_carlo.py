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
from pc_sets.pitch_classes import INTERVAL_WEIGHT_PROFILES
from pc_sets.rhythm import EXAMPLE_RHYTHM_CONFIG, SubdivisionType, AccentType
from midi import sequence_to_midi, sequence_to_midi_with_rhythm

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
        output_dir: str = "datasets",
        interval_weight_configs: Optional[List[Dict[int, float]]] = None,
        generate_random_weights: bool = False,
        weight_variation_factor: float = 0.5,
        rhythm_config_file: Optional[str] = None,
        rhythm_param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        use_rhythm: bool = False
    ):
        """Initialize the Monte Carlo simulator.
        
        Args:
            num_simulations: Number of simulations to run
            base_config_file: Path to a YAML file with base configuration (optional)
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            output_dir: Directory to store output files
            interval_weight_configs: List of interval weight configurations to cycle through
                during simulations. Each config is a dictionary mapping interval classes (1-6)
                to weight factors.
            generate_random_weights: If True, generate random interval weights for each simulation
            weight_variation_factor: Controls the amount of variation in random weights (0.0-1.0)
            rhythm_config_file: Path to a YAML file with rhythm configuration
            rhythm_param_ranges: Dictionary mapping rhythm parameter names to (min, max) tuples
            use_rhythm: Whether to apply rhythm to the generated sequences
        """
        self.num_simulations = num_simulations
        self.output_dir = output_dir
        self.generate_random_weights = generate_random_weights
        self.weight_variation_factor = max(0.0, min(1.0, weight_variation_factor))
        
        # Initialize rhythm-related attributes
        self.use_rhythm = use_rhythm  # Add this line to fix the error
        self.rhythm_param_ranges = rhythm_param_ranges or {
            "subdivision": (2, 8),
            "variation_probability": (0.1, 0.8),
            "shift_probability": (0.0, 0.5)
        }
        
        # Load rhythm configuration if provided
        self.rhythm_config = None
        if rhythm_config_file:
            try:
                with open(rhythm_config_file, 'r') as f:
                    self.rhythm_config = yaml.safe_load(f)
                logger.info(f"Loaded rhythm configuration from {rhythm_config_file}")
            except Exception as e:
                logger.error(f"Failed to load rhythm configuration from {rhythm_config_file}: {e}")
                self.rhythm_config = EXAMPLE_RHYTHM_CONFIG.copy()
        elif use_rhythm:
            # Use default rhythm config if not provided but rhythm is enabled
            self.rhythm_config = EXAMPLE_RHYTHM_CONFIG.copy()
        
        # Initialize interval weight configurations
        if interval_weight_configs:
            self.interval_weight_configs = interval_weight_configs
        elif generate_random_weights:
            # We'll generate them on-the-fly during simulation
            self.interval_weight_configs = None
        else:
            # Use predefined profiles as defaults
            self.interval_weight_configs = list(INTERVAL_WEIGHT_PROFILES.values())
        
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
    
    def generate_random_interval_weights(self) -> Dict[int, float]:
        """Generate random interval weights.
        
        The weights are centered around 1.0, with variation determined by
        the weight_variation_factor. Higher variation factors result in
        greater differences between weights.
        
        Returns:
            Dict mapping interval classes (1-6) to weight factors
        """
        # Variation range depends on the weight variation factor
        var_range = 0.2 + (self.weight_variation_factor * 1.3)  # Range: 0.2-1.5
        
        # Generate weights centered around 1.0
        return {
            i: max(0.1, random.uniform(1.0 - var_range, 1.0 + var_range))
            for i in range(1, 7)
        }
    
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
        
        # Add interval weights if configured
        if self.generate_random_weights:
            # Generate new random weights for this simulation
            config["interval_weights"] = self.generate_random_interval_weights()
        elif self.interval_weight_configs:
            # Cycle through provided weight configurations
            config_index = random.randint(0, len(self.interval_weight_configs) - 1)
            config["interval_weights"] = self.interval_weight_configs[config_index]
        
        # Add rhythm parameters if enabled
        if self.use_rhythm:
            rhythm_config = self.rhythm_config.copy() if self.rhythm_config else {}
            
            # Randomize continuous rhythm parameters
            for param, (min_val, max_val) in self.rhythm_param_ranges.items():
                if param in rhythm_config and isinstance(rhythm_config[param], (int, float)):
                    if isinstance(rhythm_config[param], int):
                        rhythm_config[param] = random.randint(int(min_val), int(max_val))
                    else:
                        rhythm_config[param] = random.uniform(min_val, max_val)
            
            # Randomly select subdivision type
            if random.random() < 0.7: # 70% chance to vary subdivision type
                subdivision_types = ["regular", "swing", "dotted", "shuffle", "complex"]
                rhythm_config["subdivision_type"] = random.choice(subdivision_types)
            
            # Randomly select accent type
            if random.random() < 0.7: # 70% chance to vary accent type
                accent_types = ["downbeat", "offbeat", "syncopated"]
                rhythm_config["accent_type"] = random.choice(accent_types)
                
            # Randomly select time signature
            if random.random() < 0.5: # 50% chance to vary time signature
                common_time_sigs = [(4, 4), (3, 4), (6, 8), (5, 4), (7, 8)]
                rhythm_config["time_signature"] = random.choice(common_time_sigs)
            
            # Add to configuration
            config["rhythm"] = rhythm_config
        
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
            # If using rhythm, separate it from the main config before generation
            rhythm_config = None
            if self.use_rhythm and "rhythm" in config:
                rhythm_config = config.pop("rhythm")
            
            # Generate sequence
            sequence = generate_sequence_from_config(config)
            
            # Put rhythm back for the result
            if rhythm_config:
                config["rhythm"] = rhythm_config
            
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
                    
                    # Apply rhythm if enabled
                    if self.use_rhythm and "rhythm" in config:
                        sequence_to_midi_with_rhythm(
                            sequence, 
                            midi_path, 
                            is_melodic=is_melodic,
                            rhythm_config=config["rhythm"],
                            params=midi_params
                        )
                    else:
                        sequence_to_midi(
                            sequence, 
                            midi_path, 
                            is_melodic=is_melodic,
                            params=midi_params
                        )
                except Exception as e:
                    logger.error(f"Failed to generate MIDI for simulation {simulation_id}: {e}")
                    midi_path = None

            # Return simulation result with interval weights information
            result = {
                "id": simulation_id,
                "config": config,
                "sequence": sequence,
                "midi_file": midi_path,
                "execution_time": execution_time,
                "statistics": stats,
                "success": True
            }
            
            # Include interval weights in the result if used
            if "interval_weights" in config:
                result["interval_weights"] = config["interval_weights"]
            
            # Include rhythm configuration in the result if used
            if self.use_rhythm and "rhythm" in config:
                result["rhythm_config"] = config["rhythm"]
        
            return result
        
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
        
        # Additional analysis based on interval content
        if generation_type == "chordal" and len(sequence) > 0:
            # Check if weighted interval characteristics match our expectations
            # This can be useful for validating whether interval weights affected the output
            try:
                from pc_sets.pitch_classes import PitchClassSet
                
                # Analyze the interval content of the first and last chords
                first_chord = PitchClassSet(sequence[0])
                last_chord = PitchClassSet(sequence[-1])
                
                # Get interval profiles
                stats["first_chord_profile"] = first_chord.get_interval_profile()
                stats["last_chord_profile"] = last_chord.get_interval_profile()
                
                # Calculate profile difference between first and last chord
                dissonance_change = stats["last_chord_profile"]["dissonance_ratio"] - stats["first_chord_profile"]["dissonance_ratio"]
                consonance_change = stats["last_chord_profile"]["consonance_ratio"] - stats["first_chord_profile"]["consonance_ratio"]
                
                stats["dissonance_change"] = float(dissonance_change)
                stats["consonance_change"] = float(consonance_change)
            except Exception as e:
                logger.warning(f"Could not perform interval profile analysis: {e}")
        
        # Add rhythm analysis if available
        if hasattr(self, 'rhythm_config') and self.rhythm_config:
            try:
                # Calculate rhythmic complexity metrics
                if generation_type == "melodic":
                    # Estimate rhythmic density and syncopation based on subdivision
                    subdivision = self.rhythm_config.get("subdivision", 4)
                    stats["estimated_rhythm_density"] = min(1.0, subdivision / 8)
                    
                    # Estimate syncopation based on accent type
                    accent_type = self.rhythm_config.get("accent_type", "downbeat")
                    if accent_type == "syncopated":
                        stats["estimated_syncopation"] = 0.8
                    elif accent_type == "offbeat":
                        stats["estimated_syncopation"] = 0.6
                    else:
                        stats["estimated_syncopation"] = 0.2
            except Exception as e:
                logger.warning(f"Could not perform rhythm analysis: {e}")
        
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
                
                # Add interval weights information if available
                if "interval_weights" in sim:
                    for interval, weight in sim["interval_weights"].items():
                        key = f"weight_interval_{interval}"
                        row[key] = weight
                        all_fields.add(key)
                
                # Add rhythm information if available
                if self.use_rhythm and "rhythm_config" in sim:
                    rhythm_config = sim["rhythm_config"]
                    for key, value in rhythm_config.items():
                        # Skip complex data structures that can't be easily represented in CSV
                        if isinstance(value, (int, float, str)):
                            key_name = f"rhythm_{key}"
                            row[key_name] = value
                            all_fields.add(key_name)
                
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
    
    def generate_weight_correlation_report(self, output_path: str = None) -> Dict:
        """Generate a report analyzing correlations between interval weights and musical characteristics.
        
        This report can help understand how different interval weightings affect the 
        generated music in the Monte Carlo simulation.
        
        Args:
            output_path: Path where to save the JSON report (optional)
        
        Returns:
            Dictionary containing the correlation data
        """
        # Initialize correlation data
        correlation_data = {
            "melodic": {},
            "chordal": {}
        }
        
        # Collect data points
        melodic_data = []
        chordal_data = []
        
        for sim in self.dataset["simulations"]:
            if not sim["success"] or "interval_weights" not in sim:
                continue
            
            weights = sim["interval_weights"]
            stats = sim["statistics"]
            gen_type = sim["config"]["generation_type"]
            
            data_point = {
                f"weight_{i}": weights.get(str(i), weights.get(i, 1.0))
                for i in range(1, 7)
            }
            
            # Add statistics
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    data_point[key] = value
            
            # Add to the appropriate dataset
            if gen_type == "melodic":
                melodic_data.append(data_point)
            else:
                chordal_data.append(data_point)
        
        # Calculate correlations for melodic sequences
        if melodic_data:
            try:
                import pandas as pd
                df = pd.DataFrame(melodic_data)
                
                # Identify metrics to correlate with weights
                metrics = [col for col in df.columns if not col.startswith('weight_')]
                
                for metric in metrics:
                    correlation_data["melodic"][metric] = {}
                    for i in range(1, 7):
                        weight_key = f"weight_{i}"
                        if weight_key in df.columns:
                            corr = df[weight_key].corr(df[metric])
                            correlation_data["melodic"][metric][f"interval_{i}"] = float(corr)
            except ImportError:
                logger.warning("Pandas not available for correlation analysis")
        
        # Calculate correlations for chordal sequences
        if chordal_data:
            try:
                import pandas as pd
                df = pd.DataFrame(chordal_data)
                
                metrics = [col for col in df.columns if not col.startswith('weight_')]
                
                for metric in metrics:
                    correlation_data["chordal"][metric] = {}
                    for i in range(1, 7):
                        weight_key = f"weight_{i}"
                        if weight_key in df.columns:
                            corr = df[weight_key].corr(df[metric])
                            correlation_data["chordal"][metric][f"interval_{i}"] = float(corr)
            except ImportError:
                logger.warning("Pandas not available for correlation analysis")
        
        # Save the report if requested
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(correlation_data, f, indent=2)
                logger.info(f"Weight correlation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save weight correlation report: {e}")
        
        return correlation_data

    def export_training_dataset(self, dataset_dir: str = "training_dataset") -> str:
        """Export the simulation results as a training dataset.
        
        Creates a structured dataset where each MIDI file is associated with its
        configuration parameters as labels. This format facilitates machine learning
        applications that learn relationships between parameters and musical output.
        
        Args:
            dataset_dir: Directory to store the dataset
            
        Returns:
            Path to the saved dataset metadata file
        """
        output_dir = os.path.join(self.output_dir, dataset_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataset structure
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_samples": 0,
                "parameter_space": {},
                "feature_types": {}
            },
            "samples": []
        }
        
        # Track parameter ranges for all numeric parameters
        param_values = {}
        
        # Process each successful simulation
        for sim in self.dataset["simulations"]:
            if not sim["success"] or "midi_file" not in sim or not sim["midi_file"]:
                continue
            
            # Create a unique filename for this sample
            midi_filename = f"sample_{sim['id']}.mid"
            new_midi_path = os.path.join(output_dir, midi_filename)
            
            # Copy the MIDI file to dataset directory
            try:
                import shutil
                shutil.copy2(sim["midi_file"], new_midi_path)
                
                # Create feature vector from configuration
                features = self._extract_features(sim)
                
                # Update parameter ranges
                for name, value in features.items():
                    if isinstance(value, (int, float)):
                        if name not in param_values:
                            param_values[name] = []
                        param_values[name].append(value)
                
                # Add to dataset
                dataset["samples"].append({
                    "midi_file": midi_filename,
                    "features": features,
                    "statistics": sim["statistics"],
                    "id": sim["id"]
                })
                
                # Update sample count
                dataset["metadata"]["num_samples"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process sample {sim['id']}: {e}")
        
        # Calculate parameter space information
        for param, values in param_values.items():
            if values:
                dataset["metadata"]["parameter_space"][param] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "values": sorted(list(set(values))) if len(set(values)) < 10 else None
                }
        
        # Determine feature types
        if dataset["samples"]:
            first_sample = dataset["samples"][0]["features"]
            for name, value in first_sample.items():
                if isinstance(value, bool):
                    dataset["metadata"]["feature_types"][name] = "boolean"
                elif isinstance(value, int):
                    dataset["metadata"]["feature_types"][name] = "integer"
                elif isinstance(value, float):
                    dataset["metadata"]["feature_types"][name] = "float"
                elif isinstance(value, str):
                    dataset["metadata"]["feature_types"][name] = "categorical"
                elif isinstance(value, list):
                    dataset["metadata"]["feature_types"][name] = "array"
                elif isinstance(value, tuple):
                    dataset["metadata"]["feature_types"][name] = "tuple"
        
        # Save dataset metadata
        metadata_path = os.path.join(output_dir, "dataset.json")
        with open(metadata_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Create a CSV version for easier import into ML frameworks
        csv_path = os.path.join(output_dir, "dataset.csv")
        self._export_dataset_csv(dataset, csv_path)
        
        logger.info(f"Training dataset with {dataset['metadata']['num_samples']} samples exported to {output_dir}")
        return metadata_path

    def _extract_features(self, simulation: Dict) -> Dict:
        """Extract feature vector from simulation configuration.
        
        Args:
            simulation: A simulation result dictionary
            
        Returns:
            Dict of features extracted from the configuration
        """
        features = {}
        
        # Include generation parameters
        config = simulation["config"]
        features["generation_type"] = config["generation_type"]
        features["sequence_length"] = config["sequence_length"]
        features["randomness_factor"] = config["randomness_factor"]
        features["variation_probability"] = config["variation_probability"]
        features["progression"] = config.get("progression", False)
        features["progression_type"] = config.get("progression_type", "static")
        
        # Include rhythm parameters if available
        if "rhythm_config" in simulation:
            rhythm = simulation["rhythm_config"]
            
            # Add each rhythm parameter
            for key, value in rhythm.items():
                # Handle time signature specially since it's a tuple
                if key == "time_signature" and isinstance(value, tuple):
                    features["rhythm_time_signature_numerator"] = value[0]
                    features["rhythm_time_signature_denominator"] = value[1]
                    features["rhythm_time_signature_ratio"] = value[0] / value[1]
                # Skip complex data structures like accent patterns
                elif isinstance(value, (int, float, str, bool)):
                    features[f"rhythm_{key}"] = value
        
        # Include interval weights if available
        if "interval_weights" in simulation:
            weights = simulation["interval_weights"]
            for interval, weight in weights.items():
                features[f"weight_{interval}"] = weight
        
        return features

    def _export_dataset_csv(self, dataset: Dict, csv_path: str):
        """Export dataset to CSV format for ML frameworks.
        
        Args:
            dataset: Dataset dictionary
            csv_path: Path to save CSV file
        """
        # Extract all possible feature names
        feature_names = set()
        for sample in dataset["samples"]:
            feature_names.update(sample["features"].keys())
        
        # Sort feature names for consistency
        feature_names = sorted(list(feature_names))
        
        # Create CSV rows
        rows = []
        for sample in dataset["samples"]:
            row = {
                "midi_file": sample["midi_file"],
                "id": sample["id"]
            }
            
            # Add each feature
            for name in feature_names:
                row[name] = sample["features"].get(name, None)
            
            rows.append(row)
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ["midi_file", "id"] + feature_names
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    
def generate_variations_dataset(
    base_config_file: str,
    param_name: str,
    values: List[float],
    samples_per_value: int = 10,
    output_dir: str = "variation_dataset",
    interval_weight_profile: Optional[str] = None
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
    
    # Apply interval weight profile if specified
    if interval_weight_profile:
        if interval_weight_profile in INTERVAL_WEIGHT_PROFILES:
            base_config["interval_weights"] = interval_weight_profile
        else:
            logger.warning(f"Unknown interval weight profile '{interval_weight_profile}', using default")
    
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


def generate_rhythm_variations_dataset(
    base_config_file: str,
    rhythm_param_name: str,
    values: List[float],
    samples_per_value: int = 10,
    output_dir: str = "rhythm_variation_dataset"
) -> str:
    """Generate a dataset by varying a single rhythm parameter through a range of values."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base configuration
    with open(base_config_file, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Make sure we have a rhythm section
    if "rhythm" not in base_config:
        base_config["rhythm"] = EXAMPLE_RHYTHM_CONFIG.copy()
    
    # Create dataset structure
    dataset = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "base_config": base_config,
            "varied_parameter": f"rhythm.{rhythm_param_name}",
            "parameter_values": values,
            "samples_per_value": samples_per_value
        },
        "variations": {}
    }
    
    # Generate sequences for each parameter value
    for value in tqdm(values, desc=f"Varying rhythm.{rhythm_param_name}"):
        dataset["variations"][str(value)] = []
        
        for i in range(samples_per_value):
            # Create a new configuration with this parameter value
            config = base_config.copy()
            rhythm_config = config.get("rhythm", {}).copy()
            
            # Handle special cases for certain rhythm parameters
            if rhythm_param_name == "subdivision_type" and isinstance(value, (int, float)):
                # Convert numeric index to subdivision type
                subdivision_types = list(map(str, SubdivisionType))
                idx = min(int(value) % len(subdivision_types), len(subdivision_types) - 1)
                rhythm_config[rhythm_param_name] = subdivision_types[idx]
            elif rhythm_param_name == "accent_type" and isinstance(value, (int, float)):
                # Convert numeric index to accent type
                accent_types = list(map(str, AccentType))
                idx = min(int(value) % len(accent_types), len(accent_types) - 1)
                rhythm_config[rhythm_param_name] = accent_types[idx]
            elif rhythm_param_name == "time_signature":
                # Handle time signature as a special case
                if isinstance(value, (int, float)):
                    # Use value as numerator with common denominators
                    numerator = max(2, min(12, int(value)))
                    denominator = 4 if numerator % 3 != 0 else 8
                    rhythm_config[rhythm_param_name] = (numerator, denominator)
            else:
                # Regular numeric parameter
                rhythm_config[rhythm_param_name] = value
            
            # Store rhythm config separately from the main config
            config_for_generation = config.copy()
            
            # Remove any properties that shouldn't be passed to GenerationConfig
            if "rhythm" in config_for_generation:
                rhythm_for_midi = config_for_generation.pop("rhythm")
            else:
                rhythm_for_midi = rhythm_config
                
            # Remove midi_properties to avoid passing it to GenerationConfig
            midi_params = {"tempo": 120, "base_octave": 4}
            if "midi_properties" in config_for_generation:
                midi_params.update(config_for_generation.pop("midi_properties"))
                
            config["rhythm"] = rhythm_config  # Keep rhythm in the complete config for dataset
            
            try:
                # Generate sequence with the cleaned config
                sequence = generate_sequence_from_config(config_for_generation)
                
                # Save to dataset
                dataset["variations"][str(value)].append({
                    "sample_id": i,
                    "sequence": sequence
                })
                
                # Generate MIDI file with rhythm
                midi_path = os.path.join(output_dir, f"rhythm_{rhythm_param_name}_{value}_{i}.mid")
                is_melodic = config_for_generation["generation_type"] == "melodic"
                sequence_to_midi_with_rhythm(
                    sequence, midi_path, is_melodic=is_melodic,
                    rhythm_config=rhythm_for_midi,
                    params=midi_params
                )
                
            except Exception as e:
                logger.error(f"Failed to generate rhythm sample for {rhythm_param_name}={value}: {e}")
    
    # Save the dataset
    output_file = os.path.join(output_dir, f"rhythm_{rhythm_param_name}_variations.json")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Rhythm variations dataset saved to {output_file}")
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
    
    parser.add_argument('--interval-weight-profile', type=str, choices=list(INTERVAL_WEIGHT_PROFILES.keys()),
                      help='Use a specific interval weight profile for all simulations')
    
    parser.add_argument('--random-weights', action='store_true',
                      help='Generate random interval weights for each simulation')
    
    parser.add_argument('--weight-variation', type=float, default=0.5,
                      help='Amount of variation in random weights (0.0-1.0)')
    
    parser.add_argument('--weight-cycling', action='store_true',
                      help='Cycle through all available interval weight profiles')
    
    parser.add_argument('--correlation-report', action='store_true',
                      help='Generate a correlation report between weights and musical features')
    
    # Add rhythm-related arguments
    parser.add_argument('--use-rhythm', action='store_true',
                      help='Apply rhythm patterns to generated sequences')
    
    parser.add_argument('--rhythm-config', type=str,
                      help='Path to rhythm configuration YAML file')
    
    parser.add_argument('--rhythm-variation-mode', action='store_true',
                      help='Enable variation mode to study a single rhythm parameter')
    
    parser.add_argument('--rhythm-param-name', type=str,
                      help='Rhythm parameter name to vary in variation mode')
    
    parser.add_argument('--rhythm-param-min', type=float, default=1.0,
                      help='Minimum value for the varied rhythm parameter')
    
    parser.add_argument('--rhythm-param-max', type=float, default=8.0,
                      help='Maximum value for the varied rhythm parameter')
    
    parser.add_argument('--rhythm-param-steps', type=int, default=8,
                      help='Number of steps between min and max rhythm values')
    
    # Add dataset generation argument
    parser.add_argument('--export-dataset', action='store_true',
                      help='Export results as a training dataset')
    
    parser.add_argument('--dataset-dir', type=str, default='training_dataset',
                      help='Directory to store the training dataset')
    
    args = parser.parse_args()
    
    # Configuration validation
    if args.variation_mode and not args.param_name:
        parser.error("--param-name is required with --variation-mode")
    
    if args.variation_mode and not args.base_config:
        parser.error("--base-config is required with --variation-mode")
    
    if args.rhythm_variation_mode and not args.rhythm_param_name:
        parser.error("--rhythm-param-name is required with --rhythm-variation-mode")
    
    if args.rhythm_variation_mode and not args.base_config:
        parser.error("--base-config is required with --rhythm-variation-mode")
    
    try:
        if args.rhythm_variation_mode:
            # Run rhythm variation mode
            values = np.linspace(args.rhythm_param_min, args.rhythm_param_max, args.rhythm_param_steps)
            output_file = generate_rhythm_variations_dataset(
                args.base_config,
                args.rhythm_param_name,
                values.tolist(),
                args.samples_per_value,
                args.output_dir
            )
            print(f"Rhythm variation dataset saved to: {output_file}")
        elif args.variation_mode:
            # Run variation mode
            values = np.linspace(args.param_min, args.param_max, args.param_steps)
            
            # Add interval weight profile if specified
            if args.interval_weight_profile:
                output_file = generate_variations_dataset(
                    args.base_config,
                    args.param_name,
                    values.tolist(),
                    args.samples_per_value,
                    args.output_dir,
                    args.interval_weight_profile
                )
            else:
                output_file = generate_variations_dataset(
                    args.base_config,
                    args.param_name,
                    values.tolist(),
                    args.samples_per_value,
                    args.output_dir
                )
            print(f"Variation dataset saved to: {output_file}")
        else:
            # Setup for interval weights
            interval_weight_configs = None
            generate_random_weights = args.random_weights
            
            if args.interval_weight_profile:
                # Use specific profile for all simulations
                interval_weight_configs = [INTERVAL_WEIGHT_PROFILES[args.interval_weight_profile]]
            elif args.weight_cycling:
                # Use all available profiles
                interval_weight_configs = list(INTERVAL_WEIGHT_PROFILES.values())
            
            # Run Monte Carlo simulation
            simulator = MonteCarloSimulator(
                num_simulations=args.num_simulations,
                base_config_file=args.base_config,
                output_dir=args.output_dir,
                interval_weight_configs=interval_weight_configs,
                generate_random_weights=generate_random_weights,
                weight_variation_factor=args.weight_variation,
                rhythm_config_file=args.rhythm_config,
                use_rhythm=args.use_rhythm
            )
            
            dataset = simulator.run(parallel=not args.no_parallel)
            json_file = simulator.save_dataset()
            csv_file = simulator.export_stats_to_csv()
            
            # Generate training dataset if requested
            if args.export_dataset:
                dataset_path = simulator.export_training_dataset(args.dataset_dir)
                print(f"- Training dataset: {dataset_path}")
            
            # Generate correlation report if requested
            if args.correlation_report:
                report_path = os.path.join(args.output_dir, "weight_correlation_report.json")
                simulator.generate_weight_correlation_report(report_path)
                print(f"- Weight correlation report: {report_path}")
            
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
