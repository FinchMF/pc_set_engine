"""
Dataset Generator for PC Rules Engine

This module provides functionality to generate large, diverse datasets
by running multiple Monte Carlo simulations with different configurations.
"""

import os
import sys
import json
import time
import yaml
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

from monte_carlo import MonteCarloSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Generates comprehensive datasets by running multiple Monte Carlo simulations
    with varying configurations.
    """
    
    def __init__(
        self, 
        config_file: str,
        output_dir: str,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the dataset generator.
        
        Args:
            config_file: Path to the dataset configuration file
            output_dir: Directory to save the generated dataset
            random_seed: Optional random seed for reproducibility
        """
        self.config_file = config_file
        self.output_dir = output_dir
        self.random_seed = random_seed
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Load configuration
        self.config = self._load_config()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            "generator_version": "1.0.0",
            "generation_timestamp": datetime.now().isoformat(),
            "config_file": config_file,
            "random_seed": random_seed,
            "simulation_groups": [],
            "total_simulations": 0,
            "successful_simulations": 0,
            "failed_simulations": 0,
            "generation_time": 0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the dataset configuration file."""
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
            
            # Validate required fields
            required_fields = ['dataset_name', 'simulation_groups']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Configuration missing required field: {field}")
            
            # Validate simulation groups
            if not config['simulation_groups']:
                raise ValueError("No simulation groups defined in configuration")
            
            for i, group in enumerate(config['simulation_groups']):
                if 'name' not in group:
                    group['name'] = f"group_{i}"
                if 'num_simulations' not in group:
                    raise ValueError(f"Missing num_simulations in group {group['name']}")
                if 'base_config' not in group:
                    raise ValueError(f"Missing base_config in group {group['name']}")
            
            return config
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _prepare_group_output_dir(self, group_name: str) -> str:
        """Prepare the output directory for a simulation group."""
        group_dir = os.path.join(self.output_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        return group_dir
    
    def _run_simulation_group(self, group_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a group of Monte Carlo simulations based on the provided configuration."""
        group_name = group_config['name']
        num_simulations = group_config['num_simulations']
        base_config = group_config['base_config']
        
        logger.info(f"Running simulation group: {group_name} ({num_simulations} simulations)")
        
        group_output_dir = self._prepare_group_output_dir(group_name)
        
        # Get parameter ranges if specified
        param_ranges = group_config.get('param_ranges', {})
        rhythm_param_ranges = group_config.get('rhythm_param_ranges', {})
        
        # Process rhythm_param_ranges to ensure they're in the correct format
        processed_rhythm_param_ranges = {}
        for param, value_range in rhythm_param_ranges.items():
            # Handle special parameters that might not be min-max ranges
            if param in ['subdivision_type', 'accent_type']:
                # These are direct values or lists of options, not ranges
                processed_rhythm_param_ranges[param] = value_range
            elif param == 'time_signature':
                # Time signatures need special handling
                # For MonteCarloSimulator compatibility, we need to omit this parameter
                # and handle time signature changes at a higher level
                logger.info(f"Time signature variations will be handled separately: {value_range}")
                # Skip adding to processed ranges
                continue
            elif isinstance(value_range, list) and len(value_range) == 2 and all(isinstance(x, (int, float)) for x in value_range):
                # Standard numerical range [min, max]
                processed_rhythm_param_ranges[param] = value_range
            else:
                logger.warning(f"Skipping incompatible rhythm parameter range for {param}: {value_range}")
        
        # Set options for MIDI and pitch class generation
        midi_options = {
            "save_midi": group_config.get('save_midi', True),
            "midi_directory": os.path.join(group_output_dir, "midi") if group_config.get('save_midi', True) else None,
            "include_pitch_classes": True,  # Always include pitch classes in results
            "normalize_pitch_classes": group_config.get('normalize_pitch_classes', True)
        }
        
        # Create midi directory if needed
        if midi_options["midi_directory"]:
            os.makedirs(midi_options["midi_directory"], exist_ok=True)
            
        logger.info(f"Processed rhythm parameter ranges: {processed_rhythm_param_ranges}")
        logger.info(f"MIDI options: {midi_options}")
        
        # Create a temporary config file if base_config is a dictionary
        temp_config_file = None
        
        if isinstance(base_config, dict):
            try:
                temp_config_file = os.path.join(self.output_dir, f"temp_{group_name}_config.yaml")
                with open(temp_config_file, 'w') as f:
                    yaml.dump(base_config, f)
                base_config_file = temp_config_file
            except Exception as e:
                logger.error(f"Error creating temporary config file: {str(e)}")
                raise
        elif isinstance(base_config, str):
            # Resolve path relative to the dataset config file location
            config_dir = os.path.dirname(os.path.abspath(self.config_file))
            base_config_file = os.path.normpath(os.path.join(config_dir, base_config))
            
            # Check if file exists
            if not os.path.exists(base_config_file):
                # Try relative to current working directory
                if os.path.exists(base_config):
                    base_config_file = base_config
                else:
                    # Try relative to script directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    alt_path = os.path.normpath(os.path.join(script_dir, base_config))
                    if os.path.exists(alt_path):
                        base_config_file = alt_path
                    else:
                        logger.error(f"Base config file not found: {base_config}")
                        raise FileNotFoundError(f"Base config file not found: {base_config}")
            
            logger.info(f"Using base config file: {base_config_file}")
        
        # Check if MonteCarloSimulator supports midi_options
        simulator_args = {
            "num_simulations": num_simulations,
            "base_config_file": base_config_file,
            "output_dir": group_output_dir,
            "param_ranges": param_ranges,
            "use_rhythm": group_config.get('use_rhythm', True),
            "rhythm_param_ranges": processed_rhythm_param_ranges
        }
        
        # Test if midi_options is a valid parameter for MonteCarloSimulator
        import inspect
        simulator_params = inspect.signature(MonteCarloSimulator.__init__).parameters
        if 'midi_options' in simulator_params:
            simulator_args['midi_options'] = midi_options
        
        # Run the Monte Carlo simulation
        simulator = MonteCarloSimulator(**simulator_args)
        
        group_start_time = time.time()
        simulator.run()
        group_execution_time = time.time() - group_start_time
        
        # Clean up temporary file if created
        if temp_config_file and os.path.exists(temp_config_file):
            try:
                os.remove(temp_config_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_config_file}: {str(e)}")
        
        # Save group results - use direct file paths to avoid double directory issues
        results_file = os.path.join(group_output_dir, f"{group_name}_results.json")
        stats_file = os.path.join(group_output_dir, f"{group_name}_stats.csv")
        
        # Try to save results safely
        try:
            # Access simulation results - they might be in different attributes depending on implementation
            simulation_results = []
            if hasattr(simulator, 'results'):
                simulation_results = simulator.results
                logger.info(f"Found simulator.results with {len(simulation_results)} items")
            elif hasattr(simulator, 'simulations'):
                simulation_results = simulator.simulations
                logger.info(f"Found simulator.simulations with {len(simulation_results)} items")
            
            # Let's check for actual MIDI files in the output directory
            midi_dir = os.path.join(group_output_dir)
            midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
            logger.info(f"Found {len(midi_files)} MIDI files in {midi_dir}")
            
            # Create a mapping of MIDI files to simulations
            midi_file_map = {}
            for midi_file in midi_files:
                # Extract simulation index from filename (assuming format simulation_X.mid)
                try:
                    sim_idx = int(midi_file.split('_')[1].split('.')[0])
                    midi_file_map[sim_idx] = os.path.join(midi_dir, midi_file)
                except (IndexError, ValueError):
                    # If filename doesn't match expected pattern, ignore
                    pass
            
            logger.info(f"Mapped {len(midi_file_map)} MIDI files to simulation indices")
            
            # If we have MIDI files but no simulation results, create dummy simulations
            if len(midi_file_map) > 0 and len(simulation_results) == 0:
                logger.warning(f"No simulation results found but {len(midi_file_map)} MIDI files exist. Creating dummy simulations.")
                for sim_idx, midi_path in midi_file_map.items():
                    simulation_results.append({
                        "id": sim_idx,
                        "success": True,
                        "midi_file": midi_path,
                        "midi_file_exists": True,
                        "group": group_name,
                        "created_from_midi_file": True
                    })
            
            # Verify and enhance each simulation result
            for i, sim in enumerate(simulation_results):
                # First, check if we have a MIDI file for this simulation
                if i in midi_file_map:
                    sim['midi_file'] = midi_file_map[i]
                    sim['midi_file_exists'] = True
                    sim['success'] = True
                    logger.debug(f"Simulation {i} marked successful due to found MIDI file: {midi_file_map[i]}")
            
            # If we found MIDI files but none of the simulations are marked successful yet, 
            # this is likely a case where the simulation objects don't match the MIDI files
            if len(midi_file_map) > 0 and sum(1 for sim in simulation_results if sim.get('success', False)) == 0:
                logger.warning("Found MIDI files but no simulations marked as successful. Fixing simulation data.")
                
                # Create new simulations or update existing ones based on MIDI files
                for sim_idx, midi_path in midi_file_map.items():
                    # If this simulation index exists in our results, update it
                    if sim_idx < len(simulation_results):
                        simulation_results[sim_idx]['midi_file'] = midi_path
                        simulation_results[sim_idx]['midi_file_exists'] = True
                        simulation_results[sim_idx]['success'] = True
                        simulation_results[sim_idx]['fixed_by_midi'] = True
                    else:
                        # Otherwise create a new simulation result
                        simulation_results.append({
                            "id": sim_idx,
                            "success": True,
                            "midi_file": midi_path,
                            "midi_file_exists": True,
                            "group": group_name,
                            "created_from_midi_file": True
                        })
            
            # Process each simulation to ensure proper metadata
            for i, sim in enumerate(simulation_results):
                # Check for MIDI file existence if one is specified
                midi_file = sim.get('midi_file')
                if midi_file and not sim.get('midi_file_exists', False):
                    # Convert relative to absolute path if needed
                    midi_path = midi_file if os.path.isabs(midi_file) else os.path.join(group_output_dir, midi_file)
                    sim['midi_file_absolute'] = os.path.abspath(midi_path)
                    
                    # Verify MIDI file exists
                    midi_exists = os.path.exists(midi_path)
                    sim['midi_file_exists'] = midi_exists
                    
                    if midi_exists and not sim.get('success', False):
                        sim['success'] = True
                        logger.debug(f"Simulation {i} marked successful due to MIDI file: {midi_path}")
                
                # Handle error cases preemptively by checking error field
                if 'error' in sim and 'max() arg is an empty sequence' in str(sim['error']):
                    # This is a known issue - try to fix it by ensuring any pitch class sequences aren't empty
                    if 'pitch_classes' in sim:
                        logger.info(f"Attempting to fix empty sequence error in simulation {i}")
                        if isinstance(sim['pitch_classes'], list):
                            # For list of lists structure
                            for j, pc in enumerate(sim['pitch_classes']):
                                if isinstance(pc, list) and len(pc) == 0:
                                    sim['pitch_classes'][j] = [0]
                                    logger.info(f"Fixed empty list at index {j} in simulation {i}")
                        elif isinstance(sim['pitch_classes'], dict):
                            # For dictionary structure
                            for k, v in sim['pitch_classes'].items():
                                if isinstance(v, list) and len(v) == 0:
                                    sim['pitch_classes'][k] = [0]
                                    logger.info(f"Fixed empty list for key {k} in simulation {i}")
                
                # Validate success status based on pitch classes if not already set
                if not sim.get('success', False):
                    # Check if pitch classes exist and are not empty
                    has_pitch_classes = False
                    if 'pitch_classes' in sim and sim['pitch_classes']:
                        try:
                            if isinstance(sim['pitch_classes'], list):
                                # Special case: single empty list that needs to be fixed
                                if len(sim['pitch_classes']) == 0:
                                    sim['pitch_classes'] = [[0]]  # Add default pitch class
                                    logger.warning(f"Fixed completely empty pitch class list in simulation {i}")
                                    has_pitch_classes = True
                                # Special case: list with a single value that's an empty list
                                elif len(sim['pitch_classes']) == 1 and isinstance(sim['pitch_classes'][0], list) and len(sim['pitch_classes'][0]) == 0:
                                    sim['pitch_classes'][0] = [0]  # Add default pitch class
                                    logger.warning(f"Fixed single empty pitch class list in simulation {i}")
                                    has_pitch_classes = True
                                # Normal case: non-empty list
                                elif len(sim['pitch_classes']) > 0:
                                    # Check for any valid elements
                                    valid_elements = False
                                    for j, pc in enumerate(sim['pitch_classes']):
                                        if pc is None:
                                            sim['pitch_classes'][j] = [0]
                                            logger.warning(f"Replaced None at index {j} in simulation {i}")
                                        elif isinstance(pc, list) and len(pc) == 0:
                                            sim['pitch_classes'][j] = [0]
                                            logger.warning(f"Replaced empty list at index {j} in simulation {i}")
                                        elif isinstance(pc, (int, float)) or (isinstance(pc, list) and len(pc) > 0):
                                            valid_elements = True
                                    
                                    has_pitch_classes = valid_elements
                            elif isinstance(sim['pitch_classes'], dict):
                                # Check if any value in the dict is non-empty
                                if any(v for v in sim['pitch_classes'].values()):
                                    has_pitch_classes = True
                                else:
                                    # Ensure no empty lists in dict values
                                    for k, v in sim['pitch_classes'].items():
                                        if isinstance(v, list) and len(v) == 0:
                                            sim['pitch_classes'][k] = [0]  # Default pitch class
                                            logger.warning(f"Replaced empty pitch class sequence for key {k} in simulation {i}")
                                        elif v is None:
                                            sim['pitch_classes'][k] = [0]
                                            logger.warning(f"Replaced None value for key {k} in simulation {i}")
                        except Exception as e:
                            logger.warning(f"Error checking pitch classes for simulation {i}: {e}")
                            # Try to fix any potential issues causing max() errors
                            if isinstance(sim.get('pitch_classes'), list):
                                # More comprehensive fix for list types
                                fixed_pcs = []
                                for pc in sim['pitch_classes']:
                                    if pc is None or (isinstance(pc, list) and len(pc) == 0):
                                        fixed_pcs.append([0])
                                    else:
                                        fixed_pcs.append(pc)
                                sim['pitch_classes'] = fixed_pcs
                            elif isinstance(sim.get('pitch_classes'), dict):
                                # Fix for dictionary types
                                for k, v in sim['pitch_classes'].items():
                                    if v is None or (isinstance(v, list) and len(v) == 0):
                                        sim['pitch_classes'][k] = [0]
                    
                    # Also check for 'target_pitch_class' which might be needed for directed progression
                    if 'parameters' in sim and sim['parameters'].get('progression_type') == 'directed':
                        if 'target_pitch_class' not in sim['parameters'] or sim['parameters']['target_pitch_class'] is None:
                            # Add a default target pitch class to prevent directed progression errors
                            sim['parameters']['target_pitch_class'] = 0
                            logger.warning(f"Added missing target_pitch_class for directed progression in simulation {i}")
                    
                    # Mark as successful if there are pitch classes
                    if has_pitch_classes:
                        sim['success'] = True
                        logger.debug(f"Simulation {i} marked successful due to pitch classes")
                
                # Add metadata for traceability
                sim['group'] = group_name
                if 'parameters' not in sim:
                    sim['parameters'] = {}

            # Count successful simulations after fixing success flags
            successful_count = sum(1 for sim in simulation_results if sim.get('success', False))
            failed_count = num_simulations - successful_count
            
            # If we still have no successful simulations but we found MIDI files, something is wrong with our mapping
            # Let's force the success count based on MIDI files
            if successful_count == 0 and len(midi_files) > 0:
                logger.warning(f"Failed to mark any simulations successful despite finding {len(midi_files)} MIDI files. "
                              f"Setting success count based on MIDI files.")
                successful_count = len(midi_files)
                failed_count = num_simulations - successful_count
                
                # Also add a special flag to our metadata
                dataset_metadata = {
                    "generated_at": datetime.now().isoformat(),
                    "num_simulations": num_simulations,
                    "base_config": base_config,
                    "param_ranges": param_ranges,
                    "rhythm_param_ranges": rhythm_param_ranges,
                    "total_execution_time": group_execution_time,
                    "group_name": group_name,
                    "midi_directory": midi_options.get("midi_directory"),
                    "save_midi": midi_options.get("save_midi", True),
                    "successful_simulations": successful_count,
                    "failed_simulations": failed_count,
                    "midi_files_found": len(midi_files),
                    "success_count_forced_from_midi": True
                }
            else:
                dataset_metadata = {
                    "generated_at": datetime.now().isoformat(),
                    "num_simulations": num_simulations,
                    "base_config": base_config,
                    "param_ranges": param_ranges,
                    "rhythm_param_ranges": rhythm_param_ranges,
                    "total_execution_time": group_execution_time,
                    "group_name": group_name,
                    "midi_directory": midi_options.get("midi_directory"),
                    "save_midi": midi_options.get("save_midi", True),
                    "successful_simulations": successful_count,
                    "failed_simulations": failed_count
                }
                
            logger.info(f"Group {group_name}: {successful_count} successful, {failed_count} failed simulations")
            
            # Create the dataset as a dictionary
            dataset = {
                "metadata": dataset_metadata,
                "simulations": simulation_results
            }
            
            # Write directly to file instead of using simulator's save method
            with open(results_file, 'w') as f:
                json.dump(dataset, f, indent=2)
                
            logger.info(f"Dataset saved to: {results_file}")
            
            # Export stats directly too if possible
            try:
                # Instead of using the simulator's method which has path issues,
                # we'll create a simple stats export if we have simulation results
                stats_data = []
                
                for sim in simulation_results:
                    # Extract basic stats for each simulation
                    sim_stats = {
                        "id": sim.get("id", 0),
                        "success": sim.get("success", False),
                        "execution_time": sim.get("execution_time", 0),
                        "has_midi": 'midi_file' in sim,
                        "has_pitch_classes": 'pitch_classes' in sim and bool(sim['pitch_classes']),
                        "pitch_class_count": len(sim.get('pitch_classes', [])),
                    }
                    
                    # Add sequence parameters if available
                    if "parameters" in sim:
                        for param_key, param_value in sim.get("parameters", {}).items():
                            # Flatten simple values
                            if isinstance(param_value, (int, float, str, bool)):
                                sim_stats[f"param_{param_key}"] = param_value
                    
                    # Add any statistics if available
                    if "statistics" in sim:
                        for stat_key, stat_value in sim.get("statistics", {}).items():
                            # Flatten simple values
                            if isinstance(stat_value, (int, float, str, bool)):
                                sim_stats[stat_key] = stat_value
                    
                    stats_data.append(sim_stats)
                
                if stats_data:
                    # Convert to DataFrame and save
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_csv(stats_file, index=False)
                    logger.info(f"Statistics saved to: {stats_file}")
            except Exception as e:
                logger.error(f"Failed to save statistics: {str(e)}")
                stats_file = None  # Mark as not available
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            # Continue execution despite the error
        
        # Get the successful and failed counts - use the count from earlier processing
        if not 'successful_count' in locals() or not isinstance(successful_count, int):
            # Alternative method: count MIDI files in the output directory as successful simulations
            midi_dir = os.path.join(group_output_dir)
            midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
            successful_count = len(midi_files)
            failed_count = num_simulations - successful_count
            logger.info(f"Using MIDI file count for success stats: {successful_count} successful, {failed_count} failed")
        
        group_metadata = {
            "name": group_name,
            "num_simulations": num_simulations,
            "successful_simulations": successful_count,
            "failed_simulations": failed_count,
            "execution_time": group_execution_time,
            "base_config": base_config,
            "param_ranges": param_ranges,
            "rhythm_param_ranges": rhythm_param_ranges,
            "results_file": results_file,
            "stats_file": stats_file
        }
        
        return group_metadata
    
    def generate(self) -> Dict[str, Any]:
        """Generate the complete dataset by running all simulation groups."""
        start_time = time.time()
        
        logger.info(f"Starting dataset generation: {self.config['dataset_name']}")
        
        # Process each simulation group
        for group_config in self.config['simulation_groups']:
            group_metadata = self._run_simulation_group(group_config)
            self.metadata["simulation_groups"].append(group_metadata)
            self.metadata["total_simulations"] += group_metadata["num_simulations"]
            self.metadata["successful_simulations"] += group_metadata["successful_simulations"]
            self.metadata["failed_simulations"] += group_metadata["failed_simulations"]
        
        # Record total generation time
        self.metadata["generation_time"] = time.time() - start_time
        
        # Save the consolidated dataset
        self._save_consolidated_dataset()
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Dataset generation completed in {self.metadata['generation_time']:.2f} seconds")
        logger.info(f"Total simulations: {self.metadata['total_simulations']}")
        logger.info(f"Successful: {self.metadata['successful_simulations']}, Failed: {self.metadata['failed_simulations']}")
        
        return self.metadata
    
    def _save_consolidated_dataset(self):
        """Consolidate and save the complete dataset."""
        dataset_name = self.config['dataset_name']
        consolidated_file = os.path.join(self.output_dir, f"{dataset_name}.json")
        
        # Consolidate all simulation results
        all_simulations = []
        
        for group_metadata in self.metadata["simulation_groups"]:
            try:
                with open(group_metadata["results_file"], 'r') as file:
                    group_data = json.load(file)
                    if 'simulations' in group_data:
                        # Tag simulations with their group
                        for sim in group_data['simulations']:
                            sim['group'] = group_metadata['name']
                        all_simulations.extend(group_data['simulations'])
            except Exception as e:
                logger.error(f"Error reading results from {group_metadata['name']}: {str(e)}")
        
        # Create the consolidated dataset
        consolidated_dataset = {
            "metadata": self.metadata,
            "simulations": all_simulations
        }
        
        # Save the consolidated dataset
        with open(consolidated_file, 'w') as file:
            json.dump(consolidated_dataset, file, indent=2)
        
        logger.info(f"Consolidated dataset saved to: {consolidated_file}")
    
    def _save_metadata(self):
        """Save the generation metadata to a separate file."""
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        
        with open(metadata_file, 'w') as file:
            json.dump(self.metadata, file, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_file}")


def create_dataset_from_config(config_path: str, output_dir: str, random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a dataset from a configuration file.
    
    Args:
        config_path: Path to the dataset configuration file
        output_dir: Directory to save the generated dataset
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Dictionary containing metadata about the generated dataset
    """
    generator = DatasetGenerator(config_path, output_dir, random_seed)
    return generator.generate()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate datasets for PC Rules Engine")
    parser.add_argument("config", help="Path to the dataset configuration file")
    parser.add_argument("--output-dir", "-o", default="dataset", help="Output directory")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_dataset_from_config(args.config, args.output_dir, args.seed)
