import json
import argparse
import time
from typing import Dict, List, Union
from pathlib import Path
import sys

# Add the parent directory to sys.path to ensure module imports work correctly
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_logger, log_config, log_execution_time

# Initialize logger with both console and file logging
logger = get_logger(__name__, log_file="engine_run.log")

from pc_sets.engine import (
    generate_sequence_from_config, 
    EXAMPLE_CONFIG,
    GenerationType,
    ProgressionType
)

def get_default_config() -> Dict:
    """Return the default configuration for the pitch class engine."""
    logger.debug("Getting default configuration")
    return EXAMPLE_CONFIG.copy()

def get_melodic_config(randomness: float = 0.3, variation: float = 0.4) -> Dict:
    """
    Return a configuration for generating a melodic sequence.
    
    Args:
        randomness: Level of randomness (0.0-1.0)
        variation: Probability of applying variations (0.0-1.0)
    """
    logger.debug(f"Creating melodic config with randomness={randomness}, variation={variation}")
    config = get_default_config()
    config["generation_type"] = "melodic"
    config["start_pc"] = 0  # C
    config["target_pc"] = 7  # G
    config["randomness_factor"] = randomness
    config["variation_probability"] = variation
    config["constraints"] = {
        "max_interval": 3 + int(randomness * 3),  # Larger intervals with higher randomness
        "vary_pc": True
    }
    return config

def get_random_walk_melodic_config(randomness: float = 0.5, variation: float = 0.6) -> Dict:
    """
    Return a configuration for generating a random walk melodic sequence with 
    specified randomness levels.
    
    Args:
        randomness: Level of randomness (0.0-1.0)
        variation: Probability of applying variations (0.0-1.0)
    """
    logger.debug(f"Creating random walk config with randomness={randomness}, variation={variation}")
    config = get_default_config()
    config["generation_type"] = "melodic"
    config["start_pc"] = 0  # C
    config["progression_type"] = "random"
    config["randomness_factor"] = randomness
    config["variation_probability"] = variation
    config["constraints"] = {
        "max_interval": 2 + int(randomness * 4),  # Larger intervals with higher randomness
        "vary_pc": True
    }
    return config

def get_chord_progression_config() -> Dict:
    """Return a configuration for generating a chord progression."""
    logger.debug("Creating chord progression configuration")
    config = get_default_config()
    config["generation_type"] = "chordal"
    config["start_pc"] = [0, 4, 7]  # C major triad
    config["target_pc"] = [7, 11, 2]  # G major triad
    config["sequence_length"] = 8
    return config

def get_static_chord_config() -> Dict:
    """Return a configuration for generating static chords with variations."""
    logger.debug("Creating static chord configuration")
    config = get_default_config()
    config["generation_type"] = "chordal"
    config["start_pc"] = [0, 3, 7]  # C minor triad
    config["progression"] = False
    config["constraints"] = {"vary_chord": True}
    return config

def get_random_walk_config() -> Dict:
    """Return a configuration for generating a random walk melodic sequence."""
    logger.debug("Creating random walk configuration")
    return get_random_walk_melodic_config()

def run_engine_with_config(config: Dict) -> List[Union[int, List[int]]]:
    """
    Run the pitch class engine with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Generated sequence
    """
    logger.info("Running engine with configuration")
    start_time = time.time()
    result = generate_sequence_from_config(config)
    log_execution_time(logger, start_time, "Engine execution")
    return result

def save_sequence_to_file(sequence: List[Union[int, List[int]]], filepath: str, config: Dict = None):
    """
    Save a generated sequence to a JSON file.
    
    Args:
        sequence: The generated sequence
        filepath: Path to save the file
        config: Optional configuration used to generate the sequence
    """
    output_data = {
        "sequence": sequence,
    }
    
    if config:
        output_data["config"] = config
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Sequence saved to {filepath}")
    print(f"Sequence saved to {filepath}")

def display_sequence(sequence: List[Union[int, List[int]]], generation_type: str):
    """
    Display a generated sequence in a readable format.
    
    Args:
        sequence: The generated sequence
        generation_type: Type of sequence ("melodic" or "chordal")
    """
    pitch_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    
    print("\nGenerated Sequence:")
    print("-----------------")
    
    if generation_type.lower() == "melodic":
        for i, pc in enumerate(sequence):
            note = f"Step {i+1}: {pitch_names[pc]} (PC {pc})"
            print(note)
            logger.debug(note)
    else:  # chordal
        for i, pcs in enumerate(sequence):
            chord_notes = [pitch_names[pc] for pc in pcs]
            chord = f"Step {i+1}: [{', '.join(chord_notes)}] (PCs {pcs})"
            print(chord)
            logger.debug(chord)
    
    print("-----------------")

def main():
    """Main function to run the pitch class engine with command line arguments."""
    logger.info("Starting pitch class rules engine")
    
    parser = argparse.ArgumentParser(description='Run the Pitch Class Engine with different configurations.')
    
    parser.add_argument('--config-type', choices=[
        'default', 'melodic', 'chord-progression', 'static-chord', 'random-walk'
    ], default='default', help='Type of configuration to use')
    
    parser.add_argument('--output', '-o', type=str, help='Path to save the output JSON')
    
    parser.add_argument('--randomness', '-r', type=float, default=0.3, 
                      help='Level of randomness (0.0-1.0) for melodic generation')
    
    parser.add_argument('--variation', '-v', type=float, default=0.4,
                      help='Probability of variations (0.0-1.0) for melodic generation')
    
    parser.add_argument('--sequence-length', '-l', type=int, default=8,
                      help='Length of the generated sequence')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO', help='Set the logging level')
    
    args = parser.parse_args()
    
    # Set log level based on command line argument
    import logging
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    logger.info(f"Command-line arguments: {args}")
    
    # Clamp randomness and variation values between 0 and 1
    randomness = max(0.0, min(1.0, args.randomness))
    variation = max(0.0, min(1.0, args.variation))
    
    # Select the appropriate configuration
    if args.config_type == 'melodic':
        config = get_melodic_config(randomness=randomness, variation=variation)
    elif args.config_type == 'chord-progression':
        config = get_chord_progression_config()
    elif args.config_type == 'static-chord':
        config = get_static_chord_config()
    elif args.config_type == 'random-walk':
        config = get_random_walk_melodic_config(randomness=randomness, variation=variation)
    else:  # default
        config = get_default_config()
    
    # Set the sequence length if provided
    if args.sequence_length:
        config["sequence_length"] = args.sequence_length
    
    # Run the engine with the selected configuration
    sequence = run_engine_with_config(config)
    
    # Display the result
    generation_type = config.get("generation_type", "melodic")
    display_sequence(sequence, generation_type)
    
    # Save to file if requested
    if args.output:
        save_sequence_to_file(sequence, args.output, config)
    
    logger.info("Engine execution completed")

if __name__ == "__main__":
    main()
