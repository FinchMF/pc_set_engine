"""Command-line interface for the pitch class rules engine.

This module provides a user-friendly command-line interface for generating
musical sequences using the pitch class rules engine. It supports various
configuration presets, allows customization of randomness parameters,
and offers options for saving the generated sequences.

The module includes:
- Preset configurations for different types of sequences (melodic, chordal, etc.)
- Parameters for controlling randomness and variation
- Utilities for displaying and saving generated sequences
- Command-line argument parsing for easy configuration

Examples:
    Generate a melodic sequence from C to G with default settings:
    ```bash
    python run_engine.py --config-type melodic
    ```

    Generate a chord progression with custom sequence length:
    ```bash
    python run_engine.py --config-type chord-progression --sequence-length 6
    ```

    Generate a highly random melodic sequence and save to file:
    ```bash
    python run_engine.py --config-type melodic --randomness 0.8 --output sequence.json
    ```
"""
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
from midi import sequence_to_midi

def get_default_config() -> Dict:
    """Return the default configuration for the pitch class engine.
    
    This function provides a copy of the example configuration from the engine
    module, which can be used as a starting point for creating custom configs.
    
    Returns:
        Dict: A dictionary containing default configuration parameters.
    """
    logger.debug("Getting default configuration")
    return EXAMPLE_CONFIG.copy()

def get_melodic_config(randomness: float = 0.3, variation: float = 0.4) -> Dict:
    """Return a configuration for generating a melodic sequence.
    
    Creates a configuration for a melodic sequence that progresses from
    C to G, with customizable randomness parameters.
    
    Args:
        randomness: Level of randomness from 0.0 (deterministic) to 1.0 (highly random).
        variation: Probability of applying variations from 0.0 (none) to 1.0 (maximum).
    
    Returns:
        Dict: A dictionary containing configuration parameters for melodic generation.
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
    """Return a configuration for generating a random walk melodic sequence.
    
    Creates a configuration for a melodic sequence that randomly evolves
    without a specific target, with customizable randomness parameters.
    
    Args:
        randomness: Level of randomness from 0.0 (predictable) to 1.0 (chaotic).
        variation: Probability of applying variations from 0.0 (none) to 1.0 (maximum).
    
    Returns:
        Dict: A dictionary containing configuration parameters for random walk generation.
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
    """Return a configuration for generating a chord progression.
    
    Creates a configuration for a chord progression that moves from
    C major to G major over a specified number of steps.
    
    Returns:
        Dict: A dictionary containing configuration parameters for chord progression generation.
    """
    logger.debug("Creating chord progression configuration")
    config = get_default_config()
    config["generation_type"] = "chordal"
    config["start_pc"] = [0, 4, 7]  # C major triad
    config["target_pc"] = [7, 11, 2]  # G major triad
    config["sequence_length"] = 8
    return config

def get_static_chord_config() -> Dict:
    """Return a configuration for generating static chords with variations.
    
    Creates a configuration for generating a sequence of C minor chords
    with slight variations between them.
    
    Returns:
        Dict: A dictionary containing configuration parameters for static chord generation.
    """
    logger.debug("Creating static chord configuration")
    config = get_default_config()
    config["generation_type"] = "chordal"
    config["start_pc"] = [0, 3, 7]  # C minor triad
    config["progression"] = False
    config["constraints"] = {"vary_chord": True}
    return config

def get_random_walk_config() -> Dict:
    """Return a configuration for generating a random walk melodic sequence.
    
    Convenience function that calls get_random_walk_melodic_config with default parameters.
    
    Returns:
        Dict: A dictionary containing configuration parameters for random walk generation.
    """
    logger.debug("Creating random walk configuration")
    return get_random_walk_melodic_config()

def run_engine_with_config(config: Dict) -> List[Union[int, List[int]]]:
    """Run the pitch class engine with the given configuration.
    
    This function executes the generation process using the provided configuration
    and measures the execution time.
    
    Args:
        config: Configuration dictionary for the generation process.
        
    Returns:
        List[Union[int, List[int]]]: Generated sequence of pitch classes or lists of pitch classes.
        
    Example:
        ```python
        config = get_melodic_config()
        sequence = run_engine_with_config(config)
        ```
    """
    logger.info("Running engine with configuration")
    start_time = time.time()
    result = generate_sequence_from_config(config)
    log_execution_time(logger, start_time, "Engine execution")
    return result

def save_sequence_to_file(sequence: List[Union[int, List[int]]], filepath: str, config: Dict = None):
    """Save a generated sequence to a JSON file.
    
    Writes the sequence and optionally the configuration used to generate it
    to a JSON file at the specified location.
    
    Args:
        sequence: The generated sequence to save.
        filepath: Path where the file should be saved.
        config: Optional configuration used to generate the sequence.
        
    Example:
        ```python
        save_sequence_to_file(sequence, "output.json", config)
        ```
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
    """Display a generated sequence in a readable format.
    
    Formats and prints the generated sequence to the console, converting
    numeric pitch classes to their musical note names.
    
    Args:
        sequence: The generated sequence to display.
        generation_type: Type of sequence ("melodic" or "chordal").
        
    Example:
        ```python
        display_sequence([0, 4, 7], "melodic")
        ```
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
    
    parser.add_argument('--midi', '-m', type=str, help='Path to save the output MIDI file')
    
    parser.add_argument('--randomness', '-r', type=float, default=0.3, 
                      help='Level of randomness (0.0-1.0) for melodic generation')
    
    parser.add_argument('--variation', '-v', type=float, default=0.4,
                      help='Probability of variations (0.0-1.0) for melodic generation')
    
    parser.add_argument('--sequence-length', '-l', type=int, default=8,
                      help='Length of the generated sequence')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO', help='Set the logging level')
    
    parser.add_argument('--tempo', type=int, default=120,
                      help='Tempo in BPM for MIDI generation')
    
    parser.add_argument('--base-octave', type=int, default=4,
                      help='Base octave for MIDI generation (4 = middle C)')
    
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
    
    # Save to JSON file if requested
    if args.output:
        save_sequence_to_file(sequence, args.output, config)
    
    # Generate MIDI file if requested
    if args.midi:
        is_melodic = generation_type.lower() == "melodic"
        midi_params = {
            "tempo": args.tempo,
            "base_octave": args.base_octave,
            "note_duration": 0.5,  # Default to half-second notes
        }
        try:
            midi_path = sequence_to_midi(sequence, args.midi, is_melodic=is_melodic, params=midi_params)
            print(f"MIDI file saved to {midi_path}")
        except Exception as e:
            logger.error(f"Failed to create MIDI file: {e}")
            print(f"Error creating MIDI file: {str(e)}")
    
    logger.info("Engine execution completed")

if __name__ == "__main__":
    main()
