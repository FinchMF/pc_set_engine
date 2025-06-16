from typing import List, Dict, Union, Optional, Tuple, Any
import random
import json
import time
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_logger, log_config, log_execution_time

# Initialize logger
logger = get_logger(__name__)

from .pitch_classes import PitchClass, PitchClassSet, COMMON_SETS

class GenerationType(Enum):
    MELODIC = "melodic"
    CHORDAL = "chordal"

class ProgressionType(Enum):
    STATIC = "static"
    DIRECTED = "directed"
    RANDOM = "random"

@dataclass
class GenerationConfig:
    """Configuration for the pitch class generation pipeline."""
    start_pc: Union[int, List[int]]  # Starting pitch class or set
    generation_type: GenerationType  # Melodic or chordal
    sequence_length: int = 8  # Length of the sequence to generate
    progression: bool = False  # Whether to use progression
    target_pc: Optional[Union[int, List[int]]] = None  # Target pitch class or set for progression
    transformation_steps: Optional[int] = None  # Number of steps for the transformation
    progression_type: ProgressionType = ProgressionType.DIRECTED  # Type of progression
    allowed_operations: List[str] = None  # Operations allowed for transformations
    constraints: Dict[str, Any] = None  # Constraints to apply
    style_profile: Optional[str] = None  # Style profile (e.g., "jazz", "classical")
    randomness_factor: float = 0.2  # Level of randomness (0.0 to 1.0)
    variation_probability: float = 0.3  # Probability of applying variations
    
    def __post_init__(self):
        # Convert string values to enums
        if isinstance(self.generation_type, str):
            self.generation_type = GenerationType(self.generation_type.lower())
        
        if isinstance(self.progression_type, str):
            self.progression_type = ProgressionType(self.progression_type.lower())
            
        # Set default allowed operations if not provided
        if self.allowed_operations is None:
            self.allowed_operations = [
                "transpose", "invert", "complement", 
                "add_note", "remove_note", "substitute_note"
            ]
            
        # Set default constraints if not provided
        if self.constraints is None:
            self.constraints = {}
            
        # Ensure target_pc is provided if progression is enabled
        if self.progression and self.progression_type == ProgressionType.DIRECTED and self.target_pc is None:
            raise ValueError("Target pitch class must be provided for directed progression")
            
        # Clamp randomness_factor between 0 and 1
        self.randomness_factor = max(0.0, min(1.0, self.randomness_factor))
        
        # Clamp variation_probability between 0 and 1
        self.variation_probability = max(0.0, min(1.0, self.variation_probability))


class PitchClassEngine:
    """
    Engine for generating melodic or chordal sequences based on pitch classes.
    """
    
    def __init__(self, config: Union[GenerationConfig, Dict]):
        """
        Initialize the engine with a configuration.
        
        Args:
            config: Configuration for the generation process.
        """
        if isinstance(config, dict):
            self.config = GenerationConfig(**config)
        else:
            self.config = config
        
        # Log the configuration
        if isinstance(self.config, GenerationConfig):
            config_dict = {k: v for k, v in self.config.__dict__.items()}
            log_config(logger, config_dict)
        
        # Initialize the starting pitch class set
        if isinstance(self.config.start_pc, int):
            self.start_pcs = PitchClassSet([self.config.start_pc])
        else:
            self.start_pcs = PitchClassSet(self.config.start_pc)
        
        logger.info(f"Starting pitch class set: {self.start_pcs}")
        
        # Initialize target pitch class set if provided
        self.target_pcs = None
        if self.config.target_pc is not None:
            if isinstance(self.config.target_pc, int):
                self.target_pcs = PitchClassSet([self.config.target_pc])
            else:
                self.target_pcs = PitchClassSet(self.config.target_pc)
            logger.info(f"Target pitch class set: {self.target_pcs}")
                
        # Initialize transformation steps if not provided
        if self.config.progression and self.config.transformation_steps is None:
            self.config.transformation_steps = self.config.sequence_length - 1
            logger.debug(f"Using default transformation steps: {self.config.transformation_steps}")
    
    def generate(self) -> List[Union[int, List[int]]]:
        """
        Generate a sequence based on the configuration.
        
        Returns:
            A list of pitch classes (for melodic) or pitch class sets (for chordal).
        """
        start_time = time.time()
        
        if self.config.generation_type == GenerationType.MELODIC:
            logger.info("Generating melodic sequence")
            result = self._generate_melodic()
        else:  # GenerationType.CHORDAL
            logger.info("Generating chordal sequence")
            result = self._generate_chordal()
        
        log_execution_time(logger, start_time, "Sequence generation")
        logger.info(f"Generated sequence of length {len(result)}")
        return result
    
    def _generate_melodic(self) -> List[int]:
        """
        Generate a melodic sequence.
        
        Returns:
            A list of pitch classes representing the melody.
        """
        # Get the initial pitch class
        if len(self.start_pcs) != 1:
            # For melodic, we only want one starting note
            start_pc = list(self.start_pcs.pcs)[0].pc
        else:
            start_pc = list(self.start_pcs.pcs)[0].pc
        
        sequence = [start_pc]
        current_pc = start_pc
        
        # Access randomness parameters
        randomness = self.config.randomness_factor
        variation_prob = self.config.variation_probability
        
        if self.config.progression:
            if self.config.progression_type == ProgressionType.DIRECTED and self.target_pcs:
                # Get target pitch class for directed progression
                if len(self.target_pcs) != 1:
                    target_pc = list(self.target_pcs.pcs)[0].pc
                else:
                    target_pc = list(self.target_pcs.pcs)[0].pc
                    
                # Calculate steps needed to reach target
                steps = self.config.transformation_steps or (self.config.sequence_length - 1)
                pc_delta = (target_pc - start_pc) % 12
                step_size = pc_delta / steps if steps > 0 else 0
                
                # Generate intermediate steps
                for i in range(1, self.config.sequence_length):
                    if i < steps:
                        # Calculate the next step
                        expected_pc = (start_pc + int(i * step_size)) % 12
                        
                        # Add controlled randomness based on randomness factor
                        if random.random() < variation_prob:
                            # Calculate maximum deviation based on randomness factor
                            max_deviation = int(1 + 3 * randomness)  # 1-4 semitones based on randomness
                            deviation = random.randint(-max_deviation, max_deviation)
                            next_pc = (expected_pc + deviation) % 12
                        else:
                            next_pc = expected_pc
                    else:
                        # Near the target, we can still have some randomness
                        if random.random() < randomness * 0.5:  # Reduced randomness near target
                            next_pc = (target_pc + random.choice([-1, 1])) % 12
                        else:
                            next_pc = target_pc
                    
                    sequence.append(next_pc)
                    current_pc = next_pc
            
            elif self.config.progression_type == ProgressionType.RANDOM:
                # Generate a random walk through pitch classes with controlled randomness
                for _ in range(1, self.config.sequence_length):
                    # Base interval affected by randomness factor
                    base_interval = self.config.constraints.get("max_interval", 2)
                    max_step = base_interval + int(2 * randomness)  # Increase max step based on randomness
                    
                    # The higher the randomness, the more uniform the distribution
                    if random.random() < randomness:
                        # More random: uniform distribution of steps
                        step = random.randint(-max_step, max_step)
                    else:
                        # More controlled: favor smaller steps
                        smaller_step = random.randint(-1, 1)
                        medium_step = random.randint(-2, 2)
                        step = random.choice([smaller_step, smaller_step, medium_step])
                    
                    next_pc = (current_pc + step) % 12
                    sequence.append(next_pc)
                    current_pc = next_pc
        else:
            # Static generation - generate from the same pitch class set
            for _ in range(1, self.config.sequence_length):
                # Determine if we should vary the pitch class based on variation probability
                if random.random() < variation_prob:
                    # The level of variation depends on randomness factor
                    max_step = 1 + int(2 * randomness)  # 1-3 semitones
                    step = random.randint(-max_step, max_step)
                    next_pc = (current_pc + step) % 12
                else:
                    # Just repeat the same pitch class
                    next_pc = current_pc
                
                sequence.append(next_pc)
                current_pc = next_pc
        
        # Apply post-processing to enhance musicality based on randomness
        if randomness < 0.3:
            # Low randomness: nudge toward more tonal melodic contour
            self._enhance_melodic_contour(sequence)
        
        return sequence
    
    def _enhance_melodic_contour(self, sequence: List[int]) -> None:
        """
        Enhance the musicality of a melodic sequence by adjusting its contour.
        This is an in-place operation.
        
        Args:
            sequence: The melodic sequence to enhance
        """
        # Don't process very short sequences
        if len(sequence) <= 3:
            return
        
        # Look for unmusical leaps and adjust them
        for i in range(1, len(sequence) - 1):
            prev_pc = sequence[i-1]
            current_pc = sequence[i]
            next_pc = sequence[i+1]
            
            # Calculate intervals
            prev_interval = min((current_pc - prev_pc) % 12, (prev_pc - current_pc) % 12)
            next_interval = min((next_pc - current_pc) % 12, (current_pc - next_pc) % 12)
            
            # If we have two large leaps in the same direction, modify the second one
            if prev_interval > 3 and next_interval > 3:
                # Check if leaps are in the same direction
                if (current_pc > prev_pc and next_pc > current_pc) or \
                   (current_pc < prev_pc and next_pc < current_pc):
                    # Change direction or reduce interval
                    if random.random() < 0.7:  # 70% chance to change direction
                        # Move in opposite direction
                        sequence[i+1] = (current_pc - (next_pc - current_pc)) % 12
                    else:
                        # Reduce the size of the second leap
                        sequence[i+1] = (current_pc + (1 if next_pc > current_pc else -1)) % 12

    def _generate_chordal(self) -> List[List[int]]:
        """
        Generate a chord progression.
        
        Returns:
            A list of pitch class sets representing the chords.
        """
        # Get the initial chord
        current_pcs = self.start_pcs
        sequence = [sorted([pc.pc for pc in current_pcs.pcs])]
        
        if self.config.progression:
            if self.config.progression_type == ProgressionType.DIRECTED and self.target_pcs:
                # Calculate steps for progression
                steps = self.config.transformation_steps or (self.config.sequence_length - 1)
                
                # Generate intermediate chords moving toward target
                for i in range(1, self.config.sequence_length):
                    if i < steps:
                        # Transform the chord one step toward the target
                        current_pcs = self._transform_pcs_toward_target(
                            current_pcs, 
                            self.target_pcs, 
                            step=i, 
                            total_steps=steps
                        )
                    else:
                        current_pcs = self.target_pcs
                        
                    sequence.append(sorted([pc.pc for pc in current_pcs.pcs]))
            
            elif self.config.progression_type == ProgressionType.RANDOM:
                # Generate a random chord progression
                for _ in range(1, self.config.sequence_length):
                    # Apply a random transformation from allowed operations
                    current_pcs = self._apply_random_transformation(current_pcs)
                    sequence.append(sorted([pc.pc for pc in current_pcs.pcs]))
        else:
            # Static generation - generate variations on the same chord
            for _ in range(1, self.config.sequence_length):
                if "vary_chord" in self.config.constraints and self.config.constraints["vary_chord"]:
                    # Generate a slight variation
                    varied_pcs = self._apply_minor_variation(current_pcs)
                    sequence.append(sorted([pc.pc for pc in varied_pcs.pcs]))
                    current_pcs = varied_pcs
                else:
                    # Repeat the same chord
                    sequence.append(sorted([pc.pc for pc in current_pcs.pcs]))
        
        return sequence
    
    def _transform_pcs_toward_target(
        self, 
        current: PitchClassSet, 
        target: PitchClassSet, 
        step: int, 
        total_steps: int
    ) -> PitchClassSet:
        """
        Transform a pitch class set toward a target pitch class set.
        
        Args:
            current: The current pitch class set
            target: The target pitch class set
            step: Current step number
            total_steps: Total number of steps
            
        Returns:
            A new pitch class set that's one step closer to the target
        """
        # Determine which operation to apply
        operations = self.config.allowed_operations
        
        # If we have different cardinalities, we need to add or remove notes
        if len(current) < len(target):
            if "add_note" in operations:
                # Add a note from the target that's not in the current set
                target_pcs = [pc.pc for pc in target.pcs]
                current_pcs = [pc.pc for pc in current.pcs]
                candidates = [pc for pc in target_pcs if pc not in current_pcs]
                
                if candidates:
                    # Add one note from the target
                    new_pcs = current_pcs + [random.choice(candidates)]
                    return PitchClassSet(new_pcs)
        
        elif len(current) > len(target):
            if "remove_note" in operations:
                # Remove a note that's not in the target
                target_pcs = [pc.pc for pc in target.pcs]
                current_pcs = [pc.pc for pc in current.pcs]
                candidates = [pc for pc in current_pcs if pc not in target_pcs]
                
                if candidates:
                    # Remove one note not in the target
                    new_pcs = [pc for pc in current_pcs if pc != random.choice(candidates)]
                    return PitchClassSet(new_pcs)
        
        # For equal cardinality, we can use substitution or transposition
        if "transpose" in operations and random.random() < 0.4:
            # Calculate optimal transposition
            current_pcs = sorted([pc.pc for pc in current.pcs])
            target_pcs = sorted([pc.pc for pc in target.pcs])
            
            # Try different transpositions and see which gets us closest
            best_distance = float('inf')
            best_transposition = 0
            
            for t in range(12):
                transposed = [(pc + t) % 12 for pc in current_pcs]
                distance = sum(min((t_pc - c_pc) % 12, (c_pc - t_pc) % 12) 
                              for c_pc, t_pc in zip(sorted(transposed), sorted(target_pcs)))
                
                if distance < best_distance:
                    best_distance = distance
                    best_transposition = t
            
            return current.transpose(best_transposition)
        
        elif "substitute_note" in operations:
            # Substitute one note with a note from the target
            current_pcs = [pc.pc for pc in current.pcs]
            target_pcs = [pc.pc for pc in target.pcs]
            
            # Find notes to replace
            to_replace = [pc for pc in current_pcs if pc not in target_pcs]
            replacements = [pc for pc in target_pcs if pc not in current_pcs]
            
            if to_replace and replacements:
                # Replace one note
                new_pcs = current_pcs.copy()
                idx = new_pcs.index(random.choice(to_replace))
                new_pcs[idx] = random.choice(replacements)
                return PitchClassSet(new_pcs)
        
        # If no transformation was applied, default to a small random change
        return self._apply_minor_variation(current)
    
    def _apply_random_transformation(self, pcs: PitchClassSet) -> PitchClassSet:
        """
        Apply a random transformation to a pitch class set.
        
        Args:
            pcs: The pitch class set to transform
            
        Returns:
            A transformed pitch class set
        """
        operation = random.choice(self.config.allowed_operations)
        logger.debug(f"Applying random transformation: {operation}")
        
        if operation == "transpose":
            # Random transposition
            return pcs.transpose(random.randint(1, 11))
            
        elif operation == "invert":
            # Invert around a random axis
            return pcs.invert(random.randint(0, 11))
            
        elif operation == "complement":
            # Return the complement
            return pcs.complement()
            
        elif operation == "add_note":
            # Add a random note
            current_pcs = [pc.pc for pc in pcs.pcs]
            available = [pc for pc in range(12) if pc not in current_pcs]
            
            if available:
                new_pcs = current_pcs + [random.choice(available)]
                return PitchClassSet(new_pcs)
                
        elif operation == "remove_note":
            # Remove a random note if we have more than one
            if len(pcs) > 1:
                current_pcs = [pc.pc for pc in pcs.pcs]
                to_remove = random.choice(current_pcs)
                new_pcs = [pc for pc in current_pcs if pc != to_remove]
                return PitchClassSet(new_pcs)
                
        elif operation == "substitute_note":
            # Substitute a random note
            current_pcs = [pc.pc for pc in pcs.pcs]
            
            if current_pcs:
                idx = random.randrange(len(current_pcs))
                current_pc = current_pcs[idx]
                
                # Find a different pitch class
                candidates = [pc for pc in range(12) if pc != current_pc and pc not in current_pcs]
                
                if candidates:
                    current_pcs[idx] = random.choice(candidates)
                    return PitchClassSet(current_pcs)
        
        # If we couldn't apply the chosen operation or as a fallback
        return pcs
    
    def _apply_minor_variation(self, pcs: PitchClassSet) -> PitchClassSet:
        """
        Apply a minor variation to a pitch class set.
        
        Args:
            pcs: The pitch class set to vary
            
        Returns:
            A slightly varied pitch class set
        """
        # 50% chance to return the original
        if random.random() < 0.5:
            return pcs
        
        # Otherwise apply a minimal transformation
        current_pcs = [pc.pc for pc in pcs.pcs]
        
        variation_type = random.choice(["shift", "substitute", "add/remove"])
        
        if variation_type == "shift":
            # Shift the entire set by a small interval
            shift = random.choice([-2, -1, 1, 2])
            return pcs.transpose(shift)
            
        elif variation_type == "substitute" and len(current_pcs) > 0:
            # Replace one note with an adjacent one
            idx = random.randrange(len(current_pcs))
            current_pc = current_pcs[idx]
            
            # Shift by +/- 1 semitone
            shift = random.choice([-1, 1])
            current_pcs[idx] = (current_pc + shift) % 12
            return PitchClassSet(current_pcs)
            
        elif variation_type == "add/remove":
            if len(current_pcs) > 3 and random.random() < 0.5:
                # Remove a random note
                to_remove = random.choice(current_pcs)
                new_pcs = [pc for pc in current_pcs if pc != to_remove]
                return PitchClassSet(new_pcs)
            else:
                # Add a random note
                available = [pc for pc in range(12) if pc not in current_pcs]
                
                if available:
                    new_pcs = current_pcs + [random.choice(available)]
                    return PitchClassSet(new_pcs)
        
        # Fallback to the original set if no variation was applied
        return pcs


def generate_sequence_from_config(config_data: Dict) -> List[Union[int, List[int]]]:
    """
    Generate a sequence from a configuration dictionary.
    
    Args:
        config_data: Configuration for the generation
        
    Returns:
        Generated sequence of pitch classes or pitch class sets
    """
    logger.info("Generating sequence from configuration")
    engine = PitchClassEngine(config_data)
    return engine.generate()


# Example configuration schema:
EXAMPLE_CONFIG = {
    "start_pc": 0,  # C
    "generation_type": "melodic",  # or "chordal"
    "sequence_length": 8,  # generate 8 notes/chords
    "progression": True,  # use progression
    "target_pc": 7,  # G
    "progression_type": "directed",  # or "random" or "static"
    "allowed_operations": ["transpose", "invert", "add_note"],
    "constraints": {
        "max_interval": 3,  # maximum step size for melodic movement
        "vary_pc": True  # allow pitch class variation
    },
    "randomness_factor": 0.3,  # level of randomness (0.0-1.0)
    "variation_probability": 0.4  # probability of applying variations (0.0-1.0)
}

logger.info("Engine module initialized")

# Example usage:
# config = {
#     "start_pc": [0, 4, 7],  # C major
#     "generation_type": "chordal",
#     "sequence_length": 4,
#     "progression": True,
#     "target_pc": [7, 11, 2],  # G major
#     "progression_type": "directed"
# }
# sequence = generate_sequence_from_config(config)
# print(sequence)  # [[0, 4, 7], [0, 4, 9], [2, 7, 11], [7, 11, 2]]
