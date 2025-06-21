"""Rhythm generation and manipulation for the Pitch Class Rules Engine.

This module provides tools for generating, manipulating, and applying rhythmic patterns
to melodic and chordal sequences. It supports various time signatures, subdivisions,
accent patterns, and rhythmic transformations.

Key features:
- Time signature and subdivision-based rhythm generation
- Accent pattern application for musicality
- Configurable polyrhythms and cross-rhythms
- Rhythm vector manipulation (augmentation, diminution, displacement)
- Integration with melodic and chordal sequences

The module is designed to work alongside the pitch class generation system,
adding a temporal dimension to the generated musical content.

Examples:
    Basic rhythm generation:
    ```python
    from pc_sets.rhythm import RhythmEngine, RhythmConfig
    
    config = RhythmConfig(
        time_signature=(4, 4),
        subdivision=8,
        accent_pattern=[1, 0, 0.5, 0, 1, 0, 0.5, 0]
    )
    
    rhythm_engine = RhythmEngine(config)
    rhythm_vector = rhythm_engine.generate(length=8)
    ```

    Apply rhythm to a melodic sequence:
    ```python
    from pc_sets.engine import generate_sequence_from_config
    from pc_sets.rhythm import apply_rhythm_to_sequence
    
    # Generate a melodic sequence
    config = {"start_pc": 0, "generation_type": "melodic", "sequence_length": 8}
    sequence = generate_sequence_from_config(config)
    
    # Generate and apply a rhythm
    rhythm_config = {"time_signature": [4, 4], "subdivision": 8}
    timed_sequence = apply_rhythm_to_sequence(sequence, rhythm_config)
    ```
"""
import math
import random
from typing import List, Tuple, Dict, Union, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_logger

# Initialize logger
logger = get_logger(__name__)


class SubdivisionType(Enum):
    """Types of rhythmic subdivisions.
    
    Defines how beats are divided in the rhythm generation process.
    
    Attributes:
        REGULAR: Standard even subdivision (duplets, triplets, etc.)
        SWING: Jazz swing feel with uneven subdivision
        DOTTED: Dotted rhythm feel (long-short pattern)
        SHUFFLE: Shuffle feel with slight swing
        COMPLEX: Mixed subdivisions in one measure
    """
    REGULAR = "regular"
    SWING = "swing"
    DOTTED = "dotted"
    SHUFFLE = "shuffle"
    COMPLEX = "complex"


class AccentType(Enum):
    """Types of accent patterns.
    
    Defines how accents are applied to beats in the rhythm.
    
    Attributes:
        DOWNBEAT: Traditional accent on the downbeat
        OFFBEAT: Accent on off beats (upbeats)
        SYNCOPATED: Syncopated accent pattern (emphasis between beats)
        CUSTOM: Custom user-defined accent pattern
    """
    DOWNBEAT = "downbeat"
    OFFBEAT = "offbeat" 
    SYNCOPATED = "syncopated"
    CUSTOM = "custom"


@dataclass
class RhythmConfig:
    """Configuration for rhythm generation.
    
    This class encapsulates all parameters needed to control rhythm
    generation, including time signature, subdivision, accent patterns,
    and transformations.
    
    Attributes:
        time_signature: Tuple of (beats per measure, beat unit)
        subdivision: Number of subdivisions per beat
        subdivision_type: Type of subdivision (regular, swing, etc.)
        accent_pattern: Optional list of accent values (0.0-1.0) for each subdivision
        accent_type: Type of accent pattern to apply
        variation_probability: Probability of applying variations to the rhythm
        shift_probability: Probability of shifting accents
        dynamics_range: Tuple of (min, max) dynamic values (0.0-1.0)
        tempo: Beats per minute
        polyrhythm_ratio: Optional tuple for polyrhythm generation (e.g., 3:2)
    """
    time_signature: Tuple[int, int] = (4, 4)
    subdivision: int = 4  # e.g., 4 for sixteenth notes in 4/4
    subdivision_type: SubdivisionType = SubdivisionType.REGULAR
    accent_pattern: Optional[List[float]] = None
    accent_type: AccentType = AccentType.DOWNBEAT
    variation_probability: float = 0.2
    shift_probability: float = 0.1
    dynamics_range: Tuple[float, float] = (0.5, 1.0)
    tempo: int = 120
    polyrhythm_ratio: Optional[Tuple[int, int]] = None
    
    def __post_init__(self):
        """Initialize derived attributes and validate configuration."""
        # Convert string values to enums
        if isinstance(self.subdivision_type, str):
            self.subdivision_type = SubdivisionType(self.subdivision_type.lower())
            
        if isinstance(self.accent_type, str):
            self.accent_type = AccentType(self.accent_type.lower())
        
        # Clamp probabilities between 0 and 1
        self.variation_probability = max(0.0, min(1.0, self.variation_probability))
        self.shift_probability = max(0.0, min(1.0, self.shift_probability))
        
        # Generate accent pattern if not provided
        if self.accent_pattern is None:
            self.accent_pattern = self._generate_default_accents()
    
    def _generate_default_accents(self) -> List[float]:
        """Generate a default accent pattern based on the accent type.
        
        Returns:
            List of accent values for each subdivision.
        """
        # Calculate total number of subdivisions in a measure
        total_subdivisions = self.time_signature[0] * self.subdivision
        
        # Initialize all accents to neutral value
        accents = [0.5] * total_subdivisions
        
        if self.accent_type == AccentType.DOWNBEAT:
            # Accent the downbeats (first subdivision of each beat)
            for i in range(0, total_subdivisions, self.subdivision):
                accents[i] = 1.0
                
        elif self.accent_type == AccentType.OFFBEAT:
            # Accent the offbeats (second subdivision of each beat in common subdivisions)
            offset = self.subdivision // 2
            for i in range(0, total_subdivisions, self.subdivision):
                if i + offset < total_subdivisions:
                    accents[i] = 0.3  # Reduce downbeat
                    accents[i + offset] = 1.0  # Accent offbeat
        
        elif self.accent_type == AccentType.SYNCOPATED:
            # Create a syncopated pattern (accent weak beats/subdivisions)
            for i in range(0, total_subdivisions):
                # Accent the "and" of beats in common time
                if (i % self.subdivision) == self.subdivision // 2:
                    accents[i] = 1.0
                # Slightly accent unexpected places for syncopation
                elif i % 3 == 0 and i % self.subdivision != 0:
                    accents[i] = 0.8
                # De-emphasize traditional strong beats
                elif i % self.subdivision == 0:
                    accents[i] = 0.4
        
        return accents


class RhythmEngine:
    """Engine for generating rhythmic patterns.
    
    This class implements the core functionality for generating rhythm
    vectors according to a provided configuration. It supports various
    rhythm manipulations like augmentation, diminution, and displacement.
    
    Attributes:
        config: The rhythm generation configuration.
    """
    
    def __init__(self, config: Union[RhythmConfig, Dict]):
        """Initialize the rhythm engine with a configuration.
        
        Args:
            config: Configuration for the rhythm generation, either as a
                RhythmConfig object or a dictionary of parameters.
        """
        if isinstance(config, dict):
            self.config = RhythmConfig(**config)
        else:
            self.config = config
            
        logger.info(f"Initialized rhythm engine with time signature {self.config.time_signature}")
    
    def generate(self, length: Optional[int] = None) -> List[float]:
        """Generate a rhythm vector of note durations.
        
        Creates a list of note duration values based on the configured
        rhythm parameters. Each value represents the duration of a note
        in beats.
        
        Args:
            length: Optional length of the sequence. If not provided,
                   will generate one full measure based on time signature.
                   
        Returns:
            List of float values representing note durations.
            
        Example:
            ```python
            rhythm_engine = RhythmEngine(config)
            # Generate durations for 8 notes
            durations = rhythm_engine.generate(length=8)
            ```
        """
        # Determine the pattern length (one measure by default)
        if length is None:
            # Calculate subdivisions in one full measure
            length = self.config.time_signature[0] * self.config.subdivision
            
        # Generate base rhythm durations for one measure
        base_pattern = self._generate_base_pattern()
        
        # Extend or truncate to match the requested length
        if len(base_pattern) < length:
            # Repeat pattern to fill requested length
            repetitions = math.ceil(length / len(base_pattern))
            extended_pattern = base_pattern * repetitions
            rhythm_vector = extended_pattern[:length]
        else:
            # Truncate pattern to requested length
            rhythm_vector = base_pattern[:length]
        
        # Apply variations if configured
        if self.config.variation_probability > 0:
            rhythm_vector = self._apply_variations(rhythm_vector)
            
        logger.debug(f"Generated rhythm vector of length {len(rhythm_vector)}")
        return rhythm_vector
    
    def _generate_base_pattern(self) -> List[float]:
        """Generate the base rhythm pattern for one measure.
        
        Implements different subdivision types and creates the foundational
        rhythm pattern based on the configuration.
        
        Returns:
            List of duration values for one measure.
        """
        # Calculate total subdivisions in a measure
        beats_per_measure = self.config.time_signature[0]
        beat_unit = self.config.time_signature[1]
        subdivisions_per_beat = self.config.subdivision
        
        # Calculate duration of a single subdivision in beats
        base_duration = 1.0 / subdivisions_per_beat
        
        # Generate durations based on subdivision type
        durations = []
        
        if self.config.subdivision_type == SubdivisionType.REGULAR:
            # Regular even subdivisions
            durations = [base_duration] * (beats_per_measure * subdivisions_per_beat)
            
        elif self.config.subdivision_type == SubdivisionType.SWING:
            # Swing rhythm (typically 2:1 ratio in eighth notes)
            for _ in range(beats_per_measure):
                for i in range(0, subdivisions_per_beat, 2):
                    if i < subdivisions_per_beat - 1:
                        # Swing ratio (longer first note, shorter second)
                        durations.append(base_duration * 1.5)  # Long
                        durations.append(base_duration * 0.5)  # Short
                    else:
                        # Handle odd number of subdivisions
                        durations.append(base_duration)
                        
        elif self.config.subdivision_type == SubdivisionType.DOTTED:
            # Dotted rhythm (3:1 ratio)
            for _ in range(beats_per_measure):
                for i in range(0, subdivisions_per_beat, 2):
                    if i < subdivisions_per_beat - 1:
                        # Dotted rhythm (long-short pattern)
                        durations.append(base_duration * 1.5)  # Dotted note
                        durations.append(base_duration * 0.5)  # Short note
                    else:
                        # Handle odd number of subdivisions
                        durations.append(base_duration)
                        
        elif self.config.subdivision_type == SubdivisionType.SHUFFLE:
            # Shuffle rhythm (milder swing, typically 3:2 ratio)
            for _ in range(beats_per_measure):
                for i in range(0, subdivisions_per_beat, 2):
                    if i < subdivisions_per_beat - 1:
                        # Shuffle ratio (slightly longer first note)
                        durations.append(base_duration * 1.2)  # Slightly long
                        durations.append(base_duration * 0.8)  # Slightly short
                    else:
                        # Handle odd number of subdivisions
                        durations.append(base_duration)
                        
        elif self.config.subdivision_type == SubdivisionType.COMPLEX:
            # Mixed subdivisions (e.g., combining duplets and triplets)
            subdivision_patterns = [
                # Regular
                [base_duration] * subdivisions_per_beat,
                # Triplet feel
                [base_duration * (3/subdivisions_per_beat)] * 3,
                # Dotted rhythm
                [base_duration * 1.5, base_duration * 0.5] * (subdivisions_per_beat // 2),
            ]
            
            for _ in range(beats_per_measure):
                # Select a random subdivision pattern for this beat
                pattern = random.choice(subdivision_patterns)
                # Normalize to ensure the beat duration is preserved
                beat_total = sum(pattern)
                normalized_pattern = [d * (1.0 / beat_total) for d in pattern]
                durations.extend(normalized_pattern)
        
        # Handle polyrhythm if configured
        if self.config.polyrhythm_ratio:
            durations = self._apply_polyrhythm(durations)
            
        return durations
    
    def _apply_polyrhythm(self, durations: List[float]) -> List[float]:
        """Apply a polyrhythm to the duration pattern.
        
        Args:
            durations: The original duration pattern.
            
        Returns:
            Modified durations with polyrhythm applied.
        """
        if not self.config.polyrhythm_ratio:
            return durations
        
        polyrhythm_x, polyrhythm_y = self.config.polyrhythm_ratio
        total_beats = self.config.time_signature[0]
        
        # Calculate note durations for each rhythm layer
        layer1_duration = total_beats / polyrhythm_x
        layer2_duration = total_beats / polyrhythm_y
        
        # Create polyrhythm by combining both layers
        polyrhythm = []
        
        # Layer 1
        for i in range(polyrhythm_x):
            polyrhythm.append(layer1_duration)
            
        # Combine with original pattern by interspersing the polyrhythm
        combined = []
        poly_idx = 0
        
        for i, dur in enumerate(durations):
            if i % self.config.subdivision == 0 and poly_idx < len(polyrhythm):
                # Insert polyrhythm element on beat boundaries
                combined.append(polyrhythm[poly_idx])
                poly_idx += 1
            else:
                # Keep original duration
                combined.append(dur)
        
        return combined
    
    def _apply_variations(self, rhythm_vector: List[float]) -> List[float]:
        """Apply rhythmic variations to the pattern.
        
        Modifies the rhythm by applying occasional variations like
        ties, dotted rhythms, or syncopation based on the configured
        variation probability.
        
        Args:
            rhythm_vector: The original rhythm pattern.
            
        Returns:
            Modified rhythm with variations applied.
        """
        varied_rhythm = rhythm_vector.copy()
        
        # Apply variations based on probability
        for i in range(len(varied_rhythm) - 1):
            if random.random() < self.config.variation_probability:
                variation_type = random.choice(['tie', 'dot', 'shorten'])
                
                if variation_type == 'tie':
                    # Tie this note to the next (combine durations)
                    varied_rhythm[i] += varied_rhythm[i + 1]
                    varied_rhythm[i + 1] = 0.0  # Tied note has zero duration
                    
                elif variation_type == 'dot':
                    # Apply dotted rhythm if there's room
                    if i + 2 < len(varied_rhythm):
                        # Take half of the next note's duration
                        dot_amount = varied_rhythm[i + 1] * 0.5
                        varied_rhythm[i] += dot_amount
                        varied_rhythm[i + 1] -= dot_amount
                        
                elif variation_type == 'shorten':
                    # Shorten this note and create a rest
                    if varied_rhythm[i] > 0.1:  # Only if note is long enough
                        shortened = varied_rhythm[i] * 0.75
                        # The rest is implicit in the shorter duration
                        varied_rhythm[i] = shortened
        
        # Filter out any zero durations (tied notes)
        varied_rhythm = [dur for dur in varied_rhythm if dur > 0]
        
        return varied_rhythm
    
    def apply_rhythm_to_sequence(self, 
                               sequence: List[Union[int, List[int]]],
                               is_melodic: bool = True) -> List[Dict]:
        """Apply the generated rhythm to a pitch class sequence.
        
        Combines the pitch information with rhythm information to create
        a sequence of notes with durations, which can be used for MIDI
        generation or other output formats.
        
        Args:
            sequence: A melodic or chordal sequence from the pitch class engine.
            is_melodic: Whether the sequence is melodic (True) or chordal (False).
            
        Returns:
            List of dictionaries containing pitch and duration information.
            
        Example:
            ```python
            melodic_sequence = [0, 4, 7, 4]  # C-E-G-E
            rhythm_engine = RhythmEngine(rhythm_config)
            timed_sequence = rhythm_engine.apply_rhythm_to_sequence(melodic_sequence)
            # Result: [{"pitch": 0, "duration": 0.5}, {"pitch": 4, "duration": 0.25}, ...]
            ```
        """
        # Generate rhythm pattern of appropriate length
        rhythm = self.generate(length=len(sequence))
        
        # Match rhythm to sequence
        result = []
        
        for i, pc in enumerate(sequence):
            # Get the corresponding rhythm value (or default if index is out of bounds)
            duration = rhythm[i] if i < len(rhythm) else 0.25
            
            # Create note or chord with duration
            if is_melodic:
                # Single note for melodic sequence
                result.append({
                    "pitch": pc, 
                    "duration": duration,
                    "velocity": self._get_velocity_for_note(i)
                })
            else:
                # Chord for chordal sequence
                result.append({
                    "pitches": pc,
                    "duration": duration,
                    "velocity": self._get_velocity_for_note(i)
                })
        
        return result
    
    def _get_velocity_for_note(self, position: int) -> int:
        """Determine the velocity (volume) for a note based on accent pattern.
        
        Args:
            position: Position in the sequence.
            
        Returns:
            Velocity value between 0-127 for MIDI.
        """
        # Get the accent pattern and wrap around if needed
        if self.config.accent_pattern:
            pattern_position = position % len(self.config.accent_pattern)
            accent = self.config.accent_pattern[pattern_position]
        else:
            accent = 0.7  # Default accent if no pattern
        
        # Apply random variation to accent
        if random.random() < self.config.shift_probability:
            accent_shift = random.uniform(-0.1, 0.1)
            accent = max(0.0, min(1.0, accent + accent_shift))
        
        # Map accent (0.0-1.0) to MIDI velocity (0-127)
        min_vel, max_vel = 40, 127  # Reasonable MIDI velocity range
        velocity = int(min_vel + accent * (max_vel - min_vel))
        
        return velocity
    
    def augment(self, rhythm_vector: List[float], factor: float = 2.0) -> List[float]:
        """Augment a rhythm by stretching the durations.
        
        Args:
            rhythm_vector: Original rhythm durations.
            factor: Multiplication factor for durations.
            
        Returns:
            Augmented (lengthened) rhythm.
        """
        return [duration * factor for duration in rhythm_vector]
    
    def diminish(self, rhythm_vector: List[float], factor: float = 2.0) -> List[float]:
        """Diminish a rhythm by compressing the durations.
        
        Args:
            rhythm_vector: Original rhythm durations.
            factor: Division factor for durations.
            
        Returns:
            Diminished (shortened) rhythm.
        """
        return [duration / factor for duration in rhythm_vector]
    
    def displace(self, rhythm_vector: List[float], offset: int = 1) -> List[float]:
        """Displace a rhythm by shifting the pattern.
        
        Args:
            rhythm_vector: Original rhythm durations.
            offset: Number of positions to shift.
            
        Returns:
            Displaced rhythm pattern.
        """
        if not rhythm_vector:
            return []
        
        # Rotate the pattern by the offset
        return rhythm_vector[offset:] + rhythm_vector[:offset]


def apply_rhythm_to_sequence(
    sequence: List[Union[int, List[int]]],
    rhythm_config: Union[Dict, RhythmConfig],
    is_melodic: bool = True
) -> List[Dict]:
    """Apply rhythm to a pitch class sequence.
    
    Convenience function to apply rhythm to a sequence without
    explicitly creating a RhythmEngine instance.
    
    Args:
        sequence: A melodic or chordal sequence from the pitch class engine.
        rhythm_config: Configuration for rhythm generation.
        is_melodic: Whether the sequence is melodic (True) or chordal (False).
        
    Returns:
        List of dictionaries containing pitch and rhythm information.
    """
    engine = RhythmEngine(rhythm_config)
    return engine.apply_rhythm_to_sequence(sequence, is_melodic)


def get_common_rhythm_patterns() -> Dict[str, List[float]]:
    """Get a dictionary of common rhythm patterns.
    
    Returns predefined rhythm patterns for various musical styles and meters.
    
    Returns:
        Dictionary mapping pattern names to lists of duration values.
    """
    return {
        # 4/4 patterns
        "basic_4_4": [1.0, 1.0, 1.0, 1.0],
        "basic_eighth_4_4": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "waltz_3_4": [1.0, 0.5, 0.5, 1.0],
        "bossa_nova": [0.75, 0.25, 0.5, 0.5, 0.75, 0.25, 0.5, 0.5],
        "son_clave": [0.5, 0.5, 0.75, 0.75, 0.5, 1.0],
        "rumba_clave": [0.75, 0.5, 0.75, 0.5, 0.5],
        
        # 3/4 patterns
        "waltz_basic": [1.0, 0.5, 0.5, 1.0],
        "waltz_eighth": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        
        # 6/8 patterns
        "compound_basic": [1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
        "compound_swung": [1.5, 0.5, 1.0, 1.5, 0.5],
        
        # Jazz patterns
        "swing_eighth": [0.66, 0.33, 0.66, 0.33, 0.66, 0.33, 0.66, 0.33],
        "bebop": [0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5],
        
        # Classical patterns
        "alberti_bass": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        "siciliana": [0.75, 0.25, 0.5, 0.5],
    }


# Example rhythm configuration
EXAMPLE_RHYTHM_CONFIG = {
    "time_signature": (4, 4),
    "subdivision": 4,  # Sixteenth notes
    "subdivision_type": "regular",
    "accent_type": "downbeat",
    "variation_probability": 0.2,
    "shift_probability": 0.1,
    "dynamics_range": (0.6, 1.0),
    "tempo": 120
}

logger.info("Rhythm module initialized")
