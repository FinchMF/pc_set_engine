"""Pitch class set theory implementation for music analysis and generation.

This module provides classes and utilities for working with pitch classes and pitch class sets
in post-tonal music theory. It supports operations such as:
- Pitch class creation and manipulation
- Pitch class set creation and analysis
- Normal form and prime form calculation
- Forte number identification
- Interval vector calculation
- Set operations (transpose, invert, complement, etc.)

Common pitch class sets like triads, seventh chords, and scales are provided
in the COMMON_SETS dictionary for convenience.

Examples:
    Create and analyze a C minor 7th chord:
    ```
    cm7 = PitchClassSet([0, 3, 7, 10])
    print(cm7.forte_number)  # "4-26"
    print(cm7.interval_vector)  # [0, 1, 1, 1, 2, 1]
    ```

    Transform a chord:
    ```
    cmaj = PitchClassSet([0, 4, 7])  # C major
    fmaj = cmaj.transpose(5)  # Transpose to F major
    cmaj_inv = cmaj.invert()  # Invert the C major chord
    ```
"""
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union
import time
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_logger, log_execution_time

# Initialize logger
logger = get_logger(__name__)

class PitchClass:
    """A class representing a single pitch class from 0-11 (C-B).
    
    Pitch classes represent the twelve distinct notes in traditional Western music,
    independent of their octave. The values 0-11 represent C through B.
    
    Attributes:
        pc (int): The pitch class value (0-11).
        PITCH_CLASS_NAMES (List[str]): Class attribute mapping pitch class numbers to name strings.
    """
    PITCH_CLASS_NAMES = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 
                         'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    
    def __init__(self, pc: int):
        """Initialize a pitch class.
        
        Args:
            pc (int): The pitch class value (will be normalized to 0-11 using modulo 12).
        """
        self.pc = pc % 12
    
    def __repr__(self) -> str:
        """Get the string representation of the pitch class.
        
        Returns:
            str: A string showing the pitch class value and name.
        """
        return f"PC({self.pc}: {self.PITCH_CLASS_NAMES[self.pc]})"
    
    def __eq__(self, other) -> bool:
        """Check if this pitch class equals another.
        
        Args:
            other: Another PitchClass or an integer.
            
        Returns:
            bool: True if the pitch classes are equal, False otherwise.
        """
        if isinstance(other, PitchClass):
            return self.pc == other.pc
        return self.pc == other
    
    def __hash__(self) -> int:
        """Get the hash value for this pitch class.
        
        Returns:
            int: Hash value.
        """
        return hash(self.pc)
    
    def transpose(self, n: int) -> 'PitchClass':
        """Transpose this pitch class by n semitones.
        
        Args:
            n (int): The number of semitones to transpose by.
            
        Returns:
            PitchClass: A new PitchClass object representing the transposed value.
        """
        return PitchClass((self.pc + n) % 12)
    
    def invert(self, axis: int = 0) -> 'PitchClass':
        """Invert this pitch class around a specified axis.
        
        Args:
            axis (int, optional): The axis of inversion (default is 0, which is C).
            
        Returns:
            PitchClass: A new PitchClass object representing the inverted value.
        """
        return PitchClass((axis * 2 - self.pc) % 12)


class PitchClassSet:
    """A class representing a set of pitch classes.
    
    This class implements pitch class set theory operations, providing methods
    for analyzing and manipulating collections of pitch classes. Features include
    normal form and prime form calculation, Forte number identification, and
    interval vector calculation.
    
    Attributes:
        pcs (Set[PitchClass]): The set of pitch class objects.
        _normal_form (List[int], optional): Cached normal form.
        _prime_form (List[int], optional): Cached prime form.
        _forte_number (str, optional): Cached Forte number.
        _interval_vector (List[int], optional): Cached interval vector.
    """
    _FORTE_CATALOG = {}  # This will be populated with the Forte catalog data
    
    def __init__(self, pcs: Union[List[int], Set[int], Tuple[int, ...], np.ndarray]):
        """Initialize a pitch class set.
        
        Args:
            pcs (Union[List[int], Set[int], Tuple[int, ...], np.ndarray]): A collection of pitch class values
            (0-11) representing the pitch class set. This can be a list, set, tuple, or numpy array.
        """
        # Convert all inputs to a set of PitchClass objects
        if isinstance(pcs, np.ndarray):
            pcs = pcs.tolist()
        
        self.pcs = {PitchClass(pc) for pc in pcs}
        self._normal_form = None
        self._prime_form = None
        self._forte_number = None
        self._interval_vector = None
        
        logger.debug(f"Created {self}")
    
    def __repr__(self):
        pc_list = sorted([pc.pc for pc in self.pcs])
        return f"PCS{pc_list}"
    
    def __contains__(self, item):
        if isinstance(item, int):
            item = PitchClass(item)
        return item in self.pcs
    
    def __len__(self):
        return len(self.pcs)
    
    def __eq__(self, other):
        if isinstance(other, PitchClassSet):
            return self.pcs == other.pcs
        return False

    @property
    def cardinality(self) -> int:
        """Get the cardinality (number of pitch classes) of the set.
        
        Returns:
            int: The number of pitch classes in the set.
        """
        return len(self.pcs)
    
    @property
    def normal_form(self) -> List[int]:
        """Calculate and return the normal form of the pitch class set.
        
        The normal form is the most "compact" arrangement of the pitch classes,
        starting with the smallest possible interval from the first to last pitch class.
        Results are cached for efficiency.
        
        Returns:
            List[int]: The pitch classes in normal form.
        """
        start_time = time.time()
        
        if self._normal_form is not None:
            return self._normal_form
        
        if len(self.pcs) == 0:
            return []
        
        # Get all rotations of the set
        pc_list = sorted([pc.pc for pc in self.pcs])
        rotations = []
        for i in range(len(pc_list)):
            rotated = pc_list[i:] + pc_list[:i]
            # Normalize to start from 0
            normalized = [(pc - rotated[0]) % 12 for pc in rotated]
            rotations.append(normalized)
        
        # Find the most compact rotation
        most_compact = None
        for rotation in rotations:
            if most_compact is None or (
                rotation[-1] < most_compact[-1] or 
                (rotation[-1] == most_compact[-1] and rotation < most_compact)
            ):
                most_compact = rotation
        
        # Restore the original pitch classes but in the most compact ordering
        index = rotations.index(most_compact)
        self._normal_form = pc_list[index:] + pc_list[:index]
        
        log_execution_time(logger, start_time, f"Normal form calculation for {self}")
        return self._normal_form
    
    @property
    def prime_form(self) -> List[int]:
        """Calculate and return the prime form of the pitch class set.
        
        The prime form is the most compact arrangement of the pitch class set,
        comparing both the normal form and its inversion and selecting the most
        compact version. Prime form is used for classification and comparison of sets.
        
        Returns:
            List[int]: The pitch classes in prime form.
        """
        if self._prime_form is not None:
            return self._prime_form
        
        if len(self.pcs) == 0:
            return []
        
        # Get the normal form
        normal = self.normal_form
        
        # Calculate the inversion of the normal form
        inversion = [(12 - pc) % 12 for pc in normal]
        inversion.sort()
        
        # Find rotations for the inversion
        inversion_rotations = []
        for i in range(len(inversion)):
            rotated = inversion[i:] + inversion[:i]
            # Normalize to start from 0
            normalized = [(pc - rotated[0]) % 12 for pc in rotated]
            inversion_rotations.append(normalized)
        
        # Find the most compact inversion
        most_compact_inv = min(inversion_rotations, key=lambda x: (x[-1], x))
        
        # Compare normal form with the most compact inversion
        normal_normalized = [(pc - normal[0]) % 12 for pc in normal]
        
        if most_compact_inv < normal_normalized:
            self._prime_form = most_compact_inv
        else:
            self._prime_form = normal_normalized
        
        return self._prime_form
    
    @property
    def forte_number(self) -> str:
        """Get the Forte number for this pitch class set.
        
        The Forte number is a standard classification system for pitch class sets
        in post-tonal theory. It consists of the cardinality followed by a dash and
        a unique identifier (e.g., "3-11" for a major or minor triad).
        
        Returns:
            str: The Forte number, or "Unknown" if not found in the catalog.
        """
        if self._forte_number is not None:
            return self._forte_number
            
        # Get the prime form and look it up in the Forte catalog
        prime = tuple(self.prime_form)
        
        # Initialize Forte catalog if not already done
        if not self._FORTE_CATALOG:
            self._init_forte_catalog()
            
        self._forte_number = self._FORTE_CATALOG.get(prime, "Unknown")
        return self._forte_number
    
    @property
    def interval_vector(self) -> List[int]:
        """Calculate the interval vector of the pitch class set.
        
        The interval vector is a 6-element array showing the frequency of each
        interval class (1-6) in the pitch class set. This provides a measure of
        the "sound" of the set independent of specific pitch classes.
        
        Returns:
            List[int]: The interval vector with 6 elements.
        """
        if self._interval_vector is not None:
            return self._interval_vector
        
        pcs = sorted([pc.pc for pc in self.pcs])
        vector = [0] * 6  # Six interval classes (1-6)
        
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                # Calculate the interval class
                interval = min((pcs[j] - pcs[i]) % 12, (pcs[i] - pcs[j]) % 12)
                if interval > 0:  # Ignore unisons
                    vector[interval - 1] += 1
                    
        self._interval_vector = vector
        return self._interval_vector
    
    @classmethod
    def _init_forte_catalog(cls):
        """Initialize the Forte catalog with mappings from prime forms to Forte numbers.
        
        This is an internal method that populates the _FORTE_CATALOG dictionary
        used for identifying pitch class sets by their standard names.
        """
        logger.info("Initializing Forte catalog")
        # This is a simplified version; the full catalog would be more extensive
        cls._FORTE_CATALOG = {
            # Trichords
            (0, 1, 3): "3-2",
            (0, 1, 4): "3-3",
            (0, 1, 5): "3-4",
            (0, 1, 6): "3-5",
            (0, 2, 3): "3-6",
            (0, 2, 4): "3-7",
            (0, 2, 5): "3-8", 
            (0, 2, 6): "3-9",
            (0, 2, 7): "3-11",
            (0, 3, 4): "3-1",
            (0, 3, 6): "3-10",
            (0, 3, 7): "3-12",
            (0, 4, 8): "3-13",
            
            # Tetrachords
            (0, 1, 2, 3): "4-1",
            (0, 1, 2, 4): "4-2",
            (0, 1, 2, 5): "4-4",
            (0, 1, 2, 6): "4-5",
            (0, 1, 2, 7): "4-6",
            (0, 1, 3, 4): "4-3",
            (0, 1, 3, 5): "4-11",
            (0, 1, 3, 6): "4-13",
            (0, 1, 3, 7): "4-Z29",
            (0, 1, 4, 5): "4-7",
            (0, 1, 4, 6): "4-Z15",
            (0, 1, 4, 7): "4-18",
            (0, 1, 4, 8): "4-19",
            (0, 1, 5, 6): "4-8",
            (0, 1, 5, 8): "4-20",
            (0, 1, 6, 7): "4-9",
            (0, 2, 3, 5): "4-10",
            (0, 2, 3, 6): "4-12",
            (0, 2, 3, 7): "4-14",
            (0, 2, 4, 6): "4-21",
            (0, 2, 4, 7): "4-22",
            (0, 2, 4, 8): "4-24",
            (0, 2, 5, 7): "4-23",
            (0, 2, 5, 8): "4-27",
            (0, 2, 6, 8): "4-25",
            (0, 3, 4, 7): "4-17",
            (0, 3, 5, 8): "4-26",
            (0, 3, 6, 9): "4-28",
            
            # Pentachords (selected few for demonstration)
            (0, 1, 2, 3, 4): "5-1",
            (0, 1, 2, 3, 6): "5-4",
            (0, 1, 3, 5, 7): "5-24",
            (0, 2, 4, 6, 8): "5-33",
            
            # Hexachords (selected few)
            (0, 1, 2, 3, 4, 5): "6-1",
            (0, 1, 2, 3, 4, 7): "6-2",
            (0, 1, 2, 3, 4, 8): "6-Z3",
            (0, 1, 2, 4, 5, 8): "6-15",
            (0, 2, 4, 6, 8, 10): "6-35",  # Whole tone scale
            
            # Special sets
            (0, 1, 3, 4, 6, 7, 9, 10): "8-28",  # Octatonic scale
            (0, 2, 4, 5, 7, 9, 11): "7-35",     # Major scale
            (0, 2, 3, 5, 7, 8, 10): "7-34",     # Harmonic minor scale
            (0, 1, 3, 5, 6, 8, 10): "7-31",     # Harmonic major scale
        }
        logger.info("Forte catalog initialized")
    
    def transpose(self, n: int) -> 'PitchClassSet':
        """Transpose the pitch class set by n semitones.
        
        Args:
            n (int): The number of semitones to transpose by.
            
        Returns:
            PitchClassSet: A new PitchClassSet object representing the transposed set.
        """
        return PitchClassSet([((pc.pc + n) % 12) for pc in self.pcs])
    
    def invert(self, axis: int = 0) -> 'PitchClassSet':
        """Invert the pitch class set around a specified axis.
        
        Args:
            axis (int, optional): The axis of inversion (default is 0, which is C).
            
        Returns:
            PitchClassSet: A new PitchClassSet object representing the inverted set.
        """
        return PitchClassSet([((axis * 2 - pc.pc) % 12) for pc in self.pcs])
    
    def complement(self) -> 'PitchClassSet':
        """Get the complement of the pitch class set.
        
        The complement contains all pitch classes not in the original set.
        
        Returns:
            PitchClassSet: A new PitchClassSet object representing the complement.
        """
        current_pcs = {pc.pc for pc in self.pcs}
        complement_pcs = {pc for pc in range(12) if pc not in current_pcs}
        return PitchClassSet(complement_pcs)
    
    def is_subset(self, other: 'PitchClassSet') -> bool:
        """Check if this set is a subset of another pitch class set.
        
        Args:
            other (PitchClassSet): Another pitch class set.
            
        Returns:
            bool: True if this is a subset of other, False otherwise.
        """
        return all(pc in other.pcs for pc in self.pcs)
    
    def is_superset(self, other: 'PitchClassSet') -> bool:
        """Check if this set is a superset of another pitch class set.
        
        Args:
            other (PitchClassSet): Another pitch class set.
            
        Returns:
            bool: True if this is a superset of other, False otherwise.
        """
        return all(pc in self.pcs for pc in other.pcs)
    
    def intersection(self, other: 'PitchClassSet') -> 'PitchClassSet':
        """Get the intersection of this set with another pitch class set.
        
        Args:
            other (PitchClassSet): Another pitch class set.
            
        Returns:
            PitchClassSet: A new PitchClassSet containing elements in both sets.
        """
        return PitchClassSet([pc.pc for pc in self.pcs if pc in other.pcs])
    
    def union(self, other: 'PitchClassSet') -> 'PitchClassSet':
        """Get the union of this set with another pitch class set.
        
        Args:
            other (PitchClassSet): Another pitch class set.
            
        Returns:
            PitchClassSet: A new PitchClassSet containing elements from either set.
        """
        return PitchClassSet([pc.pc for pc in self.pcs] + [pc.pc for pc in other.pcs])
    
    def is_z_related(self, other: 'PitchClassSet') -> bool:
        """Check if this set is Z-related to another pitch class set.
        
        Z-related sets have the same interval vector but different prime forms,
        meaning they have the same intervallic content but different structures.
        
        Args:
            other (PitchClassSet): Another pitch class set.
            
        Returns:
            bool: True if the sets are Z-related, False otherwise.
        """
        return (self.interval_vector == other.interval_vector and 
                self.prime_form != other.prime_form)

    @classmethod
    def from_name(cls, name: str) -> 'PitchClassSet':
        """Create a pitch class set from a string of note names.
        
        Args:
            name (str): Space-separated note names (e.g., "C E G").
            
        Returns:
            PitchClassSet: A new PitchClassSet containing the specified notes.
            
        Example:
            ```
            cmaj = PitchClassSet.from_name("C E G")
            ```
        """
        note_to_pc = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        notes = name.split()
        pcs = [note_to_pc[note] for note in notes if note in note_to_pc]
        return cls(pcs)


# Common named sets for reference
COMMON_SETS = {
    "major_triad": PitchClassSet([0, 4, 7]),  # C major
    "minor_triad": PitchClassSet([0, 3, 7]),  # C minor
    "diminished_triad": PitchClassSet([0, 3, 6]),  # C diminished
    "augmented_triad": PitchClassSet([0, 4, 8]),  # C augmented
    "major7": PitchClassSet([0, 4, 7, 11]),  # Cmaj7
    "dominant7": PitchClassSet([0, 4, 7, 10]),  # C7
    "minor7": PitchClassSet([0, 3, 7, 10]),  # Cm7
    "half_diminished7": PitchClassSet([0, 3, 6, 10]),  # Cm7b5
    "diminished7": PitchClassSet([0, 3, 6, 9]),  # Cdim7
    "whole_tone": PitchClassSet([0, 2, 4, 6, 8, 10]),  # Whole tone scale
    "octatonic": PitchClassSet([0, 1, 3, 4, 6, 7, 9, 10]),  # Octatonic scale
    "chromatic": PitchClassSet(range(12)),  # Chromatic scale
    "major_scale": PitchClassSet([0, 2, 4, 5, 7, 9, 11]),  # C major scale
    "harmonic_minor": PitchClassSet([0, 2, 3, 5, 7, 8, 11]),  # C harmonic minor
}

logger.info(f"Loaded {len(COMMON_SETS)} common pitch class sets")

# Example usage:
# cmin7 = PitchClassSet([0, 3, 7, 10])
# print(cmin7.forte_number)  # "4-26"
# print(cmin7.interval_vector)  # [0, 1, 1, 1, 2, 1]
# print(cmin7.normal_form)
# print(cmin7.prime_form)
# 
# # Transpose a set
# fmin7 = cmin7.transpose(5)  # F minor 7 chord
# 
# # Get a common set
# major_scale = COMMON_SETS["major_scale"]
