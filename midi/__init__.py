"""MIDI utilities for the pitch class rules engine.

This package provides functionality to convert pitch class sequences
to MIDI files for playback and further processing.
"""
import os
from pathlib import Path

# Create the midi_files directory
MIDI_FILES_DIR = Path(__file__).parent.parent / "midi_files"
MIDI_FILES_DIR.mkdir(exist_ok=True)

from .translator import (
    sequence_to_midi,
    load_and_convert_sequence,
    enhance_with_rhythm,
    sequence_to_midi_with_rhythm,
    DEFAULT_PARAMS,
    DEFAULT_MIDI_DIR
)

__all__ = [
    'sequence_to_midi',
    'load_and_convert_sequence',
    'enhance_with_rhythm',
    'sequence_to_midi_with_rhythm',
    'DEFAULT_PARAMS',
    'DEFAULT_MIDI_DIR'
]
