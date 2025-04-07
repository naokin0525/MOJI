# src/data/__init__.py
"""
Data handling package for the SVG Handwriting Generation project.

This package includes modules for:
- Dataset definition and loading (dataset.py)
- SVG file parsing and processing (svg_utils.py)
- Custom .moj dataset format parsing (moj_parser.py)
- Interacting with the GlyphWiki API (glyphwiki_api.py)
"""

# Expose key components directly from the package level
from .dataset import HandwritingDataset, create_dataloader
from .svg_utils import parse_svg_file, simplify_strokes, normalize_strokes, strokes_to_sequence_tensor
from .moj_parser import parse_moj_file
from .glyphwiki_api import fetch_glyph_data

# Define what `from src.data import *` imports
__all__ = [
    "HandwritingDataset",
    "create_dataloader",
    "parse_svg_file",
    "simplify_strokes",
    "normalize_strokes",
    "strokes_to_sequence_tensor",
    "parse_moj_file",
    "fetch_glyph_data",
]

# You can add package-level initialization code here if needed in the future.