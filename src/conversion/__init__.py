# src/conversion/__init__.py
"""
Format Conversion package for the SVG Handwriting Generation project.

Contains modules for:
- Converting generated SVGs to raster formats (PNG, JPG) (to_raster.py)
- Converting generated SVGs (per character) into font files (OTF, TTF) (to_font.py)
"""

from .to_raster import convert_svg_to_raster
from .to_font import convert_svgs_to_font

__all__ = [
    "convert_svg_to_raster",
    "convert_svgs_to_font",
]