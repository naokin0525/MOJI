"""
Image and Font Converter Utility

This script provides functions for converting generated SVG files into other formats
like PNG and JPG, as well as for generating OpenType (.otf) and TrueType (.ttf) fonts
from the learned handwriting model.
"""

import logging
import cairosvg
from PIL import Image
from fonttools.fontBuilder import FontBuilder
from fonttools.pens.ttGlyphPen import TTGlyphPen
from svgpathtools import parse_path

from src.config import (
    PNG_SCALE_FACTOR,
    FONT_UNITS_PER_EM,
    FONT_FAMILY_NAME,
    FONT_STYLE_NAME,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def svg_to_png(svg_path: str, png_path: str):
    """Converts an SVG file to a PNG file."""
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path, scale=PNG_SCALE_FACTOR)
        logging.info(f"Successfully converted {svg_path} to {png_path}")
    except Exception as e:
        logging.error(f"Failed to convert SVG to PNG: {e}")


def svg_to_jpg(svg_path: str, jpg_path: str):
    """Converts an SVG file to a JPG file by first rendering to PNG."""
    temp_png_path = jpg_path + ".temp.png"
    try:
        svg_to_png(svg_path, temp_png_path)
        img = Image.open(temp_png_path).convert("RGB")
        img.save(jpg_path, "jpeg")
        logging.info(f"Successfully converted {svg_path} to {jpg_path}")
    except Exception as e:
        logging.error(f"Failed to convert SVG to JPG: {e}")
    finally:
        import os

        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)


def generate_font(model, character_map: dict, output_path: str):
    """
    Generates an OTF/TTF font from a trained model and a character set.

    Args:
        model: The trained generation model.
        character_map (dict): A dictionary mapping characters to their vector representations.
                              (This is a simplified interface; in practice, you'd generate
                               the SVG path for each character on the fly).
        output_path (str): The path to save the font file (.otf or .ttf).
    """
    logging.info("Starting font generation...")
    try:
        fb = FontBuilder(FONT_UNITS_PER_EM)
        fb.setupGlyphOrder([".notdef"] + list(character_map.keys()))
        fb.setupCharacterMap(character_map)

        # This is a simplified representation. A real implementation needs
        # to generate a canonical SVG for each character from the model.
        # For now, we assume `character_map` gives us pre-generated SVG path strings.
        glyphs = {}
        for char, svg_path_string in character_map.items():
            pen = TTGlyphPen(None)
            path = parse_path(svg_path_string)
            path.draw(pen)
            glyphs[char] = pen.glyph()

        fb.setupGlyphMetrics({".notdef": (600, 0)})  # Default width for .notdef
        for char, glyph in glyphs.items():
            # You would need to calculate glyph width properly
            fb.setupGlyphMetrics({char: (glyph.width, 0)})

        fb.glyphs = glyphs
        fb.setupHorizontalMetrics()
        # Naming and metadata
        fb.setupNameTable(
            {"familyName": FONT_FAMILY_NAME, "styleName": FONT_STYLE_NAME}
        )
        fb.setupOS2()
        fb.setupPost()
        fb.setupFpgm()
        fb.setupPrep()

        fb.save(output_path)
        logging.info(f"Font successfully saved to {output_path}")

    except Exception as e:
        logging.error(f"Font generation failed: {e}")
        raise
