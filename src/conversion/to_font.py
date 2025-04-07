# src/conversion/to_font.py
"""
Utility to convert generated character SVGs into font files (TTF/OTF).

Uses the 'fontTools' library. This is a complex process and this implementation
is a basic starting point. It handles simple SVG paths (lines, possibly curves
via cu2qu) and sets up minimal required font tables. Does NOT handle complex
SVG features, advanced typography, or automatic metric calculation robustly.

Requires external dependencies: `pip install fonttools`
"""

import logging
import os
from typing import Dict, List, Tuple
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.transformPen import TransformPen
from fontTools.svgLib.path import SVGPath # EXPERIMENTAL path parser
from fontTools.misc.transform import Identity
from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib.tables._c_m_a_p import cmap_format_4
from fontTools.ttLib.tables._h_m_t_x import Hmtx
from fontTools.ttLib.tables._g_l_y_f import Glyph # For type hinting

# Local implementation for converting cubic Bezier curves to quadratic Bezier curves
class QuadraticConverterPen:
    def __init__(self, outPen, maxErr=1.0):
        self.outPen = outPen
        self._currentPoint = None
        self.maxErr = maxErr
    def moveTo(self, pt):
        self.outPen.moveTo(pt)
        self._currentPoint = pt
    def lineTo(self, pt):
        self.outPen.lineTo(pt)
        self._currentPoint = pt
    def curveTo(self, cp1, cp2, pt):
        # Convert cubic curve defined by (P0, cp1, cp2, pt) into a quadratic curve approximation.
        P0 = self._currentPoint
        Q0 = P0
        Q2 = pt
        Q1 = ((3*cp1[0] - P0[0] + 3*cp2[0] - pt[0]) / 4,
              (3*cp1[1] - P0[1] + 3*cp2[1] - pt[1]) / 4)
        self.outPen.qCurveTo(Q1, Q2)
        self._currentPoint = pt
    def qCurveTo(self, *points):
        self.outPen.qCurveTo(*points)
        self._currentPoint = points[-1]
    def closePath(self):
        self.outPen.closePath()
    def endPath(self):
        self.outPen.endPath()

# Import custom utilities and exceptions
try:
    from ..utils.error_handling import ConversionError, DataError
    from ..data.svg_utils import parse_svg_file, calculate_bounding_box, normalize_strokes # Reuse parsing and normalization logic
except ImportError:
     class ConversionError(Exception): pass
     class DataError(Exception): pass
     def parse_svg_file(fp, bp): logger.error("SVG Utils not found for font conversion."); return []
     def calculate_bounding_box(s): return 0,0,1,1
     def normalize_strokes(s, **kwargs): return s


logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_UNITS_PER_EM = 1000 # Common value for font resolution

def _convert_svg_strokes_to_glyph(strokes: List[List[Tuple[float, float]]],
                                 glyph_name: str,
                                 units_per_em: int = DEFAULT_UNITS_PER_EM,
                                 target_height: int | None = None, # Optional: Normalize height
                                 ) -> Glyph | None:
    """
    Converts SVG stroke data (list of lists of points) into a fontTools Glyph object.

    This is a simplified conversion. Assumes input strokes are normalized or scaled appropriately.
    Handles lines and attempts cubic-to-quadratic conversion for curves if needed later.

    Args:
        strokes: Stroke data [(x, y), ...]. Coordinates should be in font units.
        glyph_name: Name for the glyph (e.g., 'A', 'uniXXXX').
        units_per_em: Font resolution.
        target_height: If provided, normalize the glyph height to this value within UPM.

    Returns:
        fontTools Glyph object or None on failure.
    """
    if not strokes or not any(strokes):
        logger.warning(f"No stroke data provided for glyph '{glyph_name}'. Skipping.")
        return None

    # --- Optional Normalization within Font Units ---
    # If SVGs weren't pre-normalized to UPM scale, do it here.
    # Example: Normalize height to target_height (e.g., 750 for typical cap height in 1000 UPM)
    if target_height is not None:
        min_x, min_y, max_x, max_y = calculate_bounding_box(strokes)
        height = max_y - min_y
        if height > 1e-6: # Avoid division by zero
             scale = target_height / height
             # Center vertically? Adjust based on desired baseline alignment.
             # Simple scale and shift y to be >= 0
             translate_y = -min_y * scale
             translate_x = -min_x * scale # Shift x >= 0 as well

             scaled_strokes = []
             for stroke in strokes:
                  scaled_strokes.append([((x * scale) + translate_x, (y * scale) + translate_y) for x, y in stroke])
             strokes = scaled_strokes
        else:
             # Handle zero height case (e.g., horizontal line) - center it?
             pass # Keep original strokes for now if height is zero

    # --- Create Glyph Pens ---
    # TTGlyphPen is for drawing into TrueType 'glyf' table format (quadratic Beziers)
    try:
        glyph_pen = TTGlyphPen(glyphSet=None) # glyphSet not needed for single glyph drawing

        # Wrap TTGlyphPen with our QuadraticConverterPen to automatically convert
        # cubic Bezier commands to quadratic Bezier curves.
        # Future code calling pen.curveTo will trigger the conversion.
        pen = QuadraticConverterPen(glyph_pen, maxErr=1.0)

    except Exception as e:
        logger.error(f"Failed to initialize fontTools pens for glyph '{glyph_name}': {e}", exc_info=True)
        return None

    # --- Draw Strokes onto Pen ---
    try:
        for stroke in strokes:
            if len(stroke) < 1: continue

            start_point = stroke[0]
            pen.moveTo(start_point) # Move to start of stroke

            if len(stroke) > 1:
                for point in stroke[1:]:
                    # Currently only handles LineTo as parse_svg_file outputs line segments
                    pen.lineTo(point)
                    # --- Future Enhancement for Curves ---
                    # If stroke data included curve type info:
                    # if point_type == 'cubic': pen.curveTo(cp1, cp2, end_point)
                    # elif point_type == 'quadratic': pen.qCurveTo(cp1, end_point)
                    # else: pen.lineTo(point)

            # Close path if stroke seems closed? SVG Z command handling needed in parser.
            # pen.closePath() # Not typically used for handwritten strokes unless explicitly closed loops

        # End drawing for this glyph
        pen.endPath() # Though TTGlyphPen might not strictly require it

        # Get the glyph object
        glyph = glyph_pen.glyph()
        glyph.recalcBounds(glyfTable=None) # Recalculate bounds after drawing

        return glyph

    except Exception as e:
        logger.error(f"Error drawing strokes for glyph '{glyph_name}': {e}", exc_info=True)
        return None


def convert_svgs_to_font(svg_files_map: Dict[str, str],
                         output_font_path: str,
                         font_name: str = "GeneratedHandwriting",
                         style_name: str = "Regular",
                         units_per_em: int = DEFAULT_UNITS_PER_EM,
                         ascent: int | None = None, # Optional override
                         descent: int | None = None, # Optional override (negative value)
                         default_advance_width: int | None = None):
    """
    Converts a collection of character SVG files into a TTF font file.

    Args:
        svg_files_map (Dict[str, str]): Dictionary mapping character strings (e.g., 'A', '猫')
                                        to their corresponding SVG file paths.
        output_font_path (str): Path to save the output font (.ttf recommended).
        font_name (str): The family name of the font.
        style_name (str): Style name (e.g., 'Regular', 'Bold').
        units_per_em (int): Font resolution (units per em square).
        ascent (int, optional): Font ascender value. Calculated if None.
        descent (int, optional): Font descender value (negative). Calculated if None.
        default_advance_width (int, optional): Horizontal advance width for all glyphs.
                                               Calculated per glyph if None.

    Raises:
        ConversionError: If font generation fails.
        DataError: If SVG parsing fails for any input file.
    """
    logger.info(f"Starting font generation for '{font_name} {style_name}' -> {output_font_path}")

    # --- FontBuilder Setup ---
    # FontBuilder helps set up the basic tables.
    # Requires glyph map (name -> glyph obj), cmap, metrics.
    glyph_map: Dict[str, Glyph] = {}
    unicode_map: Dict[int, str] = {} # Unicode codepoint -> glyph name
    glyph_metrics: Dict[str, Tuple[int, int]] = {} # name -> (advanceWidth, leftSideBearing)

    # Add the mandatory .notdef glyph
    glyph_map['.notdef'] = TTGlyphPen(None).glyph() # Empty glyph

    all_min_y = float('inf')
    all_max_y = float('-inf')
    all_max_x = float('-inf')

    # --- Process Each SVG Glyph ---
    for char, svg_path in svg_files_map.items():
        if len(char) != 1:
            logger.warning(f"Character key '{char}' is not a single character. Skipping for font cmap.")
            continue

        glyph_name = f"uni{ord(char):04X}" # Use standard Unicode naming convention
        logger.debug(f"Processing char '{char}' (Glyph: {glyph_name}) from {svg_path}")

        # 1. Parse SVG
        try:
            # parse_svg_file returns list of strokes, points are (x, y)
            # Assuming parse_svg_file handles bezier sampling into lines for now
            strokes = parse_svg_file(svg_path, bezier_points=10) # Bezier points irrelevant if sampled
            if not strokes or not any(strokes):
                 logger.warning(f"No valid strokes found in SVG for char '{char}'. Skipping glyph.")
                 continue

            # --- Coordinate Transformation/Normalization (Crucial!) ---
            # Coordinates from SVG need to be mapped to the font's UPM grid.
            # Option 1: Assume SVGs are already drawn on a grid matching UPM (e.g., 0-1000).
            # Option 2: Normalize SVG content to fit within UPM, respecting aspect ratio & baseline.
            # Using normalize_strokes from svg_utils - Requires careful parameterization.
            # Let's normalize height to ~75% of UPM, assuming baseline is y=0.
            # This needs refinement based on actual SVG content scale and desired font metrics.
            target_glyph_height = int(units_per_em * 0.75)
            # Normalize strokes based on their *own* bounding box to fit target height
            # This aligns tops but baseline might vary. Better: normalize globally later?
            # Let's try normalizing within the glyph conversion:
            glyph = _convert_svg_strokes_to_glyph(strokes, glyph_name, units_per_em, target_height=target_glyph_height)

            if glyph is None:
                 logger.warning(f"Failed to convert strokes to glyph object for '{char}'. Skipping.")
                 continue

        except DataError as e:
            logger.error(f"Failed to parse SVG file {svg_path} for char '{char}': {e}")
            # Decide whether to skip or raise error for the whole process
            continue # Skip this character
        except Exception as e:
            logger.error(f"Unexpected error processing SVG {svg_path} for char '{char}': {e}", exc_info=True)
            continue # Skip this character

        # Store glyph and map Unicode
        glyph_map[glyph_name] = glyph
        unicode_map[ord(char)] = glyph_name

        # --- Calculate Metrics (Basic) ---
        # Use glyph bounding box for basic metrics. Needs refinement for typography.
        # Ensure bounds are calculated if not done already
        if not hasattr(glyph, 'xMin'): glyph.recalcBounds(glyfTable=None)

        xmin, ymin, xmax, ymax = glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax

        # Update global font bounds
        all_min_y = min(all_min_y, ymin)
        all_max_y = max(all_max_y, ymax)
        all_max_x = max(all_max_x, xmax) # Used for potential advance width default

        # Horizontal Metrics: (advanceWidth, leftSideBearing)
        lsb = xmin # Left side bearing is typically the xmin of the bounding box
        # Advance width: typically width of bbox (xmax-xmin) + some side bearing padding.
        # Or use a fixed width for monospace, or calculate based on xmax.
        adv_width = default_advance_width if default_advance_width is not None else int(xmax + lsb * 0.5) # Heuristic: add half lsb as right bearing
        adv_width = max(1, adv_width) # Ensure not zero

        glyph_metrics[glyph_name] = (adv_width, lsb)


    # Ensure .notdef has metrics
    if '.notdef' not in glyph_metrics:
         glyph_metrics['.notdef'] = (int(units_per_em * 0.6), 0) # Default width for missing glyph

    if len(glyph_map) <= 1: # Only .notdef
         raise ConversionError("No valid glyphs were generated from the input SVGs.")

    # --- Determine Global Font Metrics ---
    font_ascent = ascent if ascent is not None else int(all_max_y)
    font_descent = descent if descent is not None else int(all_min_y) # Should be negative
    if font_descent > 0: font_descent = -int(units_per_em * 0.2) # Ensure descent is negative, fallback
    logger.info(f"Calculated Font Metrics: Ascent={font_ascent}, Descent={font_descent}")

    # --- Build Font using FontBuilder ---
    try:
        fb = FontBuilder(unitsPerEm=units_per_em, isTTF=True) # Build TTF structure
        fb.setupGlyphOrder(list(glyph_map.keys())) # Define glyph order
        fb.setupCharacterMap(unicode_map) # Setup cmap
        fb.setupGlyf(glyph_map) # Add glyph data
        fb.setupHorizontalMetrics(glyph_metrics) # Add hmtx
        fb.setupHorizontalHeader(ascent=font_ascent, descent=font_descent) # Setup hhea
        fb.setupNameTable({ # Setup name table (required entries)
             'familyName': font_name,
             'styleName': style_name,
             'uniqueFontIdentifier': f"FontTools:{font_name}-{style_name}",
             'fullName': f"{font_name} {style_name}",
             'version': "Version 1.000",
             'psName': f"{font_name.replace(' ','')}-{style_name}" # PostScript name, no spaces
         })
        fb.setupOS2( # Setup OS/2 table with basic metrics
            usWeightClass=400, # Normal weight
            usWidthClass=5, # Medium width
            sTypoAscender=font_ascent,
            sTypoDescender=font_descent,
            sTypoLineGap=max(0, int(units_per_em * 0.1)), # Suggest line gap
            usWinAscent=abs(font_ascent),
            usWinDescent=abs(font_descent),
            # Need to calculate avgCharWidth, xAvgCharWidth if possible
        )
        fb.setupPost() # Setup post table
        # setupMaxp() is often called internally by setupGlyf/others or at build time

        # Save the font
        logger.info(f"Saving font to {output_font_path}...")
        fb.save(output_font_path)
        logger.info(f"Font successfully generated and saved.")

    except ImportError as e:
         logger.error(f"Missing required library fontTools or cu2qu: {e}. Please install.")
         raise ConversionError(f"Missing dependency: {e}. Ensure fontTools and cu2qu are installed.") from e
    except Exception as e:
        logger.error(f"Failed to build or save the font: {e}", exc_info=True)
        raise ConversionError(f"Font generation failed: {e}") from e

# --- Example Usage ---
# if __name__ == "__main__":
#     # Create dummy SVG files for 'A' and 'B'
#     dummy_svg_dir = "dummy_char_svgs"
#     os.makedirs(dummy_svg_dir, exist_ok=True)
#     svg_map = {}

#     svg_a = '<svg viewBox="0 0 100 100"><path d="M 10 90 L 50 10 L 90 90 M 30 60 H 70"/></svg>'
#     svg_b = '<svg viewBox="0 0 100 100"><path d="M 10 10 V 90 H 50 C 80 90 80 60 50 60 H 10 M 10 50 H 50 C 80 50 80 20 50 10 H 10"/></svg>'

#     path_a = os.path.join(dummy_svg_dir, "A.svg")
#     path_b = os.path.join(dummy_svg_dir, "B.svg")
#     with open(path_a, "w") as f: f.write(svg_a)
#     with open(path_b, "w") as f: f.write(svg_b)
#     svg_map['A'] = path_a
#     svg_map['B'] = path_b

#     # Set up logging
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
#     output_font = "MyHandwritingFont.ttf"

#     print("\n--- Testing SVG to Font Conversion ---")
#     try:
#         convert_svgs_to_font(svg_map, output_font, font_name="MyHandwriting", style_name="Test")
#         print(f"Font conversion attempted. Check for '{output_font}'.")
#     except Exception as e:
#         print(f"  Font Conversion failed: {e}")

#     # Clean up
#     # import shutil
#     # shutil.rmtree(dummy_svg_dir)
#     # if os.path.exists(output_font): os.remove(output_font)
#     # print("\nCleaned up test files.")