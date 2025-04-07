# src/data/moj_parser.py
"""
Parser for the custom .moj handwriting data format.

Assumes .moj files are JSON containing character metadata and stroke data.
The stroke data can be either pre-extracted points with pressure/time,
or raw SVG path strings.
"""

import json
import os
import logging

# Import utilities and custom exceptions
try:
    from .svg_utils import _parse_path_d # Use internal function for path parsing if needed
    from ..utils.error_handling import DataError
except ImportError:
    # Define minimal fallbacks if modules aren't available yet
     class DataError(Exception):
         pass
     def _parse_path_d(d_string, bezier_points=10):
         logging.error("SVG parsing utility not available for moj_parser.")
         return [] # Return empty

logger = logging.getLogger(__name__)

# Define expected structure keys (adjust if format differs)
KEY_CHAR = "character"
KEY_LABEL = "label" # e.g., Unicode
KEY_STROKES_POINTS = "strokes" # List of lists of points like [{"x":_, "y":_, "p":_, "t":_}, ...]
KEY_STROKES_SVG = "svg_paths" # List of SVG d-strings ["M...L...", "M...C..."]

def parse_moj_file(file_path: str, bezier_points: int = 10) -> tuple[str | None, list[list[tuple]] | None]:
    """
    Parses a .moj file (assumed JSON) to extract character label and stroke data.

    Handles two potential stroke data formats within the JSON:
    1.  `KEY_STROKES_POINTS`: Pre-extracted points [(x, y, p, t), ...].
    2.  `KEY_STROKES_SVG`: Raw SVG path d-strings ["M...", "C..."].

    Args:
        file_path (str): Path to the .moj file.
        bezier_points (int): Number of points to sample for Bezier curves if parsing svg_paths.

    Returns:
        tuple[str | None, list[list[tuple]] | None]:
            - Character label (e.g., Unicode string or character itself). None on failure.
            - List of strokes, where each stroke is a list of points.
              Points are tuples, format depends on source: (x, y, p, t) if from points,
              (x, y) if from SVG paths. Returns None on failure.

    Raises:
        DataError: If the file is not valid JSON or lacks required structure.
    """
    label = None
    strokes = None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract label (prioritize 'label', fallback to 'character')
        if KEY_LABEL in data:
            label = data[KEY_LABEL]
        elif KEY_CHAR in data:
            label = data[KEY_CHAR]
        else:
            # Fallback: use filename without extension if no label found
            label = os.path.splitext(os.path.basename(file_path))[0]
            logger.warning(f"No '{KEY_LABEL}' or '{KEY_CHAR}' key found in {file_path}. Using filename '{label}' as label.")

        # Extract strokes based on available key
        if KEY_STROKES_POINTS in data:
            logger.debug(f"Parsing stroke points from '{KEY_STROKES_POINTS}' key in {file_path}")
            raw_strokes_data = data[KEY_STROKES_POINTS]
            if not isinstance(raw_strokes_data, list):
                raise DataError(f"'{KEY_STROKES_POINTS}' value in {file_path} is not a list.")

            strokes = []
            for stroke_data in raw_strokes_data:
                if not isinstance(stroke_data, list):
                     logger.warning(f"Found non-list stroke element in '{KEY_STROKES_POINTS}' of {file_path}. Skipping.")
                     continue
                stroke_points = []
                for point_data in stroke_data:
                    if isinstance(point_data, dict) and 'x' in point_data and 'y' in point_data:
                        # Extract x, y, pressure (p), time (t) if available
                        x = point_data.get('x')
                        y = point_data.get('y')
                        p = point_data.get('p', 0.5) # Default pressure if missing
                        t = point_data.get('t', 0.0) # Default time if missing
                        stroke_points.append((float(x), float(y), float(p), float(t)))
                    else:
                        logger.warning(f"Invalid point format in stroke data of {file_path}. Skipping point: {point_data}")
                if stroke_points: # Only add stroke if it has valid points
                     strokes.append(stroke_points)

        elif KEY_STROKES_SVG in data:
            logger.debug(f"Parsing stroke paths from '{KEY_STROKES_SVG}' key in {file_path}")
            svg_paths = data[KEY_STROKES_SVG]
            if not isinstance(svg_paths, list):
                 raise DataError(f"'{KEY_STROKES_SVG}' value in {file_path} is not a list.")

            strokes = []
            for d_string in svg_paths:
                 if isinstance(d_string, str):
                     # Use the SVG path parsing utility
                     # Note: _parse_path_d returns list of strokes; a single d_string can have multiple 'M' commands.
                     parsed_sub_strokes = _parse_path_d(d_string, bezier_points)
                     # The result points are only (x, y) as SVG paths lack pressure/time
                     strokes.extend(parsed_sub_strokes)
                 else:
                      logger.warning(f"Found non-string element in '{KEY_STROKES_SVG}' of {file_path}. Skipping path.")

        else:
            raise DataError(f"No stroke data found in {file_path}. Expected key '{KEY_STROKES_POINTS}' or '{KEY_STROKES_SVG}'.")

        if not strokes:
             logger.warning(f"Parsed stroke data resulted in an empty list for {file_path}.")
             # Return None if no valid strokes were parsed
             return label, None

        return label, strokes

    except FileNotFoundError:
        logger.error(f".moj file not found: {file_path}")
        return None, None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in .moj file: {file_path} - {e}")
        raise DataError(f"Invalid JSON in {file_path}") from e
    except DataError as e: # Re-raise DataErrors from validation
        logger.error(f"Data structure error in {file_path}: {e}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred parsing .moj file {file_path}: {e}", exc_info=True)
        raise DataError(f"Failed to process .moj file {file_path}: {e}") from e

# --- Example Usage ---
# if __name__ == "__main__":
#     # Create dummy .moj files for testing

#     # Format 1: Points with p/t
#     moj_content_points = """
#     {
#       "character": "A",
#       "label": "U+0041",
#       "strokes": [
#         [ {"x": 10, "y": 90, "p": 0.2, "t": 0.1}, {"x": 50, "y": 10, "p": 0.8, "t": 0.3}, {"x": 90, "y": 90, "p": 0.3, "t": 0.5} ],
#         [ {"x": 30, "y": 50, "p": 0.5, "t": 0.6}, {"x": 70, "y": 50, "p": 0.6, "t": 0.7} ]
#       ]
#     }
#     """
#     # Format 2: SVG Paths
#     moj_content_svg = """
#     {
#       "character": "B",
#       "label": "U+0042",
#       "svg_paths": [
#         "M 10 10 V 90 H 50 Q 80 90 80 70 Q 80 50 50 50",
#         "M 10 50 H 50"
#       ]
#     }
#     """
#     # Format 3: Invalid
#     moj_content_invalid = """
#     {
#       "character": "C",
#       "label": "U+0043"
#       // Missing strokes key
#     }
#     """

#     dummy_files = {"test_points.moj": moj_content_points, "test_svg.moj": moj_content_svg, "test_invalid.moj": moj_content_invalid}
#     for filename, content in dummy_files.items():
#         with open(filename, "w") as f:
#             f.write(content)

#     # Set up basic logging for testing this module
#     logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

#     print("--- Testing MOJ Parser ---")
#     for filename in dummy_files:
#         print(f"\nParsing {filename}:")
#         try:
#             label, strokes = parse_moj_file(filename)
#             if label is not None and strokes is not None:
#                 print(f"  Label: {label}")
#                 print(f"  Found {len(strokes)} strokes.")
#                 if strokes:
#                      print(f"  First point of first stroke: {strokes[0][0]}")
#                      print(f"  Point format length: {len(strokes[0][0])}") # Should be 4 for points, 2 for svg
#             elif label is not None and strokes is None:
#                 print(f"  Label: {label}, but no valid strokes parsed.")
#             else:
#                 print("  Failed to parse.")
#         except DataError as e:
#             print(f"  Caught expected DataError: {e}")
#         except Exception as e:
#             print(f"  Caught unexpected error: {e}")

#     # Clean up dummy files
#     for filename in dummy_files:
#         os.remove(filename)