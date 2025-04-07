# src/data/svg_utils.py
"""
Utilities for parsing and processing SVG handwriting data.

Handles extraction of stroke coordinates from SVG <path> elements,
simplification, normalization, and conversion to sequential tensor format.
"""

import xml.etree.ElementTree as ET
import re
import numpy as np
import logging
import torch # For tensor conversion

# Import custom exception
try:
    from ..utils.error_handling import DataError
except ImportError:
    # Define a minimal fallback if error_handling isn't available yet
     class DataError(Exception):
        pass

logger = logging.getLogger(__name__)

# --- Constants ---
# Number of points to sample for Bézier curves. Adjust for desired precision/complexity.
# This might need to be adaptive based on curve length in a more advanced version.
DEFAULT_BEZIER_POINTS = 10
# Common SVG namespaces
SVG_NS = {'svg': 'http://www.w3.org/2000/svg'}

# --- Path 'd' attribute parsing ---
# Regex to find path commands and their coordinates
PATH_COMMAND_RE = re.compile(r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)')
# Regex to extract numbers (floats/ints) from coordinate strings
NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+|[-+]?\d+')

def _parse_path_d(d_string: str, bezier_points: int = DEFAULT_BEZIER_POINTS) -> list[list[tuple[float, float]]]:
    """
    Parses an SVG path 'd' attribute string into a list of strokes.

    Each stroke is a list of (x, y) coordinate tuples.
    Handles M/m, L/l, H/h, V/v, C/c, Q/q, Z/z commands.
    Approximates Bezier curves (C/c, Q/q) by sampling points.
    Ignores S/s, T/t, A/a for simplicity in handwriting context (can be added if needed).

    Args:
        d_string (str): The 'd' attribute content.
        bezier_points (int): Number of line segments to approximate a Bezier curve.

    Returns:
        list[list[tuple[float, float]]]: A list of strokes, where each stroke is a list of (x, y) points.
                                          Returns an empty list if parsing fails.
    """
    strokes = []
    current_stroke = []
    last_point = np.array([0.0, 0.0])
    start_point = np.array([0.0, 0.0]) # For 'Z' command
    last_control_point = None # For 'S', 'T' commands (if implemented)

    try:
        for command, coord_str in PATH_COMMAND_RE.findall(d_string):
            coords = [float(n) for n in NUMBER_RE.findall(coord_str)]
            coord_pairs = list(zip(coords[::2], coords[1::2])) # Group into (x, y)

            is_relative = command.islower()
            command_upper = command.upper()

            # Process segments based on command
            segment_points = []
            idx = 0
            while idx < len(coords):
                current_pos = np.copy(last_point)
                offset = current_pos if is_relative else np.array([0.0, 0.0])

                if command_upper == 'M': # MoveTo
                    target = np.array(coords[idx:idx+2]) + offset
                    if current_stroke: # If already drawing, start a new stroke
                        strokes.append(current_stroke)
                    current_stroke = [tuple(target)] # Start new stroke with this point
                    start_point = np.copy(target)
                    last_point = target
                    idx += 2
                    # Subsequent points in 'M' are treated as 'L'
                    command_upper = 'L'
                    is_relative = command.islower() # Keep relative flag for subsequent implicit L
                    offset = current_pos if is_relative else np.array([0.0, 0.0])

                elif command_upper == 'L': # LineTo
                    target = np.array(coords[idx:idx+2]) + offset
                    segment_points.append(tuple(target))
                    last_point = target
                    idx += 2

                elif command_upper == 'H': # Horizontal LineTo
                    target_x = coords[idx] + offset[0]
                    target = np.array([target_x, current_pos[1]])
                    segment_points.append(tuple(target))
                    last_point = target
                    idx += 1

                elif command_upper == 'V': # Vertical LineTo
                    target_y = coords[idx] + offset[1]
                    target = np.array([current_pos[0], target_y])
                    segment_points.append(tuple(target))
                    last_point = target
                    idx += 1

                elif command_upper == 'C': # Cubic Bezier CurveTo
                    if len(coords) - idx < 6: break # Need 3 points
                    cp1 = np.array(coords[idx:idx+2]) + offset
                    cp2 = np.array(coords[idx+2:idx+4]) + offset
                    target = np.array(coords[idx+4:idx+6]) + offset

                    # Sample points along the curve (excluding start point)
                    for t in np.linspace(0, 1, bezier_points + 1)[1:]:
                        pt = (1-t)**3 * current_pos + 3*(1-t)**2*t * cp1 + 3*(1-t)*t**2 * cp2 + t**3 * target
                        segment_points.append(tuple(pt))

                    last_point = target
                    last_control_point = cp2 # Store for potential 'S' command
                    idx += 6

                elif command_upper == 'Q': # Quadratic Bezier CurveTo
                    if len(coords) - idx < 4: break # Need 2 points
                    cp = np.array(coords[idx:idx+2]) + offset
                    target = np.array(coords[idx+2:idx+4]) + offset

                    # Sample points along the curve (excluding start point)
                    for t in np.linspace(0, 1, bezier_points + 1)[1:]:
                         pt = (1-t)**2 * current_pos + 2*(1-t)*t * cp + t**2 * target
                         segment_points.append(tuple(pt))

                    last_point = target
                    last_control_point = cp # Store for potential 'T' command
                    idx += 4

                # --- S, T, A commands omitted for simplicity ---

                elif command_upper == 'Z': # ClosePath
                    if np.any(last_point != start_point): # Only add line if not already closed
                         segment_points.append(tuple(start_point))
                    last_point = start_point
                    # Technically, Z closes the current subpath. For simplicity here,
                    # we treat it as ending the current stroke. A new M is expected after Z.
                    idx = len(coords) # Consume remaining coordinates if any (shouldn't be any)

                else:
                    logger.warning(f"Unsupported SVG path command '{command}' in d='{d_string}'. Skipping.")
                    # Advance index past potentially problematic coordinates. This is heuristic.
                    command_len_guess = {'S': 4, 'T': 2, 'A': 7}.get(command_upper, 2) # Typical coord counts
                    idx += command_len_guess * (len(coords) // command_len_guess) if len(coords) > 0 else 0


            if segment_points:
                 if not current_stroke: # Handle paths starting without M (implicitly M 0 0)
                     current_stroke = [tuple(last_point)] # Start at current origin (likely 0,0)
                 current_stroke.extend(segment_points)

        # Add the last stroke if it exists
        if current_stroke:
            strokes.append(current_stroke)

    except Exception as e:
        logger.error(f"Failed to parse SVG path d='{d_string}': {e}", exc_info=True)
        return [] # Return empty list on failure

    # Filter out empty strokes that might occur due to parsing issues or empty paths
    strokes = [s for s in strokes if len(s) > 1] # Require at least 2 points for a stroke segment
    return strokes


def parse_svg_file(file_path: str, bezier_points: int = DEFAULT_BEZIER_POINTS) -> list[list[tuple[float, float]]]:
    """
    Parses an SVG file to extract stroke data.

    Looks for <path> elements and extracts their geometry.
    Assumes stroke order corresponds to element order in the SVG.

    Args:
        file_path (str): Path to the SVG file.
        bezier_points (int): Number of points to sample for Bezier curves within paths.

    Returns:
        list[list[tuple[float, float]]]: A list of strokes extracted from the SVG.
                                          Each stroke is a list of (x, y) points.
                                          Returns empty list if file not found or no paths found.
    Raises:
        DataError: If the file cannot be parsed as XML.
    """
    all_strokes = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Find all path elements within the SVG
        # This might need refinement based on specific SVG structures (e.g., paths inside <g> tags)
        path_elements = root.findall('.//svg:path', SVG_NS)
        if not path_elements:
            # Fallback for SVGs without explicit namespace
            path_elements = root.findall('.//path')

        if not path_elements:
             logger.warning(f"No <path> elements found in SVG file: {file_path}")
             # Could add support for <polyline>, <line> etc. here if needed
             return []

        for path in path_elements:
            d_string = path.get('d')
            if d_string:
                strokes_from_path = _parse_path_d(d_string, bezier_points)
                # A single <path> element can contain multiple strokes (separated by M commands)
                all_strokes.extend(strokes_from_path)
            else:
                 logger.warning(f"Found <path> element without 'd' attribute in {file_path}. Skipping.")

    except ET.ParseError as e:
        logger.error(f"Failed to parse XML in SVG file: {file_path} - {e}")
        raise DataError(f"Invalid XML structure in SVG file: {file_path}") from e
    except FileNotFoundError:
        logger.error(f"SVG file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred parsing SVG file {file_path}: {e}", exc_info=True)
        raise DataError(f"Failed to process SVG file {file_path}: {e}") from e

    return all_strokes


# --- Stroke Processing ---

def simplify_strokes(strokes: list[list[tuple[float, float]]], tolerance: float = 1.0) -> list[list[tuple[float, float]]]:
    """
    Simplifies strokes using the Ramer-Douglas-Peucker algorithm. (Optional)
    Reduces the number of points in a stroke while preserving the shape.

    Args:
        strokes (list[list[tuple[float, float]]]): List of strokes (lists of points).
        tolerance (float): The maximum distance between the original curve and its
                           simplification. Higher values mean more simplification.

    Returns:
        list[list[tuple[float, float]]]: Simplified list of strokes.
    """
    # Implementation requires RDP algorithm. Can use a library like `rdp` or implement manually.
    # Example using the `rdp` library (requires `pip install rdp`)
    try:
        from rdp import rdp
        simplified_strokes = []
        for stroke in strokes:
            if len(stroke) > 2:
                simplified_stroke = rdp(np.array(stroke), epsilon=tolerance)
                # Ensure the simplified stroke still has at least start and end points
                if len(simplified_stroke) >= 2:
                     simplified_strokes.append([tuple(p) for p in simplified_stroke])
                elif stroke: # If simplification reduced to <2 points, keep original start/end
                     simplified_strokes.append([stroke[0], stroke[-1]])
            elif stroke: # Keep strokes with 1 or 2 points as is
                simplified_strokes.append(stroke)
        return simplified_strokes
    except ImportError:
        logger.warning("The 'rdp' library is not installed. Skipping stroke simplification.")
        return strokes
    except Exception as e:
        logger.error(f"Error during stroke simplification: {e}. Returning original strokes.", exc_info=True)
        return strokes


def calculate_bounding_box(strokes: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    """Calculates the min/max x/y coordinates covering all strokes."""
    if not strokes or not any(strokes):
        return 0, 0, 0, 0
    all_points = np.array([p for stroke in strokes for p in stroke])
    if all_points.size == 0:
        return 0, 0, 0, 0
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    return min_x, min_y, max_x, max_y


def normalize_strokes(strokes: list[list[tuple[float, float]]], target_size: int = 256, keep_aspect_ratio: bool = True, margin: float = 0.05) -> list[list[tuple[float, float]]]:
    """
    Normalizes strokes to fit within a target bounding box (e.g., target_size x target_size).

    Args:
        strokes (list[list[tuple[float, float]]]): List of strokes (lists of points).
        target_size (int): The size of the target square bounding box.
        keep_aspect_ratio (bool): Whether to maintain the original aspect ratio.
        margin (float): Percentage of target_size to leave as margin around the strokes.

    Returns:
        list[list[tuple[float, float]]]: Normalized list of strokes. Returns original if strokes are empty.
    """
    if not strokes or not any(strokes):
        return strokes

    min_x, min_y, max_x, max_y = calculate_bounding_box(strokes)
    width = max_x - min_x
    height = max_y - min_y

    # Handle cases where width or height is zero (single point or straight line)
    if width == 0 and height == 0:
        # Center the single point
        center_x = target_size / 2
        center_y = target_size / 2
        return [[(center_x, center_y)] * len(stroke) for stroke in strokes] # Map all points to center
    elif width == 0:
        width = height # Treat as square for scaling if only vertical
    elif height == 0:
        height = width # Treat as square for scaling if only horizontal


    # Calculate scale factor
    actual_target_size = target_size * (1.0 - 2 * margin)
    scale = actual_target_size / max(width, height)
    if not keep_aspect_ratio:
        scale_x = actual_target_size / width
        scale_y = actual_target_size / height
    else:
        scale_x = scale_y = scale

    # Calculate translation
    offset_x = target_size * margin
    offset_y = target_size * margin
    # Adjust offset to center the content if aspect ratio is kept
    if keep_aspect_ratio:
        if width > height:
            offset_y += (actual_target_size - height * scale) / 2.0
        else:
            offset_x += (actual_target_size - width * scale) / 2.0

    translate_x = offset_x - min_x * scale_x
    translate_y = offset_y - min_y * scale_y

    # Apply transformation
    normalized_strokes = []
    for stroke in strokes:
        normalized_stroke = []
        for x, y in stroke:
            new_x = x * scale_x + translate_x
            new_y = y * scale_y + translate_y
            normalized_stroke.append((new_x, new_y))
        normalized_strokes.append(normalized_stroke)

    return normalized_strokes

# --- Conversion to Sequence Format ---

def strokes_to_sequence_tensor(strokes: list[list[tuple[float, float]]],
                              max_seq_len: int,
                              normalize: bool = True,
                              normalization_size: int = 256,
                              include_pressure_time: bool = False) -> torch.Tensor | None:
    """
    Converts a list of strokes into a fixed-length sequence tensor for model input.

    Each step in the sequence represents a point and its state.
    Format: [delta_x, delta_y, pen_down_state, end_of_stroke_state, end_of_sequence_state]
    If include_pressure_time is True (and data is available, e.g. from .moj):
    Format: [delta_x, delta_y, pressure, time_delta, pen_down_state, end_of_stroke_state, end_of_sequence_state]

    Args:
        strokes (list[list[tuple[float, float]]]): List of strokes (lists of points).
            Points might be (x, y) or (x, y, p, t) if pressure/time available.
        max_seq_len (int): The target length for the output sequence (padding/truncation).
        normalize (bool): Whether to normalize strokes before processing.
        normalization_size (int): Target size if normalizing.
        include_pressure_time (bool): Flag to include pressure/time dimensions. Assumes
                                     points are tuples of len >= 4 if True.

    Returns:
        torch.Tensor | None: A tensor of shape (max_seq_len, feature_dim) or None if input is empty/invalid.
                            feature_dim is 5 or 7 depending on include_pressure_time.
    """
    if not strokes or not any(strokes):
        logger.warning("Attempted to convert empty strokes to sequence tensor.")
        return None

    # Optional normalization
    if normalize:
        proc_strokes = normalize_strokes(strokes, target_size=normalization_size)
    else:
        proc_strokes = strokes

    # Determine feature dimension
    feature_dim = 7 if include_pressure_time else 5
    sequence = []

    last_x, last_y = 0.0, 0.0
    last_t = 0.0 # Track time for delta_t calculation

    # Start with an initial "pen up" state? Some models expect this. Let's omit for now.
    # sequence.append([0.0] * feature_dim) # Example: Start at 0,0 pen up.

    total_points = sum(len(s) for s in proc_strokes)
    if total_points == 0: return None

    point_count = 0
    for stroke_idx, stroke in enumerate(proc_strokes):
        if not stroke: continue # Skip empty strokes

        for point_idx, point in enumerate(stroke):
            # Extract data from point tuple
            current_x, current_y = point[0], point[1]
            current_p = point[2] if include_pressure_time and len(point) > 2 else 0.5 # Default pressure 0.5
            current_t = point[3] if include_pressure_time and len(point) > 3 else 0.0 # Default time 0.0

            delta_x = current_x - last_x
            delta_y = current_y - last_y
            delta_t = current_t - last_t

            pen_down = 1.0 # Pen is down during a stroke
            end_of_stroke = 0.0
            end_of_sequence = 0.0

            # Build the feature vector for this point
            if include_pressure_time:
                features = [delta_x, delta_y, current_p, delta_t, pen_down, end_of_stroke, end_of_sequence]
            else:
                features = [delta_x, delta_y, pen_down, end_of_stroke, end_of_sequence]

            sequence.append(features)
            point_count += 1

            last_x, last_y = current_x, current_y
            last_t = current_t

            # Check if sequence length limit is reached
            if point_count >= max_seq_len - 1: # Need space for potential final end_of_sequence marker
                 logger.warning(f"Stroke sequence truncated at point {point_count} due to max_seq_len {max_seq_len}.")
                 break # Stop processing this stroke

        if point_count >= max_seq_len - 1: break # Stop processing further strokes

        # After each stroke (except the last one), add a "pen up" transition state?
        # This depends heavily on model architecture. A common way:
        # The *last* point of the stroke implicitly signals the end.
        # Let's modify the *last* point of the stroke instead of adding a new one.
        if sequence and stroke_idx < len(proc_strokes) - 1:
             sequence[-1][feature_dim - 2] = 1.0 # Mark end_of_stroke=1 on the last point

    # Add final "end of sequence" marker
    if sequence:
        sequence[-1][feature_dim - 1] = 1.0 # Mark end_of_sequence=1 on the very last point recorded

        # Padding
        pad_len = max_seq_len - len(sequence)
        if pad_len > 0:
            # Pad with zeros, but the last padding vector could also mark end_of_sequence
            # Or simply use an attention mask later. Let's pad with zeros.
            padding = [[0.0] * feature_dim for _ in range(pad_len)]
            sequence.extend(padding)
        elif pad_len < 0: # Truncation (already handled partially above)
             sequence = sequence[:max_seq_len]
             # Ensure the very last element correctly marks end_of_sequence if truncated
             if sequence:
                 # Reset potentially incorrect flags from truncated points
                 sequence[-1][feature_dim - 3] = 0.0 # pen_down = 0 ? Maybe keep pen state?
                 sequence[-1][feature_dim - 2] = 0.0 # end_of_stroke = 0
                 sequence[-1][feature_dim - 1] = 1.0 # end_of_sequence = 1
    else:
        # Handle case where sequence is still empty (e.g., only empty strokes input)
        return torch.zeros((max_seq_len, feature_dim))


    return torch.tensor(sequence, dtype=torch.float32)


# --- Example Usage ---
# if __name__ == "__main__":
#     # Requires a sample SVG file named 'test.svg' in the same directory
#     test_svg_path = "test.svg"
#     if os.path.exists(test_svg_path):
#         print(f"--- Testing SVG Parsing ({test_svg_path}) ---")
#         raw_strokes = parse_svg_file(test_svg_path)
#         print(f"Found {len(raw_strokes)} strokes.")
#         # for i, s in enumerate(raw_strokes): print(f"  Stroke {i+1}: {len(s)} points")

#         print("\n--- Testing Normalization ---")
#         normalized_strokes = normalize_strokes(raw_strokes, target_size=100, margin=0)
#         if normalized_strokes:
#             min_x, min_y, max_x, max_y = calculate_bounding_box(normalized_strokes)
#             print(f"Normalized bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}] (approx 0-100)")

#         print("\n--- Testing Simplification (requires rdp) ---")
#         simplified_strokes = simplify_strokes(raw_strokes, tolerance=2.0)
#         # print(f"Simplified to {len(simplified_strokes)} strokes.")
#         # for i, s in enumerate(simplified_strokes): print(f"  Stroke {i+1}: {len(s)} points")


#         print("\n--- Testing Sequence Tensor Conversion ---")
#         sequence_tensor = strokes_to_sequence_tensor(raw_strokes, max_seq_len=500, normalize=True, normalization_size=256)

#         if sequence_tensor is not None:
#             print(f"Generated sequence tensor shape: {sequence_tensor.shape}")
#             # print("Sample data (first 5 steps):")
#             # print(sequence_tensor[:5])
#             # print("Sample data (last 5 steps):")
#             # print(sequence_tensor[-5:])
#         else:
#             print("Failed to generate sequence tensor.")

#     else:
#         print(f"Test SVG file '{test_svg_path}' not found. Skipping tests.")