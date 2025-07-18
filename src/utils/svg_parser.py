"""
SVG Parser Utility

This script handles the parsing of SVG files to extract stroke data.
It reads SVG <path> elements, samples points along each path, and normalizes
the coordinates into a sequence of (dx, dy, pressure) points suitable for
a deep learning model.
"""

import logging
from lxml import etree
import numpy as np
from svgpathtools import parse_path

from src.config import MAX_POINTS_PER_STROKE, EPSILON

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_svg_file(file_path: str):
    """
    Parses an SVG file and extracts a sequence of points from its paths.

    Each point in the sequence is represented as a tuple:
    (delta_x, delta_y, pressure, pen_down, end_of_stroke)
    - (delta_x, delta_y): The offset from the previous point.
    - pressure: Stroke pressure (default to 1.0 if not available).
    - pen_down: A binary flag (1 for drawing, 0 for pen up between strokes).
    - end_of_stroke: A binary flag indicating the last point of a stroke.

    Args:
        file_path (str): The path to the SVG file.

    Returns:
        np.ndarray: A numpy array of shape (N, 5) representing the stroke sequence,
                    or None if parsing fails.
    """
    try:
        tree = etree.parse(file_path)
        root = tree.getroot()
        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        paths = root.xpath('//svg:path', namespaces=namespace)

        if not paths:
            logging.warning(f"No paths found in SVG file: {file_path}")
            return None

        all_strokes = []
        last_point = np.array([0.0, 0.0])

        for path_element in paths:
            path_data = path_element.get('d')
            if not path_data:
                continue

            path = parse_path(path_data)
            num_samples = min(MAX_POINTS_PER_STROKE, int(path.length()))
            if num_samples < 2:
                continue

            points = []
            for i in range(num_samples):
                # Sample points uniformly along the path length
                p = path.point(i / (num_samples - 1))
                points.append([p.real, p.imag])
            
            points = np.array(points)

            # Convert to delta coordinates (dx, dy)
            deltas = np.diff(points, axis=0)
            deltas = np.insert(deltas, 0, points[0] - last_point, axis=0)

            # Create the 5-dimensional representation
            # (dx, dy, pressure, pen_down, end_of_stroke)
            stroke_data = np.zeros((len(deltas), 5))
            stroke_data[:, :2] = deltas
            stroke_data[:, 2] = 1.0  # Default pressure
            stroke_data[:, 3] = 1.0  # Pen is down for the whole stroke
            stroke_data[0, 3] = 0.0 # Pen lifts to move to the start of this new stroke
            stroke_data[-1, 4] = 1.0 # Mark the end of this stroke

            all_strokes.append(stroke_data)
            last_point = points[-1]

        if not all_strokes:
            return None
        
        # The first stroke's "pen up" flag should be 1 (start of drawing)
        all_strokes[0][0, 3] = 1.0

        return np.vstack(all_strokes)

    except Exception as e:
        logging.error(f"Failed to parse SVG file {file_path}: {e}")
        return None