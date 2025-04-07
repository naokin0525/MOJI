# src/generation/stroke_simulation.py
"""
Utilities to add realistic variations to generated handwriting strokes.

Includes functions for adding jitter, simulating pressure/speed effects (heuristically),
and potentially smoothing.
"""

import numpy as np
import random
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Type alias for clarity
Point = Tuple[float, float]
Stroke = List[Point]
Strokes = List[Stroke]
PointWithWidth = Tuple[float, float, float] # x, y, width
StrokeWithWidth = List[PointWithWidth]
StrokesWithWidth = List[StrokeWithWidth]

def add_jitter(strokes: Strokes, intensity: float = 0.5) -> Strokes:
    """
    Adds random noise (jitter) to stroke coordinates.

    Args:
        strokes (Strokes): List of strokes (lists of (x, y) points).
        intensity (float): Controls the magnitude of the random offsets.
                           Scales the standard deviation of the noise.

    Returns:
        Strokes: Strokes with jitter applied.
    """
    if not strokes or intensity <= 0:
        return strokes

    jittered_strokes = []
    for stroke in strokes:
        if not stroke:
            jittered_strokes.append([])
            continue

        jittered_stroke = []
        # Apply correlated noise: determine a base offset for the stroke start?
        # Simpler: independent noise per point for now.
        for x, y in stroke:
            # Sample from a normal distribution, scale by intensity
            offset_x = random.gauss(0, intensity)
            offset_y = random.gauss(0, intensity)
            jittered_stroke.append((x + offset_x, y + offset_y))
        jittered_strokes.append(jittered_stroke)

    return jittered_strokes

def _calculate_velocity(stroke: Stroke) -> np.ndarray:
    """Helper to estimate point-wise velocity (magnitude of displacement)."""
    if len(stroke) < 2:
        return np.zeros(len(stroke))

    points = np.array(stroke)
    # Calculate displacement vectors (point[i] - point[i-1])
    displacements = np.diff(points, axis=0) # Shape (n-1, 2)
    # Calculate magnitude (Euclidean distance) -> speed proxy
    velocities = np.linalg.norm(displacements, axis=1) # Shape (n-1,)

    # Pad velocities array to match number of points (e.g., repeat first/last)
    # Simple padding: Assign velocity[i] to point[i+1]. Pad start with 0.
    padded_velocities = np.concatenate(([0.0], velocities))
    return padded_velocities


def simulate_pressure_width(strokes: Strokes, base_width: float = 1.0, variation: float = 0.5) -> StrokesWithWidth:
    """
    Simulates variable stroke width based on heuristic properties like velocity.
    Lower velocity (e.g., on curves) often corresponds to higher pressure/width.

    Args:
        strokes (Strokes): List of strokes (lists of (x, y) points).
        base_width (float): The target average stroke width.
        variation (float): Controls the amount of width variation.
                           Value from 0 (no variation) to 1 (high variation).

    Returns:
        StrokesWithWidth: List of strokes where points are (x, y, width).
                          Returns original structure with base_width if variation is 0.
    """
    if variation <= 0:
        # Return original structure but with width added
        return [[(p[0], p[1], base_width) for p in s] for s in strokes]

    strokes_with_width = []
    all_velocities = []
    valid_stroke_indices = [i for i, s in enumerate(strokes) if len(s) >= 2]

    if not valid_stroke_indices: # Handle case with no valid strokes for velocity calc
         return [[(p[0], p[1], base_width) for p in s] for s in strokes]

    # Calculate velocities for all valid strokes first to find global min/max
    for i in valid_stroke_indices:
         all_velocities.extend(_calculate_velocity(strokes[i]))

    if not all_velocities: # Handle empty velocity list
         return [[(p[0], p[1], base_width) for p in s] for s in strokes]

    min_vel = np.min(all_velocities)
    max_vel = np.max(all_velocities)
    vel_range = max_vel - min_vel

    # Avoid division by zero if range is tiny (e.g., single speed stroke)
    if vel_range < 1e-6:
        vel_range = 1.0

    for i, stroke in enumerate(strokes):
        stroke_with_width = []
        if len(stroke) < 2: # Handle single point or empty strokes
            stroke_with_width = [(p[0], p[1], base_width) for p in stroke]
        else:
            velocities = _calculate_velocity(stroke)
            for j, point in enumerate(stroke):
                # Normalize velocity (0 to 1, approximately)
                norm_velocity = (velocities[j] - min_vel) / vel_range
                # Invert relationship: higher velocity -> lower width
                # Apply variation factor. Map norm_velocity [0,1] to width modulation [1-v, 1+v]
                # Width = base_width * (1 + variation * (1 - 2 * norm_velocity)) # Linear inverse map
                # Smoother variation using sigmoid or power function might be better
                # Simple approach: More pressure (width) when slower
                width_factor = 1.0 + variation * (1.0 - norm_velocity) # Factor around 1.0

                # Clamp width factor to avoid extreme values (e.g., negative width)
                min_factor = max(0.1, 1.0 - variation) # Ensure width > 0
                max_factor = 1.0 + variation
                width_factor = np.clip(width_factor, min_factor, max_factor)

                current_width = base_width * width_factor
                stroke_with_width.append((point[0], point[1], current_width))

        strokes_with_width.append(stroke_with_width)

    return strokes_with_width

# TODO: Implement smoothing if desired (e.g., using Bezier fitting, spline interpolation)
# def smooth_strokes(strokes: Strokes, method='bezier') -> Strokes: ...


def apply_stroke_variations(strokes: Strokes,
                            base_width: float,
                            jitter_intensity: float = 0.0,
                            pressure_variation: float = 0.0
                            ) -> Tuple[Strokes, StrokesWithWidth | None]:
    """
    Applies selected simulations to the generated strokes.

    Args:
        strokes (Strokes): The raw generated strokes [(x, y), ...].
        base_width (float): The base stroke width for SVG output.
        jitter_intensity (float): Intensity for coordinate jitter (0 to disable).
        pressure_variation (float): Intensity for simulated pressure/width variation (0 to disable).

    Returns:
        Tuple[Strokes, StrokesWithWidth | None]:
            - strokes_processed: Strokes after jitter (and potentially smoothing).
                                 These are the coordinates to use for SVG path data.
            - strokes_with_width: Strokes with simulated width per point [(x, y, width), ...].
                                  None if pressure_variation is 0. Used for variable width rendering if supported.
    """
    processed_strokes = strokes
    strokes_with_width = None

    # 1. Apply Jitter
    if jitter_intensity > 0:
        processed_strokes = add_jitter(processed_strokes, jitter_intensity)
        logger.debug(f"Applied jitter with intensity {jitter_intensity}")

    # 2. Apply Smoothing (Optional - NYI)
    # if smoothing_factor > 0:
    #    processed_strokes = smooth_strokes(processed_strokes, ...)
    #    logger.debug("Applied smoothing")

    # 3. Simulate Pressure/Width (operates on jittered/smoothed coordinates)
    if pressure_variation > 0:
        strokes_with_width = simulate_pressure_width(processed_strokes, base_width, pressure_variation)
        logger.debug(f"Applied pressure simulation with variation {pressure_variation}")

    return processed_strokes, strokes_with_width