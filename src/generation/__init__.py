# src/generation/__init__.py
"""
Generation package for the SVG Handwriting Generation project.

Contains modules for:
- The main handwriting generation orchestrator (generator.py)
- Style control and parameterization (style_control.py)
- Simulation of realistic stroke variations (stroke_simulation.py)
"""

from .generator import HandwritingGenerator

__all__ = [
    "HandwritingGenerator",
]