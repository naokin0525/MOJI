# src/training/__init__.py
"""
Training package for the SVG Handwriting Generation project.

Contains modules for:
- Loss function definitions (losses.py)
- The main training loop orchestrator (trainer.py)
"""

from .trainer import Trainer
from .losses import (
    calculate_reconstruction_loss,
    calculate_kl_divergence_loss,
    calculate_discriminator_loss,
    calculate_generator_loss
)

__all__ = [
    "Trainer",
    "calculate_reconstruction_loss",
    "calculate_kl_divergence_loss",
    "calculate_discriminator_loss",
    "calculate_generator_loss",
]