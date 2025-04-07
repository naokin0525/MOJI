# src/models/__init__.py
"""
Models package for the SVG Handwriting Generation project.

Contains definitions for the core VAE-GAN model and its components
(Encoder, Decoder/Generator, Discriminator, Sequence Models).
"""

from .vaegan import HandwritingVAEGAN
from .encoder import StrokeEncoderVAE
from .decoder import StrokeDecoderGenerator
from .discriminator import StrokeSequenceDiscriminator
from .sequence_model import StrokeRNN, StrokeTransformerEncoder # Expose sequence modules if needed externally

# Define what `from src.models import *` imports
__all__ = [
    "HandwritingVAEGAN",
    "StrokeEncoderVAE",
    "StrokeDecoderGenerator",
    "StrokeSequenceDiscriminator",
    "StrokeRNN",
    "StrokeTransformerEncoder",
]