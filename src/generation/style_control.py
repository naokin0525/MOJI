# src/generation/style_control.py
"""
Manages handwriting style parameters.

Defines presets for different styles ('casual', 'formal', etc.) which control
aspects like jitter intensity and simulated pressure variation.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Define presets for simulation parameters based on style name
# Values are indicative and should be tuned based on visual results.
STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    'default': { # Used as fallback
        'jitter_intensity': 0.4,    # Moderate jitter
        'pressure_variation': 0.3, # Moderate width variation
        # 'smoothing_factor': 0.0,  # No smoothing
        # 'connection_bias': 0.0,   # For potential cursive logic
        'description': 'A standard, balanced style.'
    },
    'casual': {
        'jitter_intensity': 0.7,    # Higher jitter
        'pressure_variation': 0.5, # More width variation
        # 'smoothing_factor': 0.1,
        'description': 'Informal, slightly messy style with noticeable variations.'
    },
    'formal': {
        'jitter_intensity': 0.1,    # Very low jitter
        'pressure_variation': 0.15, # Low width variation
        # 'smoothing_factor': 0.5,  # More smoothing for clean lines
        'description': 'Neat, clean style with minimal variation.'
    },
    'cursive': {
        'jitter_intensity': 0.3,
        'pressure_variation': 0.4,
        # 'smoothing_factor': 0.3,
        # 'connection_bias': 0.8, # Higher tendency to connect (needs specific logic)
        'description': 'Flowing style, potentially connected (connection not fully implemented).'
    },
    # Add more styles as needed
    'technical': {
        'jitter_intensity': 0.05,
        'pressure_variation': 0.05,
        'description': 'Extremely precise, almost like drafting.'
     }
}

# --- Style Parameter Retrieval ---

def get_style_parameters(style_name: str | None) -> dict:
    """
    Retrieves simulation parameters associated with a given style name.

    Args:
        style_name (str | None): The name of the desired style (case-insensitive).
                                If None or not found, returns 'default' parameters.

    Returns:
        dict: A dictionary containing parameters like 'jitter_intensity',
              'pressure_variation', etc.
    """
    if style_name is None:
        logger.debug("No style specified, using 'default' style parameters.")
        return STYLE_PRESETS['default'].copy() # Return copy to prevent modification

    style_key = style_name.lower()

    if style_key in STYLE_PRESETS:
        logger.debug(f"Using style parameters for '{style_key}'.")
        return STYLE_PRESETS[style_key].copy()
    else:
        logger.warning(f"Unknown style name: '{style_name}'. Falling back to 'default' style.")
        return STYLE_PRESETS['default'].copy()

# --- Style Modification (Future Extensions) ---

# Placeholder for functions that might modify latent vectors based on style
# def apply_style_to_latent(z: torch.Tensor, style_params: dict) -> torch.Tensor:
#     """ Placeholder: Modifies latent vector z based on style """
#     # Requires a defined mapping from style params to latent space adjustments
#     logger.warning("Latent vector style modification is not implemented yet.")
#     return z