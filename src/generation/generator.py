# src/generation/generator.py
"""
Core handwriting generation logic.

Takes text input, uses the trained model, applies style and simulation,
and outputs an SVG string.
"""

import random
import torch
import numpy as np
import logging
from typing import List, Tuple

try:
    from ..models.vaegan import HandwritingVAEGAN
    from .style_control import get_style_parameters
    from .stroke_simulation import apply_stroke_variations
    from ..utils.error_handling import GenerationError
except ImportError:
    # Fallbacks for partial imports
    class HandwritingVAEGAN(torch.nn.Module): pass
    class GenerationError(Exception): pass
    def get_style_parameters(s): return {}
    def apply_stroke_variations(s, **kwargs): return s, None


logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]
Stroke = List[Point]
Strokes = List[Stroke]
PointWithWidth = Tuple[float, float, float]
StrokeWithWidth = List[PointWithWidth]
StrokesWithWidth = List[StrokeWithWidth]


class HandwritingGenerator:
    """
    Generates handwriting SVG from text using a trained model and style settings.
    """
    def __init__(self,
                 model: HandwritingVAEGAN,
                 device: torch.device,
                 style: str = 'default',
                 random_variation_scale: float = 1.0, # Scale factor for sampling noise (0 = deterministic mean)
                 stroke_width: float = 1.0,
                 stroke_color: str = "black",
                 seed: int | None = None,
                 # --- Parameters for tensor_to_strokes conversion ---
                 # These should match the feature indices used during training/tensor creation
                 dx_idx: int = 0,
                 dy_idx: int = 1,
                 pen_down_idx: int = 2,
                 eos_stroke_idx: int = 3, # End of stroke flag index
                 eos_seq_idx: int = 4,    # End of sequence flag index
                 pt_included: bool = False # Was pressure/time included in model output?
                 ):
        """
        Initializes the HandwritingGenerator.

        Args:
            model: The trained HandwritingVAEGAN model instance.
            device: The torch device (CPU or CUDA).
            style: Name of the desired style (e.g., 'casual', 'formal').
            random_variation_scale: Controls noise in latent sampling (0=mean, 1=stddev).
            stroke_width: Base width for SVG strokes.
            stroke_color: Color for SVG strokes.
            seed: Optional random seed for reproducibility.
            dx_idx, dy_idx, ...: Indices of features in the model's output tensor.
            pt_included: Whether pressure/time are present in the model output tensor.
        """
        self.model = model.to(device)
        self.model.eval() # Ensure model is in evaluation mode
        self.device = device
        self.latent_dim = model.latent_dim
        self.style_name = style
        self.random_variation_scale = random_variation_scale
        self.base_stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.seed = seed

        # Store feature indices
        self.dx_idx = dx_idx
        self.dy_idx = dy_idx
        self.pen_down_idx = pen_down_idx
        self.eos_stroke_idx = eos_stroke_idx
        self.eos_seq_idx = eos_seq_idx
        self.pt_included = pt_included # Affects tensor interpretation

        # Set seed if provided
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(self.seed)
            logger.info(f"Generator random seed set to: {self.seed}")

        # Load style parameters that control simulation
        self.style_params = get_style_parameters(self.style_name)
        logger.info(f"Generator initialized with style '{self.style_name}'. Params: {self.style_params}")


    @torch.no_grad() # Disable gradient calculations for generation
    def _generate_char_sequence(self, char: str) -> torch.Tensor | None:
        """Generates the raw sequence tensor for a single character."""
        # TODO: Handle conditional generation if model supports it (e.g., char embedding -> z)
        # For now, generate unconditionally from random noise z.

        # Sample latent vector z
        if self.random_variation_scale <= 0:
            # Use zero vector for deterministic output (represents mean of prior)
             z = torch.zeros(1, self.latent_dim, device=self.device)
        else:
             # Sample from standard normal, scale variance if needed (though scaling stddev is typical)
             z = torch.randn(1, self.latent_dim, device=self.device) * self.random_variation_scale

        # TODO: Apply style modification to z if implemented
        # z = apply_style_to_latent(z, self.style_params)

        try:
            # Generate sequence tensor from the model's decoder/generator
            raw_sequence_tensor = self.model.generate(z) # Shape: (1, max_seq_len, features)
            return raw_sequence_tensor.squeeze(0) # Return shape (max_seq_len, features)
        except Exception as e:
             logger.error(f"Model generation failed for character '{char}' (or from z): {e}", exc_info=True)
             raise GenerationError(f"Model inference failed for char '{char}'") from e


    def _tensor_to_strokes(self, sequence_tensor: torch.Tensor, pen_down_threshold=0.5) -> Strokes:
        """Converts a raw sequence tensor back into list of (x, y) strokes."""
        if sequence_tensor is None or len(sequence_tensor) == 0:
            return []

        strokes: Strokes = []
        current_stroke: Stroke = []
        current_pos = np.array([0.0, 0.0]) # Start at origin
        pen_is_down = False

        sequence = sequence_tensor.cpu().numpy() # Move to CPU and convert to numpy

        for point_data in sequence:
            # Extract data based on stored indices
            dx = point_data[self.dx_idx]
            dy = point_data[self.dy_idx]
            pen_down_signal = point_data[self.pen_down_idx]
            # eos_stroke_signal = point_data[self.eos_stroke_idx] # May not be needed if pen_down is reliable
            eos_seq_signal = point_data[self.eos_seq_idx]

            # Update absolute position
            current_pos += np.array([dx, dy])

            is_currently_down = pen_down_signal > pen_down_threshold

            if is_currently_down:
                current_stroke.append(tuple(current_pos)) # Add point with absolute coords
                pen_is_down = True
            else:
                # Pen is up
                if pen_is_down:
                    # Pen was just lifted, end the current stroke if it has points
                    if current_stroke:
                         strokes.append(current_stroke)
                    current_stroke = [] # Start a new empty stroke
                pen_is_down = False

            # Check for end of sequence marker
            if eos_seq_signal > 0.5:
                break # Stop processing this sequence

        # Add the last stroke if the sequence ended while pen was down
        if pen_is_down and current_stroke:
            strokes.append(current_stroke)

        # Filter out potential empty strokes
        strokes = [s for s in strokes if s]
        return strokes


    def _strokes_to_svg_path(self, strokes: Strokes, stroke_width: float) -> str:
        """Converts list of strokes into SVG path data string(s)."""
        if not strokes:
            return ""

        path_data_elements = []
        for stroke in strokes:
            if not stroke or len(stroke) < 1: continue

            # Start path with Move command
            d = f"M {stroke[0][0]:.2f} {stroke[0][1]:.2f}"
            # Add Line commands for subsequent points
            if len(stroke) > 1:
                 d += " L " + " ".join([f"{p[0]:.2f} {p[1]:.2f}" for p in stroke[1:]])

            # Create path element with constant width for now
            path_str = f'<path d="{d}" stroke="{self.stroke_color}" stroke-width="{stroke_width:.2f}" fill="none" stroke-linecap="round" stroke-linejoin="round" />'
            path_data_elements.append(path_str)

            # --- TODO: Variable Width Handling ---
            # If strokes_with_width were generated and needed variable width SVG:
            # 1. Segment stroke based on width changes.
            # 2. Generate multiple <path> elements with different stroke-width attributes.
            # OR
            # 3. Convert centerline + width data into a filled outline polygon/path. (More complex)

        return "\n".join(path_data_elements)


    def _assemble_full_svg(self, svg_elements: List[str], width: int=500, height: int=100) -> str:
        """Wraps generated SVG path elements in a full SVG document structure."""
        # TODO: Calculate actual bounding box from elements for better width/height.
        # TODO: Implement proper character layout instead of just overlaying.
        content = "\n".join(svg_elements)

        # Basic SVG wrapper
        svg_string = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <title>Generated Handwriting</title>
  <g>
    {content}
  </g>
</svg>
"""
        return svg_string


    def generate_handwriting(self, text: str) -> str:
        """
        Generates an SVG string representing the input text in handwriting.

        Args:
            text (str): The text to convert to handwriting.

        Returns:
            str: A string containing the complete SVG document.

        Raises:
            GenerationError: If generation fails for any character.
        """
        logger.info(f"Generating handwriting for text: '{text}' with style '{self.style_name}'")
        all_char_svg_elements = []
        current_x_offset = 0 # Basic layout: advance X position (NYI properly)

        for char in text:
            if char.isspace():
                # Handle spaces (e.g., add horizontal offset)
                # TODO: Implement proper space handling based on font metrics/heuristics
                current_x_offset += 20 # Arbitrary space width for now
                logger.debug(f"Skipping space, advancing offset.")
                continue

            logger.debug(f"Generating character: '{char}'")
            # 1. Generate raw tensor sequence from model
            try:
                raw_tensor = self._generate_char_sequence(char)
                if raw_tensor is None:
                    logger.warning(f"Generation returned None for character '{char}'. Skipping.")
                    continue
            except GenerationError as e:
                logger.error(f"Failed to generate sequence for character '{char}': {e}")
                # Option: Skip character or raise error? Raise for now.
                raise e

            # 2. Convert tensor to coordinate strokes
            strokes_xy = self._tensor_to_strokes(raw_tensor)
            if not strokes_xy:
                 logger.warning(f"Tensor to strokes conversion yielded no strokes for '{char}'. Skipping.")
                 continue

            # 3. Apply stroke simulation based on style
            jitter = self.style_params.get('jitter_intensity', 0.0)
            pressure_var = self.style_params.get('pressure_variation', 0.0)
            # Note: apply_stroke_variations currently returns (strokes, strokes_with_width|None)
            simulated_strokes, _ = apply_stroke_variations( # Ignore width data for now
                 strokes_xy,
                 self.base_stroke_width,
                 jitter_intensity=jitter * self.random_variation_scale, # Scale jitter by random variation factor?
                 pressure_variation=pressure_var
            )

            # 4. Convert simulated strokes to SVG path element(s)
            # Use base_stroke_width for constant width output
            svg_char_paths = self._strokes_to_svg_path(simulated_strokes, self.base_stroke_width)

            # 5. TODO: Layout - Apply transform to position the character
            # For now, just collect paths; they will overlay at origin
            # Correct implementation would wrap svg_char_paths in a <g transform="translate(x, 0)">
            # and calculate 'x' based on previous char width.
            if svg_char_paths:
                all_char_svg_elements.append(svg_char_paths)


        # 6. Assemble the final SVG document
        final_svg = self._assemble_full_svg(all_char_svg_elements)
        logger.info("Handwriting generation complete.")
        return final_svg