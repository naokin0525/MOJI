"""
Handwriting Generation Script.

This program uses a trained VAE-GAN model to generate handwriting for a given
text string and saves it as an SVG file. It allows for adjustments like
stroke width and random variation.
"""

import argparse
import logging
import torch
import numpy as np
from src.config import (
    DEVICE, LATENT_DIM, DEFAULT_STROKE_WIDTH, DEFAULT_RANDOM_VARIATION,
    DEFAULT_LINE_HEIGHT, DEFAULT_CHAR_SPACING
)
from src.models.vae_gan import VAE_GAN
from src.utils.image_converter import svg_to_png, svg_to_jpg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sequence_to_svg_path(sequence: np.ndarray, stroke_width: float) -> str:
    """
    Converts a numerical stroke sequence back into an SVG path string.
    """
    path_str = f'<path d="'
    current_pos = np.array([0.0, 0.0])
    
    for point in sequence:
        dx, dy, pressure, pen_down, end_of_stroke = point
        
        if pen_down == 0:  # Pen up, move to the next position
            current_pos += np.array([dx, dy])
            path_str += f' M {current_pos[0]:.2f} {current_pos[1]:.2f}'
        else: # Pen down, draw a line
            start_pos = current_pos
            end_pos = start_pos + np.array([dx, dy])
            path_str += f' L {end_pos[0]:.2f} {end_pos[1]:.2f}'
            current_pos = end_pos
            
    path_str += f'" stroke="black" stroke-width="{stroke_width * pressure:.2f}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    return path_str

def generate_handwriting(args):
    """Main generation function."""
    try:
        # Load the trained model
        model = VAE_GAN()
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logging.info(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        logging.error(f"Model file not found at: {args.model_path}")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    full_svg_content = ''
    current_x_offset = 0

    with torch.no_grad():
        for char in args.text:
            if char == ' ':
                current_x_offset += DEFAULT_CHAR_SPACING * 3
                continue

            # Generate a random latent vector
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            
            # Add random variation to the latent vector for style diversity
            z += args.random_variation * torch.randn_like(z)

            # Generate the sequence from the latent vector
            # Max length is a hyperparameter; 200 should be enough for a character
            generated_sequence = model.generator(z, max_seq_len=200)
            
            # Move to CPU and convert to numpy
            sequence_np = generated_sequence.cpu().squeeze(0).numpy()
            
            # Post-processing: stop at the first "end of character" signal if any
            # (A more robust model would have an explicit end token)
            
            # Convert sequence to an SVG path string
            svg_path = sequence_to_svg_path(sequence_np, args.stroke_width)
            
            # Wrap the path in a group to apply the offset
            full_svg_content += f'<g transform="translate({current_x_offset}, 0)">\n{svg_path}\n</g>\n'
            
            # Update offset for the next character
            # A proper way is to get the bounding box, here we use a fixed advance
            max_x = np.cumsum(sequence_np[:, 0]).max()
            current_x_offset += max_x + DEFAULT_CHAR_SPACING

    # Create the final SVG file
    svg_header = f'<svg width="{current_x_offset}" height="{DEFAULT_LINE_HEIGHT}" xmlns="http://www.w3.org/2000/svg">\n'
    svg_footer = '</svg>'
    final_svg = svg_header + full_svg_content + svg_footer

    try:
        with open(args.output_path, 'w') as f:
            f.write(final_svg)
        logging.info(f"Handwriting saved to {args.output_path}")
        
        # Optional: Convert to other formats
        output_lower = args.output_path.lower()
        if output_lower.endswith('.png'):
            svg_to_png(args.output_path, args.output_path)
        elif output_lower.endswith('.jpg') or output_lower.endswith('.jpeg'):
            svg_to_jpg(args.output_path, args.output_path)

    except IOError as e:
        logging.error(f"Failed to write to output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate handwriting from a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth) file.")
    parser.add_argument("--text", type=str, required=True, help="The text string to generate.")
    parser.add_argument("--output_path", type=str, default="output/svg/generated.svg", help="Path to save the output SVG, PNG, or JPG file.")
    parser.add_argument("--random_variation", type=float, default=DEFAULT_RANDOM_VARIATION, help="Amount of random noise to add for style variation.")
    parser.add_argument("--stroke_width", type=float, default=DEFAULT_STROKE_WIDTH, help="Base width of the stroke.")

    args = parser.parse_args()
    generate_handwriting(args)