# scripts/generate.py
"""
Main generation script for the SVG Handwriting Generation model.

This script handles:
- Parsing command-line arguments for generation parameters.
- Setting up logging and device (CPU/GPU).
- Loading a pre-trained HandwritingVAEGAN model.
- Initializing the handwriting generator.
- Generating SVG handwriting based on input text and style parameters.
- Saving the output as SVG and optionally converting to PNG/JPG.
"""

import argparse
import os
import sys
import logging
import torch
import re
from datetime import datetime

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# --- Import Custom Modules ---
# These will be created in subsequent steps.
try:
    from src.utils.logger import setup_logging
    from src.utils.config import load_config, get_config_value
    from src.utils.error_handling import GenerationError, ModelLoadError, ConversionError
    from src.models.vaegan import HandwritingVAEGAN # To instantiate the model structure
    from src.generation.generator import HandwritingGenerator # Handles the actual generation process
    from src.conversion.to_raster import convert_svg_to_raster # For PNG/JPG output
except ImportError as e:
    print(f"Error: Failed to import necessary modules. Ensure the project structure is correct and all files exist. Details: {e}")
    sys.exit(1)

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments for the generation script."""
    parser = argparse.ArgumentParser(description="Generate SVG handwriting using a trained AI model.")

    # --- Paths ---
    # Modified model_path based on user example interpretation: points to the model *file*
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file (.pth).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(project_root, "output"),
        help="Directory path to save the generated output files.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(project_root, "config", "default_config.yaml"),
        help="Path to the configuration file (YAML) - used for fallback model parameters if not in checkpoint.",
    )

    # --- Generation Parameters ---
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text string to convert to handwriting.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="casual",
        # TODO: Define available styles more formally, potentially load from model/config
        help="Desired handwriting style (e.g., 'casual', 'formal', 'cursive'). Exact available styles depend on the model.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of different handwriting samples to generate for the input text.",
    )
    parser.add_argument(
        "--random_variation",
        action="store_true", # Use flag presence as per user example: --random_variation true
        help="Apply random variations for a more natural, less repetitive look.",
    )
    parser.add_argument(
        "--stroke_width",
        type=float,
        default=1.0,
        help="Stroke width for the generated SVG paths.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="svg",
        choices=["svg", "png", "jpg"],
        help="Desired output format.",
    )
    # Add seed for reproducibility if needed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility of variations."
    )

    # --- Hardware & Logging ---
    parser.add_argument(
        "--device",
        type=str,
        default="auto", # auto, cpu, cuda, cuda:0, etc.
        help="Device to use for generation ('auto', 'cpu', 'cuda'). 'auto' selects GPU if available.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=os.path.join(project_root, "generation.log"),
        help="Path to the output log file.",
    )

    return parser.parse_args()

# --- Helper Functions ---
def sanitize_filename(text, max_len=50):
    """Sanitizes a string to be used as a filename."""
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", text)
    # Truncate if too long
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].strip()
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Handle empty string after sanitization
    if not sanitized:
        return "generated_output"
    return sanitized

# --- Main Execution ---
def main():
    """Main function to orchestrate the generation process."""
    args = parse_arguments()

    # 1. Setup Logging
    setup_logging(args.log_level, args.log_file)
    logging.info("Starting generation script...")
    logging.info(f"Command line arguments: {vars(args)}")

    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_path, exist_ok=True)
        logging.info(f"Output directory: {args.output_path}")
    except OSError as e:
        logging.error(f"Failed to create output directory '{args.output_path}': {e}")
        sys.exit(1)

    # 2. Load Configuration (primarily for fallbacks)
    try:
        config = load_config(args.config_file)
        logging.info(f"Loaded configuration from {args.config_file}")
    except FileNotFoundError:
        logging.warning(f"Configuration file not found at {args.config_file}. Relying on model checkpoint and CLI args.")
        config = {}

    # 3. Setup Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    if not torch.cuda.is_available() and "cuda" in device.type:
         logging.warning("CUDA specified but not available. Falling back to CPU.")
         device = torch.device("cpu")

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        # if device.type == 'cuda': torch.cuda.manual_seed(args.seed) # Add if using CUDA randomness extensively
        logging.info(f"Using random seed: {args.seed}")

    # 4. Load Model
    model = None # Define model variable in outer scope
    try:
        if not os.path.exists(args.model_path):
            raise ModelLoadError(f"Model file not found at: {args.model_path}")

        logging.info(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)

        # --- Crucial Part: Get model hyperparameters ---
        # Best practice: Hyperparameters should be saved *in* the checkpoint file during training.
        if 'model_hyperparams' not in checkpoint:
            logging.warning(f"Model hyperparameters not found in checkpoint '{args.model_path}'. "
                            "Attempting to load from config or use defaults. This may fail if architecture mismatch.")
            # Fallback to config/defaults (less reliable)
            cfg = lambda key, default=None: get_config_value(key, args, config, default)
            model_hyperparams = {
                'input_dim': cfg('input_dim', 5), # Example default
                'output_dim': cfg('output_dim', 5), # Example default
                'latent_dim': cfg('latent_dim', 128),
                'sequence_model_type': cfg('sequence_model_type', 'rnn'),
                # Add any other necessary hyperparameters here
            }
        else:
            model_hyperparams = checkpoint['model_hyperparams']
            logging.info(f"Loaded model hyperparameters from checkpoint: {model_hyperparams}")

        # Instantiate model with loaded hyperparameters
        # Ensure HandwritingVAEGAN constructor matches these keys
        model = HandwritingVAEGAN(**model_hyperparams).to(device)

        # Load the state dictionary
        if 'model_state_dict' not in checkpoint:
             raise ModelLoadError(f"Model state_dict not found in checkpoint '{args.model_path}'.")

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        logging.info("Model loaded and set to evaluation mode successfully.")

    except ModelLoadError as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error(f"Model file not found at path: {args.model_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        sys.exit(1)


    # 5. Initialize Handwriting Generator
    try:
        logging.info("Initializing Handwriting Generator...")
        # The generator needs the trained model and generation parameters
        generator = HandwritingGenerator(
            model=model,
            device=device,
            style=args.style,
            random_variation=args.random_variation,
            stroke_width=args.stroke_width,
            seed=args.seed # Pass seed for reproducibility within generator
            # Potentially pass other style parameters or fine-tuning data if needed
        )
        logging.info("Handwriting Generator initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Handwriting Generator: {e}", exc_info=True)
        sys.exit(1)

    # 6. Generate Handwriting Samples
    output_files = []
    base_filename = sanitize_filename(args.text)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(args.num_samples):
        sample_suffix = f"_sample{i+1}" if args.num_samples > 1 else ""
        # Use timestamp and sample number for uniqueness if base_filename is short or common
        unique_filename = f"{base_filename}{sample_suffix}_{timestamp}"
        svg_filepath = os.path.join(args.output_path, f"{unique_filename}.svg")

        try:
            logging.info(f"Generating sample {i+1}/{args.num_samples} for text: '{args.text}'")

            # --- Call the core generation function ---
            # This function should return the SVG content as a string
            svg_content = generator.generate_handwriting(args.text)

            # Save the SVG file
            with open(svg_filepath, "w", encoding="utf-8") as f:
                f.write(svg_content)
            logging.info(f"Saved SVG output to: {svg_filepath}")
            output_files.append(svg_filepath)

            # 7. Optional Conversion to Raster Format (PNG/JPG)
            if args.output_format in ["png", "jpg"]:
                raster_filepath = os.path.join(args.output_path, f"{unique_filename}.{args.output_format}")
                try:
                    logging.info(f"Converting {svg_filepath} to {args.output_format.upper()}...")
                    # Assuming convert_svg_to_raster takes input SVG path, output path, and format
                    convert_svg_to_raster(svg_filepath, raster_filepath, format=args.output_format)
                    logging.info(f"Saved {args.output_format.upper()} output to: {raster_filepath}")
                    output_files.append(raster_filepath)
                except ConversionError as e:
                    logging.error(f"Failed to convert SVG to {args.output_format.upper()}: {e}")
                except Exception as e:
                     logging.error(f"Unexpected error during {args.output_format.upper()} conversion: {e}", exc_info=True)

        except GenerationError as e:
            logging.error(f"Error during handwriting generation for sample {i+1}: {e}")
            # Decide whether to continue with other samples or exit
            # continue
        except Exception as e:
            logging.error(f"An unexpected error occurred during generation or saving sample {i+1}: {e}", exc_info=True)
            # continue # Optional: attempt to generate next sample

    logging.info(f"Generation process completed. Generated files: {output_files}")


if __name__ == "__main__":
    main()