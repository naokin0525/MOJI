"""
Main training script for the SVG Handwriting Generation model.

This script handles:
- Parsing command-line arguments.
- Setting up logging and device (CPU/GPU).
- Loading and preparing the dataset.
- Initializing the VAE-GAN model.
- Setting up optimizers and loss functions.
- Running the training loop.
- Saving model checkpoints and the final model.
"""

import argparse
import os
import sys
import logging
import torch

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# --- Import Custom Modules ---
try:
    from src.utils.logger import setup_logging
    from src.utils.config import load_config, get_config_value
    from src.utils.error_handling import TrainingError, DataError
    from src.data.dataset import create_dataloader
    from src.models.vaegan import HandwritingVAEGAN
    from src.training.trainer import Trainer
except ImportError as e:
    print(f"Error: Failed to import necessary modules. Ensure the project structure is correct and all files exist. Details: {e}")
    sys.exit(1)

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train an AI model for SVG handwriting generation.")

    # Removed redundant positional argument "program_path"

    # --- Paths ---
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the directory containing the training dataset (e.g., .moj files or SVG folders).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(project_root, "models"),
        help="Directory path to save the trained model checkpoints and final model.",
    )
    parser.add_argument(
        "--model_file_name",
        type=str,
        default="handwriting_model.pth",
        help="Base filename for the saved model artifacts.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(project_root, "config", "default_config.yaml"),
        help="Path to the configuration file (YAML). CLI args override config file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a model checkpoint to resume training or for fine-tuning.",
    )

    # --- Training Parameters ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--learning_rate_g",
        type=float,
        help="Learning rate for the Generator (VAE parts).",
    )
    parser.add_argument(
        "--learning_rate_d",
        type=float,
        help="Learning rate for the Discriminator.",
    )
    parser.add_argument(
        "--vae_weight",
        type=float,
        default=1.0,
        help="Weight for the VAE reconstruction loss component.",
    )
    parser.add_argument(
        "--gan_weight",
        type=float,
        default=0.5,
        help="Weight for the GAN adversarial loss component.",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        help="Weight for the KL divergence loss component (VAE).",
    )

    # --- Model Hyperparameters ---
    parser.add_argument(
        "--latent_dim",
        type=int,
        help="Dimensionality of the latent space.",
    )
    parser.add_argument(
        "--sequence_model_type",
        choices=['rnn', 'transformer'],
        help="Type of sequence model for strokes (RNN or Transformer).",
    )

    # --- Hardware & Logging ---
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training ('auto', 'cpu', 'cuda', 'cuda:idx'). 'auto' selects GPU if available.",
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
        default=os.path.join(project_root, "training.log"),
        help="Path to the output log file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of worker processes for data loading.",
    )

    # --- Feature Flags ---
    parser.add_argument(
        "--use_glyphwiki",
        action="store_true",
        help="Enable integration with GlyphWiki API for Kanji structure analysis (requires internet).",
    )
    parser.add_argument(
        "--enable_transfer_learning",
        action="store_true",
        help="Enable transfer learning mode (expects personalized samples and a base model via --checkpoint_path).",
    )

    return parser.parse_args()

# --- Main Execution ---
def main():
    """Main function to orchestrate the training process."""
    args = parse_arguments()

    # 1. Setup Logging
    setup_logging(args.log_level, args.log_file)
    logging.info("Starting training script...")
    logging.info(f"Command line arguments: {vars(args)}")

    # 2. Load Configuration & Combine with Args
    try:
        config = load_config(args.config_file)
        logging.info(f"Loaded configuration from {args.config_file}")
    except FileNotFoundError:
        logging.warning(f"Configuration file not found at {args.config_file}. Using CLI args and defaults.")
        config = {}

    cfg = lambda key, default=None: get_config_value(key, args, config, default)

    epochs = cfg('epochs', 100)
    batch_size = cfg('batch_size', 32)
    lr_g = cfg('learning_rate_g', 0.0002)
    lr_d = cfg('learning_rate_d', 0.0002)
    vae_weight = cfg('vae_weight', 1.0)
    gan_weight = cfg('gan_weight', 0.5)
    kl_weight = cfg('kl_weight', 0.1)
    latent_dim = cfg('latent_dim', 128)
    sequence_model_type = cfg('sequence_model_type', 'rnn')
    num_workers = cfg('num_workers', os.cpu_count() // 2 if os.cpu_count() else 1)

    logging.info(f"Effective Configuration: Epochs={epochs}, BatchSize={batch_size}, LR_G={lr_g}, LR_D={lr_d}, VAE_W={vae_weight}, GAN_W={gan_weight}, KL_W={kl_weight}, LatentDim={latent_dim}, SeqModel={sequence_model_type}, Workers={num_workers}")

    # 3. Setup Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    if not torch.cuda.is_available() and "cuda" in device.type:
         logging.warning("CUDA specified but not available. Falling back to CPU.")
         device = torch.device("cpu")

    # 4. Load Dataset
    try:
        logging.info(f"Loading dataset from: {args.dataset_path}")
        train_loader = create_dataloader(
            dataset_path=args.dataset_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            use_glyphwiki=args.use_glyphwiki,
        )
        logging.info(f"Dataset loaded successfully. Found {len(train_loader.dataset)} samples.")
    except DataError as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        sys.exit(1)

    # 5. Initialize Model
    try:
        logging.info("Initializing the Handwriting VAE-GAN model...")
        data_sample, _ = next(iter(train_loader))
        placeholder_input_dim = 5
        placeholder_output_dim = placeholder_input_dim

        model = HandwritingVAEGAN(
            input_dim=placeholder_input_dim,
            output_dim=placeholder_output_dim,
            latent_dim=latent_dim,
            sequence_model_type=sequence_model_type,
        ).to(device)

        logging.info(f"Model initialized: {model.__class__.__name__}")
        logging.debug(f"Model Architecture:\n{model}")

    except Exception as e:
        logging.error(f"Failed to initialize the model: {e}", exc_info=True)
        sys.exit(1)

    # 6. Load Checkpoint (if specified)
    start_epoch = 0
    if args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            try:
                logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
                checkpoint = torch.load(args.checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                logging.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
                if args.enable_transfer_learning:
                    logging.info("Transfer learning mode enabled. Model weights loaded, optimizers will be reset.")
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}. Starting training from scratch.", exc_info=True)
                start_epoch = 0
        else:
            logging.warning(f"Checkpoint path specified ({args.checkpoint_path}) but file not found. Starting training from scratch.")

    # 7. Setup Trainer
    try:
        logging.info("Setting up the Trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            device=device,
            learning_rate_g=lr_g,
            learning_rate_d=lr_d,
            vae_weight=vae_weight,
            gan_weight=gan_weight,
            kl_weight=kl_weight,
            model_save_path=args.model_path,
            model_file_name=args.model_file_name,
        )
        logging.info("Trainer setup complete.")
    except Exception as e:
        logging.error(f"Failed to set up the Trainer: {e}", exc_info=True)
        sys.exit(1)

    # 8. Run Training Loop
    try:
        logging.info(f"Starting training from epoch {start_epoch} for {epochs} total epochs...")
        trainer.train(num_epochs=epochs, start_epoch=start_epoch)
        logging.info("Training finished successfully.")
    except TrainingError as e:
        logging.error(f"A controlled training error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user (KeyboardInterrupt). Saving final state...")
        trainer.save_model(epoch="interrupted")
        logging.info("Model state saved. Exiting.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        try:
            trainer.save_model(epoch="error")
            logging.info("Attempted to save model state after error.")
        except Exception as save_e:
            logging.error(f"Could not save model state after error: {save_e}")
        sys.exit(1)

if __name__ == "__main__":
    main()