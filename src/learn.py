"""
Main Training Script for the Handwriting Synthesis Model.

This program orchestrates the training process for the VAE-GAN model.
It handles:
- Parsing command-line arguments for configuration.
- Loading and preparing the dataset.
- Initializing the model, optimizers, and loss functions.
- Executing the main training loop.
- Calculating and backpropagating the combined VAE and GAN losses.
- Saving model checkpoints periodically.
"""

import argparse
import os
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    MODEL_DIR,
    RANDOM_SEED,
    KLD_WEIGHT,
    BATCH_SIZE,
    LEARNING_RATE_GENERATOR,
    LEARNING_RATE_DISCRIMINATOR,
    BETA1,
    BETA2,
    LOG_INTERVAL,
    SAVE_MODEL_EPOCH_INTERVAL,
)
from src.models.vae_gan import VAE_GAN
from src.utils.dataset_loader import (
    HandwritingDataset,
    create_or_load_dataset,
    collate_fn,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


def loss_function(reconstructed_x, x, mu, logvar):
    """
    Calculates the VAE loss, which is a combination of reconstruction loss and KL divergence.
    """
    # Reconstruction loss (Mean Squared Error)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction="sum")

    # KL divergence loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld_loss


def train(args):
    """Main training function."""
    # ------------------------------------------------------------------------
    # 1. Setup and Configuration
    # ------------------------------------------------------------------------
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # ------------------------------------------------------------------------
    # 2. Load Dataset
    # ------------------------------------------------------------------------
    logging.info("Loading dataset...")
    try:
        raw_data = create_or_load_dataset(args.dataset_path)
        dataset = HandwritingDataset(raw_data)
        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Dataset loading failed: {e}")
        return

    # ------------------------------------------------------------------------
    # 3. Initialize Model and Optimizers
    # ------------------------------------------------------------------------
    logging.info(f"Using device: {DEVICE}")
    model = VAE_GAN().to(DEVICE)

    # Optimizers for Generator (VAE encoder + decoder) and Discriminator
    optimizer_G = optim.Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()),
        lr=LEARNING_RATE_GENERATOR,
        betas=(BETA1, BETA2),
    )
    optimizer_D = optim.Adam(
        model.discriminator.parameters(),
        lr=LEARNING_RATE_DISCRIMINATOR,
        betas=(BETA1, BETA2),
    )

    adversarial_loss = torch.nn.BCELoss().to(DEVICE)

    logging.info("Starting training...")
    # ------------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_g_loss = 0
        total_d_loss = 0

        for batch_idx, batch in enumerate(data_loader):
            real_data = batch["strokes"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)

            batch_size = real_data.size(0)

            # Create labels for adversarial loss
            real_labels = torch.full((batch_size, 1), 1.0, device=DEVICE)
            fake_labels = torch.full((batch_size, 1), 0.0, device=DEVICE)

            # ---------------------------------
            # Train Discriminator
            # ---------------------------------
            optimizer_D.zero_grad()

            # Train with real data
            real_validity = model.discriminator(real_data, lengths)
            d_loss_real = adversarial_loss(real_validity, real_labels)

            # Train with fake data
            reconstructed_data, _, _ = model(real_data, lengths)
            fake_validity = model.discriminator(reconstructed_data.detach(), lengths)
            d_loss_fake = adversarial_loss(fake_validity, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # ---------------------------------
            # Train Generator (VAE)
            # ---------------------------------
            optimizer_G.zero_grad()

            reconstructed_data, mu, logvar = model(real_data, lengths)

            # VAE Loss
            recon_loss, kld_loss = loss_function(
                reconstructed_data, real_data, mu, logvar
            )
            vae_loss = recon_loss + KLD_WEIGHT * kld_loss

            # GAN Loss for Generator
            fake_validity = model.discriminator(reconstructed_data, lengths)
            g_loss_gan = adversarial_loss(fake_validity, real_labels)

            # Combined Generator Loss
            g_loss = (args.vae_weight * vae_loss) + (args.gan_weight * g_loss_gan)
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            if batch_idx % LOG_INTERVAL == 0:
                logging.info(
                    f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(data_loader)} | "
                    f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}"
                )

        avg_g_loss = total_g_loss / len(data_loader)
        avg_d_loss = total_d_loss / len(data_loader)
        logging.info(f"--- End of Epoch {epoch} ---")
        logging.info(f"Average Generator Loss: {avg_g_loss:.4f}")
        logging.info(f"Average Discriminator Loss: {avg_d_loss:.4f}")

        # --------------------------------------------------------------------
        # 5. Save Model Checkpoint
        # --------------------------------------------------------------------
        if epoch % SAVE_MODEL_EPOCH_INTERVAL == 0:
            model_save_path = os.path.join(
                args.model_path, f"{args.model_file_name}_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved to {model_save_path}")

    # Save final model
    final_model_path = os.path.join(args.model_path, args.model_file_name)
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the handwriting synthesis model."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the root of the raw dataset directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_DIR,
        help="Directory to save the trained models.",
    )
    parser.add_argument(
        "--model_file_name",
        type=str,
        default="handwriting_model.pth",
        help="Name of the final model file.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--vae_weight",
        type=float,
        default=0.5,
        help="Weight for the VAE loss component.",
    )
    parser.add_argument(
        "--gan_weight",
        type=float,
        default=0.5,
        help="Weight for the GAN loss component.",
    )

    args = parser.parse_args()
    train(args)
