# src/training/trainer.py
"""
Trainer class to handle the training loop for the Handwriting VAE-GAN model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
import time
from collections import defaultdict

try:
    from ..models.vaegan import HandwritingVAEGAN
    from .losses import (calculate_reconstruction_loss, calculate_kl_divergence_loss,
                       calculate_discriminator_loss, calculate_generator_loss,
                       create_padding_mask)
    from ..utils.error_handling import TrainingError
except ImportError:
    # Fallbacks for partial imports
    class HandwritingVAEGAN(torch.nn.Module): pass
    class TrainingError(Exception): pass
    def calculate_reconstruction_loss(*args, **kwargs): return torch.tensor(0.0)
    def calculate_kl_divergence_loss(*args, **kwargs): return torch.tensor(0.0)
    def calculate_discriminator_loss(*args, **kwargs): return torch.tensor(0.0)
    def calculate_generator_loss(*args, **kwargs): return torch.tensor(0.0)
    def create_padding_mask(*args, **kwargs): return None


logger = logging.getLogger(__name__)

class Trainer:
    """Orchestrates the training process for the HandwritingVAEGAN model."""

    def __init__(self,
                 model: HandwritingVAEGAN,
                 train_loader: DataLoader,
                 device: torch.device,
                 learning_rate_g: float = 0.0002,
                 learning_rate_d: float = 0.0002,
                 beta1: float = 0.5, # Adam optimizer beta1 parameter (common for GANs)
                 beta2: float = 0.999, # Adam optimizer beta2 parameter
                 vae_weight: float = 1.0,
                 gan_weight: float = 0.5,
                 kl_weight: float = 0.1,
                 reconstruction_loss_type: str = 'mse', # 'mse' or 'bce'
                 grad_clip_value: float | None = 1.0, # Optional gradient clipping
                 checkpoint_dir: str = "checkpoints",
                 model_filename_base: str = "handwriting_vaegan",
                 save_every_n_epochs: int = 5 # How often to save checkpoints
                ):
        """
        Initializes the Trainer.

        Args:
            model: The HandwritingVAEGAN model instance.
            train_loader: DataLoader for the training dataset.
            device: The torch device (CPU or CUDA).
            learning_rate_g: Learning rate for the Generator/VAE optimizer.
            learning_rate_d: Learning rate for the Discriminator optimizer.
            beta1: Adam optimizer beta1.
            beta2: Adam optimizer beta2.
            vae_weight: Weight for the VAE part (recon + KL) in the generator loss.
            gan_weight: Weight for the adversarial part in the generator loss.
            kl_weight: Weight for the KL divergence term within the VAE loss.
            reconstruction_loss_type: Type of loss for sequence reconstruction ('mse' or 'bce').
            grad_clip_value: Max norm for gradient clipping (None to disable).
            checkpoint_dir: Directory to save model checkpoints.
            model_filename_base: Base name for saved model files.
            save_every_n_epochs: Frequency of saving checkpoints.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.lr_g = learning_rate_g
        self.lr_d = learning_rate_d
        self.beta1 = beta1
        self.beta2 = beta2
        self.vae_weight = vae_weight
        self.gan_weight = gan_weight
        self.kl_weight = kl_weight
        self.recon_loss_type = reconstruction_loss_type
        self.grad_clip_value = grad_clip_value
        self.checkpoint_dir = checkpoint_dir
        self.model_filename_base = model_filename_base
        self.save_every_n_epochs = save_every_n_epochs

        # --- Define Optimizers ---
        # Generator optimizer optimizes Encoder and Decoder parameters
        g_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        self.optimizer_g = optim.Adam(g_params, lr=self.lr_g, betas=(self.beta1, self.beta2))

        # Discriminator optimizer optimizes only Discriminator parameters
        d_params = list(model.discriminator.parameters())
        self.optimizer_d = optim.Adam(d_params, lr=self.lr_d, betas=(self.beta1, self.beta2))

        logger.info("Trainer initialized.")
        logger.info(f" Optimizers: Adam (G: lr={self.lr_g}, D: lr={self.lr_d}, beta1={self.beta1})")
        logger.info(f" Loss Weights: VAE={self.vae_weight}, GAN={self.gan_weight}, KL={self.kl_weight}")
        logger.info(f" Recon Loss: {self.recon_loss_type.upper()}")
        if self.grad_clip_value: logger.info(f" Gradient Clipping: Enabled (value={self.grad_clip_value})")


    def _get_padding_mask(self, batch_sequences: torch.Tensor) -> torch.Tensor | None:
        """Creates padding mask based on EOS flag (assumed to be last feature)."""
        try:
            # Assumes EOS is the last feature in the sequence tensor
            eos_index = -1
            mask = create_padding_mask(batch_sequences, eos_index=eos_index)
            # Ensure mask is on the correct device
            return mask.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create padding mask: {e}. Proceeding without mask.", exc_info=True)
            return None

    def train_epoch(self, epoch_num: int) -> dict:
        """Runs a single training epoch."""
        self.model.train() # Set model to training mode
        epoch_losses = defaultdict(float)
        start_time = time.time()
        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            # --- Data Preparation ---
            # Expecting (sequence_tensor, label, glyph_tensor) from dataloader
            # We primarily need the sequence_tensor for training here
            real_sequences, _, _ = batch_data
            real_sequences = real_sequences.to(self.device)
            batch_size = real_sequences.size(0)

            # Create padding mask if needed (e.g., for Transformer or loss calculation)
            # Mask is True for valid steps, False for padding
            padding_mask = self._get_padding_mask(real_sequences)

            # --- Train Discriminator ---
            self.optimizer_d.zero_grad()

            # Real samples
            d_real_logits = self.model.discriminate(real_sequences, src_key_padding_mask=~padding_mask if padding_mask is not None else None) # Transformer mask is True for ignore

            # Fake samples
            # Generate latent vectors (either random noise or from encoder)
            with torch.no_grad(): # Don't track gradients for z generation here
                 # Option 1: Use random noise
                 z_noise = torch.randn(batch_size, self.model.latent_dim, device=self.device)
                 # Option 2: Use encoder output (potentially better starting point?)
                 # mu, logvar = self.model.encoder(real_sequences)
                 # z_encoded = self.model.reparameterize(mu, logvar)
                 # z_for_fake = z_encoded # Or z_noise
                 z_for_fake = z_noise

                 # Generate fake sequence and detach from generator's graph
                 fake_sequences = self.model.generate(z_for_fake).detach()

            d_fake_logits = self.model.discriminate(fake_sequences, src_key_padding_mask=None) # Assume generated sequences are full length

            # Calculate Discriminator loss
            loss_d = calculate_discriminator_loss(d_real_logits, d_fake_logits)

            # Backpropagate and update Discriminator
            loss_d.backward()
            # Optional gradient clipping for discriminator
            # if self.grad_clip_value: torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.grad_clip_value)
            self.optimizer_d.step()


            # --- Train Generator (Encoder + Decoder) ---
            self.optimizer_g.zero_grad()

            # VAE Forward Pass (calculate mu, logvar, z, and reconstruction)
            reconstructed_sequences, mu, logvar, z_vae = self.model(real_sequences, src_key_padding_mask=~padding_mask if padding_mask is not None else None)

            # VAE Losses
            loss_recon = calculate_reconstruction_loss(reconstructed_sequences, real_sequences, mask=padding_mask, loss_type=self.recon_loss_type)
            loss_kl = calculate_kl_divergence_loss(mu, logvar)

            # Adversarial Loss for Generator
            # Generate fake sequences *without detaching* from graph
            # Use z sampled during VAE pass (z_vae) or fresh noise (z_noise)? Using z_vae couples VAE/GAN more.
            fake_sequences_for_g = self.model.generate(z_vae)
            d_fake_logits_for_g = self.model.discriminate(fake_sequences_for_g, src_key_padding_mask=None) # Pass fake sequence to D

            loss_g_adv = calculate_generator_loss(d_fake_logits_for_g)

            # Combine Generator/VAE losses
            # loss_g_total = (self.vae_weight * loss_recon) + (self.vae_weight * self.kl_weight * loss_kl) + (self.gan_weight * loss_g_adv)
            loss_g_vae = self.vae_weight * (loss_recon + self.kl_weight * loss_kl)
            loss_g_gan = self.gan_weight * loss_g_adv
            loss_g_total = loss_g_vae + loss_g_gan

            # Backpropagate and update Generator (Encoder + Decoder)
            loss_g_total.backward()
            if self.grad_clip_value: torch.nn.utils.clip_grad_norm_(list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), self.grad_clip_value)
            self.optimizer_g.step()


            # --- Logging losses for this batch ---
            epoch_losses['loss_d'] += loss_d.item()
            epoch_losses['loss_g_total'] += loss_g_total.item()
            epoch_losses['loss_recon'] += loss_recon.item()
            epoch_losses['loss_kl'] += loss_kl.item()
            epoch_losses['loss_g_adv'] += loss_g_adv.item()

            if (batch_idx + 1) % 50 == 0: # Log progress every 50 batches
                 logger.debug(f" Epoch [{epoch_num+1}][{batch_idx+1}/{num_batches}] | "
                             f"Loss D: {loss_d.item():.4f} | Loss G: {loss_g_total.item():.4f} "
                             f"(Recon: {loss_recon.item():.4f}, KL: {loss_kl.item():.4f}, Adv: {loss_g_adv.item():.4f})")


        # --- Calculate average losses for the epoch ---
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        end_time = time.time()
        logger.info(f"Epoch [{epoch_num+1}] completed in {end_time - start_time:.2f}s.")
        logger.info(f" Average Losses: Loss D: {avg_losses['loss_d']:.4f} | Loss G: {avg_losses['loss_g_total']:.4f} "
                     f"(Recon: {avg_losses['loss_recon']:.4f}, KL: {avg_losses['loss_kl']:.4f}, Adv: {avg_losses['loss_g_adv']:.4f})")

        return avg_losses


    def save_checkpoint(self, epoch: int | str, is_best: bool = False):
        """Saves model checkpoint."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            # Include hyperparameters needed to reconstruct the model instance
            'model_hyperparams': self.model.get_hyperparameters() # Assumes model has this method
        }

        filename = f"{self.model_filename_base}_epoch_{epoch}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

        if is_best:
            best_filename = f"{self.model_filename_base}_best.pth"
            best_filepath = os.path.join(self.checkpoint_dir, best_filename)
            torch.save(state, best_filepath)
            logger.info(f"Best checkpoint updated: {best_filepath}")


    def train(self, num_epochs: int, start_epoch: int = 0):
        """Main training loop."""
        logger.info(f"Starting training from epoch {start_epoch + 1} for {num_epochs} epochs...")
        best_loss = float('inf') # Placeholder for tracking best model (e.g., based on recon loss)

        for epoch in range(start_epoch, num_epochs):
            try:
                epoch_losses = self.train_epoch(epoch)

                # --- Checkpoint Saving ---
                # Simple saving logic: save every N epochs and save final
                is_last_epoch = (epoch == num_epochs - 1)
                if (epoch + 1) % self.save_every_n_epochs == 0 or is_last_epoch:
                    self.save_checkpoint(epoch + 1)

                # Example: Save 'best' model based on reconstruction loss
                current_recon_loss = epoch_losses.get('loss_recon', float('inf'))
                if current_recon_loss < best_loss:
                    logger.info(f"New best reconstruction loss ({current_recon_loss:.4f}) achieved at epoch {epoch+1}. Saving best model.")
                    best_loss = current_recon_loss
                    self.save_checkpoint(epoch + 1, is_best=True)


            except TrainingError as e:
                 logger.error(f"Controlled training error occurred in epoch {epoch + 1}: {e}")
                 logger.info("Attempting to save final state before stopping...")
                 self.save_checkpoint(f"error_epoch_{epoch+1}")
                 raise e # Re-raise after saving
            except Exception as e:
                 logger.error(f"Unexpected error during epoch {epoch + 1}: {e}", exc_info=True)
                 logger.info("Attempting to save final state due to unexpected error...")
                 self.save_checkpoint(f"error_epoch_{epoch+1}")
                 # Re-raise a TrainingError to signal failure
                 raise TrainingError(f"Unexpected error stopped training at epoch {epoch + 1}") from e


        logger.info("Training finished.")
        # Final save is handled within the loop on the last epoch