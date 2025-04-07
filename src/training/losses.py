# src/training/losses.py
"""
Loss functions for training the VAE-GAN handwriting model.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def calculate_reconstruction_loss(predicted_seq: torch.Tensor,
                                  target_seq: torch.Tensor,
                                  mask: torch.Tensor | None = None,
                                  loss_type: str = 'mse') -> torch.Tensor:
    """
    Calculates the reconstruction loss between predicted and target sequences.

    Args:
        predicted_seq (torch.Tensor): Predicted sequence (batch, seq_len, features).
        target_seq (torch.Tensor): Ground truth sequence (batch, seq_len, features).
        mask (torch.Tensor, optional): Boolean mask (batch, seq_len) where True indicates
                                      valid (non-padded) time steps. Defaults to None (no masking).
        loss_type (str): Type of loss ('mse' or 'bce'). Defaults to 'mse'.

    Returns:
        torch.Tensor: Scalar tensor representing the mean reconstruction loss.
    """
    batch_size, seq_len, features = target_seq.shape
    if predicted_seq.shape != target_seq.shape:
         # Pad or truncate predicted_seq if necessary, though they should match from decoder
         pred_len = predicted_seq.shape[1]
         if pred_len > seq_len:
             predicted_seq = predicted_seq[:, :seq_len, :]
         elif pred_len < seq_len:
              padding = torch.zeros(batch_size, seq_len - pred_len, features, device=predicted_seq.device)
              predicted_seq = torch.cat([predicted_seq, padding], dim=1)

    if loss_type.lower() == 'mse':
        # Calculate element-wise MSE loss without reduction
        loss = F.mse_loss(predicted_seq, target_seq, reduction='none') # (batch, seq_len, features)
    elif loss_type.lower() == 'bce':
        # Assumes predicted_seq contains logits, target_seq contains probabilities (0 or 1)
        loss = F.binary_cross_entropy_with_logits(predicted_seq, target_seq, reduction='none')
    else:
        raise ValueError(f"Unsupported reconstruction loss type: {loss_type}")

    # Sum loss across the feature dimension
    loss = loss.sum(dim=-1) # (batch, seq_len)

    if mask is not None:
        if mask.shape != loss.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match loss shape {loss.shape}")
        # Apply mask: loss is calculated only for valid time steps
        loss = loss * mask.float() # Zero out loss for padded steps
        # Calculate mean loss over valid steps per sequence
        # Avoid division by zero if a sequence has no valid steps (shouldn't happen with EOS handling)
        num_valid_steps = mask.sum(dim=1).clamp(min=1)
        mean_loss_per_sequence = loss.sum(dim=1) / num_valid_steps
    else:
        # If no mask, average over all time steps
        mean_loss_per_sequence = loss.mean(dim=1) # (batch)

    # Return the mean loss across the batch
    return mean_loss_per_sequence.mean()


def calculate_kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Calculates the KL divergence loss between the learned latent distribution N(mu, var)
    and a standard normal distribution N(0, 1).

    Args:
        mu (torch.Tensor): Mean of the latent distribution (batch, latent_dim).
        logvar (torch.Tensor): Log variance of the latent distribution (batch, latent_dim).

    Returns:
        torch.Tensor: Scalar tensor representing the mean KL divergence loss.
    """
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where log(sigma^2) = logvar
    #       sigma^2 = exp(logvar)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # Sum over latent_dim
    return kl_div.mean() # Average over batch dimension


def calculate_discriminator_loss(d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates the standard GAN discriminator loss (non-saturating).

    Args:
        d_real_logits (torch.Tensor): Discriminator output logits for real samples (batch, 1).
        d_fake_logits (torch.Tensor): Discriminator output logits for fake samples (batch, 1).

    Returns:
        torch.Tensor: Scalar tensor representing the discriminator loss.
    """
    # Real samples loss: -log(D(x)) -> minimize BCE(logits, 1s)
    real_loss = F.binary_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits))

    # Fake samples loss: -log(1 - D(G(z))) -> minimize BCE(logits, 0s)
    fake_loss = F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))

    # Total discriminator loss
    d_loss = (real_loss + fake_loss) / 2.0
    return d_loss


def calculate_generator_loss(d_fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates the standard GAN generator loss (non-saturating).
    Generator aims to make discriminator classify fake samples as real.

    Args:
        d_fake_logits (torch.Tensor): Discriminator output logits for fake samples generated
                                     by the generator (batch, 1).

    Returns:
        torch.Tensor: Scalar tensor representing the generator's adversarial loss.
    """
    # Generator loss: -log(D(G(z))) -> minimize BCE(logits, 1s)
    g_loss = F.binary_cross_entropy_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))
    return g_loss

# --- Helper to create padding mask based on EOS flag ---
def create_padding_mask(sequence_tensor: torch.Tensor, eos_index: int = -1) -> torch.Tensor:
    """
    Creates a boolean padding mask based on the End-of-Sequence (EOS) flag.
    Assumes EOS flag is 1.0 at the last valid step and 0.0 otherwise.
    Padded steps after the first EOS should also have EOS=0.0.

    Args:
        sequence_tensor (torch.Tensor): Input tensor (batch, seq_len, features).
        eos_index (int): Index of the EOS flag in the feature dimension. Defaults to -1.

    Returns:
        torch.Tensor: Boolean mask (batch, seq_len), True for valid steps, False for padding.
    """
    batch_size, seq_len, _ = sequence_tensor.shape
    # Find the index of the first occurrence of EOS=1 along seq_len dimension
    eos_flags = sequence_tensor[:, :, eos_index] # (batch, seq_len)
    # Check if any EOS flag is set (handle all-padding case)
    has_eos = torch.any(eos_flags > 0.5, dim=1) # (batch,) - Use threshold for float comparison
    # Find the index of the first EOS flag (argmax returns first max index)
    # Add 1 to include the EOS step itself. Result shape (batch,)
    # Add small epsilon to handle potential all-zero case safely before argmax
    first_eos_idx = torch.argmax(eos_flags + 1e-9, dim=1) + 1

    # Create range tensor (0, 1, ..., seq_len-1) for comparison
    indices = torch.arange(seq_len, device=sequence_tensor.device).unsqueeze(0).expand(batch_size, -1) # (batch, seq_len)

    # Mask is True where index < first_eos_idx
    mask = indices < first_eos_idx.unsqueeze(1)

    # If a sequence had no EOS flag at all, the argmax defaults to 0, making first_eos_idx=1.
    # We need to ensure these sequences get a mask of all False (or handle them specifically).
    # Let's ensure that sequences without EOS are masked entirely if that's desired,
    # or treated as full length if padding wasn't based on EOS.
    # Assuming sequences *should* have an EOS, mask out sequences where none was found.
    mask = mask & has_eos.unsqueeze(1)

    return mask # (batch, seq_len)