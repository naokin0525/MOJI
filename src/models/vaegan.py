# src/models/vaegan.py
"""
Main VAE-GAN hybrid model for handwriting generation.

Combines the Encoder, Decoder/Generator, and Discriminator components.
"""

import torch
import torch.nn as nn
import logging

try:
    from .encoder import StrokeEncoderVAE
    from .decoder import StrokeDecoderGenerator
    from .discriminator import StrokeSequenceDiscriminator
except ImportError:
    # Fallbacks for partial imports
    class StrokeEncoderVAE(nn.Module): pass
    class StrokeDecoderGenerator(nn.Module): pass
    class StrokeSequenceDiscriminator(nn.Module): pass

logger = logging.getLogger(__name__)

class HandwritingVAEGAN(nn.Module):
    """
    VAE-GAN model combining Encoder, Decoder/Generator, and Discriminator.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dim: int, # Shared hidden dim for components, can be overridden
                 output_dim: int | None = None, # Defaults to input_dim if None
                 sequence_model_type: str = 'rnn',
                 rnn_type: str = 'LSTM', # Used if sequence_model_type is 'rnn'
                 num_layers: int = 2, # Shared layer count, can be overridden
                 num_heads: int = 8, # Used if sequence_model_type is 'transformer'
                 dim_feedforward: int = 512, # Used if sequence_model_type is 'transformer'
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 bidirectional_encoder: bool = True,
                 bidirectional_discriminator: bool = True,
                 # Allow overriding specific component args if needed
                 encoder_args: dict | None = None,
                 decoder_args: dict | None = None,
                 discriminator_args: dict | None = None
                ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim # Typically input and output dims match

        # --- Default arguments for components ---
        default_encoder_args = {
            'input_dim': input_dim, 'hidden_dim': hidden_dim, 'latent_dim': latent_dim,
            'sequence_model_type': sequence_model_type, 'rnn_type': rnn_type,
            'num_layers': num_layers, 'num_heads': num_heads, 'dim_feedforward': dim_feedforward,
            'dropout': dropout, 'bidirectional_encoder': bidirectional_encoder, 'max_seq_len': max_seq_len
        }
        default_decoder_args = {
            'latent_dim': latent_dim, 'output_dim': output_dim, 'hidden_dim': hidden_dim,
            'max_seq_len': max_seq_len, 'rnn_type': rnn_type, # Assuming RNN decoder
            'num_layers': num_layers, 'dropout': dropout
        }
        default_discriminator_args = {
            'input_dim': output_dim, # Discriminator sees generated/real sequences
            'hidden_dim': hidden_dim, 'sequence_model_type': sequence_model_type,
            'rnn_type': rnn_type, 'num_layers': num_layers, 'num_heads': num_heads,
            'dim_feedforward': dim_feedforward, 'dropout': dropout,
            'bidirectional_discriminator': bidirectional_discriminator, 'max_seq_len': max_seq_len
        }

        # --- Update args with overrides ---
        final_encoder_args = default_encoder_args.copy()
        if encoder_args: final_encoder_args.update(encoder_args)

        final_decoder_args = default_decoder_args.copy()
        if decoder_args: final_decoder_args.update(decoder_args)

        final_discriminator_args = default_discriminator_args.copy()
        if discriminator_args: final_discriminator_args.update(discriminator_args)


        # --- Instantiate Components ---
        self.encoder = StrokeEncoderVAE(**final_encoder_args)
        logger.info(f"Initialized Encoder with args: {final_encoder_args}")

        self.decoder = StrokeDecoderGenerator(**final_decoder_args)
        logger.info(f"Initialized Decoder/Generator with args: {final_decoder_args}")

        self.discriminator = StrokeSequenceDiscriminator(**final_discriminator_args)
        logger.info(f"Initialized Discriminator with args: {final_discriminator_args}")

        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + epsilon * std
        epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> tuple:
        """
        Defines the forward pass for VAE training (reconstruction).

        Args:
            x (torch.Tensor): Input sequence (batch, seq_len, input_dim).
            src_key_padding_mask (torch.Tensor, optional): Padding mask for Transformer Encoder.

        Returns:
            tuple: Contains:
                - reconstructed_x (torch.Tensor): Output sequence from decoder (batch, max_seq_len, output_dim).
                - mu (torch.Tensor): Latent mean (batch, latent_dim).
                - logvar (torch.Tensor): Latent log variance (batch, latent_dim).
                - z (torch.Tensor): Sampled latent vector (batch, latent_dim).
        """
        mu, logvar = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        z = self.reparameterize(mu, logvar)
        # Use teacher forcing for reconstruction by passing original sequence 'x'
        reconstructed_x = self.decoder(z, target_sequence=x)
        return reconstructed_x, mu, logvar, z

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generates sequences from given latent vectors using the decoder/generator.

        Args:
            z (torch.Tensor): Latent vectors (batch, latent_dim).

        Returns:
            torch.Tensor: Generated sequences (batch, max_seq_len, output_dim).
        """
        # Call decoder without target sequence for autoregressive generation
        return self.decoder(z, target_sequence=None)

    def discriminate(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Passes sequences through the discriminator.

        Args:
            x (torch.Tensor): Input sequence (batch, seq_len, input_dim/output_dim).
            src_key_padding_mask (torch.Tensor, optional): Padding mask for Transformer Discriminator.

        Returns:
            torch.Tensor: Discriminator logits (batch, 1).
        """
        return self.discriminator(x, src_key_padding_mask=src_key_padding_mask)

    def get_hyperparameters(self) -> dict:
         """
         Returns a dictionary of hyperparameters needed to reconstruct this model.
         Useful for saving along with the state_dict.
         """
         # Collect args used for initialization (assuming they are stored or accessible)
         # This requires careful tracking of the actual args used.
         # Example structure:
         return {
             # These need to be accurately retrieved from the actual init args
             'input_dim': self.encoder.input_dim, # Example access
             'latent_dim': self.latent_dim,
             'hidden_dim': self.encoder.hidden_dim, # Assuming shared hidden_dim access
             'output_dim': self.decoder.output_dim,
             'sequence_model_type': self.encoder.sequence_model_type, # Assuming consistent type
             'rnn_type': getattr(self.encoder.sequence_encoder, 'rnn_type', None) if self.encoder.sequence_model_type == 'rnn' else None,
             'num_layers': getattr(self.encoder.sequence_encoder, 'num_layers', None), # Need consistent access
             'num_heads': getattr(self.encoder.sequence_encoder, 'transformer_encoder', {}).layers[0].self_attn.num_heads if self.encoder.sequence_model_type == 'transformer' else None, # Example complex access
             'dim_feedforward': getattr(self.encoder.sequence_encoder, 'transformer_encoder', {}).layers[0].linear1.out_features if self.encoder.sequence_model_type == 'transformer' else None, # Example
             'dropout': getattr(self.encoder.sequence_encoder, 'dropout', 0.1), # Need consistent access
             'max_seq_len': self.decoder.max_seq_len,
             'bidirectional_encoder': getattr(self.encoder.sequence_encoder, 'bidirectional', None) if self.encoder.sequence_model_type == 'rnn' else None,
             'bidirectional_discriminator': getattr(self.discriminator.sequence_processor, 'bidirectional', None) if self.discriminator.sequence_model_type == 'rnn' else None,
             # Note: Accessing nested attributes like this is fragile.
             # It's better to store the init args dicts directly in __init__.
         }