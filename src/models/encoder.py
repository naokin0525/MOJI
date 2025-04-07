# src/models/encoder.py
"""
Encoder component of the VAE for handwriting sequences.

Encodes a sequence into parameters (mean, log-variance) of a latent distribution.
"""

import torch
import torch.nn as nn

try:
    from .sequence_model import StrokeRNN, StrokeTransformerEncoder
except ImportError:
    # Fallback for potential circular or partial imports during setup
    class StrokeRNN(nn.Module): pass
    class StrokeTransformerEncoder(nn.Module): pass

class StrokeEncoderVAE(nn.Module):
    """
    Encodes a stroke sequence tensor into latent space parameters (mu, logvar).
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int, # Used for RNN hidden size or Transformer model dim
                 latent_dim: int,
                 sequence_model_type: str = 'rnn', # 'rnn' or 'transformer'
                 rnn_type: str = 'LSTM', # if sequence_model_type is 'rnn'
                 num_layers: int = 2,    # if sequence_model_type is 'rnn'
                 num_heads: int = 8,     # if sequence_model_type is 'transformer'
                 dim_feedforward: int = 512, # if sequence_model_type is 'transformer'
                 dropout: float = 0.1,
                 bidirectional_encoder: bool = True, # Often beneficial for encoders
                 max_seq_len: int = 512 # Needed for Transformer positional encoding
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_model_type = sequence_model_type.lower()

        if self.sequence_model_type == 'rnn':
            self.sequence_encoder = StrokeRNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rnn_type=rnn_type,
                dropout=dropout,
                bidirectional=bidirectional_encoder
            )
            encoder_output_dim = hidden_dim * (2 if bidirectional_encoder else 1)
        elif self.sequence_model_type == 'transformer':
            self.sequence_encoder = StrokeTransformerEncoder(
                input_dim=input_dim,
                model_dim=hidden_dim, # Use hidden_dim as model_dim for transformer
                num_heads=num_heads,
                num_encoder_layers=num_layers, # Re-use num_layers for transformer layers
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            encoder_output_dim = hidden_dim # Transformer output dim is model_dim
        else:
            raise ValueError(f"Unsupported sequence_model_type: {sequence_model_type}")

        # Layers to map sequence representation to latent space parameters
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input sequence tensor (batch, seq_len, input_dim).
            src_key_padding_mask (torch.Tensor, optional): Padding mask for Transformer. (batch, seq_len).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - mu (torch.Tensor): Mean of the latent distribution (batch, latent_dim).
                - logvar (torch.Tensor): Log variance of the latent distribution (batch, latent_dim).
        """
        batch_size = x.size(0)

        # Pass through sequence encoder
        if self.sequence_model_type == 'rnn':
            # Use the final hidden state of the last layer
            # output: (batch, seq_len, hidden*dirs), hidden_state: (h_n, c_n) or h_n
            # h_n shape: (num_layers * num_dirs, batch, hidden_dim)
            _, hidden_state = self.sequence_encoder(x)
            if isinstance(hidden_state, tuple): # LSTM case (h_n, c_n)
                h_n = hidden_state[0]
            else: # GRU case h_n
                h_n = hidden_state

            # Concatenate final states of forward and backward RNNs if bidirectional
            if self.sequence_encoder.rnn.bidirectional:
                 # h_n is (num_layers*2, batch, hidden_dim), we want final layer -> (-2, -1)
                 # Reshape to (num_layers, 2, batch, hidden) and take last layer [ -1]
                 h_n = h_n.view(self.sequence_encoder.num_layers, 2, batch_size, self.hidden_dim)[-1]
                 # Concatenate forward and backward states -> (batch, hidden_dim * 2)
                 final_hidden = torch.cat((h_n[0], h_n[1]), dim=1)
            else:
                # h_n is (num_layers, batch, hidden_dim), take last layer [-1]
                final_hidden = h_n[-1]
            # `final_hidden` shape: (batch, hidden_dim * num_directions)

        elif self.sequence_model_type == 'transformer':
            # Use mean pooling over sequence dimension as representation
            encoded_seq = self.sequence_encoder(x, src_key_padding_mask=src_key_padding_mask) # (batch, seq_len, model_dim)
            # Apply mask before pooling if provided
            if src_key_padding_mask is not None:
                 # Invert mask: True means keep, False means mask out
                 inverted_mask = ~src_key_padding_mask.unsqueeze(-1).expand_as(encoded_seq)
                 masked_encoded_seq = encoded_seq * inverted_mask
                 # Sum over sequence dimension and divide by actual sequence length
                 summed = masked_encoded_seq.sum(dim=1)
                 non_padding_count = inverted_mask.sum(dim=1)
                 non_padding_count = non_padding_count.clamp(min=1) # Avoid division by zero
                 final_hidden = summed / non_padding_count
            else:
                 final_hidden = encoded_seq.mean(dim=1) # (batch, model_dim)
            # Alternative: Use a [CLS] token if added to the input sequence


        # Calculate mu and logvar
        mu = self.fc_mu(final_hidden)
        logvar = self.fc_logvar(final_hidden)

        return mu, logvar