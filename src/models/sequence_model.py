# src/models/sequence_model.py
"""
Sequence processing modules (RNN and Transformer) for handwriting strokes.
"""

import torch
import torch.nn as nn
import math

class StrokeRNN(nn.Module):
    """
    A standard RNN layer (LSTM or GRU) for processing stroke sequences.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 rnn_type: str = 'LSTM', # 'LSTM' or 'GRU'
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.dropout = dropout if num_layers > 1 else 0 # Dropout only between layers

        RNNClass = getattr(nn, rnn_type.upper())

        self.rnn = RNNClass(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=self.dropout,
            bidirectional=bidirectional,
            batch_first=True # Expect input as (batch, seq_len, features)
        )

    def forward(self, x: torch.Tensor, h_0: torch.Tensor | None = None, c_0: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple]:
        """
        Forward pass through the RNN.

        Args:
            x (torch.Tensor): Input sequence tensor (batch, seq_len, input_dim).
            h_0 (torch.Tensor, optional): Initial hidden state. Defaults to zeros.
            c_0 (torch.Tensor, optional): Initial cell state (for LSTM). Defaults to zeros.

        Returns:
            tuple[torch.Tensor, tuple]:
                - output (torch.Tensor): Output sequence (batch, seq_len, hidden_dim * num_directions).
                - hidden_state (tuple): Final hidden state(s) (h_n, c_n for LSTM).
        """
        # Initialize hidden state if not provided
        num_directions = 2 if self.bidirectional else 1
        batch_size = x.size(0)
        device = x.device

        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        initial_state = h_0

        if self.rnn_type.upper() == 'LSTM':
            if c_0 is None:
                c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
            initial_state = (h_0, c_0)

        # Pass through RNN
        # PackedSequence could be used here for variable lengths, but assumes padding is handled externally for now
        output, hidden_state = self.rnn(x, initial_state)

        return output, hidden_state


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Use batch dim 1 for broadcasting
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer so it's part of state_dict but not parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x = x + self.pe[:, :x.size(1)] # Add positional encoding
        # Updated for batch dimension:
        x = x + self.pe[:, :x.size(1), :].to(x.device) # Ensure PE is on correct device
        return self.dropout(x)


class StrokeTransformerEncoder(nn.Module):
    """
    A Transformer Encoder layer for processing stroke sequences.
    """
    def __init__(self,
                 input_dim: int,
                 model_dim: int, # d_model
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 512 # Needed for positional encoding
                 ):
        super().__init__()
        self.model_dim = model_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Expect input as (batch, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through the Transformer Encoder.

        Args:
            src (torch.Tensor): Input sequence tensor (batch, seq_len, input_dim).
            src_key_padding_mask (torch.Tensor, optional): Mask indicating padding tokens
                                                            (batch, seq_len). True where padded.

        Returns:
            torch.Tensor: Encoded sequence (batch, seq_len, model_dim).
        """
        src = self.input_projection(src) * math.sqrt(self.model_dim) # Project and scale
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output

# Note: Transformer Decoder is omitted here but would be needed for a full Transformer VAE/GAN.
# The current plan uses an RNN Decoder.