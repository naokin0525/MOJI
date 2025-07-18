"""
Recurrent Neural Network (RNN) for Sequential Stroke Modeling.

This script defines the core sequence processing model, which can be used
as a building block for both the encoder and decoder of the VAE. It uses an
LSTM to handle the variable-length sequences of stroke points.
"""

import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    """
    A sequence model based on an LSTM for processing handwriting stroke data.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        """
        Initializes the SequenceModel.

        Args:
            input_dim (int): The number of features for each point in the sequence.
                             e.g., 5 for (dx, dy, pressure, pen_down, end_of_stroke).
            hidden_dim (int): The number of features in the hidden state of the LSTM.
            num_layers (int): The number of recurrent layers.
            dropout (float): Dropout probability.
        """
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input/output tensors are (batch, seq, feature)
            dropout=dropout,
            bidirectional=True # Bidirectional processing can capture context better
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x, lengths):
        """
        Forward pass for the sequence model.

        Args:
            x (torch.Tensor): A batch of padded stroke sequences.
                              Shape: (batch_size, seq_len, input_dim).
            lengths (torch.Tensor): A tensor of the original sequence lengths
                                    before padding. Shape: (batch_size,).

        Returns:
            torch.Tensor: The output features from the LSTM.
                          Shape: (batch_size, seq_len, hidden_dim * 2).
        """
        # Pack padded sequence to handle variable lengths efficiently
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Forward pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_x)

        # Unpack the sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        return output, (hidden, cell)