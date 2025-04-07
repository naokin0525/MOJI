# src/models/decoder.py
"""
Decoder component of the VAE / Generator component of the GAN.

Decodes a latent vector `z` into a sequence of stroke points.
Currently implemented using an RNN (LSTM/GRU).
"""

import torch
import torch.nn as nn
import logging

try:
    from .sequence_model import StrokeRNN # Only need RNN for this implementation
except ImportError:
    class StrokeRNN(nn.Module): pass

logger = logging.getLogger(__name__)

class StrokeDecoderGenerator(nn.Module):
    """
    Decodes a latent vector into a stroke sequence tensor using an RNN.
    Acts as VAE Decoder and GAN Generator.
    """
    def __init__(self,
                 latent_dim: int,
                 output_dim: int, # Dimension of each point in the output sequence
                 hidden_dim: int,
                 max_seq_len: int,
                 rnn_type: str = 'LSTM',
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 # sos_token_embedding_dim: int = None # Optional explicit start token
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # Layer to map latent vector z to initial RNN hidden state
        # RNN state shape: (num_layers * num_directions, batch, hidden_dim)
        # We use num_directions=1 for decoder RNN.
        self.fc_init_hidden = nn.Linear(latent_dim, num_layers * hidden_dim)
        if rnn_type.upper() == 'LSTM':
            self.fc_init_cell = nn.Linear(latent_dim, num_layers * hidden_dim)

        # The RNN decoder layer
        # Input dim to RNN should match output dim unless using separate embedding for input tokens
        # For direct prediction, input_dim = output_dim
        self.decoder_rnn = StrokeRNN(
            input_dim=output_dim, # Takes previous predicted point as input
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout,
            bidirectional=False # Decoder RNN is typically unidirectional
        )

        # Final linear layer to map RNN output to point features
        self.fc_output = nn.Linear(hidden_dim, output_dim)

        # Optional: Define a start-of-sequence (SOS) token if needed for generation start
        # self.sos_token = nn.Parameter(torch.randn(1, 1, output_dim)) # Learnable SOS token


    def forward(self, z: torch.Tensor, target_sequence: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            z (torch.Tensor): Latent vectors (batch, latent_dim).
            target_sequence (torch.Tensor, optional): Ground truth sequence for teacher forcing
                                                      during VAE training (batch, seq_len, output_dim).
                                                      If None, generates sequence via sampling.

        Returns:
            torch.Tensor: Generated or reconstructed sequence (batch, max_seq_len, output_dim).
        """
        batch_size = z.size(0)
        device = z.device

        # 1. Initialize RNN state from latent vector z
        hidden = self.fc_init_hidden(z) # (batch, num_layers * hidden_dim)
        # Reshape to (num_layers, batch, hidden_dim)
        hidden = hidden.view(self.num_layers, batch_size, self.hidden_dim)

        if self.decoder_rnn.rnn_type.upper() == 'LSTM':
            cell = self.fc_init_cell(z)
            cell = cell.view(self.num_layers, batch_size, self.hidden_dim)
            hidden_state = (hidden, cell)
        else: # GRU
            hidden_state = hidden

        # 2. Generation process
        outputs = []

        # Determine sequence length for generation/teacher forcing
        seq_len = self.max_seq_len
        if target_sequence is not None:
            # Use target sequence length for teacher forcing, up to max_seq_len
            seq_len = min(target_sequence.size(1), self.max_seq_len)
            # Prepare teacher forcing inputs (shifted right, with SOS token prepended)
            # Simple approach: use target points directly as inputs to predict the *next* point
            rnn_input_sequence = target_sequence[:, :seq_len-1, :] # Use points 0 to L-2 to predict 1 to L-1
             # Prepend a start token (e.g., zeros or a learned SOS)
            start_token = torch.zeros(batch_size, 1, self.output_dim, device=device)
            # start_token = self.sos_token.expand(batch_size, -1, -1) # If using learned SOS
            rnn_input_sequence = torch.cat([start_token, rnn_input_sequence], dim=1)


        # --- Teacher Forcing Path (Training) ---
        if target_sequence is not None:
            # Process the whole sequence at once using teacher forcing
            rnn_output, _ = self.decoder_rnn(rnn_input_sequence, *hidden_state if isinstance(hidden_state, tuple) else (hidden_state,))
            # rnn_output shape: (batch, seq_len, hidden_dim)

            # Map RNN output to predicted point features
            predictions = self.fc_output(rnn_output) # (batch, seq_len, output_dim)

            # Pad if target sequence was shorter than max_seq_len
            if predictions.size(1) < self.max_seq_len:
                 padding_len = self.max_seq_len - predictions.size(1)
                 padding = torch.zeros(batch_size, padding_len, self.output_dim, device=device)
                 # Mark end-of-sequence on the padding? Or handle in loss function via masks.
                 # Let's assume loss handles padding.
                 predictions = torch.cat([predictions, padding], dim=1)

            return predictions

        # --- Autoregressive Generation Path (Sampling/Inference) ---
        else:
            # Initialize input with start token (e.g., zeros)
            current_input = torch.zeros(batch_size, 1, self.output_dim, device=device)
            # current_input = self.sos_token.expand(batch_size, -1, -1) # If using learned SOS

            for t in range(self.max_seq_len):
                # Pass current input and hidden state through one step of RNN
                rnn_output, hidden_state = self.decoder_rnn(current_input, *hidden_state if isinstance(hidden_state, tuple) else (hidden_state,))
                # rnn_output shape: (batch, 1, hidden_dim)

                # Map RNN output to prediction for the next point
                prediction = self.fc_output(rnn_output) # (batch, 1, output_dim)
                outputs.append(prediction)

                # Use the prediction as the input for the next step
                current_input = prediction

                # Optional: Check if an end-of-sequence state was predicted and stop early
                # requires specific handling of state dimensions in output_dim
                # e.g., if output_dim[-1] is EOS probability > threshold -> break

            # Concatenate all predicted steps
            generated_sequence = torch.cat(outputs, dim=1) # (batch, max_seq_len, output_dim)
            return generated_sequence