# src/models/discriminator.py
"""
Discriminator component of the GAN for handwriting sequences.

Classifies a sequence as real or fake.
"""

import torch
import torch.nn as nn

try:
    from .sequence_model import StrokeRNN, StrokeTransformerEncoder
except ImportError:
    class StrokeRNN(nn.Module): pass
    class StrokeTransformerEncoder(nn.Module): pass


class StrokeSequenceDiscriminator(nn.Module):
    """
    Discriminates between real and generated stroke sequences.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int, # Used for RNN hidden size or Transformer model dim
                 sequence_model_type: str = 'rnn', # 'rnn' or 'transformer'
                 rnn_type: str = 'LSTM', # if sequence_model_type is 'rnn'
                 num_layers: int = 2,    # if sequence_model_type is 'rnn' / transformer
                 num_heads: int = 8,     # if sequence_model_type is 'transformer'
                 dim_feedforward: int = 512, # if sequence_model_type is 'transformer'
                 dropout: float = 0.1,
                 bidirectional_discriminator: bool = True, # Usually good for discrimination
                 max_seq_len: int = 512 # Needed for Transformer positional encoding
                ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_model_type = sequence_model_type.lower()

        # Sequence processing layer
        if self.sequence_model_type == 'rnn':
            self.sequence_processor = StrokeRNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rnn_type=rnn_type,
                dropout=dropout,
                bidirectional=bidirectional_discriminator
            )
            processor_output_dim = hidden_dim * (2 if bidirectional_discriminator else 1)
        elif self.sequence_model_type == 'transformer':
            self.sequence_processor = StrokeTransformerEncoder(
                input_dim=input_dim,
                model_dim=hidden_dim, # Use hidden_dim as model_dim
                num_heads=num_heads,
                num_encoder_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            processor_output_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported sequence_model_type: {sequence_model_type}")

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(processor_output_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1) # Output a single logit (real/fake score)
            # No sigmoid here, typically handled by BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input sequence tensor (batch, seq_len, input_dim).
            src_key_padding_mask (torch.Tensor, optional): Padding mask for Transformer. (batch, seq_len).

        Returns:
            torch.Tensor: Logits indicating real/fake probability (batch, 1).
        """
        batch_size = x.size(0)

        # Process sequence
        if self.sequence_model_type == 'rnn':
            # Use the final hidden state like in the encoder
            _, hidden_state = self.sequence_processor(x)
            if isinstance(hidden_state, tuple): h_n = hidden_state[0] # LSTM
            else: h_n = hidden_state # GRU

            if self.sequence_processor.rnn.bidirectional:
                 h_n = h_n.view(self.sequence_processor.num_layers, 2, batch_size, self.hidden_dim)[-1]
                 final_representation = torch.cat((h_n[0], h_n[1]), dim=1)
            else:
                 final_representation = h_n[-1]
            # `final_representation` shape: (batch, hidden_dim * num_directions)

        elif self.sequence_model_type == 'transformer':
            # Use mean pooling like in the encoder
            processed_seq = self.sequence_processor(x, src_key_padding_mask=src_key_padding_mask)
            if src_key_padding_mask is not None:
                 inverted_mask = ~src_key_padding_mask.unsqueeze(-1).expand_as(processed_seq)
                 masked_seq = processed_seq * inverted_mask
                 summed = masked_seq.sum(dim=1)
                 count = inverted_mask.sum(dim=1).clamp(min=1)
                 final_representation = summed / count
            else:
                 final_representation = processed_seq.mean(dim=1)
             # `final_representation` shape: (batch, model_dim)

        # Classify
        logits = self.classification_head(final_representation) # (batch, 1)
        return logits