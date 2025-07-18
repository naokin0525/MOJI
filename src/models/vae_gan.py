"""
Hybrid VAE-GAN Model for Handwriting Generation.

This file defines the main deep learning architecture, combining a
Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN).

- The Encoder (part of VAE) uses an RNN to compress a stroke sequence into a latent vector.
- The Generator/Decoder (part of VAE and GAN) uses an RNN to generate a stroke sequence
  from a latent vector.
- The Discriminator (part of GAN) uses an RNN to distinguish between real and
  generated stroke sequences.
"""

import torch
import torch.nn as nn
from src.models.rnn import SequenceModel
from src.config import INPUT_DIM, LATENT_DIM, RNN_HIDDEN_DIM, RNN_NUM_LAYERS, RNN_DROPOUT

class Encoder(nn.Module):
    """Encodes a sequence of strokes into a latent space representation."""
    def __init__(self):
        super(Encoder, self).__init__()
        self.rnn = SequenceModel(INPUT_DIM, RNN_HIDDEN_DIM, RNN_NUM_LAYERS, RNN_DROPOUT)
        # Bidirectional LSTM means hidden_dim * 2
        self.fc_mu = nn.Linear(RNN_HIDDEN_DIM * 2, LATENT_DIM)
        self.fc_logvar = nn.Linear(RNN_HIDDEN_DIM * 2, LATENT_DIM)

    def forward(self, x, lengths):
        _, (hidden, _) = self.rnn(x, lengths)
        # Concatenate the final hidden states from both directions
        hidden_forward_backward = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        mu = self.fc_mu(hidden_forward_backward)
        logvar = self.fc_logvar(hidden_forward_backward)
        return mu, logvar

class Generator(nn.Module):
    """Generates a sequence of strokes from a latent vector."""
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_to_hidden = nn.Linear(LATENT_DIM, RNN_HIDDEN_DIM * RNN_NUM_LAYERS)
        self.rnn = nn.LSTM(INPUT_DIM, RNN_HIDDEN_DIM, RNN_NUM_LAYERS, batch_first=True)
        self.fc_out = nn.Linear(RNN_HIDDEN_DIM, INPUT_DIM)

    def forward(self, z, max_seq_len):
        # Prepare initial hidden and cell states from the latent vector
        hidden = self.latent_to_hidden(z).view(RNN_NUM_LAYERS, z.size(0), RNN_HIDDEN_DIM)
        cell = torch.zeros_like(hidden) # Start with zero cell state

        # Initial input point (start of sequence token)
        batch_size = z.size(0)
        input_point = torch.zeros(batch_size, 1, INPUT_DIM, device=z.device)
        input_point[:, :, 3] = 1 # Set pen_down to 1 to start drawing

        outputs = []
        for _ in range(max_seq_len):
            output, (hidden, cell) = self.rnn(input_point, (hidden, cell))
            output_point = self.fc_out(output)
            outputs.append(output_point)
            input_point = output_point # Use the output as the next input (autoregressive)

        return torch.cat(outputs, dim=1)

class Discriminator(nn.Module):
    """Distinguishes between real and generated stroke sequences."""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.rnn = SequenceModel(INPUT_DIM, RNN_HIDDEN_DIM, RNN_NUM_LAYERS, RNN_DROPOUT)
        # Bidirectional LSTM output * 2
        self.fc = nn.Sequential(
            nn.Linear(RNN_HIDDEN_DIM * 2, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid() # Output a probability
        )

    def forward(self, x, lengths):
        rnn_output, _ = self.rnn(x, lengths)
        # Use the output of the last valid time step for classification
        # Gather the last relevant output based on lengths
        last_outputs = rnn_output[torch.arange(len(lengths)), lengths - 1]
        validity = self.fc(last_outputs)
        return validity

class VAE_GAN(nn.Module):
    """The main class that combines the VAE and GAN components."""
    def __init__(self):
        super(VAE_GAN, self).__init__()
        self.encoder = Encoder()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to allow backpropagation through a random node."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, lengths):
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        # Generate a sequence of the same max length as the input batch
        reconstructed_x = self.generator(z, x.size(1))
        return reconstructed_x, mu, logvar