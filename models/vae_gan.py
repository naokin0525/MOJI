import torch
import torch.nn as nn

class HandwritingModel(nn.Module):
    def __init__(self, char_vocab_size, embedding_dim, hidden_dim, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim + 4, hidden_dim, batch_first=True)  # 4 = (dx, dy, pressure, pen_up)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 4)  # Predict dx, dy, pressure, pen_up

    def encode(self, char_emb, sequence):
        inputs = torch.cat((char_emb.unsqueeze(1).repeat(1, sequence.size(1), 1), sequence), dim=2)
        _, (h, _) = self.encoder(inputs)
        mu = self.fc_mu(h[-1])
        logvar = self.fc_logvar(h[-1])
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, char_emb, z, sequence_length):
        z = z.unsqueeze(1).repeat(1, sequence_length, 1)
        inputs = torch.cat((char_emb.unsqueeze(1).repeat(1, sequence_length, 1), z), dim=2)
        outputs, _ = self.decoder(inputs)
        return self.fc_out(outputs)

    def forward(self, char, sequence):
        char_emb = self.embedding(char)
        mu, logvar = self.encode(char_emb, sequence)
        z = self.reparameterize(mu, logvar)
        recon_sequence = self.decode(char_emb, z, sequence.size(1))
        return recon_sequence, mu, logvar