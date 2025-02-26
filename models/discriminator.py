import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_size, char_vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, 50)
        self.model = nn.Sequential(
            nn.Conv2d(1 + 50, 64, 4, 2, 1),  # Input: image + char embedding channels
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * (img_size // 4) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, char):
        char_emb = self.embedding(char).view(char.size(0), 50, 1, 1).repeat(1, 1, img.size(2), img.size(3))
        input = torch.cat((img, char_emb), dim=1)
        return self.model(input)