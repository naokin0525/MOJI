import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vae_gan import HandwritingModel
from models.discriminator import Discriminator
from datasets.moj_dataset import MojDataset
from utils.svg_utils import sequence_to_image

def parse_args():
    parser = argparse.ArgumentParser(description="Train handwriting model")
    parser.add_argument("program_path", type=str, help="Program directory")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to .moj dataset")
    parser.add_argument("--model_file_name", type=str, default="model.pt", help="Model file name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--vae_weight", type=float, default=0.5, help="VAE loss weight")
    parser.add_argument("--gan_weight", type=float, default=0.5, help="GAN loss weight")
    return parser.parse_args()

def train(model, discriminator, dataloader, args, device):
    optimizer_g = optim.Adam(model.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        for batch in dataloader:
            chars, sequences = batch['char'].to(device), batch['sequence'].to(device)
            batch_size = chars.size(0)

            # VAE Forward
            recon_seq, mu, logvar = model(chars, sequences)
            recon_loss = ((recon_seq - sequences) ** 2).mean()
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + args.vae_weight * kl_loss

            # GAN Forward
            real_imgs = torch.stack([sequence_to_image(seq) for seq in sequences]).to(device)
            fake_imgs = torch.stack([sequence_to_image(seq) for seq in recon_seq]).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_out = discriminator(real_imgs, chars)
            fake_out = discriminator(fake_imgs.detach(), chars)
            d_loss = criterion(real_out, real_labels) + criterion(fake_out, fake_labels)
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_out = discriminator(fake_imgs, chars)
            g_loss = criterion(fake_out, real_labels)
            total_loss = vae_loss + args.gan_weight * g_loss
            total_loss.backward()
            optimizer_g.step()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss.item():.4f}")

    torch.save(model.state_dict(), f"{args.model_path}/{args.model_file_name}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = MojDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = HandwritingModel(char_vocab_size=dataset.vocab_size, embedding_dim=50, 
                            hidden_dim=256, latent_dim=64).to(device)
    discriminator = Discriminator(img_size=64, char_vocab_size=dataset.vocab_size).to(device)
    
    train(model, discriminator, dataloader, args, device)

if __name__ == "__main__":
    main()