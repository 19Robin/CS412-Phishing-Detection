# src/vae_gan.py
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from scipy.sparse import issparse

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(VAE_Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def vae_gan(X, y, epochs=100, batch_size=32, latent_dim=100):
    """
    VAE-GAN with BERT embeddings for phishing email augmentation.
    Args:
        X: Input features
        y: Labels
        epochs: Number of training epochs
        batch_size: Batch size
        latent_dim: Latent space dimension
    Returns:
        Augmented features and labels
    """
    if issparse(X):
        X = X.toarray()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    encoder = VAE_Encoder(input_dim, latent_dim).to(device)
    decoder = VAE_Decoder(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    vae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    X_tensor = torch.FloatTensor(X).to(device)

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            real_data = X_tensor[i:i + batch_size]

            # VAE: Encode and decode
            mu, logvar = encoder(real_data)
            z = reparameterize(mu, logvar)
            recon_data = decoder(z)

            # Train discriminator
            d_real = discriminator(real_data)
            d_fake = discriminator(recon_data.detach())
            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train VAE
            d_fake = discriminator(recon_data)
            vae_loss = criterion(d_fake, torch.ones_like(d_fake)) + 0.5 * (
                        mu ** 2 + torch.exp(logvar) - logvar - 1).sum()
            vae_optimizer.zero_grad()
            vae_loss.backward()
            vae_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss.item():.4f}, VAE Loss: {vae_loss.item():.4f}")

    # Generate synthetic samples
    z = torch.randn(len(X), latent_dim).to(device)
    X_fake = decoder(z).detach().cpu().numpy()
    y_fake = np.ones(len(X_fake))

    return np.vstack([X, X_fake]), np.hstack([y, y_fake])


if __name__ == "__main__":
    X = np.random.rand(100, 5000)
    y = np.array([0] * 90 + [1] * 10)
    X_aug, y_aug = vae_gan(X, y, epochs=10)
    print("Augmented shape:", X_aug.shape)