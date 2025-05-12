# src/mcmc_gan.py
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import issparse


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


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


def mcmc_sampling(generator, noise_dim, num_samples, device, steps=10, step_size=0.1):
    """MCMC sampling to refine generator outputs."""
    z = torch.randn(num_samples, noise_dim).to(device)
    for _ in range(steps):
        z_proposal = z + step_size * torch.randn_like(z)
        fake_data = generator(z)
        fake_data_proposal = generator(z_proposal)
        # Simplified acceptance (can be enhanced with discriminator scores)
        accept = torch.rand(num_samples).to(device) < 0.5
        z = torch.where(accept.view(-1, 1), z_proposal, z)
    return generator(z)


def mcmc_gan(X, y, epochs=100, batch_size=32, noise_dim=100):
    """
    MCMC-GAN: Uses MCMC sampling for diverse sample generation.
    Args:
        X: Input features
        y: Labels
        epochs: Number of training epochs
        batch_size: Batch size
        noise_dim: Noise dimension
    Returns:
        Augmented features and labels
    """
    if issparse(X):
        X = X.toarray()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    generator = Generator(noise_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            real_data = X_tensor[i:i + batch_size]
            real_labels = y_tensor[i:i + batch_size]

            # Train discriminator
            z = torch.randn(min(batch_size, len(X) - i), noise_dim).to(device)
            fake_data = generator(z)
            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data.detach())

            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            fake_data = generator(z)
            d_fake = discriminator(fake_data)
            g_loss = criterion(d_fake, torch.ones_like(d_fake))
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Generate samples using MCMC
    X_fake = mcmc_sampling(generator, noise_dim, len(X), device).detach().cpu().numpy()
    y_fake = np.ones(len(X_fake))

    return np.vstack([X, X_fake]), np.hstack([y, y_fake])


if __name__ == "__main__":
    X = np.random.rand(100, 5000)
    y = np.array([0] * 90 + [1] * 10)
    X_aug, y_aug = mcmc_gan(X, y, epochs=10)
    print("Augmented shape:", X_aug.shape)