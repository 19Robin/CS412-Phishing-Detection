# src/smotified_gan.py
import torch
import torch.nn as nn
import numpy as np
from imblearn.over_sampling import SMOTE
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
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)


def gradient_penalty(discriminator, real_data, fake_data, device):
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1) ** 2).mean()


def smotified_gan(X, y, epochs=100, batch_size=32, noise_dim=100, gp_lambda=10):
    """
    SMOTified-GAN: Combines SMOTE with Wasserstein GAN for data augmentation.
    Args:
        X: Input features (sparse or dense)
        y: Labels (0 or 1)
        epochs: Number of training epochs
        batch_size: Batch size
        noise_dim: Dimension of noise input
        gp_lambda: Gradient penalty coefficient
    Returns:
        Augmented features and labels
    """
    # Convert sparse to dense if necessary
    if issparse(X):
        X = X.toarray()

    # Apply SMOTE
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    input_dim = X_smote.shape[1]
    generator = Generator(noise_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

    # Convert data to tensor
    X_smote = torch.FloatTensor(X_smote).to(device)

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(X_smote), batch_size):
            # Train discriminator
            real_data = X_smote[i:i + batch_size]
            z = torch.randn(min(batch_size, len(X_smote) - i), noise_dim).to(device)
            fake_data = generator(z)

            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data.detach())

            gp = gradient_penalty(discriminator, real_data, fake_data, device)
            d_loss = d_fake.mean() - d_real.mean() + gp_lambda * gp

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            fake_data = generator(z)
            d_fake = discriminator(fake_data)
            g_loss = -d_fake.mean()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Generate synthetic samples (same size as SMOTE output)
    z = torch.randn(len(X_smote) - len(X), noise_dim).to(device)
    X_fake = generator(z).detach().cpu().numpy()
    y_fake = np.ones(len(X_fake))  # Phishing class

    return np.vstack([X, X_fake]), np.hstack([y, y_fake])


if __name__ == "__main__":
    # Example usage
    X = np.random.rand(100, 5000)  # Dummy data
    y = np.array([0] * 90 + [1] * 10)  # Imbalanced
    X_aug, y_aug = smotified_gan(X, y, epochs=10)
    print("Augmented shape:", X_aug.shape)