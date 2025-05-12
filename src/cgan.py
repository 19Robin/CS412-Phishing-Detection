# src/cgan.py
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import issparse


class CGAN_Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim):
        super(CGAN_Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        input = torch.cat([z, labels], dim=1)
        return self.model(input)


class CGAN_Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super(CGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        input = torch.cat([x, labels], dim=1)
        return self.model(input)


def cgan(X, y, epochs=100, batch_size=32, noise_dim=100):
    """
    Conditional GAN for phishing email data augmentation.
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
    label_dim = 1  # Binary labels
    generator = CGAN_Generator(noise_dim, label_dim, input_dim).to(device)
    discriminator = CGAN_Discriminator(input_dim, label_dim).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(device)

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            real_data = X_tensor[i:i + batch_size]
            real_labels = y_tensor[i:i + batch_size]

            # Train discriminator
            z = torch.randn(min(batch_size, len(X) - i), noise_dim).to(device)
            fake_data = generator(z, real_labels)
            d_real = discriminator(real_data, real_labels)
            d_fake = discriminator(fake_data.detach(), real_labels)

            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            fake_data = generator(z, real_labels)
            d_fake = discriminator(fake_data, real_labels)
            g_loss = criterion(d_fake, torch.ones_like(d_fake))
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Generate synthetic phishing samples
    z = torch.randn(len(X), noise_dim).to(device)
    y_fake = torch.ones(len(X), 1).to(device)  # Phishing class
    X_fake = generator(z, y_fake).detach().cpu().numpy()
    y_fake = y_fake.cpu().numpy().flatten()

    return np.vstack([X, X_fake]), np.hstack([y, y_fake])


if __name__ == "__main__":
    X = np.random.rand(100, 5000)
    y = np.array([0] * 90 + [1] * 10)
    X_aug, y_aug = cgan(X, y, epochs=10)
    print("Augmented shape:", X_aug.shape)