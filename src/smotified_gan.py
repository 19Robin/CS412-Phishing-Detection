import numpy as np
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os

# Build generator model
def build_generator(latent_dim, feature_dim):
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(feature_dim, activation='tanh')
    ])
    return model

# Build discriminator model
def build_discriminator(feature_dim):
    model = Sequential([
        Dense(512, input_dim=feature_dim),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# SMOTified-GAN implementation
def smotified_gan(X, y, epochs=2000, batch_size=32, save_path='models/smotified_gan/'):
    feature_dim = X.shape[1]
    latent_dim = 100

    # Apply SMOTE
    smote = SMOTE(k_neighbors=3, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Build generator and discriminator
    generator = build_generator(latent_dim, feature_dim)
    discriminator = build_discriminator(feature_dim)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

    # Build GAN
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_features = generator(gan_input)
    gan_output = discriminator(generated_features)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    # Training loop
    for epoch in range(epochs):
        # Generate synthetic samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_samples = generator.predict(noise)

        # Train discriminator
        real_samples = X[np.random.randint(0, X.shape[0], batch_size)]
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    # Save models
    os.makedirs(save_path, exist_ok=True)
    generator.save(os.path.join(save_path, 'generator.h5'))
    discriminator.save(os.path.join(save_path, 'discriminator.h5'))

    return X_smote, y_smote