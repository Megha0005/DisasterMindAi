
import torch
import torch.nn as nn


class DisasterAutoencoder(nn.Module):

    def __init__(self, input_features=8, latent_dim=3):
        super().__init__()

        # Encoder: compress input into latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, latent_dim)
        )

        # Decoder: reconstruct original input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_features)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return reconstructed, latent
