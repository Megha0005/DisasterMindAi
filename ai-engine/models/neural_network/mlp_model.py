
import torch
import torch.nn as nn


class DisasterMLP(nn.Module):

    def __init__(self, input_features=8):
        super().__init__()

        self.network = nn.Sequential(

            # Input layer
            nn.Linear(input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Hidden layer 1
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Hidden layer 2
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output layer
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
