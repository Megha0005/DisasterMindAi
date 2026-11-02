
import torch
import torch.nn as nn


class DisasterRBM(nn.Module):

    def __init__(self, visible_units=8, hidden_units=32):

        super().__init__()

        self.visible_units = visible_units
        self.hidden_units = hidden_units

        self.W = nn.Parameter(
            torch.randn(visible_units, hidden_units) * 0.01
        )

        self.visible_bias = nn.Parameter(
            torch.zeros(visible_units)
        )

        self.hidden_bias = nn.Parameter(
            torch.zeros(hidden_units)
        )

    # Gaussian visible -> Bernoulli hidden
    def sample_hidden(self, visible):

        hidden_prob = torch.sigmoid(
            torch.matmul(visible, self.W) + self.hidden_bias
        )

        hidden_sample = torch.bernoulli(hidden_prob)

        return hidden_prob, hidden_sample

    # Bernoulli hidden -> Gaussian visible
    def sample_visible(self, hidden):

        visible_mean = (
            torch.matmul(hidden, self.W.T) + self.visible_bias
        )

        visible_sample = visible_mean + torch.randn_like(visible_mean)

        return visible_mean, visible_sample

    # Contrastive Divergence
    def contrastive_divergence(self, visible, k=1):

        hidden_prob, hidden_sample = self.sample_hidden(visible)

        for _ in range(k):

            visible_mean, visible_sample = self.sample_visible(
                hidden_sample
            )

            hidden_prob_negative, hidden_sample = self.sample_hidden(
                visible_sample
            )

        return (
            hidden_prob,
            visible_mean,
            visible_sample,
            hidden_prob_negative
        )

    # Free energy
    def free_energy(self, visible):

        visible_term = 0.5 * torch.sum(
            (visible - self.visible_bias) ** 2,
            dim=1
        )

        hidden_term = torch.sum(
            torch.nn.functional.softplus(
                torch.matmul(visible, self.W) + self.hidden_bias
            ),
            dim=1
        )

        return visible_term - hidden_term

    # Generate synthetic feature patterns
    def generate(self, num_samples=10, steps=100):

        device = self.W.device

        visible = torch.randn(
            num_samples,
            self.visible_units,
            device=device
        )

        with torch.no_grad():

            for _ in range(steps):

                _, hidden = self.sample_hidden(visible)

                _, visible = self.sample_visible(hidden)

        return visible


    # Calculate Free Energy
    def free_energy(self, visible):

        visible_term = 0.5 * torch.sum(
            (visible - self.visible_bias) ** 2,
            dim=1
        )

        hidden_term = torch.sum(
            torch.nn.functional.softplus(
                torch.matmul(visible, self.W) + self.hidden_bias
            ),
            dim=1
        )

        return visible_term - hidden_term
