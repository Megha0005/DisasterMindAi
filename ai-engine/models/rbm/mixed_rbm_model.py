
import torch
import torch.nn as nn


class MixedDisasterRBM(nn.Module):

    def __init__(
        self,
        gaussian_units=8,
        binary_units=3,
        hidden_units=32
    ):
        super().__init__()

        self.gaussian_units = gaussian_units
        self.binary_units = binary_units
        self.hidden_units = hidden_units

        self.W_gaussian = nn.Parameter(
            torch.randn(gaussian_units, hidden_units) * 0.01
        )

        self.W_binary = nn.Parameter(
            torch.randn(binary_units, hidden_units) * 0.01
        )

        self.visible_bias_gaussian = nn.Parameter(
            torch.zeros(gaussian_units)
        )

        self.visible_bias_binary = nn.Parameter(
            torch.zeros(binary_units)
        )

        self.hidden_bias = nn.Parameter(
            torch.zeros(hidden_units)
        )

    def sample_hidden(self, gaussian, binary):

        activation = (
            gaussian @ self.W_gaussian
            + binary @ self.W_binary
            + self.hidden_bias
        )

        probability = torch.sigmoid(activation)
        sample = torch.bernoulli(probability)

        return probability, sample

    def sample_visible(self, hidden):

        gaussian_mean = (
            hidden @ self.W_gaussian.T
            + self.visible_bias_gaussian
        )

        gaussian_sample = (
            gaussian_mean + torch.randn_like(gaussian_mean)
        )

        binary_probability = torch.sigmoid(
            hidden @ self.W_binary.T
            + self.visible_bias_binary
        )

        binary_sample = torch.bernoulli(binary_probability)

        return (
            gaussian_mean,
            gaussian_sample,
            binary_probability,
            binary_sample
        )

    def free_energy(self, gaussian, binary):

        gaussian_energy = 0.5 * torch.sum(
            (gaussian - self.visible_bias_gaussian) ** 2,
            dim=1
        )

        hidden_activation = (
            gaussian @ self.W_gaussian
            + binary @ self.W_binary
            + self.hidden_bias
        )

        hidden_energy = torch.sum(
            torch.nn.functional.softplus(hidden_activation),
            dim=1
        )

        binary_energy = -torch.sum(
            binary * self.visible_bias_binary,
            dim=1
        )

        return gaussian_energy + binary_energy - hidden_energy

    def generate(self, num_samples=1000, steps=500):

        device = self.W_gaussian.device

        gaussian = torch.randn(
            num_samples,
            self.gaussian_units,
            device=device
        )

        binary = torch.bernoulli(
            torch.full(
                (num_samples, self.binary_units),
                0.5,
                device=device
            )
        )

        with torch.no_grad():

            for _ in range(steps):

                _, hidden = self.sample_hidden(
                    gaussian, binary
                )

                (
                    _,
                    gaussian,
                    _,
                    binary
                ) = self.sample_visible(hidden)

        return gaussian, binary
