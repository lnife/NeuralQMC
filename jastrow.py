import torch
import torch.nn as nn
from distances import electron_distances


class JastrowFactor(nn.Module):

    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(
            torch.tensor(0.25)
        )  # cusp condition coefficient, intentially choosing the condition thats closer to cusp value
        self.b = nn.Parameter(torch.tensor(0.1))  # screening parameter

    def forward(self, pos):

        _, _, r12 = electron_distances(pos)  # electron-electron distance

        j = (self.a * r12) / (1 + self.b * r12)  # simple Pade Jastrow

        return j.unsqueeze(1)  # match neural network shape

