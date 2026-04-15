import torch
import torch.nn as nn
from .distances import electron_distances


class JastrowFactor(nn.Module):

    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(
            torch.tensor(0.5)
            # this MOTHERFUCKER, i was using parameter for triplet state (0.25) on a ground state system of He
        )  # cusp condition coefficient for fucking!!! singlet ground state, im tired i wanna sleep now!
        self.b = nn.Parameter(torch.tensor(0.1))  # screening parameter

    def forward(self, pos):

        _, _, r12 = electron_distances(pos)  # electron-electron distance

        # Ref: Jastrow, Phys. Rev. 98, 1479 (1955)
        j = (self.a * r12) / (1 + self.b * r12)  # simple Pade Jastrow
        # Ref: Umrigar, Nightingale, and Runge, J. Chem. Phys. 99, Appendix B, 2865 (1993)

        return j.unsqueeze(1)  # match neural network shape
