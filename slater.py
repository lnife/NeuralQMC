import torch
import torch.nn as nn


class SlaterDeterminant(nn.Module):

    def __init__(self):
        super().__init__()

        # neural network producing orbitals
        self.orbitals = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
        )

    def forward(self, pos):

        r1 = pos[:, 0:3]  # electron 1
        r2 = pos[:, 3:6]  # electron 2

        phi_r1 = self.orbitals(r1)  # orbital values for electron1
        phi_r2 = self.orbitals(r2)  # orbital values for electron2

        # build 2x2 Slater matrix
        M11 = phi_r1[:, 0]
        M12 = phi_r1[:, 1]

        M21 = phi_r2[:, 0]
        M22 = phi_r2[:, 1]

        det = M11 * M22 - M12 * M21

        return torch.log(torch.abs(det) + 1e-8).unsqueeze(1)

