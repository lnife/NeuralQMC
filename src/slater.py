import torch
import torch.nn as nn
from .distances import electron_distances


class SpatialAnsatz(nn.Module):
    """
    Symmetric spatial wavefunction for the Helium ground state (singlet).
    Includes an explicit nuclear cusp term: exp(-Z * (r1 + r2)).
    """

    def __init__(self, Z=2):
        super().__init__()
        self.Z = Z

        # Neural network representing the orbital correction log(phi_net(r))
        # We use a simple MLP that takes 3D coordinates.
        self.orbital_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, pos):
        # Slater determinant reference:
        # J. C. Slater, Phys. Rev. 34, 1293 (1929)

        # I tried my best slater aint working im removing the slater equation

        #       r1 = pos[:, 0:3]  # electron 1
        #       r2 = pos[:, 3:6]  # electron 2
        #
        #       phi_r1 = self.orbitals(r1)  # orbital values for electron1
        #       phi_r2 = self.orbitals(r2)  # orbital values for electron2
        #
        #       # build 2x2 Slater matrix
        #       M11 = phi_r1[:, 0]
        #       M12 = phi_r1[:, 1]
        #
        #       M21 = phi_r2[:, 0]
        #       M22 = phi_r2[:, 1]
        #
        #       det = M11 * M22 - M12 * M21

        # pos shape: (batch_size, 6) -> [x1, y1, z1, x2, y2, z2]
        r1_vec = pos[:, 0:3]
        r2_vec = pos[:, 3:6]

        r1, r2, _ = electron_distances(pos)

        # Hartree product reference:
        # D. R. Hartree, Math. Proc. Cambridge Philos. Soc. 24, 89 (1928)

        # & # Ref: Chauhan and Harbola, arXiv:1506.00912 (2015)

        # I thought i had balls, i do, or more likely my machine had no balls
        # because i was trying to train the model from scratch, IT WAS NOT CONVERGING!!!!!!

        # Log of the nuclear cusp part: -Z * (r1 + r2)
        # Ref: Kato, On the Eigenfunctions of Many-Particle Systems in Quantum Mechanics, Eq. (2.16), p. 15
        log_cusp = -self.Z * (r1 + r2)

        # Log of the neural network orbitals: log(phi(r1)) + log(phi(r2))
        # We unsqueeze to ensure (batch, 1) shape
        log_phi1 = self.orbital_net(r1_vec).squeeze(1)
        log_phi2 = self.orbital_net(r2_vec).squeeze(1)

        # Total log spatial part
        log_spatial = log_cusp + log_phi1 + log_phi2

        return log_spatial.unsqueeze(1)
