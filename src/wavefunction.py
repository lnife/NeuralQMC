import torch
import torch.nn as nn

from .slater import SpatialAnsatz
from .jastrow import JastrowFactor


class WaveNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.spatial = SpatialAnsatz()  # symmetric spatial part
        self.jastrow = JastrowFactor()  # correlation

    def forward(self, x):

        log_spatial = self.spatial(x)
        jastrow = self.jastrow(x)

        return (
            log_spatial + jastrow
        )  # Ref: Pfau et al., Phys. Rev. Research 2, 033429 (2020)

