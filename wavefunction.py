import torch
import torch.nn as nn

from jastrow import JastrowFactor
from slater import SlaterDeterminant


class WaveNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.slater = SlaterDeterminant()  # antisymmetric part
        self.jastrow = JastrowFactor()     # correlation

    def forward(self,x):

        log_slater = self.slater(x)
        jastrow = self.jastrow(x)

        return log_slater + jastrow