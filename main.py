import torch

from src import config

from src.wavefunction import WaveNet
from src.train import train
from src.sampler import metropolis_step


def main():

    device = torch.device(config.DEVICE)

    model = WaveNet().to(device)

    walkers = torch.randn(config.BATCH_SIZE, 6, device=device)

    # ← ADD THIS BLOCK HERE
    print("Thermalizing walkers...")
    with torch.no_grad():
        for _ in range(500):
            walkers, _ = metropolis_step(model, walkers, config.STEP_SIZE)

    train(model, walkers, config)

    print("expected helium ground state ≈ -2.903 Hartree")


if __name__ == "__main__":

    main()
