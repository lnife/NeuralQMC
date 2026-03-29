import torch
import torch.optim as optim

from sampler import metropolis_step
from hamiltonian import local_energy


def train(model, walkers, config):

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10000, gamma=0.5
    )  # so that i dont over shoot

    for step in range(config.TRAIN_STEPS):

        walkers, _ = metropolis_step(model, walkers, config.STEP_SIZE)
        energy = local_energy(model, walkers, config.NUCLEAR_CHARGE)
        logpsi = model(walkers).squeeze(1)
        loss = ((energy.detach() - energy.mean().detach()) * logpsi).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step()

        if (step + 1) % 200 == 0:

            print(
                f"step {step+1}  E = {energy.mean().item():.4f}  var = {energy.var().item():.4f}"
            )
