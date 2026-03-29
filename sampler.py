import torch


def metropolis_step(model, walkers, step_size):

    proposal = walkers + torch.randn_like(walkers) * step_size

    with torch.no_grad():

        log_old = model(walkers)
        log_new = model(proposal)

        log_ratio = 2 * (log_new - log_old)

        log_ratio = torch.clamp(log_ratio, -50, 50)

        ratio = torch.exp(log_ratio).squeeze()

    accept = torch.rand(walkers.shape[0], device=walkers.device) < ratio
    accept_rate = accept.float().mean().item()
    walkers[accept] = proposal[accept]

    return walkers, accept_rate

