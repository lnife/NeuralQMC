import torch
from distances import electron_distances


def local_energy(model, pos, Z):

    pos = pos.clone().detach().requires_grad_(True)

    logpsi = model(pos)

    grad = torch.autograd.grad(
        logpsi, pos, grad_outputs=torch.ones_like(logpsi), create_graph=True
    )[0]

    lap = torch.zeros(pos.shape[0], device=pos.device)

    for i in range(6):

        g = grad[:, i]

        g2 = torch.autograd.grad(  # using correct formula older code was also correct but this one correct for log-space wavefunction
            g,
            pos,
            grad_outputs=torch.ones_like(g),
            create_graph=False,  # my cpu isnt strong enough so leaving this as false if your is strong, make it True
            retain_graph=True,
        )[
            0
        ][
            :, i
        ]

        lap += g2

    grad_sq = (grad**2).sum(dim=1)

    kinetic = -0.5 * (lap + grad_sq)

    r1, r2, r12 = electron_distances(pos)

    potential = -Z / r1 - Z / r2 + 1 / r12

    return kinetic + potential
