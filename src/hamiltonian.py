import torch
from .distances import electron_distances


def local_energy(model, pos, Z):
    """
    Computes local energy E_L = H psi / psi.
    Using the log-space formula for kinetic energy:
    T = -0.5 * (nabla^2 log|psi| + |nabla log|psi||^2)
    """

    # We need to track gradients for the Laplacian
    pos = pos.clone().detach().requires_grad_(True)
    logpsi = model(pos)

    # First derivative of logpsi with respect to position
    grad = torch.autograd.grad(
        logpsi, pos, grad_outputs=torch.ones_like(logpsi), create_graph=True
    )[0]

    # Sum of second derivatives (Laplacian)
    lap = torch.zeros(pos.shape[0], device=pos.device)

    # For Helium (6 coordinates), we iterate over each coordinate
    # This is more memory-efficient than computing the full Hessian.
    for i in range(pos.shape[1]):
        grad_i = grad[:, i]
        # Second derivative for coordinate i
        # retain_graph=True is needed to reuse the graph for subsequent coordinates
        grad_ii = torch.autograd.grad(
            grad_i,
            pos,
            grad_outputs=torch.ones_like(grad_i),
            retain_graph=(i < pos.shape[1] - 1),
            create_graph=False,
        )[0][:, i]
        lap += grad_ii

    # Square of the gradient norm
    grad_sq = (grad**2).sum(dim=1)

    # Local kinetic energy
    kinetic = -0.5 * (lap + grad_sq)

    # Local potential energy
    r1, r2, r12 = electron_distances(pos)
    potential = -Z / r1 - Z / r2 + 1 / r12  # Ref: Hylleraas, Z. Phys. 54, 347 (1929)

    return kinetic + potential
