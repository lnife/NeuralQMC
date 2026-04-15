import torch

def electron_distances(pos):

    eps = 1e-8

    r1 = torch.sqrt((pos[:,0:3]**2).sum(dim=1)+eps)
    r2 = torch.sqrt((pos[:,3:6]**2).sum(dim=1)+eps)

    r12 = torch.sqrt(((pos[:,0:3]-pos[:,3:6])**2).sum(dim=1)+eps)

    return r1,r2,r12