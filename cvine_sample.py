import torch
from torch import distributions as dist
from torch.autograd import Variable


def h_gauss(u0, u1, rho, m):
    """
    Gaussian hfunction
    """
    z0 = m.icdf(u0)
    z1 = m.icdf(u1)
    h = (z1 - rho * z0)/torch.sqrt(1-rho.pow(2))

    return m.cdf(h)


def hinv_gauss(u0, u1, rho, m):
    """
    Gaussian inverse function
    """
    z0 = m.icdf(u0)
    z1 = m.icdf(u1)
    h = z1 * torch.sqrt(1 - rho.pow(2)) + rho * z0

    return m.cdf(h)


def cvine_sample(u, rho, device, normal=True):
    d = u.shape[1]
    x = torch.ones(u.shape)
    v = torch.ones(u.shape[0], d, d)
    x[:, 0] = u[:, 0]
    v[:, 0, 0] = u[:, 0]
    x = Variable(x).to(device)
    v = Variable(v).to(device)

    m = dist.Normal(torch.Tensor([0.0]).to(device),
                    torch.Tensor([1.0]).to(device))
    for i in range(1, d):
        v[:, i, 0] = u[:, i]
        for k in range(i-1, -1, -1):
            v[:, i, 0] = hinv_gauss(v[:, k, k], v[:, i, 0], rho[:, int(k + i * (i - 1)/2)], m)
        x[:, i] = v[:, i, 0]

        if i == (d - 1):
            break

        for j in range(0, i):
            v[:, i, j + 1] = h_gauss(v[:, j, j], v[:, i, j], rho[:, int(j + i * (i - 1)/2)], m)
    if normal:
        x = m.icdf(x)

    return x
