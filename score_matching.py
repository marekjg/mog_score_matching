from tqdm import tqdm
import visdom

import torch
from torch import nn
from torch import optim


# From what I understand, grad(grad_x.sum(), x) would sum dx_i * dx_j terms
# which is different than the trace of the hessian (dx_i * dx_i partials)
def score_matching_loss(x, energy, lambada=0.00001):
    grad_x = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
    grads_x_x = []
    for i in range(x.shape[1]):
        grads_x_x.append(
            torch.autograd.grad(grad_x[:, i].sum(),
                                x, create_graph=True)[0][:, i])
    grad_x_x = torch.stack(grads_x_x, dim=1)
    J = -grad_x_x.sum() + grad_x.pow(2).sum() + lambada * grad_x_x.pow(2).sum()
    return J


# Sample from learned distribution
def langevin_rollout(x, energy_function, n_steps=60, step_size=100, device='cuda:0'):
    x.requires_grad_(True)
    for i in range(n_steps):
        energy = energy_function(x)
        x_grad = torch.autograd.grad(energy.sum(), x)[0]
        x_grad.clamp_(-0.01, 0.01)
        with torch.no_grad():
            x -= step_size * x_grad / 2
            x += torch.randn(*x.shape).to(device) / (i + 1)
    x.requires_grad_(False)
    return x


def generate_batch(n_samples=128):
    batch = []
    means = [(-2, -2), (-4, 4), (4, 8), (-12, -7)]
    for i in range(4):
        x, y = means[i]
        sample = torch.randn(n_samples // 4, 2)
        sample[:, 0] += x
        sample[:, 1] += y
        batch.append(sample)
    return torch.cat(batch, dim=0)


device = 'cuda:0'
net = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 1)).to(device)
optimizer = optim.Adam(net.parameters())
viz = visdom.Visdom()

x = torch.rand(1024, 2).to(device)
x = langevin_rollout(x, net)
viz.scatter(x)

losses = []
n_steps = 4000
for i in tqdm(n_steps):
    x = generate_batch().to(device)
    x.requires_grad_(True)
    energy = net(x)

    loss = score_matching_loss(x, energy)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = torch.randn(4096, 2).to(device)
x = langevin_rollout(x, net)

viz.line(losses)
viz.scatter(x)
