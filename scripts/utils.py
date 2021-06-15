import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def save_toy_samples(netG, args, z=None):
    if z is None:
        z = torch.randn(args.n_points, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu().numpy()

    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_fake[:, 0], x_fake[:, 1])
    return fig

def visualize_results(netG,netE,orig_data, args, iters):
    netG.eval()
    netE.eval()
    z = torch.randn(args.n_points, args.z_dim).cuda()
    samples = netG(z).detach().cpu().numpy()
    points = orig_data.detach().cpu().numpy()
    plt.clf()
    ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
    ax.scatter(samples[:, 0], samples[:, 1], s=1)

    ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
    ax.scatter(points[:, 0], points[:, 1], s=1)
    netE.cpu()
    ax = plt.subplot(1, 4, 3, aspect="equal")
    plt_toy_density(lambda x: -netE(x), ax,
                         low=-4, high=4,
                         title="p(x)")

    ax = plt.subplot(1, 4, 4, aspect="equal")
    plt_toy_density(lambda x: -netE(x), ax,
                         low=-4, high=4,
                         exp=False, title="log p(x)")

    plt.savefig(("{}/Dx_%08d.png" % iters).format(str(args.save_path / 'images')))
    netE.cuda()
    netE.train()
    netG.train()
def plt_toy_density(self,logdensity, ax, npts=100,
                    title="$q(x)$", device="cpu", low=-4, high=4, exp=True):
    """
    Plot density of toy data.
    """
    side = np.linspace(low, high, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)

    logpx = logdensity(x).squeeze()

    if exp:
        logpx = logpx - logpx.logsumexp(0)
        px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
        px = px / px.sum()
    else:
        logpx = logpx - logpx.logsumexp(0)
        px = logpx.cpu().detach().numpy().reshape(npts, npts)

    ax.imshow(px)
    ax.set_title(title)
def save_samples(netG, args):
    netG.eval()
    z = torch.randn(64, args.z_dim).cuda()
    #x_fake = netG(z).detach().cpu()[:, :3]
    x_fake= netG(z).detach().cpu().reshape(args.batch_size,args.n_stack,args.size,args.size)[:, :3]
    img = make_grid(x_fake, normalize=True)
    netG.train()
    return img


def save_energies(netE, args, n_points=100, beta=1.):
    x = np.linspace(-4, 4, n_points)
    y = np.linspace(-4, 4, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))

    with torch.no_grad():
        grid = torch.from_numpy(grid).float().cuda()
        e_grid = netE(grid) * beta

    p_grid = F.log_softmax(-e_grid, 0).exp()
    e_grid = e_grid.cpu().numpy().reshape((n_points, n_points))
    p_grid = p_grid.cpu().numpy().reshape((n_points, n_points))

    fig1 = plt.Figure()
    ax1 = fig1.add_subplot(111)
    im = ax1.imshow(e_grid, origin='lower',aspect="equal")
    fig1.colorbar(im)

    plt.clf()
    fig2 = plt.Figure()
    ax2 = fig2.add_subplot(111)
    im = ax2.imshow(p_grid, origin='lower',aspect="equal")
    fig2.colorbar(im)

    return fig1, fig2
