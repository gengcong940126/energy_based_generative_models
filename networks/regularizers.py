import torch


def gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand_like(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def score_penalty(netE, data, beta=1.):
    data.requires_grad_(True)
    energy = netE(data) * beta
    score = torch.autograd.grad(
        outputs=energy, inputs=data,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    score=score.flatten(start_dim=1)
    return (score.norm(2, dim=1) ** 2).mean()

def gp_sm(netE, x,G_z):
    G_z.requires_grad_()
    D_x = netE(G_z)
    G_mean=G_z.mean(dim=0)
    x_mean=x.mean(dim=0)
    #M=((G_z-G_mean).flatten(start_dim=1).norm(2,dim=-1)**2).mean().sqrt()
    M = ((x_mean - G_mean).flatten().norm(2, dim=-1) ** 2).mean().sqrt()
    #x.requires_grad_()
    gradients = torch.autograd.grad(
        outputs=D_x,
        inputs=G_z,
        grad_outputs=torch.ones_like(D_x),
        allow_unused=True,
        create_graph=True,retain_graph=True
    )[0]
    gradients = gradients.flatten(start_dim=1)
    v = torch.randn(G_z.shape).cuda().flatten(start_dim=1)
    gradients2 = torch.autograd.grad(
        outputs=(v * gradients).sum(dim=1),
        inputs=G_z,
        grad_outputs=torch.ones_like((v * gradients).sum(dim=1)),
        allow_unused=True,
        create_graph=True,retain_graph=True
    )[0]
    gradients2 = gradients2.flatten(start_dim=1)
    J = (gradients.norm(2, dim=1) ** 2).mean()+ (gradients2 * v).sum(dim=1).mean()*2
    gp_loss=J*M.detach()
    # L2 norm
    #gp_loss = J + (gradients2 * v).sum(dim=1).mean()
    return gp_loss
