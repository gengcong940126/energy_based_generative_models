import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def EnergyModel_fc(sn=False):
    """
    Large MLP EBM.
    """
    if sn:
        return nn.Sequential(
            #nn.Flatten(start_dim=1),
            nn.utils.spectral_norm(nn.Linear(2352, 2000)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Linear(2000, 1000)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Linear(1000, 500)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Linear(500, 250)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Linear(250, 250)),
            nn.LeakyReLU(0.2),
            nn.Linear(250, 1, bias=True)
        )
    else:
        return nn.Sequential(
            #nnj.Flatten(start_dim=1),
            nn.Linear(2352, 2000),
            nn.PReLU(),
            nn.Linear(2000, 1000),
            nn.PReLU(),
            nn.Linear(1000, 500),
            nn.PReLU(),
            nn.Linear(500, 250),
            nn.PReLU(),
            nn.Linear(250, 250),
            nn.PReLU(),
            nn.Linear(250, 1, bias=True))
def Generator_fc(z_dim):
    """
    Large MLP generator.
    """
    final_act = nn.Tanh()

    return nn.Sequential(
        nn.Linear(z_dim, 500, bias=True),
        nn.BatchNorm1d(500,momentum=0.1),
        nn.PReLU(),
        nn.Linear(500, 1000, bias=True),
        nn.BatchNorm1d(1000,momentum=0.1),
        nn.PReLU(),
        nn.Linear(1000, 2000, bias=True),
        nn.BatchNorm1d(2000,momentum=0.1),
        nn.PReLU(),
        nn.Linear(2000, 2352, bias=True),
        final_act)


class Generator(nn.Module):
    def __init__(self, input_dim=1, z_dim=128, dim=512):
        super().__init__()
        self.expand = nn.Linear(z_dim, 2 * 2 * dim)
        self.main = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.PReLU(),
            nn.ConvTranspose2d(dim, dim // 2, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.PReLU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, 5, 2, 2),
            nn.BatchNorm2d(dim // 4),
            nn.PReLU(),
            nn.ConvTranspose2d(dim // 4, dim // 8, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(dim // 8),
            nn.PReLU(),
            nn.ConvTranspose2d(dim // 8, input_dim, 5, 2, 2, output_padding=1),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, z):
        x = self.expand(z).view(z.size(0), -1, 2, 2)
        return self.main(x)


class EnergyModel(nn.Module):
    def __init__(self, input_dim=1, dim=512):
        super().__init__()
        self.expand = nn.Linear(2 * 2 * dim, 1)
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, dim // 8, 5, 2, 2),
            nn.PReLU(),
            nn.Conv2d(dim // 8, dim // 4, 5, 2, 2),
            nn.PReLU(),
            nn.Conv2d(dim // 4, dim // 2, 5, 2, 2),
            nn.PReLU(),
            nn.Conv2d(dim // 2, dim, 5, 2, 2),
            nn.PReLU()
        )
        self.apply(weights_init)

    def forward(self, x, return_fmap=False):
        out = self.main(x).view(x.size(0), -1)
        energies = self.expand(out).squeeze(-1)
        if return_fmap:
            return out, energies
        else:
            return energies


class StatisticsNetwork(nn.Module):
    def __init__(self, input_dim=1, z_dim=128, dim=512):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, dim // 8, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, dim, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.expand = nn.Linear(2 * 2 * dim, z_dim)
        self.classify = nn.Sequential(
            nn.Linear(z_dim * 2, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, 1),
        )
        self.apply(weights_init)

    def forward(self, x, z):
        out = self.main(x).view(x.size(0), -1)
        out = self.expand(out)
        out = torch.cat([out, z], -1)
        return self.classify(out).squeeze(-1)
