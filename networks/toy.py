import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 2)
        )

    def forward(self, z):
        return self.main(z)

class Generator2(nn.Module):
    def __init__(self, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.PReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.PReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, 2)
        )

    def forward(self, z):
        return self.main(z)
class EnergyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(2, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        return self.main(x).squeeze(-1)
class EnergyModel2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(2, dim),
            nn.PReLU(),
            nn.Linear(dim, dim),
            nn.PReLU(),
            nn.Linear(dim, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.main(x).squeeze(-1)

class StatisticsNetwork(nn.Module):
    def __init__(self, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(2 + z_dim, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(dim, 1)
        )

    def forward(self, x, z):
        x = torch.cat([x, z], -1)
        return self.main(x).squeeze(-1)
