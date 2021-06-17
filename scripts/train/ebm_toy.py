import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import time
import numpy as np
from tensorboardX import SummaryWriter
import json
import torch
import sys
sys.path.append('./')
sys.path.append('scripts/')

from utils import save_toy_samples, save_energies, visualize_results
from data.toy import inf_train_gen
from networks.toy import Generator, EnergyModel, StatisticsNetwork,Generator2,EnergyModel2
from functions import train_generator, train_energy_model

"""
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export PORT=6006
        export CUDA_HOME=/opt/cuda/cuda-10.2
        export TIME_STR=1
        python scripts/train/ebm_toy.py --dataset twomoon --save_path logs


    :return:
    """
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--save_path', required=True)

    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=100)

    parser.add_argument('--energy_model_iters', type=int, default=1)
    parser.add_argument('--generator_iters', type=int, default=1)
    parser.add_argument('--mcmc_iters', type=int, default=0)
    parser.add_argument('--lamda', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--mode', type=str, default='0gp')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--iters', type=int, default=150000)
    parser.add_argument('--n_points', type=int, default=1600)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--save_interval', type=int, default=5000)

    args = parser.parse_args()
    return args


args = parse_args()
#root = Path(args.save_path)
root = Path(os.path.join(args.save_path+ '/' + args.dataset + '_soft/%02d' % args.energy_model_iters + '/%03d' % int(time.time())))
#################################################
# Create Directories
#################################################
if root.exists():
    os.system('rm -rf %s' % str(root))

os.makedirs(str(root))
os.system('mkdir -p %s' % str(root / 'models'))
os.system('mkdir -p %s' % str(root / 'images'))
with open("{}/args.txt".format(root), 'w') as f:
    json.dump(args.__dict__, f, indent=4, sort_keys=True)
writer = SummaryWriter(str(root))
#################################################
itr = inf_train_gen(args.dataset, args.batch_size)

netG = Generator2(args.z_dim, args.dim).cuda()
netE = EnergyModel2(args.dim).cuda()
netH = StatisticsNetwork(args.z_dim, args.dim).cuda()
# netE.load_state_dict(torch.load('logs/twomoon_soft/01/1623963113/models/netE.pt'))
# netG.load_state_dict(torch.load('logs/twomoon_soft/01/1623963113/models/netG.pt'))
# netH.load_state_dict(torch.load('logs/twomoon_soft/01/1623963113/models/netD.pt'))
params = {'lr': 2e-4, 'betas': (0.0, 0.9)}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
optimizerG = torch.optim.Adam(netG.parameters(), **params)
optimizerH = torch.optim.Adam(netH.parameters(), **params)

#################################################
# Dump Original Data
#################################################
orig_data = inf_train_gen(args.dataset, args.n_points).__next__()
fig = plt.Figure()
ax = fig.add_subplot(111)
ax.scatter(orig_data[:, 0], orig_data[:, 1])
writer.add_figure('originals', fig, 0)
#plt.savefig(("{}/data.png" ).format(str(Path(args.save_path) / 'images')))
##################################################

start_time = time.time()
e_costs = []
g_costs = []
for iters in range(args.iters):

    for i in range(args.generator_iters):
        train_generator(
            netG, netE, netH,
            optimizerG, optimizerH,
            args, g_costs
        )

    for i in range(args.energy_model_iters):
        x_real = torch.from_numpy(itr.__next__()).cuda()
        train_energy_model(
            x_real,
            netG, netE, optimizerE,
            args, e_costs
        )

    _, loss_mi = np.mean(g_costs[-args.generator_iters:], 0)
    d_real, d_fake, penalty = np.mean(e_costs[-args.energy_model_iters:], 0)

    writer.add_scalar('loss_fake', d_fake, iters)
    writer.add_scalar('loss_real', d_real, iters)
    writer.add_scalar('loss_penalty', penalty, iters)
    writer.add_scalar('loss_mi', loss_mi, iters)

    if iters % args.log_interval == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'D_costs: {} G_costs: {} Time: {:5.3f}'.format(
                  iters, args.iters,
                  (args.log_interval * iters) / args.iters,
                  np.asarray(e_costs).mean(0),
                  np.asarray(g_costs).mean(0),
                  (time.time() - start_time) / args.log_interval
              ))
        #visualize_results(netG,netE,orig_data, args, iters)
        fig_samples = save_toy_samples(netG, args)
        e_fig, p_fig = save_energies(netE, args)

        writer.add_figure('samples', fig_samples, iters)
        writer.add_figure('energy', e_fig, iters)
        writer.add_figure('density', p_fig, iters)

        e_costs = []
        g_costs = []
        start_time = time.time()

    if iters % args.save_interval == 0:
        torch.save(
            netG.state_dict(),
            root / 'models/netG.pt'
        )
        torch.save(
            netE.state_dict(),
            root / 'models/netE.pt'
        )
        torch.save(
            netH.state_dict(),
            root / 'models/netD.pt'
        )
