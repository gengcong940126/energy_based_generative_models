from pathlib import Path
import argparse
import os
import sys
import time
import numpy as np
import json
from easydict import EasyDict
import yaml
import torch
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter

sys.path.append("./")
sys.path.append("scripts/")

from evals import tf_fid_is_score
from utils import save_samples
from data.cifar import inf_train_gen
from networks.cifar import Generator, EnergyModel, StatisticsNetwork
from functions import train_generator, train_energy_model

"""
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export PORT=6006
        export CUDA_HOME=/opt/cuda/cuda-10.2
        export TIME_STR=1
        python scripts/train/ebm_cifar.py --save_path logs/cifar


    :return:
    """
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)

    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--dim", type=int, default=512)

    parser.add_argument("--energy_model_iters", type=int, default=5)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument('--fid_cache', type=str,default='/home/congen/code/AGE/data/tf_fid_stats_cifar10_32.npz')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    args = parser.parse_args()
    return args


args = parse_args()
root = Path(os.path.join(args.save_path+  '/%02d' % args.energy_model_iters + '/%03d' % int(time.time())))
#################################################
# Create Directories
#################################################
if root.exists():
    os.system("rm -rf %s" % str(root))

os.makedirs(str(root))
os.system("mkdir -p %s" % str(root / "models"))
os.system("mkdir -p %s" % str(root / "images"))
with open("{}/args.txt".format(root), 'w') as f:
    json.dump(args.__dict__, f, indent=4, sort_keys=True)
writer = SummaryWriter(str(root))
#################################################

itr = inf_train_gen(args.batch_size)
netG = Generator(args.z_dim, args.dim).cuda()
netE = EnergyModel(args.dim).cuda()
netH = StatisticsNetwork(args.z_dim, args.dim).cuda()

params = {"lr": 1e-4, "betas": (0.5, 0.9)}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
optimizerG = torch.optim.Adam(netG.parameters(), **params)
optimizerH = torch.optim.Adam(netH.parameters(), **params)

########################################################################
# Dump Original Data
########################################################################
for i in range(8):
    orig_data = itr.__next__()
    #save_image(orig_data, root / 'images/orig.png', normalize=True)
    img = make_grid(orig_data, normalize=True)
    writer.add_image("samples/original", img, i)
########################################################################

start_time = time.time()
e_costs = []
g_costs = []
for iters in range(args.iters):

    for i in range(args.generator_iters):
        train_generator(netG, netE, netH, optimizerG, optimizerH, args, g_costs)

    for i in range(args.energy_model_iters):
        x_real = itr.__next__().cuda()
        train_energy_model(x_real, netG, netE, optimizerE, args, e_costs)

    _, loss_mi = np.mean(g_costs[-args.generator_iters :], 0)
    d_real, d_fake, penalty = np.mean(e_costs[-args.energy_model_iters :], 0)

    writer.add_scalar("energy/fake", d_fake, iters)
    writer.add_scalar("energy/real", d_real, iters)
    writer.add_scalar("loss/penalty", penalty, iters)
    writer.add_scalar("loss/mi", loss_mi, iters)

    if iters % args.log_interval == 0:
        print(
            "Train Iter: {}/{} ({:.0f}%)\t"
            "D_costs: {} G_costs: {} Time: {:5.3f}".format(
                iters,
                args.iters,
                (args.log_interval * iters) / args.iters,
                np.asarray(e_costs).mean(0),
                np.asarray(g_costs).mean(0),
                (time.time() - start_time) / args.log_interval,
            )
        )
        img = save_samples(netG, args)
        writer.add_image("samples/generated", img, iters)

        e_costs = []
        g_costs = []
        start_time = time.time()

    if iters % args.save_interval == 0:
        is_score, fid_score= tf_fid_is_score(args, netG, z_dim=args.z_dim)
        # print("-" * 100)
        # print("Inception Score: mean = {} std = {}".format(mean, std))
        # print("-" * 100)
        writer.add_scalar('inception_score', is_score[0], iters)
        writer.add_scalar('inception_score_std', is_score[1], iters)
        writer.add_scalar('fid_score', fid_score, iters)
        #writer.add_scalar("inception_score/mean", mean, iters)
        #writer.add_scalar("inception_score/std", std, iters)

        torch.save(netG.state_dict(), root / "models/netG.pt")
        torch.save(netE.state_dict(), root / "models/netE.pt")
        torch.save(netH.state_dict(), root / "models/netH.pt")
