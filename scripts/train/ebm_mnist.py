from pathlib import Path
import argparse
import os
import time
import numpy as np
import torch
import json
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
import sys

sys.path.append("./")
sys.path.append("scripts/")

from evals import ModeCollapseEval
from utils import save_samples
from data.mnist import inf_train_gen
from networks.mnist import Generator, EnergyModel, StatisticsNetwork,EnergyModel_fc,Generator_fc
from functions import train_generator, train_energy_model

def is_debugging():
  import sys
  gettrace = getattr(sys, 'gettrace', None)

  if gettrace is None:
    assert 0, ('No sys.gettrace')
  elif gettrace():
    return True
  else:
    return False
"""
    Usage:

        export CUDA_VISIBLE_DEVICES=6
        export PORT=6007
        export CUDA_HOME=/opt/cuda/cuda-10.2
        export TIME_STR=1
        python scripts/train/ebm_mnist.py --save_path logs/stackmnist


    :return:
    """
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--n_stack", type=int, default=3)

    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--energy_model_iters", type=int, default=1)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--mcmc_iters", type=int, default=0)
    parser.add_argument("--lamda", type=float, default=10)
    parser.add_argument("--alpha", type=float, default=0.01)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=60000)
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
if is_debugging()==False:
    os.makedirs(str(root))
    os.system("mkdir -p %s" % str(root / "models"))
    os.system("mkdir -p %s" % str(root / "images"))
    with open("{}/args.txt".format(root), 'w') as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=True)
writer = SummaryWriter(str(root))
#################################################

mc_eval = ModeCollapseEval(args.n_stack, args.z_dim)
itr = inf_train_gen(args.batch_size, n_stack=args.n_stack)
netG = Generator(args.n_stack, args.z_dim, args.dim).cuda()
#netG = Generator_fc(args.z_dim).cuda()
netE = EnergyModel(args.n_stack, args.dim).cuda()
#netE = EnergyModel_fc().cuda()
netH = StatisticsNetwork(args.n_stack, args.z_dim, args.dim).cuda()
params = {"lr": 2e-4, "betas": (0.0, 0.9),"weight_decay": 1e-5}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
optimizerG = torch.optim.Adam(netG.parameters(), **params)
optimizerH = torch.optim.Adam(netH.parameters(), **params)

########################################################################
# Dump Original Data
########################################################################
for i in range(8):
    orig_data = itr.__next__()
    # save_image(orig_data, root / 'images/orig.png', normalize=True)
    img = make_grid(orig_data[:, :3], normalize=True)
    writer.add_image("samples/original", img, i)
########################################################################

torch.backends.cudnn.benchmark = True

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
        netG.eval()
        print("-" * 100)
        n_modes, kld = mc_eval.count_modes(netG)
        print("-" * 100)
        netG.train()

        writer.add_scalar("metrics/mode_count", n_modes, iters)
        writer.add_scalar("metrics/kl_divergence", kld, iters)

        torch.save(netG.state_dict(), root / "models/netG.pt")
        torch.save(netE.state_dict(), root / "models/netE.pt")
        torch.save(netH.state_dict(), root / "models/netH.pt")
