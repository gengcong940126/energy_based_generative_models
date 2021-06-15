from pathlib import Path
import argparse
from tqdm import tqdm
import os

import torch
from torchvision.utils import save_image
from PIL import Image

import sys

sys.path.append("./")
sys.path.append("scripts/")
from collections import OrderedDict
from sampler import MALA_corrected_sampler
#from inception_score import get_inception_score
from networks.cifar import Generator, EnergyModel
from evals import tf_fid_is_score_eval,tf_fid_is_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", required=True)
    parser.add_argument("--dump_path", default="logs/cifar_ebm/cifar_samples")

    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument('--fid_cache', type=str, default='/home/congen/code/AGE/data/tf_fid_stats_cifar10_32.npz')
    parser.add_argument("--mcmc_iters", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--temp", type=float, default=0.02)

    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--n_samples", type=int, default=50000)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)
if not Path(root).exists():
    Path(root).mkdir()
# if not Path(args.dump_path).exists():
#     Path(args.dump_path).mkdir()

netG = Generator(args.z_dim, args.dim).cuda()
netE = EnergyModel(args.dim).cuda()

netG.eval()
netE.eval()

netG.load_state_dict(torch.load('/home/congen/code/geoml_gan/models/cifar10/EBM/128/05/1617909431/epoch099_g.pkl'))
netE.load_state_dict(torch.load('/home/congen/code/geoml_gan/models/cifar10/EBM/128/05/1617909431/epoch099_d.pkl'))
# ckpt=torch.load('/home/congen/code/VERA/tmp/cifar10_2/save_model/_075000.pt')
# new_state_dict_G = OrderedDict()
# for k, va in ckpt["model"]["g"].items():
#     name = k[2:]   # remove `vgg.`，即只取vgg.0.weights的后面几位
#     new_state_dict_G[name] = va
# netG.load_state_dict(new_state_dict_G,strict=False)
# netE.load_state_dict(ckpt["model"]["logp_net"],strict=False)
#netG.load_state_dict(torch.load('//home/congen/code/energy_based_generative_models/logs/cifar/05/1617870312/models/netG.pt'))
#netE.load_state_dict(torch.load('/home/congen/code/energy_based_generative_models/logs/cifar/05/1617870312/models/netE.pt'))


images = []
for i in tqdm(range(args.n_samples // args.batch_size)):
    z = MALA_corrected_sampler(netG, netE, args)
    x = netG(z).detach()
    images.append(x)

    if i == 0:  # Debugging
        save_image(x.cpu(), root / "generated.png", normalize=True)
images = torch.cat(images, 0).cpu().numpy()
#save_image(images[0].cpu(), Path(args.dump_path) / "generated_mcmc.png", normalize=True)
#mean, std = get_inception_score(images)
is_score, fid_score= tf_fid_is_score_eval(args, images)
#is_score, fid_score= tf_fid_is_score(args, netG)
print("-" * 100)
print(
    "FID and Inception Score: alpha = {} temp={} mcmc_iters = {} mean = {} std = {} fid= {}".format(
        args.alpha, args.temp, args.mcmc_iters, is_score[0], is_score[1],fid_score))

print("-" * 100)

##########################
# Dumping images for FID #
##########################
#images = ((images * 0.5 + 0.5) * 255).astype("uint8")
# for i, img in tqdm(enumerate(images)):
#     Image.fromarray(img.transpose(1, 2, 0)).save(args.dump_path + "/image_%05d.png" % i)
#Image.fromarray(images[0].transpose(1, 2, 0)).save(args.dump_path + "/image_mcmc.png")
#os.system(
   # "python TTUR/fid.py %s TTUR/fid_stats_cifar10_train.npz --gpu 0" % args.dump_path
#)
