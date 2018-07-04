import os
import argparse
import random
from datetime import datetime

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable

from tqdm import tqdm

from model import PSGANGenerator as Generator

torch.backends.cudnn.benchmark = True

def save_image(imgs, output_dir="log", img_name="output", img_ext=".png"):
    vutils.save_image(imgs.data, "{}".format(os.path.join(output_dir, img_name+img_ext)))

def train(args):
    def to_var(x, volatile=False, requires_grad=False):
        if torch.cuda.is_available() and not args.nogpu:
            x = x.cuda(args.gpu_device_num)
        return Variable(x, volatile=volatile, requires_grad=requires_grad)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("\nsaving at {}\n".format(args.save_dir))
    print("initializing...")

    # if args.layer_num is 5 and args.base_conv_channel is 64 then
    # gen_layer: [Z_dim, 512, 256, 128,  64, 3]
    gen_layers = [args.zl_dim+args.zg_dim+args.zp_dim]+[args.base_conv_channel*(2**(args.layer_num-n)) for n in range(2, args.layer_num+1)]+[3]
    print("generator channels: ", gen_layers)

    if torch.cuda.is_available() and not args.nogpu:
        generator = Generator(conv_channels=gen_layers,
                              kernel_size=args.kernel_size,
                              local_noise_dim=args.zl_dim,
                              global_noise_dim=args.zg_dim,
                              periodic_noise_dim=args.zp_dim,
                              spatial_size=args.spatial_size,
                              hidden_noise_dim=args.mlp_hidden_dim).cuda(args.gpu_device_num)
    else:
        generator = Generator(conv_channels=gen_layers,
                              kernel_size=args.kernel_size,
                              local_noise_dim=args.zl_dim,
                              global_noise_dim=args.zg_dim,
                              periodic_noise_dim=args.zp_dim,
                              spatial_size=args.spatial_size,
                              hidden_noise_dim=args.mlp_hidden_dim)

    generator.eval()

    print("loading pretrained parameter... ", end="")
    generator.load_trained_param(args.trained, print_debug=args.show_parameters)
    print("done.")

    if args.show_parameters:
        for idx, m in enumerate(model.modules()):
            print(idx, '->', m)

        print(args)

    # for sampling
    random_noise = to_var(generator.generate_noise(batch_size=args.sample_num,
                                                   local_dim=args.zl_dim,
                                                   global_dim=args.zg_dim,
                                                   periodic_dim=args.zp_dim,
                                                   spatial_size=args.spatial_size,
                                                   tile=args.tile),
                        volatile=False)

    # generate fake image
    fake_img = generator(random_noise)

    save_image(fake_img.mul(0.5).add(0.5).cpu(), output_dir=args.save_dir, img_name="sample_from_random_noise")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # setting
    parser.add_argument('--trained', type=str, default="trained_model", help='trained parameter path of generator.')

    # detail settings
    parser.add_argument('--zl_dim', type=int, default=40, help='size of local part noise dimension')   # set default same as author's implementation
    parser.add_argument('--zg_dim', type=int, default=20, help='size of global part noise dimension')  # set default same as author's implementation
    parser.add_argument('--zp_dim', type=int, default=3, help='size of periodic part noise dimension') # set default same as author's implementation
    parser.add_argument('--mlp_hidden_dim', type=int, default=60, help='size of periodic part noise dimension')
    parser.add_argument('--spatial_size', type=int, default=16, help='size of spatial dimension')
    # for pytorch there is no pad="same", if you need use 5 or other sizes, you might need add torch.nn.functional.pad in the model.
    parser.add_argument('--kernel_size', type=int, default=4, help='size of kernels')
    parser.add_argument('--layer_num', type=int, default=5, help='number of layers')
    parser.add_argument('--base_conv_channel', type=int, default=64, help='base channel number of convolution layer')
    parser.add_argument('--tile', type=int, default=4, help='')

    parser.add_argument('--save_dir', type=str, default="./log/sampled", help='directory of saving sampled image')

    parser.add_argument('--epochs', type=int, default=10000, help="train epoch num.")
    parser.add_argument('--sample_num', type=int, default=1, help="sample size")
    parser.add_argument('--num_workers', type=int, default=8, help="worker # of data loader")

    parser.add_argument('--gpu_device_num', type=int, default=0, help="device number of gpu")
        
    # option
    parser.add_argument('-nogpu', action="store_true", default=False, help="don't use gpu")
    parser.add_argument('-show_parameters', action="store_true", default=False, help='show model parameters')
    
    args = parser.parse_args()
    
    train(args)
