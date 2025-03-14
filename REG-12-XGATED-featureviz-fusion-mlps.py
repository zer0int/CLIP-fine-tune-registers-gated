"""
Adapted from Hamid Kazemi's ViT-Visualization:
https://github.com/hamidkazemi22/vit-visualization
"""

import os
import random
import collections
import argparse
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision.transforms import Resize
from safetensors.torch import load_file
from colorama import Fore, Style

import INFERclipregXGATED as clip
from INFERclipregXGATED.model import QuickGELU
from torch.nn import ReLU

# Custom imports
from cliptools import LossArray, TotalVariation
from cliptools import ViTFeatHook, ViTEnsFeatHook, ViTFusionEnsFeatHook, ViTREGEnsFeatHook
from cliptools import ClipGeLUHook, ClipReLUHook, REGClipGeLUHook, LongClipGeLUHook
from cliptools import Clip, Tile, Jitter, RepeatBatch, ColorJitter, GaussianNoise
from cliptools import ClipViTWrapper as ClipWrapper
from cliptools import new_init, save_intermediate_step, save_image, fix_random_seed

# Suppress warnings spam from torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='REG-CLIP Fusion MLP feature activation max visualization')
    parser.add_argument('--use_model', default='models/ViT-L-14-REG-GATED-balanced-ckpt12.safetensors', help="Path to REG-XGATED CLIP model")
    parser.add_argument('--layer_range', default="0-10", type=str, help="Which layers to visualize; continuous range ('5-10') or discrete values ('5,6,8'). Default: 0-10")
    parser.add_argument('--feature_range', default="0-10", type=str, help="Which features to visualize; continuous range ('50-90') or discrete values ('500,1000)'. Default: 0-10")
    parser.add_argument('--steps', default=400, type=int, help="Number of image optimization steps; default: 400")
    parser.add_argument('--lr', default=1.0, type=float, help="Learning Rate; default: 1.0")
    parser.add_argument('--tv', default=1.0, type=float, help="Total Variation Loss; default: 1.0")
    parser.add_argument('--coeff', default=0.00005, type=float, help="For tv*coeff. 0.00005 -> sharp and noisy image; 0.05 -> soft, blurry; default: 0.00005")
    parser.add_argument("--output_folder", default='FeatureViz/FUSION-MLP', help="Folder to save output image; default: FeatureViz/FUSION-MLP")
    parser.add_argument("--save_intermediate", action='store_true', help="Save intermediate steps, too, for a quick look. Saves to folder: 'steps'")
    parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior")
    return parser.parse_args()

args = parse_arguments()
steps_folder = args.output_folder
os.makedirs(steps_folder, exist_ok=True)

clipmodel = args.use_model
clipname = os.path.splitext(clipmodel)[0]
clipname = clipname.replace("@", "-").replace("models/", "").replace("/", "")
print(clipname)

if args.deterministic:
    fix_random_seed()


class ImageNetVisualizer:
    def __init__(self, loss_array: LossArray, pre_aug: nn.Module = None,
                 post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1, save_every: int = 200, saver: bool = True,
                 print_every: int = 5, **_):
        self.loss = loss_array
        self.saver = saver
        print(self.saver)

        self.pre_aug = pre_aug
        self.post_aug = post_aug

        self.save_every = save_every
        self.print_every = print_every
        self.steps = steps
        self.lr = lr

    def __call__(self, img: torch.tensor = None, optimizer: optim.Optimizer = None, layer: int = None, feature: int = None, clipname: str = None):
        if not img.is_cuda or img.device != torch.device('cuda:0'):
            img = img.to('cuda:0')
        if not img.requires_grad:
            img.requires_grad_()
        
        # ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']        
        # Default:
        # optimizer = optimizer if optimizer is not None else optim.Adam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        optimizer = optimizer if optimizer is not None else optim.Adamax([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        print(f'#i\t{self.loss.header()}', flush=True)

        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            if i % self.print_every == 0:
                print(f'{i}\t{self.loss}', flush=True)
            if i % self.save_every == 0 and self.saver is True:
                save_intermediate_step(img, i, layer, feature, clipname)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            self.loss.reset()

        optimizer.state = collections.defaultdict(dict)
        return img


def get_clip_dimensions(model, preprocess):
    model = model.eval()
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            input_dims = transform.size
            break
    num_layers = None
    num_features = None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features
    if hasattr(model.visual.transformer, 'intermediate_fusion_mlps'):
        num_fusions = len(model.visual.transformer.intermediate_fusion_mlps)
        last_fusion = model.visual.transformer.intermediate_fusion_mlps[-1]
        if hasattr(last_fusion, '0'):
            fusion_layer = last_fusion[0]
            num_fusion_features = fusion_layer.out_features

    return input_dims, num_layers, num_features, num_fusions, num_fusion_features


def load_clip_model(device: str = 'cuda') -> torch.nn.Module:
    #global preprocess
    
    if args.use_model.endswith(".safetensors"):
        print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
        
        # Load ViT-L/14 explicitly
        model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

        # Load the safetensors state_dict and apply
        state_dict = load_file(clipmodel)
        model.load_state_dict(state_dict)

    else:
        print("Detected non-.safetensors file. Attempting to load as a pickle...")
        
        # Load normally as per the existing logic
        model, preprocess = clip.load(clipmodel, device=device, jit=False)        
    
    premodel = model
    model = ClipWrapper(model).to(device).float()
    return model, premodel, preprocess


def parse_range(range_str):
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return list(map(int, range_str.split(',')))

def generate_visualizations(model, clipname, layer_range_str, feature_range_str, image_size, tv, lr, steps, print_every, save_every, saver, coefficient):
    layer_range = parse_range(layer_range_str)
    feature_range = parse_range(feature_range_str)

    for layer in layer_range:
        for feature in feature_range:
            print(Fore.MAGENTA + Style.BRIGHT + f"Generating visualization for Layer {layer}, Feature {feature}..." + Fore.RESET)
            loss = LossArray()
            loss += ViTFusionEnsFeatHook(ClipReLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
            loss += TotalVariation(2, image_size, coefficient * tv)

            pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                            GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
            image = new_init(image_size, 1)

            visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient)
            image.data = visualizer(image, layer=layer, feature=feature, clipname=clipname)

            save_image(image, f'{steps_folder}/{clipname}_L{layer}_F{feature}.png')

def main():
    args = parse_arguments()

    model, premodel, preprocess = load_clip_model()
    input_dims, num_layers, num_features, num_fusions, num_fusion_features = get_clip_dimensions(premodel, preprocess)
    image_size = input_dims
    print(f"\nSelected input dimension for {clipmodel}:" + Fore.GREEN + Style.BRIGHT + f" {input_dims}" + Fore.RESET)
    print(f"Layers:" + Fore.GREEN + Style.BRIGHT + f"0-{num_layers-1} with 0-{num_features-1} Features / Layer" + Fore.RESET)
    print(f"Fusion Layers:" + Fore.GREEN + Style.BRIGHT + f" 0-{num_fusions-1} with 0-{num_fusion_features-1} Features / Layer\n" + Fore.RESET)

    layer_range_str = args.layer_range
    feature_range_str = args.feature_range
    
    tv = args.tv
    lr = args.lr
    coefficient=args.coeff

    steps = args.steps
    print_every = 10
    save_every = 10

    saver = False
    if args.save_intermediate:
        saver = True

    generate_visualizations(model, clipname, layer_range_str, feature_range_str, image_size, tv, lr, steps, print_every, save_every, saver, coefficient)

if __name__ == '__main__':
    main()
    


