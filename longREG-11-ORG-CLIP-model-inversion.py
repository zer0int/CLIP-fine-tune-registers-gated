"""
Adapted from Hamid Kazemi's CLIP Inversion:
https://github.com/hamidkazemi22/CLIPInversion
"""

import os
import argparse
import kornia.augmentation as kaugs
import kornia
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from cliptools import fix_random_seed, ColorJitter, RepeatBatch, JitterDims, TotalVariation, Normalization, Scale, freeze_module
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style
import copy
from safetensors.torch import load_file

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import longmodel as clip

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Inverting CLIP!')
    parser.add_argument('--num_iters', default=3400, type=int)
    parser.add_argument('--save_every', default=100, type=int)
    parser.add_argument('--print_every', default=50, type=int)
    parser.add_argument('--batch_size', default=9, type=int)
    parser.add_argument('-p', '--text_prompt', action='append', type=str, default=['catsits', 'sometimes shoes provides cat resting', 'cats shoe snuggles sits resting', 'cats shoeposing'])
    parser.add_argument('-e', '--extra_prompts', action='append', type=str, default=[])
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--tv', default=0.005, type=float)
    parser.add_argument('--jitter', action='store_true')
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--eps', default=2 / 255)
    parser.add_argument('--optimizer', default='adamw')
    parser.add_argument('--bri', type=float, default=0.4)
    parser.add_argument('--con', type=float, default=0.4)
    parser.add_argument('--sat', type=float, default=0.4)
    parser.add_argument('--l1', type=float, default=0.)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--cg_std', type=float, default=0.)
    parser.add_argument('--cg_mean', type=float, default=0.)
    parser.add_argument('--model_name', default='models/LongCLIP-L.safetensors')
    parser.add_argument('--prompt_id', type=int, default=0)
    parser.add_argument('--center_crop', type=bool, default=True, help="10% zoom and center crop during upscaling to preserve details")
    parser.add_argument('--use_best', type=bool, default=True, help="Use best embeds (loss) instead of last step embeds")
    parser.add_argument('--use_image', type=str, default=None, help="Path to image file; uses text embedding of 'CLIP opinion' instead of prompt")
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args()

args = parse_arguments()

if args.deterministic:
    fix_random_seed()
    print(Fore.CYAN + Style.BRIGHT + "Setting deterministic behavior." + Fore.RESET)


# CLIP Model Loader
def load_clip_model(model_name, device):
    model, preprocess = clip.load(model_name, device)
    return model.eval().float(), preprocess

# Image Loader
def load_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

# Augmentation Pipeline
def augment(into, augs):
    return augs(into)

# Gradient Ascent Functions
def clip_encode_text(model, text, many_tokens, prompt):
    x = torch.matmul(text, model.token_embedding.weight)
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ model.text_projection
    return x

# Entertain user by printing CLIP's 'opinion' rants about image to console
def checkin(loss, tx, lll, tok, bests, imagename):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]
    
    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')
        tokens = j.split()
        unique_tokens.update(tokens)
        
    os.makedirs("TOK", exist_ok=True)
    with open(f"TOK/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(torch.nn.Module):
    def __init__(self, batch_size, many_tokens, prompt):
        super(Pars, self).__init__()
        self.batch_size = batch_size
        self.many_tokens = many_tokens
        self.prompt = prompt
        
        # Initialize parameters
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt_embeddings = torch.zeros(batch_size, len(prompt), 49408).cuda()
        for jk, pt in enumerate(prompt):
            self.prompt_embeddings[:, jk, pt] = 1 
        
        self.update_padding()

    def update_padding(self):
        """Update the padding tokens based on current number of active tokens."""
        pad_length = 248 - (self.many_tokens + len(self.prompt) + 1)
        self.pad = torch.zeros(self.batch_size, pad_length, 49408).cuda()
        self.pad[:, :, 49407] = 1

    def diversity_penalty(self, new_tokens, existing_tokens, min_sim=0.6, max_sim=0.9):
        """
        Penalize new tokens for being too similar (>max_sim) or too dissimilar (<min_sim) to existing tokens.
        """
        # Compute cosine similarity between new tokens and existing tokens
        cosine_sim = F.cosine_similarity(new_tokens.unsqueeze(1), existing_tokens, dim=-1)

        # Identify where similarity is outside the acceptable range
        too_similar = (cosine_sim > max_sim).float()
        too_dissimilar = (cosine_sim < min_sim).float()

        # Penalize both cases
        penalty = too_similar * (cosine_sim - max_sim) ** 2  # Penalty for being too similar
        penalty += too_dissimilar * (min_sim - cosine_sim) ** 2  # Penalty for being too dissimilar

        # Return the mean penalty across all comparisons
        return penalty.mean()

    def add_tokens(self, num_new_tokens, model, image, optimizer, prompt, many_tokens, nom, augment):
        """Add more tokens with refined gradient-based initialization."""
        # Compute gradients for the current tokens
        loss, _, _ = ascend_txt(image, model, self, many_tokens, prompt, nom, augment)
        loss = loss.mean()  # Mean over the batch
        loss.backward()  # Compute gradients
        gradients = self.normu.grad  # Gradients w.r.t. current tokens

        # Weight gradients by their norm
        gradient_weights = gradients.norm(dim=-1, keepdim=True)  # Compute gradient magnitudes
        weighted_gradients = gradients * gradient_weights  # Scale gradients by magnitude
        weighted_mean = weighted_gradients.mean(dim=1, keepdim=True)  # Compute weighted mean

        # Use the weighted gradient mean to initialize new tokens
        new_tokens = weighted_mean.repeat(1, num_new_tokens, 1)
        new_tokens += torch.normal(mean=0, std=0.01, size=new_tokens.shape).cuda()

        # Apply diversity penalty to ensure new tokens are distinct but related
        existing_tokens = self.normu  # Existing token embeddings
        penalty = self.diversity_penalty(new_tokens, existing_tokens)
        new_tokens -= penalty * 0.1  # Adjust tokens based on penalty weight

        # Update normu with the new tokens
        self.normu = torch.nn.Parameter(torch.cat([self.normu, new_tokens], dim=1))
        self.many_tokens += num_new_tokens
        self.update_padding()

    def forward(self):
        self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
        fin = torch.cat([self.start, self.prompt_embeddings, self.soft, self.pad], 1)
        return fin



# Gradient Ascent
def ascend_txt(image, model, lats, many_tokens, prompt, nom, augment):
    iii = nom(augment(image[:,:3,:,:].expand(lats.normu.shape[0], -1, -1, -1)))
    iii = model.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, prompt)
    return -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1), tx, lll

# Loop with AMP
def train(image, model, lats, many_tokens, prompt, optimizer, nom, augment):
    with autocast():
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll


def generate_target_text_embeddings(img_path, model, lats, optimizer, training_iterations, checkin_step, many_tokens, prompt, nom, augment, tok, bests, args):
    if args.use_best:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = load_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
        print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

        best_loss = float('inf')  # Initialize the best loss as infinity
        best_text_embeddings = None  # Placeholder for the best text embeddings

        for j in range(training_iterations):
            # Adjust active tokens dynamically at specific steps
            if j == 50:
                num_new_tokens = 1
                print(Fore.YELLOW + Style.BRIGHT + f"Adding {num_new_tokens} tokens at step {j}..." + Fore.RESET)
                lats.add_tokens(num_new_tokens, model, img, optimizer, prompt, many_tokens, nom, augment)
                
                # Reinitialize the optimizer and scheduler with updated parameters
                optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

            # Training step
            loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment)
            current_loss = loss.mean().item()
            
            # Update best embeddings if current loss is better
            if current_loss < best_loss:
                best_loss = current_loss
                best_text_embeddings = copy.deepcopy(tx.detach())
                print(Fore.RED + Style.BRIGHT + f"New best loss: {best_loss:.3f}" + Fore.RESET)
                checkin(loss, tx, lll, tok, bests, img_name)
                print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)
            
            # Update learning rate
            scheduler.step()

            # Print learning rate for monitoring
            if j % checkin_step == 0:
                current_lr = optimizer.param_groups[0]['lr']
                #print(Fore.CYAN + f"Iteration {j}: Current Learning Rate: {current_lr:.5f}" + Fore.RESET)
                print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
                checkin(loss, tx, lll, tok, bests, img_name)
        
        # Save the best embeddings to disk
        torch.save(best_text_embeddings, f"txtembeds/{img_name}_text_embedding.pt")
        print(Fore.MAGENTA + Style.BRIGHT + "\nBest text embedding saved to 'txtembeds'.\nTokens (CLIP 'opinion') saved to 'TOK'.\n" + Fore.RESET)
        
        return img, best_text_embeddings, img_path

    else:     
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = load_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
        print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)
        for j in range(training_iterations):
            loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment)
            if j % checkin_step == 0:
                print(Fore.GREEN + f"Iteration {j}: Average Loss: {loss.mean().item()}" + Fore.RESET)
                checkin(loss, tx, lll, tok, bests, img_name)
       
        target_text_embedding = tx.detach()
        os.makedirs("txtembeds", exist_ok=True)
        torch.save(target_text_embedding, f"txtembeds/{img_name}_text_embedding.pt")
        print(Fore.MAGENTA + Style.BRIGHT + "\nText embedding saved to 'txtembeds'.\nTokens (CLIP 'opinion') saved to 'TOK'.\n" + Fore.RESET)
        
        return img, target_text_embedding, img_path


# Inversion Functions
def get_optimizer(image, lr, optimizer_type):
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam([image], lr=lr)
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW([image], lr=lr)
    else:
        optimizer = torch.optim.LBFGS([image], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
    return optimizer, scheduler

def forward(image, model, normalizer, color_jitter, text_features_map, tv_module, args, pre_aug, aug):
    image_input = pre_aug(image)
    image_input = aug(image_input)
    scale = Scale(model.visual.input_resolution)
    image_input = scale(image_input)
    image_input = color_jitter(image_input)
    image_input = normalizer(image_input)
    image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    l2_loss = torch.norm(image_features - text_features_map[model], dim=1)
    loss = torch.mean(l2_loss)
    return loss, l2_loss

def center_crop(img, crop_size):
    _, _, h, w = img.shape
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    return img[:, :, start_y:start_y+crop_size, start_x:start_x+crop_size]

def get_unique_save_path(base_path, embedding_idx, lr, tv, cg_std, cg_mean):
    """
    Ensures a unique save path by inserting a numbered directory before {embedding_idx}.
    If the number already exists, it increments until a new directory is found.
    """
    number = 1
    while True:
        numbered_path = os.path.join(base_path, str(number), str(embedding_idx),
                                     f"{lr}_{tv}_{cg_std}_{cg_mean}")
        if not os.path.exists(numbered_path):
            os.makedirs(numbered_path, exist_ok=True)
            return numbered_path
        number += 1  # Increment if directory already exists

def run_inversion(args, models, text_features_map, tv_module, normalizer, color_jitter, pre_aug, aug, embedding_idx):
    for model in models:
        freeze_module(model)
    
    image = torch.rand((1, 3, args.img_size, args.img_size)).cuda()
    image.requires_grad_()
    
    optimizer, scheduler = get_optimizer(image, args.lr, args.optimizer)
    softmax = nn.Softmax(dim=1)
    change_scale_schedule = [900, 1800]
    print(Fore.YELLOW + Style.BRIGHT + f"Running Inversion...\n" + Fore.RESET)
    # Determine the base save path
    if args.use_image:
        base_path = f'images/Long-ORG/{os.path.splitext(os.path.basename(args.use_image))[0]}'
    else:
        base_path = 'images/Long-ORG'

    # Get a unique path
    save_path = get_unique_save_path(base_path, embedding_idx, args.lr, args.tv, args.cg_std, args.cg_mean)

    os.makedirs(save_path, exist_ok=True)    
   
    for i in range(args.num_iters):
        max_grad_norm = 1.
        if i in change_scale_schedule:
            if args.center_crop:
                base_res = image.shape[2] * 2
                increment = base_res * 0.10
                new_res = int(round(base_res + increment))
                if new_res >= model.visual.input_resolution:
                    new_res = model.visual.input_resolution
                    if base_res > new_res:
                        base_res = new_res
                if args.jitter:
                    jitter.lim *= 2
                up_sample = Scale(new_res)
                image = up_sample(image.detach())
                image = center_crop(image, base_res)
                image.requires_grad_(True)
                optimizer, scheduler = get_optimizer(image, args.lr, args.optimizer)

            else:
                new_res = image.shape[2] * 2
                if args.jitter:
                    jitter.lim = jitter.lim * 2
                if new_res >= model.visual.input_resolution:
                    new_res = model.visual.input_resolution
                up_sample = Scale(new_res)
                image = up_sample(image.detach())
                image.requires_grad_(True)
                optimizer, scheduler = get_optimizer(image, args.lr, args.optimizer)
    
        def closure():
            optimizer.zero_grad()
            other_loss = tv_module(image)
            loss = args.tv * other_loss
            image_input = image
            l1_loss = torch.norm(image_input, p=1)
            loss = loss + args.l1 * l1_loss
            for model in models:
                xent_loss, scores = forward(image_input, model, normalizer, color_jitter, text_features_map, tv_module, args, pre_aug, aug)
                loss = loss + xent_loss * (1 / len(models))
            loss.backward()
            clip_grad_norm_([image], max_grad_norm)
            image.data = torch.clip(image.data, 0, 1)
            if i % args.print_every == 0:
                print(f'{i:04d}: loss is {loss:.4f}, xent: {xent_loss:.4f}, tv: {other_loss:.4f}, l1: {l1_loss:.4f}')
            if i % args.save_every == 0:
                path = os.path.join(save_path, f'{i}.png')
                torchvision.utils.save_image(image, path, normalize=True, scale_each=True)
            return loss
    
        optimizer.step(closure)
        if i >= 3400:
            scheduler.step()
    
    path = os.path.join(save_path, 'final.png')
    torchvision.utils.save_image(image, path, normalize=True, scale_each=True)
    print(Fore.MAGENTA + Style.BRIGHT + "\nInversion results saved to 'images'.\n" + Fore.RESET)



# Main loop
def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    normalizer = Normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).cuda()
    
    model, preprocess = load_clip_model(args.model_name, device)
    models = [model]
    
    tok = clip.simple_tokenizer.SimpleTokenizer()
    bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}
    prompt = clip.tokenize('''''').numpy().tolist()[0]
    prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]
    
    lats = Pars(args.batch_size, 4, prompt).cuda()
    optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
   
    augs = torch.nn.Sequential(
        kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
    ).cuda()
    
    seq = []
    if args.jitter:
        jitter = Jitter(lim=32, modeldims=model.visual.input_resolution)
        seq.append(jitter)
    seq.append(RepeatBatch(args.batch_size))
    pre_aug = nn.Sequential(*seq)
    
    if args.use_image:
        clipembeds_folder = "txtembeds"
        #target_text_embedding = torch.load(os.path.join(clipembeds_folder, "catdog_embedding.pt")).to(device)# TODO: implement pre-made embeds loading
        img, target_text_embedding, img_path = generate_target_text_embeddings(
            args.use_image, model, lats, optimizer, 340, 10, 4, prompt, normalizer, augs, tok, bests, args
        )
        text_inputs = target_text_embedding
        args.prompt = f"Image-based embedding from {args.use_image}"
    else:
        args.prompt = ' '.join(args.text_prompt)
        print(f'Using prompt: <{args.prompt}>')
        text_inputs = torch.cat([clip.tokenize(f"{c}") for c in args.prompt]).to(device)
    
    text_features_map = {}
    for model in models:
        if args.use_image:
            text_feature = text_inputs
        else:
            text_feature = model.encode_text(text_inputs)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            target_text_embedding = text_feature.detach()
        text_features_map[model] = text_feature
    
    color_jitter = ColorJitter(args.batch_size, True, mean=args.cg_mean, std=args.cg_std)
    tv_module = TotalVariation()
    
    # New loop to process all indices in target_text_embedding
    if args.use_image:
        num_embeddings = target_text_embedding.size(0)
        for embedding_idx in range(num_embeddings):
            print(f"Processing embedding index: {embedding_idx}")
            
            # Select the embedding for the current index
            current_embedding = target_text_embedding[embedding_idx:embedding_idx + 1]

            # Update text_features_map for the current embedding
            text_features_map = {model: current_embedding for model in models}

            # Run inversion with the current embedding index
            run_inversion(args, models, text_features_map, tv_module, normalizer, color_jitter, pre_aug, augs, embedding_idx)
    else:
        num_embeddings = target_text_embedding.size(0)
        for embedding_idx in range(num_embeddings):
            print(f"Processing embedding index: {embedding_idx}")
            
            # Select the embedding for the current index
            current_embedding = target_text_embedding[embedding_idx:embedding_idx + 1]

            # Update text_features_map for the current embedding
            text_features_map = {model: current_embedding for model in models}

            # Run inversion with the current embedding index
            run_inversion(args, models, text_features_map, tv_module, normalizer, color_jitter, pre_aug, augs, embedding_idx)


if __name__ == "__main__":
    main()