import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from safetensors.torch import load_file
import argparse

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import INFERclipregXGATED as clip
import clip as orgclip

# Load CLIP ViT-L/14 original model for comparison
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize Long-CLIP patch-text cosine similarity')
    parser.add_argument('--base_model', default="ViT-L/14", help="Path to a ViT-L/14 model, pickle (.pt) or .safetensors")
    parser.add_argument('--use_model', default="models/ViT-L-14-REG-GATED-balanced-ckpt12.safetensors", help="Path to a ViT-L/14 model, pickle (.pt) or .safetensors")
    parser.add_argument('--token_folder', default="EX-tokens-vis", help="Folder with gradient ascent .txt files of CLIP's opinions (or yours)")
    parser.add_argument('--image_folder', default="EX-image-vis", help="Folder with images, matching for .txt files: 'image.png' -> 'tokens_image.txt'")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

# Folder paths
image_folder = args.image_folder
tokens_folder = args.token_folder

if model_name_or_path.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    
    # Load ViT-L/14 explicitly
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    # Load the safetensors state_dict and apply
    state_dict = load_file(model_name_or_path)
    model.load_state_dict(state_dict)

else:
    print("Detected non-.safetensors file. Attempting to load as a pickle...")
    
    # Load normally as per the existing logic
    model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

if args.base_model.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    
    # Load ViT-L/14 explicitly
    modelorg, preprocess = orgclip.load("ViT-L/14", device=device, jit=False)

    # Load the safetensors state_dict and apply
    state_dict = load_file(args.base_model)
    modelorg.load_state_dict(state_dict)

else:
    print("Detected non-.safetensors file. Attempting to load as a pickle...")
    
    # Load normally as per the existing logic
    modelorg, preprocess = orgclip.load(args.base_model, device=device, jit=False)


model = model.float()
modelorg = modelorg.float()

# Function to encode image features for CLIP model
def clip_encode_image(model, image_input):
    with torch.no_grad():
        x = model.visual.conv1(image_input)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  

        cls_token = model.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        register_tokens = model.visual.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)

        x = torch.cat([cls_token, register_tokens, x], dim=1)
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        return x

# Function to encode image features for original CLIP
def orgclip_encode_image(modelorg, image_input):
    with torch.no_grad():
        x = modelorg.visual.conv1(image_input)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  

        cls_token = modelorg.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + modelorg.visual.positional_embedding.to(x.dtype)
        x = modelorg.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  
        x = modelorg.visual.transformer(x)
        x = x.permute(1, 0, 2)
        return x

# Loop through each image file in the image folder
for image_filename in os.listdir(image_folder):
    # Check if file is an image based on extension
    if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue

    # Construct corresponding tokens filename (e.g., "cat.png" -> "tokens_cat.txt")
    name_without_ext = os.path.splitext(image_filename)[0]
    tokens_filename = f"tokens_{name_without_ext}.txt"
    tokens_filepath = os.path.join(tokens_folder, tokens_filename)

    # If tokens file doesn't exist, skip this image
    if not os.path.exists(tokens_filepath):
        print(f"Skipping {image_filename}: tokens file {tokens_filename} not found.")
        continue

    # Read tokens from file (tokens are separated by a space)
    with open(tokens_filepath, "r") as f:
        tokens = f.read().strip().split(" ")
    # Remove any empty tokens (in case of extra spaces)
    tokens = [token for token in tokens if token]

    # Load image and preprocess
    image_path = os.path.join(image_folder, image_filename)
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Pre-compute image features for both models once per image
    x_myclip = clip_encode_image(model, image_input)
    x_orgclip = orgclip_encode_image(modelorg, image_input)

    # Loop through each token from the tokens file
    for obj in tokens:
        text_prompt = [f"a {obj}"]  # Format prompt

        # Encode text features for both models
        text_features_myclip = model.encode_text(clip.tokenize(text_prompt).to(device)).float()
        text_features_myclip /= text_features_myclip.norm(dim=-1, keepdim=True)

        text_features_orgclip = modelorg.encode_text(orgclip.tokenize(text_prompt).to(device)).float()
        text_features_orgclip /= text_features_orgclip.norm(dim=-1, keepdim=True)

        # Process each model
        for clip_model, x, text_features, model_name in zip(
            [model, modelorg],
            [x_myclip.clone(), x_orgclip.clone()],  # Clone to avoid in-place modifications
            [text_features_myclip, text_features_orgclip],
            ["regclip", "orgclip"]
        ):
            # Compute number of patches (excluding CLS token)
            N_patches = x.shape[1] - 1
            h = w = int(np.sqrt(N_patches))

            # Apply patch skipping rule based on the model
            if model_name == "regclip":
                x = x[:, 5:, :]  # Skip CLS and register tokens
            else:
                x = x[:, 1:, :]  # Skip only CLS token

            # Compute per-patch cosine similarity
            cos_sim_values = []
            for i in range(x.shape[1]):  
                patch = x[:, i, :]
                patch = clip_model.visual.ln_post(patch)
                patch = patch @ clip_model.visual.proj
                patch /= patch.norm(dim=-1, keepdim=True)

                cos_sim = torch.nn.functional.cosine_similarity(patch.view(1, -1), text_features.view(1, -1))
                cos_sim_values.append(cos_sim)

            cos_sim = torch.stack(cos_sim_values).cpu().detach().numpy()
            cos_sim = cos_sim.reshape(h, w)

            # Normalize heatmap
            cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min())

            # Ensure image is in correct format
            image_np = np.array(image)

            # Resize heatmap to match image size
            heatmap = np.uint8(255 * cos_sim)
            heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Plot results
            plt.figure(figsize=(8, 8))
            plt.imshow(image_np)
            plt.imshow(heatmap, cmap="jet_r", alpha=0.5)
            plt.axis("off")
            plt.title(f"Model: {model_name} -- Text: '{text_prompt[0]}'", fontsize=25)
            plt.tight_layout()

            # Save the image
            os.makedirs("plots/REG-XGATED/patch-cos", exist_ok=True)
            save_path = f"plots/REG-XGATED/patch-cos/{name_without_ext}_{obj}_{model_name}_seg.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            print(f"Saved: {save_path}")

    print(f"Processed image: {image_filename}")

print("Batch processing complete!")
