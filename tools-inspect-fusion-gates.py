import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import argparse
from safetensors.torch import load_file

import INFERclipregXGATED as clip
from cliptools import fix_random_seed

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description='REG-CLIP Fusion Gate Analysis')
    parser.add_argument('--use_model', default='models/ViT-L-14-REG-GATED-balanced-ckpt12.pt', help="Path to REG-XGATED CLIP model")
    parser.add_argument('--dataset', default="F:/AI_DATASET/ILSVRC2012/val", type=str, help="Path to image dataset (samples from folder and subfolders)")
    parser.add_argument('--num_images', default=1000, type=int, help="Number of images to sample from the dataset, default: 1000")
    parser.add_argument('--save_to', default="fusion_gate_plots", type=str, help="Output folder path, default: 'fusion_gate_plots'")
    parser.add_argument("--save_images", action='store_true', help="Save sampled images along with the results plot") 
    parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior (including fixed random seed for fetching images)")
    return parser.parse_args()

args = parse_arguments()

num_im = args.num_images
CHECKPOINT_PATH = args.use_model
DATASET_FOLDER = args.dataset
SAVE_FOLDER = args.save_to
os.makedirs(SAVE_FOLDER, exist_ok=True)

if args.deterministic:
    fix_random_seed()

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

if CHECKPOINT_PATH.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    
    # Load ViT-L/14 explicitly
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    # Load the safetensors state_dict and apply
    state_dict = load_file(CHECKPOINT_PATH)
    model.load_state_dict(state_dict)

else:
    print("Detected non-.safetensors file. Attempting to load as a pickle...")
    
    # Load normally as per the existing logic
    model, preprocess = clip.load(CHECKPOINT_PATH, device=device, jit=False)

model = model.eval().float()

# Get number of Fusion MLP layers
num_fusion_layers = len(model.visual.transformer.intermediate_fusion_mlps)


# Extract all Fusion MLP activations
def clip_encode_image(model, image_tensor):
    with torch.no_grad():
        x = model.visual.conv1(image_tensor)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls_token = model.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        register_tokens = model.visual.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)

        # Concatenate tokens: CLS, REG, and patch tokens
        x = torch.cat([cls_token, register_tokens, x], dim=1)
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # shape: [seq_len, batch, width]

        # Store activations
        fusion_activations_per_layer = {}

        for i, block in enumerate(model.visual.transformer.resblocks):
            x = block(x)
            if i >= model.visual.transformer.gate_start_layer:
                cls_token = x[0]  # [batch, width]
                reg_tokens = x[1:1 + model.visual.num_registers]  # [num_registers, batch, width]
                reg_summary = reg_tokens.mean(dim=0)  # [batch, width]
                fusion_input = torch.cat([cls_token, reg_summary], dim=-1)  # [batch, 2048]

                fusion_mlp_layer = i - model.visual.transformer.gate_start_layer

                if fusion_mlp_layer < num_fusion_layers:
                    # Full Fusion MLP
                    gate_hidden = model.visual.transformer.intermediate_fusion_mlps[fusion_mlp_layer][0](fusion_input)  # [batch, 1024]
                    gate_hidden = torch.nn.functional.relu(gate_hidden)  # Activation function
                    gate = torch.sigmoid(model.visual.transformer.intermediate_fusion_mlps[fusion_mlp_layer][2](gate_hidden))  # [batch, 1]

                    # Store per-layer fusion activation
                    fusion_activations_per_layer[f"layer_{fusion_mlp_layer}"] = gate.cpu().numpy()

                    # Apply fusion
                    fused = gate * cls_token + (1 - gate) * reg_summary
                    x = torch.cat([fused.unsqueeze(0), x[1:]], dim=0)

        x = x.permute(1, 0, 2)  # shape: [batch, seq_len, width]

        # Final CLS and register token processing
        cls_token = model.visual.ln_post(x[:, 0, :])
        reg_tokens = model.visual.ln_post(x[:, 1:1 + model.visual.num_registers, :])
        reg_summary = reg_tokens.mean(dim=1)
        fusion_input = torch.cat([cls_token, reg_summary], dim=-1)

        # Compute final fusion gate (fully applying the final MLP)
        final_hidden = model.visual.fusion_mlp[0](fusion_input)
        final_hidden = torch.nn.functional.silu(final_hidden)
        final_gate = torch.sigmoid(model.visual.fusion_mlp[2](final_hidden))

        return {
            "fusion_activations": fusion_activations_per_layer,  # Per-layer activations
            "final_fusion_gate": final_gate.cpu().numpy(),       # Final fusion gate output
            "register_tokens": reg_tokens.cpu().numpy()          # Final register token representations
        }


# Collect random images from all subfolders in DATASET_FOLDER
all_images = []
for root, _, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            all_images.append(os.path.join(root, file))

random.shuffle(all_images)
selected_images = all_images[:num_im]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
])

fusion_data_layers = []

for img_path in selected_images:
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract fusion activations
    results = clip_encode_image(model, image_tensor)
    
    # Store data
    fusion_data_layers.append({
        "image": img_path,
        "fusion_activations": results["fusion_activations"],
        "final_fusion_gate": results["final_fusion_gate"],
        "register_tokens": results["register_tokens"]
    })

    # Save the processed image for visualization
    if args.save_images:
        image.save(os.path.join(SAVE_FOLDER, os.path.basename(img_path)))

np.save(os.path.join(SAVE_FOLDER, "fusion_gate_layers.npy"), fusion_data_layers)

valid_layers = set()
for entry in fusion_data_layers:
    valid_layers.update(entry["fusion_activations"].keys())
valid_layers = sorted(valid_layers, key=lambda x: int(x.split("_")[1]))  # Sort numerically

plt.figure(figsize=(12, 8))
for layer_key in valid_layers:
    layer_values = np.concatenate([entry["fusion_activations"].get(layer_key, np.array([])).flatten() for entry in fusion_data_layers if layer_key in entry["fusion_activations"]])

    if len(layer_values) > 0:
        plt.hist(layer_values, bins=50, alpha=0.6, label=layer_key)

plt.xlabel("Fusion MLP Gate Activation")
plt.ylabel("Frequency")
plt.title("Per-Layer Fusion MLP Gate Activations")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_FOLDER, "per_layer_fusion_gate_distrib.png"))
plt.close()

print(f"Fusion MLP activations (all layers) saved in '{SAVE_FOLDER}'.")
