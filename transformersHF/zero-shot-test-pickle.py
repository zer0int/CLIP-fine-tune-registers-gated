import torch
from PIL import Image
import torch.nn.functional as F
import INFERclipregXGATED as clip

from cliptoolsoptimized import fix_random_seed

fix_random_seed()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your custom CLIP model
model, preprocess = clip.load("ViT-L-14-REG-GATED-balanced-ckpt12.pt", device=device)
model.eval().float()

# Load and preprocess image
image = Image.open("pineapple.png").convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

# Text prompts
texts = ["pine", "apple", "pineapple", "orange", "pear", "person", "cat", "dog"]

# Tokenize and encode text
text_tokens = clip.tokenize(texts).to(device)
with torch.no_grad():
    image_embed = model.encode_image(image_tensor)       # (1, D)
    text_embeds = model.encode_text(text_tokens)         # (N, D)

# Normalize for cosine similarity
image_embed = F.normalize(image_embed, dim=-1)
text_embeds = F.normalize(text_embeds, dim=-1)

# Cosine similarities
cos_sim = image_embed @ text_embeds.T  # (1, N)
cos_sim = cos_sim.squeeze(0)           # (N,)

# Output results
for text, sim in zip(texts, cos_sim):
    print(f"Similarity with '{text}': {sim.item():.4f}")
