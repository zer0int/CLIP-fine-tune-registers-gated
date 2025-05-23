import torch
from PIL import Image
from regtransformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from cliptoolsoptimized import fix_random_seed

fix_random_seed()

# Load the locally converted model
model_path = "converted_model"
model = CLIPModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
processor = CLIPProcessor.from_pretrained(model_path)

# Put model on available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the image
image = Image.open("pineapple.png").convert("RGB")

# Define the text prompts
texts = ["pine", "apple", "pineapple", "orange", "pear", "person", "cat", "dog"]

# Preprocess
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds  # (1, D)
    text_embeds = outputs.text_embeds    # (N, D)

# Normalize for cosine similarity
image_embeds = F.normalize(image_embeds, dim=-1)
text_embeds = F.normalize(text_embeds, dim=-1)

# Compute cosine similarity
cos_sim = image_embeds @ text_embeds.T  # (1, N)
cos_sim = cos_sim.squeeze(0)  # (N,)

# Print results
for text, sim in zip(texts, cos_sim):
    print(f"Similarity with '{text}': {sim.item():.4f}")
