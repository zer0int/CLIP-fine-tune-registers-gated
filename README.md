CLIP-fine-tune-registers-gated

# CLIP Needs Registers. And Gated MLPs. And +20M params. 🧑‍💻🤖
### Fixing CLIP's modality gap via happy little accidents.

- Want a retrieval model? Low modality gap? Read on! ✅🌟
- Want zero-shot accuracy & Text Encoders for gen-AI? You're better off using my [classic CLIP-fine-tune repo](https://github.com/zer0int/CLIP-fine-tune). 👈⚠️
- Jump to HF models: [huggingface.co/zer0int/CLIP-Registers-Gated_MLP-ViT-L-14](https://huggingface.co/zer0int/CLIP-Registers-Gated_MLP-ViT-L-14)
- Jump to HF Long-CLIP models: [huggingface.co/zer0int/LongCLIP-Registers-Gated_MLP-ViT-L-14/](https://huggingface.co/zer0int/LongCLIP-Registers-Gated_MLP-ViT-L-14)
------
### Update 19/MAR/2025:
- Added fusion gate inspector (subset of image dataset)
- Compares gating towards REG vs. CLS token over layers
- Usual syntax; use `--deterministic` for same choice of images
- Example:
![visualize-gates](https://github.com/user-attachments/assets/b8fa35fe-bb89-493e-b920-bfd219fb039b)
------
### Update 14/MAR/2025:

- Added feature (activation max) visualization!
- `--use_model` by default expects `models/ViT-L-14-REG-GATED-balanced-ckpt12.safetensors`
- You can specify layers & features as a range or discrete, example:
```
python REG-12-XGATED-featureviz-fusion-mlps.py --layer_range 8-11 --feature_range 42,1000,77
```
- That would visualize feature 42 and 1000 and 77 on layer 8, 9, 10 and 11.
- If you exceed the valid range for Layers or Features, you'll get an IndexError.
- Read the green text when you run the script to see the valid range! :)

- Interesting observations: MLP Fusion Gate features are either sharp or dead (thanks, ReLU...). But if not dead, they're super intricate and detailed, no matter which layer.
- On the other hand, early layers (resblocks) in the ViT encode simple structures, lines, zigzags... Then more complex textures:

![layers-example](https://github.com/user-attachments/assets/201d7589-0eb6-4f1d-90bd-f629e6f13299)
```
python REG-12-XGATED-featureviz-normal-mlps.py --layer_range 1-23 --feature_range 42,100,1000
```
![complexity-chaos-REG-CLIP](https://github.com/user-attachments/assets/c12e21cf-5d82-4c88-aa43-bc7619ecaf14)

------
### Update: 11/MAR/2025

- Added Long-CLIP (248 tokens) version! 🎉
- Same syntax, except prepend `long` to everything. :)
- Download the original model to fine-tune it (I have a .safetensors version so you don't need to load a danger-pickle!), or download my already fine-tuned models (including Text-Encoder-Only version for t2i / t2v / gen-AI):
- 👉 [huggingface.co/zer0int/LongCLIP-Registers-Gated_MLP-ViT-L-14/](https://huggingface.co/zer0int/LongCLIP-Registers-Gated_MLP-ViT-L-14)

![modality-gap-before-after](https://github.com/user-attachments/assets/4a68d089-a2bf-4ad3-a21e-e174c5ecaddf)


### 1st commit: 09/MAR/2025
------
- This was initally an attempt to implement [Paper: Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588v2) 
- ...By just fine-tuning a pre-trained model (yes, a pretty bold (or crazy) idea! 🤣).
- Tl;dr: CLIP hoards global information in local vision (image) patches -> known phenomenon of misleading heatmaps.
- Such 'register tokens' of global information are easily identified: Norm >>100 (normal local patch: <80, ~50).

![example-outliers](https://github.com/user-attachments/assets/90aa5ee8-dc02-46cd-adf0-0c6b9fcc8d1e)

## I just want a new Text Encoder... ✨

- ...for my Text-to-Image (Text-to-Video) AI!
- Direct download for best / balanced model: [click here](https://huggingface.co/zer0int/CLIP-Registers-Gated_MLP-ViT-L-14/resolve/main/ViT-L-14-REG-TE-only-balanced-HF-format-ckpt12.safetensors?download=true)
- Enjoy! (You don't need to do anything else, they're just normal CLIP Text Encoders!)


## Now, about the full model; ViT especially. 🔍
- After initial failures (patch norms ✅, zero-shot accuracy 84.5% -> ~70% ❌ == ruined model):
- Added MLP gates with ReLU to ViT resblocks. Exacerbated patch norm outliers. 🧐
- But: CLIP learned to steer its obsessive hoarding of global information! 🤩
- Result: Modality Gap (Euclidean): (OpenAI pre-trained): 0.8276 --> (THIS): 0.4740 👈🤯
- While also: Zero-shot, retrieval, ... **outperform** original model across the board. ✅
- (Exeption: Minor reduction in linear probe accuracy for *some* datasets)
- See the 'evals_benchmarks_results' for CLIP_Benchmark (LAION) & Benchmarks (code) included here.
- Summary of what changed in the ViT:

```
OpenAI pre-trained: 	Total Parameters: 427,616,513

REG-X-GATED: 		Total Parameters: 452,815,117
|--> + 4 Register Tokens, visual.positional_embedding.shape[0]: 257 -> 261
|--> + MLP with ReLU for every layer (gating) + Final Fusion MLP
|--> + Only during fine-tuning: Geometric Parametrization

# See 'TRAINgmpCLIPregXGATED/model.py' for all details.
```

![REG-8-tsne](https://github.com/user-attachments/assets/aa1016b5-7ff6-4982-9c27-24a5315309d0)

## I want to play with the new CLIP on the block! 🥳

- Grab the full models (not Text Encoder only version) [on my HuggingFace](https://huggingface.co/zer0int/CLIP-Registers-Gated_MLP-ViT-L-14)
- The safetensors are just 'import clip' inside (model structure). Just so you don't need to load a danger-pickle. =)
- Recommended examples:

### Gradient Ascent:

- Assuming you saved the model to a 'models' subfolder in the cloned repo:

```
REG-3-XGATED-gradient-ascent.py --no_adjust --use_model models/ViT-L-14-REG-GATED-balanced-ckpt12.safetensors --deterministic
```

- `--deterministic` for reproducable results / comparison between models.
- `--no_adjust` because it's a weird softmax expansion that doesn't work too well as of yet. :)
- `--use_image path/to/image.png` to use your own image. Else uses default included example.

-----

- The 'OAI' versions of (ANY!) code are for original OpenAI / CLIP models ('import clip').
- If `--use_model` is not provided, defaults to 'ViT-L/14'.
- Of course you can use custom models as well. For example, [direct download my ViT-L/14-GmP](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/resolve/main/ViT-L-14-GmP-ft.safetensors?download=true), then:

```
python REG-3-OAI-CLIP-gradient-ascent.py --no_adjust --use_model models/ViT-L-14-GmP-ft.safetensors --deterministic
```

- ⚠️ Just ensure to load 'normal' models with 'OAI' code, and register-gated models with 'REG'. Else throws error at you. :)
- Example benefit of low modality gap, gradient ascent: Look at that loss!
![REG-3-gradient-ascent](https://github.com/user-attachments/assets/c934a99f-85c4-40d0-9d53-6447d26e5160)

### Attention Heatmaps:

- The 'EXACT' variants create large images with exact (patch / square) attention heatmaps, if you need them. Takes longer.
- Recommended (use the fast and more visually aesthetic version):

```
REG-5-XGATED-visualize-attention-heatmaps.py --use_model models/ViT-L-14-REG-GATED-balanced-ckpt12.safetensors
```

- You can also specify `--token_folder` and `--image_folder`. format is "image.png" -> "tokens_image.txt". Space as separator. Check `EX-tokens-vis` and `EX-image-vis` for default examples!
- Or just use the above mentioned gradient ascent to get a CLIP opinion (texts) about your own images; they'll be saved to a 'more-tokens' subfolder!
- Batch processing only -> put your image(s) into a folder and use that as `--image_folder path/to/myimages` and `--token_folder more-tokens` after getting CLIP opinions.

### Same syntax applies for the other code. Please check the code for details - it's well-documented (I hope)! 🤗

- Attention heatmap, OpenAI ViT-L/14 pre-trained:
![openai-coffee](https://github.com/user-attachments/assets/af8cc3dd-bfcf-4edf-b6aa-fe5206b2b572)

- Attention heatmap, REG-XGATED fine-tune:
![x-reg-coffee](https://github.com/user-attachments/assets/cc62410c-391a-4def-a3c1-669231edaf85)

### Selected visual examples (as an image is worth 16x16 words):

- Patch Cos Sim (Patch-wise Cosine Similarity with Text)
![REG-6-cos-patch](https://github.com/user-attachments/assets/2989d080-c993-4f58-8b59-87f7581e5a53)

- Modgap (Modality Gap)
![REG-8-modgap](https://github.com/user-attachments/assets/dc34e4c3-0867-4cc2-a7f3-841e666c80b4)

- Direct Ascent Synthesis / see also: [github.com/zer0int/CLIP-Direct-Ascent-Synthesis](https://github.com/zer0int/CLIP-Direct-Ascent-Synthesis)
![DAS-example](https://github.com/user-attachments/assets/98311512-35a4-438e-bd2b-cd4756321785)

### Evals
- Please see respective code for details. See the (below) or 'evals_benchmarks_results' folder for results.

-----
## I want to fine-tune my own mutant REG-XGATED CLIP. 🤓

- That means you probably already know what you're doing. 🙃
- Run REG-0 `REG-0-register-token-init-kit.py` on a large dataset (of images only)
- Gets 'natural' self-emergent CLIP register tokens as init for the +4 appended, trainable registers.
- REG-1 (finetune), REG-2 (convert Geometric Parametrization .theta .r -> back to .weight)
- Please check the (extensive!) comments inside the code for details!

### Text-To-Image, Flux.1-dev, pure CLIP guidance (no T5)
- See examples in the ComfyUI workflows folder!

![clip-vs-reg-example-flux](https://github.com/user-attachments/assets/0817b48a-b0da-44cb-9357-5c52881acd6f)

## Model Performance Overview

| Task / Dataset | Metric | ViT-L/14 OpenAI (Pre-trained) | X-GATED (ckpt20 xtreme) | X-GATED (ckpt12 balanced) | X-GATED (ckpt12 balanced, ablated) |
|----------------|--------|-------------------------------|-------------------------|---------------------------|------------------------------------|
| **VoC-2007 (Multilabel)** | mAP | 0.7615 | 0.8140 | **0.8471** | 0.8247 |
| **MSCOCO Retrieval** | Image Recall@5 | 0.2194 | **0.3565** | 0.3532 | 0.3349 |
|  | Text Recall@5 | 0.3034 | **0.5425** | 0.5278 | 0.5086 |
| **Linear Probe CIFAR-10** | Acc@1 | 0.9535 | **0.9813** | **0.9813** | 0.9811 |
|  | Acc@5 | 0.9966 | **0.9997** | **0.9997** | **0.9997** |
|  | Mean Class Recall | 0.9535 | **0.9813** | **0.9813** | 0.9811 |
| **MVT ImageNet/ObjectNet (Zero-Shot)** | Accuracy | 0.8453 | 0.8686 | **0.8830** | 0.8815 |
| **Linear Probe ILSVRC2012** | Top-1 | **69.86%** | 66.43% | 67.10% | 68.99% |
|  | Top-5 | **92.70%** | 91.52% | 91.83% | 92.64% |
| **Modality Gap Metrics** | Euclidean Gap ↓ | 0.8276 | **0.4740** | 0.5395 | 0.7486 |
|  | JSD ↓ | 0.5200 | 0.1601 | **0.1303** | 0.3310 |
|  | Wasserstein Distance ↓ | 0.4084 | **0.1742** | 0.2102 | 0.3262 |
|  | Img-Text Cos Sim (mean) ↑ | 0.2723 | **0.4926** | 0.4794 | 0.3634 |
|  | Img-Text Cos Sim (std) | 0.0362 | 0.0814 | 0.0758 | 0.0537 |
|  | Text-Text Cos Sim (mean) | 0.6807 | 0.6657 | **0.6896** | **0.6896** |
|  | Text-Text Cos Sim (std) | 0.1344 | 0.1671 | 0.1535 | 0.1535 |

*Bolded values represent the best performance for each metric.*

