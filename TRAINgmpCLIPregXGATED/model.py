from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn

"""
Geometric Parametrization inspired by this paper:

ReLU Characteristic Activation Analysis
https://arxiv.org/abs/2305.15912v4
"""

class GeometricLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GeometricLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Radial component
        self.r = nn.Parameter(torch.Tensor(out_features, 1))
        # Angular component
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))
        fan_in = self.in_features
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.r, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        u = F.normalize(self.theta, p=2, dim=1)  # Normalize theta to get unit vector u
        output = F.linear(input, self.r * u)     # Geometric parameterization
        if self.bias is not None:
            output += self.bias
        return output


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", GeometricLinear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", GeometricLinear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask = None
        if self.attn_mask is not None:
            n_ctx = x.shape[0]
            attn_mask = self.attn_mask[..., -n_ctx:, -n_ctx:].to(dtype=x.dtype, device=x.device)            
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# This class integrates the intermediate gating mechanism into the vision transformer.
class VitTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, gate_start_layer: int, num_registers: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.gate_start_layer = gate_start_layer
        self.num_registers = num_registers

        # Use ModuleList to allow non-inplace updates and per-layer operations.
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        
        if layers >= gate_start_layer:
            self.intermediate_fusion_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * width, width),
                    nn.ReLU(),
                    nn.Linear(width, 1)
                ) for _ in range(layers - gate_start_layer + 1)
            ])
            # Initialize each intermediate gating MLP
            for gate in self.intermediate_fusion_mlps:
                for m in gate.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        else:
            self.intermediate_fusion_mlps = None

    def forward(self, x: torch.Tensor):
        # x shape: [seq_len, batch, width]
        for i, block in enumerate(self.resblocks):
            x = block(x)
            if self.intermediate_fusion_mlps is not None and (i + 1) >= self.gate_start_layer:
                cls_token = x[0]                     # [batch, width]
                reg_tokens = x[1:1+self.num_registers] # [num_registers, batch, width]
                reg_summary = reg_tokens.mean(dim=0)   # [batch, width]
                fusion_input = torch.cat([cls_token, reg_summary], dim=-1)  # [batch, 2*width]
                gate_index = (i + 1) - self.gate_start_layer  # index into intermediate_fusion_mlps
                gate = torch.sigmoid(self.intermediate_fusion_mlps[gate_index](fusion_input))  # [batch, 1]
                fused = gate * cls_token + (1 - gate) * reg_summary  # fused representation
                # Instead of an in-place update, create a new tensor for x:
                x = torch.cat([fused.unsqueeze(0), x[1:]], dim=0)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, num_registers: int = 4):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.num_registers = num_registers

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        num_patches = (input_resolution // patch_size) ** 2
        num_tokens = num_patches + 1 + num_registers  # CLS + patches + registers
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_tokens, width))

        self.ln_pre = LayerNorm(width)

        # Define register tokens as an empty placeholder (filled later by state_dict)
        self.register_tokens = nn.Parameter(torch.empty(num_registers, width))

        # Use the new VitTransformer
        self.transformer = VitTransformer(width, layers, heads, gate_start_layer=14, num_registers=num_registers)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        # Final fusion MLP gate
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.ReLU(),
            nn.Linear(width, 1)
        )
        # Initialize final fusion MLP
        for m in self.fusion_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls_token = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        register_tokens = self.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)

        # Concatenate tokens: CLS, REG, and patch tokens
        x = torch.cat([cls_token, register_tokens, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # shape: [seq_len, batch, width]
        x = self.transformer(x) # Now uses the VitTransformer with intermediate gating
        x = x.permute(1, 0, 2)  # shape: [batch, seq_len, width]

        # Final fusion
        cls_token = self.ln_post(x[:, 0, :])  # CLS token after final layer normalization
        reg_tokens = self.ln_post(x[:, 1:1+self.num_registers, :])  # REG tokens normalization
        reg_summary = reg_tokens.mean(dim=1)
        fusion_input = torch.cat([cls_token, reg_summary], dim=-1)
        gate = torch.sigmoid(self.fusion_mlp(fusion_input))
        fused = gate * cls_token + (1 - gate) * reg_summary

        if self.proj is not None:
            fused = fused @ self.proj

        return fused


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)

            # Handle GeometricLinear layers
            if isinstance(block.mlp.c_fc, GeometricLinear):
                nn.init.normal_(block.mlp.c_fc.r, std=fc_std)
                nn.init.kaiming_uniform_(block.mlp.c_fc.theta, a=np.sqrt(5))
            else:
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)

            if isinstance(block.mlp.c_proj, GeometricLinear):
                nn.init.normal_(block.mlp.c_proj.r, std=proj_std)
                nn.init.kaiming_uniform_(block.mlp.c_proj.theta, a=np.sqrt(5))
            else:
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        n_ctx = text.shape[-1]
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding[:n_ctx].type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear, GeometricLinear)):
            if isinstance(l, GeometricLinear):
                l.r.data = l.r.data.half()
                l.theta.data = l.theta.data.half()
            else:
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)


def adjust_state_dict(state_dict, regtoken_path="regtokens"):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "mlp.c_fc.weight" in key:
            base_key = key.replace("weight", "")
            new_state_dict[base_key + "r"] = torch.norm(value, dim=1, keepdim=True)
            new_state_dict[base_key + "theta"] = F.normalize(value, p=2, dim=1)
        elif "mlp.c_proj.weight" in key:
            base_key = key.replace("weight", "")
            new_state_dict[base_key + "r"] = torch.norm(value, dim=1, keepdim=True)
            new_state_dict[base_key + "theta"] = F.normalize(value, p=2, dim=1)
        else:
            new_state_dict[key] = value

    old_pos_embed = state_dict["visual.positional_embedding"]
    old_size = old_pos_embed.shape[0]
    new_size = 261  # 256 [VIS] + 1 [CLS] + 4 [REG] tokens
    if old_size != new_size:
        print(f"[---! INFO !---] Expanding positional embedding from {old_size} → {new_size}")
        mean_embedding = old_pos_embed.mean(dim=0, keepdim=True)
        expanded_embedding = torch.cat([old_pos_embed, mean_embedding.repeat(4, 1)], dim=0)
        new_state_dict["visual.positional_embedding"] = expanded_embedding
    else:
        print("[---! OK !---] Positional embedding size is already correct, skipping expansion.")

    # Inject Register Tokens: Load if available, otherwise initialize randomly
    if "visual.register_tokens" not in state_dict:
        print("[---! INFO !---] Register tokens missing from state_dict, attempting to load from files...")

        reg_tokens = []
        for i in range(1, 5):
            token_path = os.path.join(regtoken_path, f"top{i}_mean.pt")
            if os.path.exists(token_path):
                reg_tokens.append(torch.load(token_path))
            else:
                print(f"[---! INFO !---] {token_path} not found. Using random initialization to instantiate model.")
                reg_tokens.append(torch.randn(1024, dtype=torch.float32))

        new_state_dict["visual.register_tokens"] = torch.stack(reg_tokens)
    else:
        print("[---! OK !---] [REG] tokens already present, skipping injection.")

    return new_state_dict


def build_model(state_dict: dict, regtoken_path="regtokens"):
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    state_dict = adjust_state_dict(state_dict, regtoken_path)
    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
