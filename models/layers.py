"""Building blocks shared by the DREAM and REPA models."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Attention, Mlp


def repa_alignment_loss(rep, target_rep, mask, buffer_size):
    """
    REPA loss: 1 - cosine similarity between projected encoder features and frozen DINOv2 patch features, over the
    visible (unmasked) image tokens. rep: [B, buffer + visible, D] (buffer tokens first); target_rep: [B, seq_len, D];
    mask: [B, seq_len] with 1 for masked tokens.
    """
    target_rep = target_rep[(1 - mask).nonzero(as_tuple=True)].reshape(rep.shape[0], -1, rep.shape[-1])
    masked_ratio = torch.sum(mask) / mask.numel()
    cos_sim = F.cosine_similarity(rep[:, buffer_size:, :], target_rep.detach(), dim=-1)
    loss = (1 - cos_sim).mean()

    if masked_ratio == 1.0:
        # no visible tokens: the mean over an empty tensor is nan
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
                            src=torch.ones(bsz, seq_len, device=order.device)).bool()
    return masking


class CrossAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
        query from decoder (x), key and value from encoder (y)
        """
        B, N, C = x.shape
        Ny = y.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.)
        x = attn.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 proj_drop=0.0, attn_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm0 = norm_layer(decoder_dim)
        self.self_attn = Attention(decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm1 = norm_layer(decoder_dim)
        self.cross_attn = CrossAttention(encoder_dim, decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(decoder_dim)
        self.mlp = Mlp(in_features=decoder_dim, hidden_features=int(decoder_dim * mlp_ratio),
                       act_layer=act_layer, drop=proj_drop)

    def forward(self, x, y):
        """
        x: decoder feature; y: encoder feature (after layernorm)
        """
        x = x + self.self_attn(self.norm0(x))
        x = x + self.cross_attn(self.norm1(x), y)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------
# CLIP text tower (from OpenCLIP)

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU, grad_checkpointing: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = grad_checkpointing

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x
