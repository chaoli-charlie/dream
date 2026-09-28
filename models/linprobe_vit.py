from functools import partial

import torch
import torch.nn as nn
from diffusers import AutoencoderKL as StableAIAutoencoderKL
from timm.models.vision_transformer import Block

from models.vae import AutoencoderKL

class MaskedGenerativeEncoderViT(nn.Module):
    """ The DREAM/MAR encoder with a classification head, for linear probing on frozen features
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_path=None, vae_embed_dim=16,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=1,
                 autoencoder_type="default",
                 pooled_tokens = "buffer",
                 use_class_emb=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_path = vae_path
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2

        # --------------------------------------------------------------------------
        # Class Embedding and CFG
        self.num_classes = class_num
        self.class_emb = nn.Embedding(1000, encoder_embed_dim)
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for i in range(encoder_depth)])
        self.initialize_weights()

        self.fc_norm = norm_layer(encoder_embed_dim)
        self.encoder_norm = norm_layer(encoder_embed_dim)
        self.head = nn.Linear(encoder_embed_dim, class_num)

        # --------------------------------------------------------------------------
        self.autoencoder_type = autoencoder_type
        print("Autoencoder type:", self.autoencoder_type)
        if self.autoencoder_type == "default":
            # MAR's KL-16 VAE
            self.vae = AutoencoderKL(embed_dim=vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=vae_path).eval()
            # hard coded scale factor
            self.scale_factor = 0.2325
        elif self.autoencoder_type == "sd":
            self.vae = StableAIAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").eval()
            self.scale_factor = 0.18215
        else:
            raise ValueError(f"Unsupported autoencoder type: {self.autoencoder_type}")

        for param in self.vae.parameters():
            param.requires_grad = False

        self.pooled_tokens = pooled_tokens
        print("Pooled tokens type:", self.pooled_tokens)

        self.use_class_emb = use_class_emb
        print("Use class embedding:", self.use_class_emb)

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def tokenize(self, x):
        # vae encoding
        with torch.no_grad():
            self.vae.eval()
            if self.autoencoder_type == "default":
                x = self.vae.encode(x).sample().mul_(self.scale_factor)
            else:
                x = self.vae.encode(x).latent_dist.sample().mul_(self.scale_factor)

        # patchify
        bsz, c, h, w = x.shape
        x = x.reshape(bsz, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size).permute([0, 1, 3, 5, 2, 4])
        x = x.reshape(bsz, c * self.patch_size**2, h // self.patch_size, w // self.patch_size)
        x = x.permute([0, 2, 3, 1]).reshape(bsz, h // self.patch_size * w // self.patch_size, c * self.patch_size**2)
        return x

    def forward_encoder(self, x):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)

        if self.use_class_emb:
            # add class embedding
            labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            class_embedding = self.class_emb(labels)
            x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        else:
            x[:, :self.buffer_size] = self.fake_latent.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)

        if self.pooled_tokens == "buffer":
            x = x[:, :self.buffer_size, :].mean(dim=1)  # global pool without cls token
        elif self.pooled_tokens == "all":
            x = x.mean(dim=1)
        elif self.pooled_tokens == "image":
            x = x[:, self.buffer_size:, :].mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooled_tokens type: {self.pooled_tokens}")
        x = self.fc_norm(x)

        return x

    def forward(self, imgs):
        # tokenize and drop
        x = self.tokenize(imgs)
        # encoder
        x = self.forward_encoder(x)
        # linear head
        x = self.head(x)
        return x

    def forward_without_head(self, imgs):
        # tokenize and drop
        x = self.tokenize(imgs)
        # encoder
        x = self.forward_encoder(x)
        return x


def mar_base_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_giant_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        encoder_embed_dim=1664, encoder_depth=24, encoder_num_heads=16,
        mlp_ratio=8192/1664, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


