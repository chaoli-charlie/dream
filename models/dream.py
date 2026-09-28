import random
from collections import OrderedDict
from functools import partial

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from tqdm import tqdm

from models.base import MaskedGenerativeModel
from models.clip_loss import CLIPLoss
from models.diffloss import DiffLoss
from models.layers import CLIPTextCfg, CrossAttentionBlock, LayerNorm, QuickGELU, Transformer, repa_alignment_loss


class NEVER_MASK():
    def rvs(self, _):
        return (0.0, 0.0)  # always return 0.0, 0.0 for never masking


class MASKING_SCHEDULER():
    """Masking Warmup: shifts the mean of a truncated Gaussian masking distribution over training."""

    def __init__(self, mask_ratio_min, mask_ratio_max, mask_ratio_std, mask_ratio_mu_start, mask_ratio_mu_end, total_steps,
                 fixed_boundary=False, warmup=0, warmup_end=None, cooldown=0, warmup_shape="linear", schedule_std=False):
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.mask_ratio_std = mask_ratio_std
        self.mask_ratio_std_start = mask_ratio_std
        self.mask_ratio_mu_start = mask_ratio_mu_start
        self.mask_ratio_mu_end = mask_ratio_mu_end
        self.warmup = warmup
        self.warmup_end = warmup_end
        self.cooldown = cooldown
        self.warmup_shape = warmup_shape
        self.schedule_std = schedule_std

        self.total_steps = total_steps # num epochs
        self.fixed_boundary = fixed_boundary

        self.print_info()

    def print_info(self):
        info = "Masking Scheduler Info:\n"
        info += f"  Mask Ratio (Min, Max, Mu, Std): ({self.mask_ratio_min}, {self.mask_ratio_max}, {self.mask_ratio_mu_start}, {self.mask_ratio_std_start})\n"
        info += f"  Warmup Steps: {self.warmup}\n"
        info += f"  Warmup End Steps: {self.warmup_end}\n"
        info += f"  Cooldown Steps: {self.cooldown}\n"
        info += f"  Warmup Shape: {self.warmup_shape}\n"
        info += f"  Schedule Std: {self.schedule_std}\n"
        info += f"  Total Steps: {self.total_steps}\n"
        info += f"  Fixed Boundary: {self.fixed_boundary}\n"
        print(info)

    def std_scheduler(self, step_idx):
        if step_idx >= self.total_steps - self.cooldown:
            return max(self.mask_ratio_std_start, 1e-6)  # Ensure a small positive value

        std_value = self.mask_ratio_std_start * (step_idx - self.warmup) / (self.total_steps - self.cooldown - self.warmup)
        return max(std_value, 1e-6)  # Ensure a small positive value

    def _interpolate_mu(self, progress):
        if self.warmup_shape == "linear":
            return self.mask_ratio_mu_start + (self.mask_ratio_mu_end - self.mask_ratio_mu_start) * progress
        elif self.warmup_shape == "cosine":
            return self.mask_ratio_mu_start + 0.5 * (self.mask_ratio_mu_end - self.mask_ratio_mu_start) * (1 - np.cos(np.pi * progress))
        else:
            raise ValueError("Unknown warmup shape: {}".format(self.warmup_shape))

    def compute_warmup_mu(self, step_idx):
        if step_idx < self.warmup:
            return self.mask_ratio_min
        elif step_idx >= self.total_steps - self.cooldown:
            return self.mask_ratio_max
        else:
            return self._interpolate_mu((step_idx - self.warmup) / (self.total_steps - self.cooldown - self.warmup))

    def compute_warmup_mu_end(self, step_idx):
        """
        ignores the cooldown. only use the warmup and the warmup_end
        """
        if step_idx < self.warmup:
            return self.mask_ratio_mu_start
        elif step_idx >= self.warmup_end:
            return self.mask_ratio_mu_end
        else:
            return self._interpolate_mu((step_idx - self.warmup) / (self.warmup_end - self.warmup))

    def rvs(self, step_idx):
        if self.schedule_std:
            self.mask_ratio_std = self.std_scheduler(step_idx)
            self.mask_ratio_std = max(self.mask_ratio_std, 1e-6)

        if self.warmup_end is not None:
            mask_ratio_mu = self.compute_warmup_mu_end(step_idx)
        else:
            mask_ratio_mu = self.compute_warmup_mu(step_idx)

        mask_ratio_mu = min(max(mask_ratio_mu, self.mask_ratio_min), self.mask_ratio_max)
        if self.fixed_boundary:
            a, b = (self.mask_ratio_min - mask_ratio_mu) / self.mask_ratio_std, (self.mask_ratio_max - mask_ratio_mu) / self.mask_ratio_std
        else:
            mask_ratio_min = max(self.mask_ratio_min, mask_ratio_mu - self.mask_ratio_std)
            mask_ratio_max = min(self.mask_ratio_max, mask_ratio_mu + self.mask_ratio_std)
            a, b = (mask_ratio_min - mask_ratio_mu) / self.mask_ratio_std, (mask_ratio_max - mask_ratio_mu) / self.mask_ratio_std
        mask_ratio = stats.truncnorm(a, b, loc=mask_ratio_mu, scale=self.mask_ratio_std).rvs(1)[0]
        return (mask_ratio, 0.0)


class UNIFORM_MASKING():
    def __init__(self, mask_ratio_min, mask_ratio_max):
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max

    def rvs(self, _):
        return (random.uniform(self.mask_ratio_min, self.mask_ratio_max), 0.0)


class DREAM(MaskedGenerativeModel):
    """ Masked autoregressive generator whose encoder is jointly trained with a CLIP-style contrastive loss
    """
    def __init__(self, text_cfg: CLIPTextCfg, img_size=256,
                 vae_stride=8,
                 patch_size=2,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=4,
                 mask_ratio_min=0.7,
                 mask_ratio_max=1.0,
                 mask_ratio_mu=0.55,
                 mask_ratio_std=0.25,
                 label_drop_prob=1.0,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 quick_gelu: bool = False,
                 clip_loss_on="image-text",
                 vl_projection="identity", # "linear", "post_linear", "mlp", "post_mlp", "mlp_stablerep", "post_mlp_stablerep"
                 txt_projection="linear",
                 ssl_mlp_dim=2048,
                 embed_dim=None,
                 variable_masking=False,
                 mask_ratio_mu_start=None,
                 mask_ratio_mu_end=None,
                 epochs=None,
                 fixed_masking_boundary=True,
                 masking_warmup=0,
                 masking_warmup_end=None,
                 masking_cooldown=0,
                 warmup_shape="linear",
                 schedule_std=False,
                 text_aligner_depth=6,
                 t5_embed_dim=1024,
                 text_max_len=128,
                 min_ratio_for_clip_loss=0.0,
                 min_masked_ratio_for_mar_loss=0.0,
                 scale_loss_by_mask=False,
                 filter_clip_loss=False,
                 depth_to_align=-1,
                 uniform_masking=False,
                 repa_depth=None,
                 repa_projector_dim=2048,
                 repa_z_dim=768,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing
        self.clip_loss_on = clip_loss_on

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # Masking ratio distribution
        self.variable_masking = variable_masking
        self.uniform_masking = uniform_masking
        if mask_ratio_min < 0:
            self.mask_ratio_generator = NEVER_MASK()
        elif uniform_masking:
            self.mask_ratio_generator = UNIFORM_MASKING(mask_ratio_min, mask_ratio_max)
        elif variable_masking:
            self.mask_ratio_generator = MASKING_SCHEDULER(mask_ratio_min, mask_ratio_max, mask_ratio_std, mask_ratio_mu_start, mask_ratio_mu_end,
                                                          total_steps=epochs, fixed_boundary=fixed_masking_boundary, warmup=masking_warmup,
                                                          warmup_end=masking_warmup_end, cooldown=masking_cooldown, warmup_shape=warmup_shape,
                                                          schedule_std=schedule_std)
        else:
            self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std, (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                        loc=mask_ratio_mu, scale=mask_ratio_std)

        self.min_ratio_for_clip_loss = min_ratio_for_clip_loss
        self.min_masked_ratio_for_mar_loss = min_masked_ratio_for_mar_loss
        # --------------------------------------------------------------------------
        # DREAM encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)
        self.depth_to_align = depth_to_align

        # --------------------------------------------------------------------------
        # DREAM decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_proj_ln = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                                proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.text_proj = nn.Linear(t5_embed_dim, decoder_embed_dim, bias=True)
        self.text_proj_ln = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.text_pos_emb = nn.Parameter(torch.zeros(1, text_max_len, decoder_embed_dim))
        self.text_aligner = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(text_aligner_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        # --------------------------------------------------------------------------
        # CLIP text tower
        act_layer = QuickGELU if quick_gelu else nn.GELU
        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
            grad_checkpointing=grad_checkpointing
        )
        self.num_pos = self.context_length = text_cfg.context_length

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(text_cfg.context_length, text_cfg.width))
        self.ln_final = LayerNorm(text_cfg.width)

        # projection layers
        if embed_dim is None:
            # to load previous models
            self.embed_dim = encoder_embed_dim
        else:
            self.embed_dim = embed_dim
        self.vl_projection = vl_projection

        if self.vl_projection == "identity":
            self.image_projection = nn.Identity()
        elif self.vl_projection in ("linear", "post_linear"):
            self.image_projection = nn.Parameter(torch.empty(encoder_embed_dim, self.embed_dim))
        elif self.vl_projection in ("mlp", "post_mlp"):
            self.image_projection = self.build_mlp(hidden_size=encoder_embed_dim, projector_dim=ssl_mlp_dim, z_dim=self.embed_dim)
        elif self.vl_projection in ("mlp_stablerep", "post_mlp_stablerep"):
            self.image_projection = self._build_mlp(in_dim=encoder_embed_dim, mlp_dim=ssl_mlp_dim, out_dim=self.embed_dim)
        else:
            raise ValueError(f'Invalid vl_projection: {self.vl_projection}')

        self.txt_projection = txt_projection
        if self.txt_projection == "linear":
            self.text_projection = nn.Parameter(torch.empty(text_cfg.width, self.embed_dim))
        elif self.txt_projection == "mlp":
            self.text_projection = self.build_mlp(hidden_size=text_cfg.width, projector_dim=ssl_mlp_dim, z_dim=self.embed_dim)
        elif self.txt_projection == "stablerep":
            self.text_projection = self._build_mlp(in_dim=text_cfg.width, mlp_dim=ssl_mlp_dim, out_dim=self.embed_dim)
        else:
            raise ValueError(f'Invalid txt_projection: {self.txt_projection}')

        if clip_loss_on == "image-text":
            self.clip_tokens = slice(None)
        elif clip_loss_on == "image-tokens-text":
            self.clip_tokens = slice(buffer_size, None)
        elif clip_loss_on == "buffer":
            self.clip_tokens = slice(None, buffer_size)
        else:
            raise NotImplementedError(f"clip_loss_on {clip_loss_on} not implemented")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.clip_loss_fn = CLIPLoss(scale_loss_by_mask=scale_loss_by_mask, filter_clip_loss=filter_clip_loss)

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        # --------------------------------------------------------------------------
        # Optional REPA alignment of the output of encoder block `repa_depth` to frozen DINOv2 patch features.
        # Created last, so that enabling it leaves the initialization and order of all other parameters unchanged.
        self.repa_depth = repa_depth
        if repa_depth is not None:
            assert 1 <= repa_depth <= encoder_depth, f"repa_depth must be in [1, {encoder_depth}]"
            self.repa_projector = self.build_mlp(encoder_embed_dim, repa_projector_dim, repa_z_dim)

    def build_mlp(self, hidden_size, projector_dim, z_dim):
        return nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

    # stablerep style mlp
    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if self.txt_projection == "linear":
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        if self.vl_projection in ("linear", "post_linear"):
            nn.init.normal_(self.image_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        if self.txt_projection == "linear":
            x = x @ self.text_projection
        else:
            x = self.text_projection(x)
        return x

    def project_image_features(self, image_features, tokens=slice(None)):
        """Pool the selected encoder tokens and map them into the joint image-text embedding space."""
        if self.vl_projection == "identity":
            return image_features[:, tokens].mean(dim=1)
        elif self.vl_projection == "linear":
            return (image_features @ self.image_projection)[:, tokens].mean(dim=1)
        elif self.vl_projection == "post_linear":
            return image_features[:, tokens].mean(dim=1) @ self.image_projection
        elif self.vl_projection in ("mlp", "mlp_stablerep"):
            return self.image_projection(image_features)[:, tokens].mean(dim=1)
        elif self.vl_projection in ("post_mlp", "post_mlp_stablerep"):
            return self.image_projection(image_features[:, tokens].mean(dim=1))
        else:
            raise ValueError(f'Invalid vl_projection: {self.vl_projection}')

    def forward_mae_encoder(self, x, text=None, mask=None, class_embedding=None, return_repa=False):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.device).to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks; rep is the representation used for the contrastive loss,
        # repa_rep the projected features aligned to DINOv2 (with REPA alignment enabled)
        rep = None
        repa_rep = None
        for i, block in enumerate(self.encoder_blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)
            if (self.depth_to_align > 0) and ((i + 1) == self.depth_to_align):
                rep = x
            if return_repa and (i + 1) == self.repa_depth:
                repa_rep = self.repa_projector(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1], -1)
        x = self.encoder_norm(x)

        if self.depth_to_align == -1:
            rep = x

        assert rep is not None

        if text is None:
            return x, rep

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)
        if return_repa:
            return x, text_features, self.logit_scale.exp(), rep, repa_rep
        return x, text_features, self.logit_scale.exp(), rep

    def forward_img_features(self, x, mask=None):
        """Projected image embedding (in the joint image-text space) for a batch of patchified latents."""
        labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        class_embedding = self.class_emb(labels)

        bsz, seq_len, embed_dim = x.shape
        if mask is None:
            mask = torch.zeros(bsz, seq_len, device=x.device) # 0 means unmasked, 1 means masked

        _, rep = self.forward_mae_encoder(x, mask=mask, class_embedding=class_embedding)
        return self.project_image_features(rep)

    def forward(self, imgs, text, labels, text_embedding_t5, est_idx=None, target_rep=None):
        """
        Returns the diffusion (MAR) loss, the contrastive loss, the REPA alignment loss (None unless REPA alignment is
        enabled and target_rep, the DINOv2 patch features of imgs, is given), and masking statistics.
        """

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))

        if self.variable_masking:
            mask = self.random_masking(x, orders, est_idx)
        else:
            mask = self.random_masking(x, orders)

        # mae encoder
        loss_repa = None
        if self.repa_depth is not None and target_rep is not None:
            x, text_features, logit_scale, rep, repa_rep = self.forward_mae_encoder(x, text, mask, class_embedding, return_repa=True)
            loss_repa = repa_alignment_loss(repa_rep, target_rep, mask, self.buffer_size)
        else:
            x, text_features, logit_scale, rep = self.forward_mae_encoder(x, text, mask, class_embedding)

        # mae decoder
        use_checkpoint = self.grad_checkpointing and not torch.jit.is_scripting()
        text_embedding = self.forward_text_aligner(text_embedding_t5, use_checkpoint=use_checkpoint)
        z = self.forward_mae_decoder(x, mask, text_embedding)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        # boolean mask indicating whether the sample has enough unmasked tokens for clip loss
        clip_loss_mask = (1 - mask.sum(dim=1) / self.seq_len) >= self.min_ratio_for_clip_loss

        image_features_proj = self.project_image_features(rep, self.clip_tokens)
        loss_clip, avg_unmasked_ratio = self.clip_loss_fn(image_features=image_features_proj, text_features=text_features, logit_scale=logit_scale,
                                                          loss_mask=clip_loss_mask, min_ratio_for_clip_loss=self.min_ratio_for_clip_loss)

        # now set a min ratio for mar loss
        batch_masked_ratio = (mask.sum(dim=1) / self.seq_len).mean().item()
        avg_masked_ratio = 1.0 # default to 100% masked for this gpu batch
        if batch_masked_ratio < self.min_masked_ratio_for_mar_loss:
            loss = loss * 0.0
            avg_masked_ratio = avg_masked_ratio * 0.0

        # compute the mean masking ratio for the batch
        mask_info = {
            "mar_mask_ratio": mask.mean().item(),
            "clip_mask_ratio": mask.mean().item(),
            "percentage_batch_with_clip": avg_unmasked_ratio,
            "percentage_batch_with_mar": avg_masked_ratio,
        }

        return loss, loss_clip, loss_repa, mask_info

    # --------------------------------------------------------------------------
    # Semantically Aligned Decoding (CLIP critic)

    @torch.no_grad()
    def _clip_scores(self, tokens, mask, text_features):
        """Cosine similarity between each (partially generated) image and its caption: [bsz], higher is better."""
        bsz = tokens.size(0)
        class_embedding = self.class_emb(torch.zeros(bsz, dtype=torch.long, device=tokens.device))
        _, rep = self.forward_mae_encoder(x=tokens, mask=mask, class_embedding=class_embedding)
        img = F.normalize(self.project_image_features(rep), dim=-1)
        return (img * text_features).sum(dim=-1)

    def _encode_critic_text(self, text):
        if text is None:
            raise ValueError("text is required for CLIP critic sampling")
        return F.normalize(self.encode_text(text), dim=-1)

    def sample_tokens_with_clip_critic(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", text_embedding=None, temperature=1.0, progress=False, text=None, num_candidates=5, clip_critic_threshold=32):
        """
        Sample tokens using CLIP critic to select the best candidate at each step.
        Uses original sampling method for iterations < clip_critic_threshold,
        then switches to CLIP-based sampling for remaining iterations.
        At each such step, samples num_candidates candidates and keeps, per image, the one with the highest CLIP score.
        """
        mask, tokens, orders, text_embedding = self._init_generation(bsz, text_embedding)
        text_features = self._encode_critic_text(text)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            z, mask_to_pred, mask, cfg_iter = self._generation_step(tokens, mask, orders, step, num_iter, cfg, cfg_schedule, text_embedding)
            pred_positions = mask_to_pred.nonzero(as_tuple=True)

            if step < clip_critic_threshold:
                # Use original sampling method (no CLIP critic)
                cur_tokens[pred_positions] = self._sample_token_latents(z, temperature, cfg_iter, cfg)
            else:
                # Track best candidate for each sample independently
                best_scores = None # shape [bsz]
                best_cand = None  # shape [N_masked, z_dim]
                rows = pred_positions[0]

                for _ in range(num_candidates):
                    cand = self._sample_token_latents(z, temperature, cfg_iter, cfg)

                    # Score the complete image so far with this candidate filled in
                    temp_tokens = cur_tokens.clone()
                    temp_tokens[pred_positions] = cand
                    scores = self._clip_scores(temp_tokens, mask, text_features)

                    if best_scores is None:
                        best_scores = scores
                        best_cand = cand.clone()
                    else:
                        better = scores > best_scores # [bsz]
                        if better.any():
                            # update only rows belonging to improved samples
                            for s in torch.where(better)[0].tolist():
                                sel = (rows == s)
                                if sel.any():
                                    best_cand[sel] = cand[sel]
                            best_scores[better] = scores[better]

                cur_tokens[pred_positions] = best_cand
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens

    def sample_tokens_with_clip_critic_once(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", text_embedding=None, temperature=1.0, progress=False, text=None, num_candidates=5, clip_critic_threshold=32):
        """
        Semantically Aligned Decoding: generate num_candidates different candidates for clip_critic_threshold steps.
        Once this step is reached, use the text encoder to pick the best candidate per image,
        and then continue the generation only for this candidate.
        """
        assert clip_critic_threshold < num_iter, "clip_critic_threshold must be less than num_iter"

        mask, tokens, orders, text_embedding = self._init_generation(bsz, text_embedding)
        text_features = self._encode_critic_text(text)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # Initialize multiple candidates for parallel generation
        candidate_tokens = tokens.unsqueeze(0).repeat(num_candidates, 1, 1, 1)
        candidate_masks = mask.unsqueeze(0).repeat(num_candidates, 1, 1)

        # generate latents
        for step in indices:
            if step < clip_critic_threshold:
                for candidate_idx in range(num_candidates):
                    cur_tokens = candidate_tokens[candidate_idx].clone()
                    z, mask_to_pred, cur_mask, cfg_iter = self._generation_step(
                        cur_tokens, candidate_masks[candidate_idx], orders, step, num_iter, cfg, cfg_schedule, text_embedding)
                    cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = self._sample_token_latents(z, temperature, cfg_iter, cfg)

                    candidate_tokens[candidate_idx] = cur_tokens
                    candidate_masks[candidate_idx] = cur_mask
                continue

            if step == clip_critic_threshold:
                # Select the best candidate using CLIP scoring (aggregate over candidates, not over batch)
                scores_all = torch.stack([self._clip_scores(candidate_tokens[c], candidate_masks[c], text_features)
                                          for c in range(num_candidates)])  # [num_candidates, bsz]
                best_idx_per_sample = scores_all.argmax(dim=0)  # [bsz]

                # Gather tokens and masks per-sample from the best candidate
                batch_indices = torch.arange(bsz, device=tokens.device)
                tokens = candidate_tokens[best_idx_per_sample, batch_indices].clone()  # [bsz, seq_len, token_dim]
                mask = candidate_masks[best_idx_per_sample, batch_indices].clone()    # [bsz, seq_len]
                del candidate_tokens, candidate_masks

            # Continue with regular generation for the selected candidate
            cur_tokens = tokens.clone()
            z, mask_to_pred, mask, cfg_iter = self._generation_step(tokens, mask, orders, step, num_iter, cfg, cfg_schedule, text_embedding)
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = self._sample_token_latents(z, temperature, cfg_iter, cfg)
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens


def dream_base_txt_conditional(**kwargs):
    model = DREAM(text_cfg=CLIPTextCfg(),
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dream_large_txt_conditional(**kwargs):
    model = DREAM(text_cfg=CLIPTextCfg(),
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dream_huge_txt_conditional(**kwargs):
    model = DREAM(text_cfg=CLIPTextCfg(),
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dream_giant_txt_conditional(**kwargs):
    model = DREAM(text_cfg=CLIPTextCfg(),
        encoder_embed_dim=1664, encoder_depth=24, encoder_num_heads=16,
        decoder_embed_dim=1664, decoder_depth=24, decoder_num_heads=16,
        mlp_ratio=8192/1664, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
