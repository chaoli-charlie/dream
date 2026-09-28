from functools import partial

import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block

from models.base import MaskedGenerativeModel
from models.diffloss import DiffLoss
from models.layers import CrossAttentionBlock, repa_alignment_loss


class REPA_MAR_TXT_CONDITIONAL(MaskedGenerativeModel):
    """ Text-conditioned MAR baseline with REPA alignment of intermediate encoder features to DINOv2
    """
    def __init__(self, img_size=256, vae_stride=8, patch_size=2,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=4,
                 mask_ratio_mu=0.55,
                 mask_ratio_std=0.25,
                 mask_ratio_min=0.7,
                 mask_ratio_max=1.0,
                 label_drop_prob=1.0, # 0.1 default
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 text_aligner_depth=6,
                 t5_embed_dim=1024,
                 text_max_len=128,
                 encoder_depth_proj=8,
                 projector_dim = 2048, # dino embedding dim
                 z_dim = 768, # dino embedding dim
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

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # masking ratio: truncated Gaussian on [mask_ratio_min, mask_ratio_max]
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std, (mask_ratio_max - mask_ratio_mu) / mask_ratio_std, loc=mask_ratio_mu, scale=mask_ratio_std)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim)) # (1, 256 + 64, 1024)

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
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


        # REPA projector, applied to the output of encoder block `encoder_depth_proj`
        self.encoder_depth_proj = encoder_depth_proj
        self.projector = self.build_mlp(encoder_embed_dim, projector_dim, z_dim)

    def build_mlp(self, hidden_size, projector_dim, z_dim):
        return nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

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

    def forward_mae_encoder(self, x, mask, class_embedding):
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

        # apply Transformer blocks; rep is the projected intermediate representation aligned to DINOv2
        for i, block in enumerate(self.encoder_blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)
            if (i + 1) == self.encoder_depth_proj:
                rep = self.projector(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1], -1)

        x = self.encoder_norm(x)

        return x, rep

    def forward(self, imgs, labels, text_embedding_t5, target_rep):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x, rep = self.forward_mae_encoder(x, mask, class_embedding)

        # REPA loss on the visible image tokens
        repa_loss = repa_alignment_loss(rep, target_rep, mask, self.buffer_size)

        # mae decoder
        use_checkpoint = self.grad_checkpointing and not torch.jit.is_scripting()
        text_embedding = self.forward_text_aligner(text_embedding_t5, use_checkpoint=use_checkpoint)
        z = self.forward_mae_decoder(x, mask, text_embedding)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss, repa_loss

def repa_mar_large_txt_conditional(**kwargs):
    model = REPA_MAR_TXT_CONDITIONAL(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model

def repa_mar_base_txt_conditional(**kwargs):
    model = REPA_MAR_TXT_CONDITIONAL(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model

def repa_mar_huge_txt_conditional(**kwargs):
    model = REPA_MAR_TXT_CONDITIONAL(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model