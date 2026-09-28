"""Masked autoregressive (MAR) machinery shared by the DREAM and REPA models.

Subclasses create the modules (z_proj, encoder/decoder blocks, text aligner, diffloss, ...) and implement
forward_mae_encoder(x, mask=..., class_embedding=...) -> (x, rep). This class holds no parameters itself,
so it does not affect checkpoint keys.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from models.layers import mask_by_order


class MaskedGenerativeModel(nn.Module):

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

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(self.fake_latent.device).long()
        return orders

    def random_masking(self, x, orders, step_idx=1):
        # generate token mask; step_idx (fractional epoch) drives the Masking Warmup schedule
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(step_idx)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_text_aligner(self, text_embedding_t5, use_checkpoint=False):
        text_embedding = self.text_proj(text_embedding_t5) + self.text_pos_emb
        text_embedding = self.text_proj_ln(text_embedding)
        for block in self.text_aligner:
            if use_checkpoint:
                text_embedding = checkpoint(block, text_embedding)
            else:
                text_embedding = block(text_embedding)
        return text_embedding

    def forward_mae_decoder(self, x, mask, text_embedding):
        """
        text-conditioned decoder, following FLUID's implementation
        """
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned
        x = self.decoder_proj_ln(x)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
            for block in self.decoder_blocks:
                x = checkpoint(block, x, text_embedding)
        else:
            for block in self.decoder_blocks:
                x = block(x, text_embedding)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    # --------------------------------------------------------------------------
    # Generation

    def _init_generation(self, bsz, text_embedding):
        device = text_embedding.device
        mask = torch.ones(bsz, self.seq_len, device=device)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device)
        orders = self.sample_orders(bsz)
        text_embedding = self.forward_text_aligner(text_embedding)
        return mask, tokens, orders, text_embedding

    def _generation_step(self, tokens, mask, orders, step, num_iter, cfg, cfg_schedule, text_embedding):
        """
        One masked-generation step.
        Returns the decoder outputs at the positions to predict (duplicated for CFG), the [bsz, seq_len]
        boolean mask of those positions, the mask for the next step, and the CFG scale for this step.
        """
        bsz = tokens.size(0)
        device = tokens.device

        # label is tensor of zeros
        class_embedding = self.class_emb(torch.zeros(bsz, dtype=torch.long, device=device))

        if not cfg == 1.0:
            tokens = torch.cat([tokens, tokens], dim=0)
            class_embedding = torch.cat([class_embedding, class_embedding], dim=0)
            mask = torch.cat([mask, mask], dim=0)

        # mae encoder
        x, _ = self.forward_mae_encoder(x=tokens, mask=mask, class_embedding=class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask, text_embedding)

        # mask ratio for the next round, following MaskGIT and MAGE.
        mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
        mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(device)

        # masks out at least one for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                 torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

        # get masking for next iteration and locations to be predicted in this iteration
        mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
        if step >= num_iter - 1:
            mask_to_pred = mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())

        mask_to_pred_z = torch.cat([mask_to_pred, mask_to_pred], dim=0) if not cfg == 1.0 else mask_to_pred
        z = z[mask_to_pred_z.nonzero(as_tuple=True)]

        # cfg schedule follow Muse
        if cfg_schedule == "linear":
            cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
        elif cfg_schedule == "constant":
            cfg_iter = cfg
        else:
            raise NotImplementedError

        return z, mask_to_pred, mask_next, cfg_iter

    def _sample_token_latents(self, z, temperature, cfg_iter, cfg):
        sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
        if not cfg == 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
        return sampled_token_latent

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", text_embedding=None, temperature=1.0, progress=False):
        mask, tokens, orders, text_embedding = self._init_generation(bsz, text_embedding)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            z, mask_to_pred, mask, cfg_iter = self._generation_step(tokens, mask, orders, step, num_iter, cfg, cfg_schedule, text_embedding)
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = self._sample_token_latents(z, temperature, cfg_iter, cfg)
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens
