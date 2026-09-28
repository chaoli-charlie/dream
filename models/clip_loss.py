import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F

import util.misc as misc


def all_gather_with_grad(x):
    if not misc.is_dist_avail_and_initialized():
        return x
    return torch.cat(torch.distributed.nn.all_gather(x), dim=0)


class CLIPLoss(nn.Module):
    def __init__(self, scale_loss_by_mask=False, filter_clip_loss=False):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.scale_loss_by_mask = scale_loss_by_mask
        self.filter_clip_loss = filter_clip_loss

    def forward(self, image_features, text_features, logit_scale, loss_mask=None, min_ratio_for_clip_loss=None):
        """
        image_features are already pooled over tokens (the mean is taken before calling this function).
        loss_mask marks samples with enough unmasked image tokens to contribute to the loss.
        Returns the loss and the fraction of the global batch that contributes to it.
        """
        neg_inf = -1e9
        eps = 1e-6

        image_embed = image_features
        text_embed = text_features
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * misc.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather with gradient
        image_embed_all = all_gather_with_grad(image_embed)
        text_embed_all = all_gather_with_grad(text_embed)
        loss_mask_all = all_gather_with_grad(loss_mask)

        avg_unmasked_ratio = loss_mask_all.sum(dim=0) / (loss_mask_all.shape[0] + eps)

        if self.filter_clip_loss:
            logits_per_image = logit_scale * image_embed_all @ text_embed_all.t()
            logits_per_text = logit_scale * text_embed_all @ image_embed_all.t()

            # exclude samples without enough unmasked tokens, both as queries and as negatives
            logits_per_image[~loss_mask_all] = neg_inf
            logits_per_text[:, ~loss_mask_all] = neg_inf

            labels_all = torch.arange(image_embed_all.shape[0], device=image_embed_all.device, dtype=torch.long)

            loss = (F.cross_entropy(logits_per_image, labels_all, reduction='none') + F.cross_entropy(logits_per_text, labels_all, reduction='none')) / 2
            loss = (loss * loss_mask_all).sum() / (loss_mask_all.sum() + eps)
        else:
            # cosine similarity as logits
            logits_per_image = logit_scale * image_embed @ text_embed_all.t()
            logits_per_text = logit_scale * text_embed @ image_embed_all.t()

            loss = (F.cross_entropy(logits_per_image, self.labels) + \
                F.cross_entropy(logits_per_text, self.labels)) / 2

            if self.scale_loss_by_mask and loss_mask_all is not None:
                loss = loss / (avg_unmasked_ratio)

            # if the mask ratio for this distributed batch is less than the minimum ratio for clip loss, set loss to 0
            unmasked_ratio_batch = loss_mask.sum() / (loss_mask.numel() + eps)
            if unmasked_ratio_batch < min_ratio_for_clip_loss:
                loss = loss * 0.0
                if torch.isnan(loss):
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss, avg_unmasked_ratio
