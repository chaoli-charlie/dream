"""Text-to-image evaluation set built directly from webdataset shards.

The first num_images samples (in sorted shard order) of a held-out set of shards provide both the prompts to
generate from and the real images that serve as the FID reference, so no caption folder or reference image folder
is needed.
"""
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloaders.shards import ShardSamples
from dataloaders.transforms import center_crop_arr


class ShardEvalSet:
    def __init__(self, shards_path, num_images, img_size=256, index_cache_dir=None):
        self.samples = ShardSamples(shards_path, index_cache_dir=index_cache_dir)
        if len(self.samples) < num_images:
            raise ValueError(f"--eval_shards_path has {len(self.samples)} samples, fewer than --num_images {num_images}")
        self.num_images = num_images
        self.img_size = img_size
        print(f"Eval set: first {num_images:,} of {len(self.samples):,} samples in {shards_path}")

        # identifies the reference statistics in torch-fidelity's cache
        shard_names = "|".join(os.path.basename(p) for p in self.samples.tar_paths)
        digest = hashlib.md5(f"{os.path.abspath(shards_path)}|{shard_names}".encode()).hexdigest()[:10]
        self.cache_name = f"dream-ref-{digest}-n{num_images}-{img_size}px"

    def captions(self):
        """(caption, filename) pairs, like CaptionDataset, for generation."""
        return EvalCaptions(self)

    def reference_images(self):
        """uint8 [3, H, W] real images, center-cropped as in training, as the FID reference."""
        return EvalReferenceImages(self)


class EvalCaptions(Dataset):
    def __init__(self, eval_set):
        self.eval_set = eval_set

    def __len__(self):
        return self.eval_set.num_images

    def __getitem__(self, idx):
        return self.eval_set.samples.caption(idx), f"{idx:06d}"


class EvalReferenceImages(Dataset):
    def __init__(self, eval_set):
        self.eval_set = eval_set

    def __len__(self):
        return self.eval_set.num_images

    def __getitem__(self, idx):
        image = center_crop_arr(self.eval_set.samples.image(idx), self.eval_set.img_size)
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
