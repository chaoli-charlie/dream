"""Pre-extract the training cache used by --use_cached: VAE latents, T5 caption embeddings and CLIP caption tokens.

Output layout (read by dataloaders.cached.CachedFolder_t5_fast):
    <cached_path>/shards/<k>/<index>.npz          k = index // files_per_folder
    <cached_path>/map_new_to_orig/<k>_mapping.json {position in folder: file name}
    <cached_path>/empty_caption_t5.npz             T5 embedding of the empty caption (for CFG caption dropout)

Each .npz holds moments / moments_flip (SD-VAE posterior parameters of the center-cropped image and its horizontal
flip), text_embedding (T5 last hidden state), captions (CLIP tokens), and with --save_samples the uint8 image and its
flip (needed to train REPA from the cache, for its DINOv2 targets). Latents and T5 embeddings are computed under the same bfloat16
autocast as the non-cached training path.

    torchrun --nproc_per_node=8 main_cache.py --data_path /path/to/cc12m --cached_path /path/to/cc12m_cache
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from diffusers import AutoencoderKL as StableAIAutoencoderKL
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5EncoderModel

import util.misc as misc
from dataloaders.clip_tokenizer import SimpleTokenizer
from dataloaders.shards import ShardSamples
from dataloaders.transforms import center_crop_arr


def get_args_parser():
    parser = argparse.ArgumentParser('Extract the latent / text-embedding training cache', add_help=False)
    parser.add_argument('--data_path', required=True, type=str, help='directory of webdataset *.tar shards (CC12M or CC3M)')
    parser.add_argument('--cached_path', required=True, type=str, help='output directory (pass it to --cached_path when training)')
    parser.add_argument('--index_cache_dir', default=None, type=str, help='where to cache the shard index (default: <data_path>/.index_cache)')
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, help='images per GPU per step')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--text_encoder', default='t5-large', type=str, choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'])
    parser.add_argument('--text_max_len', default=128, type=int)
    parser.add_argument('--text_embedding_dtype', default='float16', choices=['float16', 'float32'],
                        help='storage dtype of the T5 embeddings, which dominate the cache size')
    parser.add_argument('--save_samples', action='store_true', help='also store the uint8 images (required to train REPA with --use_cached)')
    parser.add_argument('--files_per_folder', default=5000, type=int)
    parser.add_argument('--max_samples', default=None, type=int, help='only cache the first N samples (for testing)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    return parser


def cache_file(cached_path, idx, files_per_folder):
    folder = os.path.join(cached_path, "shards", str(idx // files_per_folder))
    return folder, f"{idx:09d}.npz"


class CacheSource(Dataset):
    """Center-cropped images (uint8) with their CLIP and T5 tokens, for the given global sample indices."""

    def __init__(self, samples, indices, img_size, text_max_len):
        self.samples = samples
        self.indices = indices
        self.img_size = img_size
        self.text_max_len = text_max_len
        self.clip_tokenizer = SimpleTokenizer()
        self.text_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        # a corrupt sample is replaced by the next readable one, so that every index has a cache file
        for offset in range(100):
            src = (idx + offset) % len(self.samples)
            try:
                image = center_crop_arr(self.samples.image(src), self.img_size)
                caption = self.samples.caption(src)
                break
            except Exception as e:
                print(f"Corrupt sample {src}: {e}")
        else:
            raise RuntimeError(f"No readable sample near index {idx}")

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()  # uint8 [3, H, W]
        input_ids = self.text_tokenizer([caption], return_tensors="pt", padding="max_length",
                                        max_length=self.text_max_len, truncation=True).input_ids[0]
        return idx, image, self.clip_tokenizer([caption]).reshape(-1), input_ids


def write_mappings(cached_path, num_samples, files_per_folder):
    os.makedirs(os.path.join(cached_path, "map_new_to_orig"), exist_ok=True)
    for k in range((num_samples + files_per_folder - 1) // files_per_folder):
        first, last = k * files_per_folder, min((k + 1) * files_per_folder, num_samples)
        mapping = {str(i - first): cache_file(cached_path, i, files_per_folder)[1] for i in range(first, last)}
        with open(os.path.join(cached_path, "map_new_to_orig", f"{k}_mapping.json"), "w") as f:
            json.dump(mapping, f)


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    rank, world_size = misc.get_rank(), misc.get_world_size()

    samples = ShardSamples(args.data_path, index_cache_dir=args.index_cache_dir)
    num_samples = len(samples) if args.max_samples is None else min(args.max_samples, len(samples))
    print(f"Caching {num_samples:,} samples from {args.data_path} into {args.cached_path}")

    # this rank's indices, skipping files written by an earlier (interrupted) run
    todo = []
    my_indices = range(rank, num_samples, world_size)
    existing = {}
    for idx in my_indices:
        folder, name = cache_file(args.cached_path, idx, args.files_per_folder)
        if folder not in existing:
            existing[folder] = set(os.listdir(folder)) if os.path.isdir(folder) else set()
        if name not in existing[folder]:
            todo.append(idx)
    print(f"[rank {rank}] {len(todo):,} of {len(my_indices):,} samples left to cache", force=True)

    vae = StableAIAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    text_encoder = T5EncoderModel.from_pretrained("google-t5/{}".format(args.text_encoder)).to(device).eval()
    text_dtype = np.float16 if args.text_embedding_dtype == "float16" else np.float32

    if rank == 0:
        write_mappings(args.cached_path, num_samples, args.files_per_folder)
        tokenizer = AutoTokenizer.from_pretrained("google-t5/{}".format(args.text_encoder))
        empty_ids = tokenizer([""], return_tensors="pt", padding="max_length", max_length=args.text_max_len,
                              truncation=True).input_ids.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            empty = text_encoder(input_ids=empty_ids).last_hidden_state
        np.savez(os.path.join(args.cached_path, "empty_caption_t5.npz"),
                 empty_caption_embedding_t5=empty.float().cpu().numpy().astype(text_dtype))

    loader = torch.utils.data.DataLoader(CacheSource(samples, todo, args.img_size, args.text_max_len),
                                         batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    start = time.time()
    for step, (indices, images, clip_tokens, input_ids) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        x = images.float() / 127.5 - 1.0  # [-1, 1], as Normalize([0.5], [0.5]) in training
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            moments = vae.encode(x).latent_dist.parameters
            moments_flip = vae.encode(x.flip(dims=[3])).latent_dist.parameters
            text_embedding = text_encoder(input_ids=input_ids.to(device, non_blocking=True)).last_hidden_state

        moments = moments.float().cpu().numpy()
        moments_flip = moments_flip.float().cpu().numpy()
        text_embedding = text_embedding.float().cpu().numpy().astype(text_dtype)
        for b, idx in enumerate(indices.tolist()):
            folder, name = cache_file(args.cached_path, idx, args.files_per_folder)
            os.makedirs(folder, exist_ok=True)
            item = dict(moments=moments[b], moments_flip=moments_flip[b], text_embedding=text_embedding[b],
                        captions=clip_tokens[b].numpy())
            if args.save_samples:
                item.update(samples=images[b].cpu().numpy(), samples_flip=images[b].flip(dims=[2]).cpu().numpy())
            tmp = os.path.join(folder, f".{name}.tmp.npz")
            np.savez(tmp, **item)
            os.replace(tmp, os.path.join(folder, name))

        if step % 50 == 0:
            done = (step + 1) * args.batch_size
            print(f"[rank {rank}] {min(done, len(todo)):,}/{len(todo):,} samples, {done / (time.time() - start):.1f} samples/s", force=True)

    misc.barrier()
    print(f"[rank {rank}] done in {time.time() - start:.0f}s", force=True)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
