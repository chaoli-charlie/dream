import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedFolder_t5_fast(Dataset):
    """
    Pre-extracted training cache written by main_cache.py: VAE moments, T5 caption embeddings and CLIP-tokenized
    captions per image (and optionally the image itself, for REPA's DINOv2 targets).
    Layout: <root>/shards/<k>/*.npz, <root>/map_new_to_orig/<k>_mapping.json, <root>/empty_caption_t5.npz
    """
    def __init__(self, root, extension=".npz", num_files_per_folder=5000, random_horizontal_flip=False):
        self.root = root
        self.folder_to_fn_json = os.path.join(root, "map_new_to_orig")
        self.extension = extension
        self.num_files_per_folder = num_files_per_folder
        self.random_horizontal_flip = random_horizontal_flip

        # Only read folder names at init
        self.folders = sorted(
            os.listdir(os.path.join(self.root, "shards")),
            key=lambda x: int(x)
        )
        self.num_folders = len(self.folders)
        self.total_files = self.count_total_files()
        self._folder_maps = {}

        # retrieve the empty caption embedding
        empty_caption_path = os.path.join(self.root, "empty_caption_t5.npz")
        self.empty_caption_embedding_t5 = np.load(empty_caption_path)['empty_caption_embedding_t5'].squeeze().astype(np.float32)

        print(f"Total files: {self.total_files}, Folders: {self.num_folders}")

    def count_total_files(self):
        """
        opens up all the json files in the folder_to_fn_json directory and counts the total number of files
        """
        total_files = 0
        for folder in self.folders:
            json_path = os.path.join(self.folder_to_fn_json, "{}_mapping.json".format(folder))
            with open(json_path, 'r') as f:
                folder_map = json.load(f)
            total_files += len(folder_map)
        return total_files

    def __len__(self):
        return self.total_files

    def __getitem__(self, index):
        folder_idx = index // self.num_files_per_folder
        folder = self.folders[folder_idx]
        file_idx = index % self.num_files_per_folder

        folder_path = os.path.join(self.root, "shards", folder)
        json_path = os.path.join(self.folder_to_fn_json, "{}_mapping.json".format(folder))
        if folder not in self._folder_maps:  # per-worker cache of the folder mappings
            with open(json_path, 'r') as f:
                self._folder_maps[folder] = json.load(f)
        filename = self._folder_maps[folder][str(file_idx)]
        file_path = os.path.join(folder_path, filename)

        data = np.load(file_path)

        flip = self.random_horizontal_flip and np.random.rand() < 0.5
        moments = data['moments_flip'] if flip else data['moments']

        labels = torch.tensor(0, dtype=torch.long)

        captions = data['captions']

        text_embedding = data['text_embedding'].astype(np.float32)

        empty_caption_embedding_t5 = torch.tensor(self.empty_caption_embedding_t5, dtype=torch.float32)

        item = {
            "moments": moments,
            "labels": labels,
            "captions": captions,
            "text_embedding": text_embedding,
            "empty_caption_embedding_t5": empty_caption_embedding_t5,
        }
        if 'samples' in data:
            # images are only needed by REPA; main_cache.py stores them as uint8, convert to [-1, 1]
            samples = data['samples_flip'] if flip else data['samples']
            if samples.dtype == np.uint8:
                samples = samples.astype(np.float32) / 127.5 - 1.0
            item["samples"] = samples
        return item

class CaptionDataset(Dataset):
    def __init__(self, text_dir):
        self.text_dir = text_dir
        self.samples = sorted(os.listdir(text_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename = self.samples[index]
        with open(os.path.join(self.text_dir, filename), 'r') as f:
            caption = f.read().strip()
        filename_without_ext = os.path.splitext(filename)[0]

        return caption, filename_without_ext