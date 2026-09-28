"""CC12M as webdataset tar shards, read with random access (see dataloaders/shards.py)."""
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dataloaders.clip_tokenizer import SimpleTokenizer
from dataloaders.shards import ShardSamples

# tokenizers are used inside DataLoader workers; avoid the fork-after-parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class CC12MDataset(Dataset):
    """
    Returns the same per-sample fields as the CC3M loader: image, labels, caption (CLIP tokens), input_ids and
    empty_input_ids (T5 tokens).
    """

    def __init__(self, cc12m_path, transform, debug=False, index_cache_dir=None, text_max_len=128):
        self.samples = ShardSamples(cc12m_path, max_shards=2 if debug else None, index_cache_dir=index_cache_dir)
        print(f"CC12M: {len(self.samples):,} samples in {len(self.samples.tar_paths)} shards")

        self.transform = transform
        self.text_max_len = text_max_len
        self.clip_tokenizer = SimpleTokenizer()
        self.text_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
        self.empty_input_ids = self.text_tokenizer([""], return_tensors="pt", padding="max_length",
                                                   max_length=text_max_len, truncation=True).input_ids[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                image, caption = self.samples.image(idx), self.samples.caption(idx)
                break
            except Exception as e:
                print(f"Corrupt sample {idx}: {e}")
                idx = torch.randint(0, len(self), (1,)).item()
        else:
            raise RuntimeError("Too many corrupt samples in a row")

        input_ids = self.text_tokenizer([caption], return_tensors="pt", padding="max_length",
                                        max_length=self.text_max_len, truncation=True).input_ids[0]
        return {
            "image": self.transform(image),
            "labels": torch.tensor(0, dtype=torch.long),
            "caption": self.clip_tokenizer([caption]).reshape(-1),
            "input_ids": input_ids,
            "empty_input_ids": self.empty_input_ids,
        }


def return_cc12m_train_dataset(transform=None, debug=False, data_dir="./data/cc12m", index_cache_dir=None):
    return CC12MDataset(data_dir, transform, debug=debug, index_cache_dir=index_cache_dir)
