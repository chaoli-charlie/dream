import io
from glob import glob

import PIL
import torch
from datasets import Image, load_dataset
from transformers import AutoTokenizer

from dataloaders.clip_tokenizer import SimpleTokenizer


def t2i_process_fn(batch, transform, tokenizer, text_tokenizer, text_max_len=128, dataset=None):
    images = batch["image"]
    captions = batch["caption"]

    valid_images = []
    valid_captions = []
    valid_paths = []

    for i in range(len(images)):
        try:
            path = images[i]["path"] if images[i]["path"] is not None else f"cc3m_img_{i}"
            image = PIL.Image.open(io.BytesIO(images[i]["bytes"]) if images[i]["bytes"] is not None else path).convert("RGB")
            valid_images.append(image)
            valid_captions.append(captions[i])
            valid_paths.append(path)
        except Exception as e:
            print(f"Corrupt image at index {i}: {e}")

    if not valid_images:
        # all images are corrupt: replace with a random sample (indexing with a list returns a processed batch)
        print("All images in batch are corrupt, fetching a random sample instead...")
        return dataset[[torch.randint(0, len(dataset), (1,)).item()]]

    batch["caption_words"] = valid_captions
    batch["labels"] = torch.zeros(len(valid_images), dtype=torch.long)

    with torch.no_grad():
        # T5 tokens for the generator's text conditioning
        batch["input_ids"] = text_tokenizer(valid_captions, return_tensors="pt", padding="max_length",
                                            max_length=text_max_len, truncation=True).input_ids
        batch["empty_input_ids"] = text_tokenizer([""] * len(valid_captions), return_tensors="pt", padding="max_length",
                                                  max_length=text_max_len, truncation=True).input_ids

        # CLIP tokens for the contrastive text tower. SimpleTokenizer squeezes a single caption to [77]; keep it
        # [n, 77] so that datasets' per-row unnesting yields one [77] token sequence per sample
        batch["caption"] = tokenizer(valid_captions).reshape(len(valid_captions), -1)

    batch["image"] = [transform(image) for image in valid_images]
    batch["paths"] = valid_paths

    return batch


def return_cc3m_train_dataset(transform=None, debug=False, data_dir="./data/cc3m", cache_dir=None, num_proc=12):
    """CC3M in webdataset format: data_dir/*.tar shards with jpg/txt pairs."""
    data_files = sorted(glob(f"{data_dir}/*.tar"))
    if not data_files:
        raise FileNotFoundError(f"No *.tar shards found in {data_dir} (set --cc3m_path)")

    if debug:
        data_files = data_files[:10]

    tokenizer = SimpleTokenizer()
    text_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")

    train_dataset = load_dataset(
        "webdataset",
        data_files=data_files,
        cache_dir=cache_dir,
        split="train",
        num_proc=num_proc,
    )

    train_dataset = train_dataset.rename_column("jpg", "image")
    train_dataset = train_dataset.rename_column("txt", "caption")
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ["image", "caption"]])

    train_dataset = train_dataset.cast_column("image", Image(decode=False))
    train_dataset.set_transform(lambda batch: t2i_process_fn(batch, transform=transform, tokenizer=tokenizer,
                                                             text_tokenizer=text_tokenizer, dataset=train_dataset))

    return train_dataset
