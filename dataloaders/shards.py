"""Random-access reading of webdataset tar shards (no copy of the images into a cache).

Each shard is an uncompressed *.tar holding, per sample, an image (<key>.jpg / .png / .webp) and its caption
(<key>.txt, as written by img2dataset; a <key>.json with a "caption" field, or a "captions" list whose first entry
is used, also works). Every shard is indexed once (member offsets) and the index is cached as a small .npz, so
samples are read directly from the tars with a seek.
"""
import hashlib
import io
import json
import os
import tarfile
from collections import OrderedDict
from glob import glob

import numpy as np
import PIL.Image

IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
INDEX_VERSION = 1


def index_shard(tar_path):
    """Byte offsets and sizes of each (image, caption) pair in an uncompressed tar."""
    members = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            dirname, basename = os.path.split(m.name)
            stem, _, ext = basename.partition(".")
            members.setdefault(os.path.join(dirname, stem), {})[ext.lower()] = (m.offset_data, m.size)

    records = []
    for key in sorted(members):
        exts = members[key]
        image = next((exts[e] for e in IMAGE_EXTS if e in exts), None)
        if image is None:
            continue
        if "txt" in exts:
            caption, is_json = exts["txt"], False
        elif "json" in exts:
            caption, is_json = exts["json"], True
        else:
            continue
        records.append((image[0], image[1], caption[0], caption[1], is_json))
    if not records:
        return np.zeros((0, 4), dtype=np.int64), np.zeros(0, dtype=bool)
    arr = np.array(records, dtype=np.int64)
    return arr[:, :4], arr[:, 4].astype(bool)


def load_or_build_index(tar_path, cache_dir):
    stat = os.stat(tar_path)
    path_hash = hashlib.md5(os.path.abspath(tar_path).encode()).hexdigest()[:8]  # shards in different dirs may share names
    cache_path = os.path.join(cache_dir, f"{os.path.basename(tar_path)}.{path_hash}.v{INDEX_VERSION}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if int(cached["tar_size"]) == stat.st_size and int(cached["tar_mtime"]) == int(stat.st_mtime):
            return cached["offsets"], cached["caption_is_json"]
    offsets, caption_is_json = index_shard(tar_path)
    tmp_path = f"{cache_path}.tmp{os.getpid()}.npz"
    np.savez(tmp_path, offsets=offsets, caption_is_json=caption_is_json,
             tar_size=stat.st_size, tar_mtime=int(stat.st_mtime))
    os.replace(tmp_path, cache_path)
    return offsets, caption_is_json


def parse_caption(raw, is_json):
    if not is_json:
        return raw.decode("utf-8").strip()
    meta = json.loads(raw)
    if meta.get("caption"):
        return meta["caption"].strip()
    captions = meta.get("captions") or []
    if not captions:
        raise ValueError("json has no caption")
    return captions[0].strip()


class ShardSamples:
    """Index over all (image, caption) samples in <shards_path>/*.tar, with random access by sample index."""

    def __init__(self, shards_path, max_shards=None, index_cache_dir=None, max_open_files=256):
        tars = sorted(glob(os.path.join(shards_path, "*.tar")))
        if not tars:
            raise FileNotFoundError(f"No *.tar shards found in {shards_path}")
        if max_shards is not None:
            tars = tars[:max_shards]

        index_cache_dir = index_cache_dir or os.path.join(shards_path, ".index_cache")
        os.makedirs(index_cache_dir, exist_ok=True)

        shard_ids, offsets, caption_is_json = [], [], []
        for i, tar_path in enumerate(tars):
            off, is_json = load_or_build_index(tar_path, index_cache_dir)
            shard_ids.append(np.full(len(off), i, dtype=np.int32))
            offsets.append(off)
            caption_is_json.append(is_json)
        self.tar_paths = tars
        self.shard_ids = np.concatenate(shard_ids)
        self.offsets = np.concatenate(offsets)
        self.caption_is_json = np.concatenate(caption_is_json)

        # per-process cache of open shard files (opened lazily, so each DataLoader worker has its own handles)
        self.max_open_files = max_open_files
        self._files = OrderedDict()

    def __len__(self):
        return len(self.shard_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = OrderedDict()  # file handles are not shared across processes
        return state

    def _read(self, shard_id, offset, size):
        f = self._files.pop(shard_id, None)
        if f is None:
            f = open(self.tar_paths[shard_id], "rb")
            if len(self._files) >= self.max_open_files:
                _, oldest = self._files.popitem(last=False)
                oldest.close()
        self._files[shard_id] = f
        f.seek(offset)
        return f.read(size)

    def caption(self, idx):
        shard_id = int(self.shard_ids[idx])
        _, _, cap_off, cap_size = (int(v) for v in self.offsets[idx])
        return parse_caption(self._read(shard_id, cap_off, cap_size), bool(self.caption_is_json[idx]))

    def image(self, idx):
        shard_id = int(self.shard_ids[idx])
        img_off, img_size, _, _ = (int(v) for v in self.offsets[idx])
        return PIL.Image.open(io.BytesIO(self._read(shard_id, img_off, img_size))).convert("RGB")
