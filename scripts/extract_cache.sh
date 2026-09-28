#!/usr/bin/env bash
# Pre-extract VAE latents, T5 caption embeddings and CLIP tokens for training with --use_cached.
# Run once per node, e.g.  DATA_PATH=/path/to/cc12m CACHED_PATH=/path/to/cc12m_cache NNODES=4 NODE_RANK=0 MASTER_ADDR=<node0> bash scripts/extract_cache.sh
# Add --save_samples to also store the images (needed to train REPA from the cache).
set -euo pipefail
: "${DATA_PATH:?set DATA_PATH to a directory of webdataset *.tar shards}" "${CACHED_PATH:?set CACHED_PATH to the output directory}"

torchrun --nproc_per_node=${NPROC_PER_NODE:-8} --nnodes=${NNODES:-1} --node_rank=${NODE_RANK:-0} \
  --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-29519} \
  main_cache.py --data_path "${DATA_PATH}" --cached_path "${CACHED_PATH}" \
  --img_size 256 --text_encoder t5-large "$@"
