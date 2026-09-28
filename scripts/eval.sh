#!/usr/bin/env bash
# Text-to-image FID / IS of a DREAM checkpoint (or a REPA one with MODEL=repa_mar_large_txt_conditional).
#
# Prompts and FID reference from held-out webdataset shards (the first NUM_IMAGES samples):
#   CKPT=output/dream/checkpoint-last.pth EVAL_SHARDS=/path/to/heldout_shards CFG=3.0 bash scripts/eval.sh
# or from a folder of caption .txt files and a reference (image folder or torch-fidelity .npz statistics):
#   CKPT=... CAPTIONS=/path/to/captions REFERENCE=/path/to/images_or_stats.npz bash scripts/eval.sh
#
# CFG sets the classifier-free guidance scale (default 1.0, no guidance). Extra arguments are passed through,
# e.g.  --use_clip_critic_once  (Semantically Aligned Decoding).
# The model arguments must match those used for training (see scripts/train_*.sh); arguments of the other model family are ignored.
set -euo pipefail
: "${CKPT:?set CKPT to a checkpoint path}"
if [ -n "${EVAL_SHARDS:-}" ]; then
  EVAL_DATA=(--eval_shards_path "${EVAL_SHARDS}")
else
  : "${CAPTIONS:?set EVAL_SHARDS, or CAPTIONS and REFERENCE}" "${REFERENCE:?set REFERENCE (image folder or .npz statistics)}"
  EVAL_DATA=(--caption_dataset_path "${CAPTIONS}" --fid_path2 "${REFERENCE}")
fi
OUTPUT_DIR=${OUTPUT_DIR:-$(dirname "${CKPT}")/eval}

torchrun --nproc_per_node=${NPROC_PER_NODE:-8} --nnodes=1 --node_rank=0 --master_port=${MASTER_PORT:-29518} \
  main.py \
  --img_size 256 --vae_embed_dim 4 --vae_stride 8 --patch_size 2 \
  --model ${MODEL:-dream_huge_txt_conditional} \
  --diffloss_d 3 --diffloss_w 1024 \
  --vl_projection post_mlp_stablerep --ssl_mlp_dim 1024 --embed_dim 256 \
  --evaluate --resume "${CKPT}" --checkpoint "$(basename "${CKPT}" .pth)" --use_ema \
  --eval_bsz ${EVAL_BSZ:-48} --num_images ${NUM_IMAGES:-50000} --cfg ${CFG:-1.0} \
  "${EVAL_DATA[@]}" \
  --output_dir "${OUTPUT_DIR}" "$@"
