#!/usr/bin/env bash
# REPA baseline: align encoder block 8 of MAR-L to DINOv2-B features (4 nodes x 8 GPUs).
# Run once per node, e.g.  NNODES=4 NODE_RANK=0 MASTER_ADDR=<node0> bash scripts/train_repa.sh
set -euo pipefail
DATASET=${DATASET:-cc12m}  # the paper trains on cc12m; set DATASET=cc3m (and CC3M_PATH) for a smaller run
CC12M_PATH=${CC12M_PATH:-./data/cc12m}
CC3M_PATH=${CC3M_PATH:-./data/cc3m}
OUTPUT_DIR=${OUTPUT_DIR:-output/repa_large_layer_8}

torchrun --nproc_per_node=${NPROC_PER_NODE:-8} --nnodes=${NNODES:-4} --node_rank=${NODE_RANK:-0} \
  --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-29517} \
  main.py \
  --img_size 256 --vae_embed_dim 4 --vae_stride 8 --patch_size 2 \
  --model repa_mar_large_txt_conditional \
  --diffloss_d 3 --diffloss_w 1024 --diffusion_batch_mul 4 \
  --epochs 48 --warmup_epochs 12 --batch_size 64 --blr 1.0e-4 --weight_decay 0.02 \
  --save_last_freq 6 \
  --dataset ${DATASET} --cc12m_path ${CC12M_PATH} --cc3m_path ${CC3M_PATH} --transform_type mar \
  --weight_mar_loss 1.0 --weight_repa_loss 0.5 \
  --mask_ratio_min 0.7 --mask_ratio_max 1.0 \
  --encoder_depth_proj 8 \
  --output_dir ${OUTPUT_DIR} "$@"
