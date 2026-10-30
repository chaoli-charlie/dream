# Checkpoints

## `checkpoint-49-ema.pth`: DREAM-B trained on CC3M

> **This is not the model reported in the paper.** The paper's main results come from DREAM-L trained on **CC12M**.
> This checkpoint is a smaller **DREAM-B** trained on **CC3M** with a global batch of 256, so the numbers in the main
> README and the paper do not apply to it. Use it to try out the code, run quick experiments, or as a starting point
> for fine-tuning.

| | |
|---|---|
| Model | `dream_base_txt_conditional` (329.4M parameters, excluding the frozen T5 text encoder and VAE) |
| Training data | CC3M |
| Training | 50 epochs (epochs 0–49; this is the final one), 4 GPUs x 64 images = global batch 256 |
| Contents | EMA weights only: a single `model_ema` state dict (1.3 GB) |

Download it into this folder from the [`dream-b-cc3m-v1` release](https://github.com/chaoli-charlie/dream/releases/tag/dream-b-cc3m-v1):

```bash
wget -P checkpoints https://github.com/chaoli-charlie/dream/releases/download/dream-b-cc3m-v1/checkpoint-49-ema.pth
sha256sum checkpoints/checkpoint-49-ema.pth
# 0106eea2ff647dfe8abb43a0145c6a850fc234c90e8bf9ff9f6783de96b5f3c4
```

The file holds only the exponential moving average of the weights (`model_ema`, rate 0.9999), which is what generation
and linear probing use. The raw (non-EMA) weights, optimizer state, loss-scaler state, epoch counter and training
arguments have been removed. When a checkpoint has no `model` entry, `--resume` and `main_linprobe.py --finetune` load
the EMA weights in their place. Continuing training from it therefore starts from the EMA weights with a fresh
optimizer and learning-rate schedule. The file loads with `torch.load(..., weights_only=True)`.

### Training configuration

Same objective and masking schedule as the paper's DREAM recipe, at base scale on CC3M:

```
--model dream_base_txt_conditional --dataset cc3m --transform_type mar
--img_size 256 --vae_embed_dim 4 --vae_stride 8 --patch_size 2 --autoencoder_type sd
--diffloss_d 3 --diffloss_w 1024 --diffusion_batch_mul 4
--vl_projection post_mlp_stablerep --ssl_mlp_dim 1024 --embed_dim 256
--text_encoder t5-large --text_max_len 128 --text_drop_prob 0.1
--epochs 50 --warmup_epochs 12 --batch_size 64 --blr 1.0e-4 --lr_schedule constant
--weight_decay 0.04 --grad_clip 3.0 --ema_rate 0.9999
--weight_mar_loss 1.0 --weight_clip_loss 0.005
--variable_masking --mask_ratio_mu_start 0.0 --mask_ratio_mu_end 1.0
--mask_ratio_min 0.0 --mask_ratio_max 1.0 --mask_ratio_std 0.45
--masking_warmup 0 --masking_warmup_end 36 --fixed_masking_boundary
--min_ratio_for_clip_loss 0.25 --min_masked_ratio_for_mar_loss 0.50
--scale_loss_by_mask --filter_clip_loss
```

## Usage

The model arguments must match the ones above. `scripts/eval.sh` defaults to `dream_large_txt_conditional`, so set
`MODEL=dream_base_txt_conditional`.

### Text-to-image generation and FID

```bash
MODEL=dream_base_txt_conditional CKPT=checkpoints/checkpoint-49-ema.pth \
  CAPTIONS=/path/to/captions REFERENCE=/path/to/reference CFG=7.4 bash scripts/eval.sh

# with Semantically Aligned Decoding
MODEL=dream_base_txt_conditional CKPT=checkpoints/checkpoint-49-ema.pth \
  CAPTIONS=/path/to/captions REFERENCE=/path/to/reference CFG=7.4 bash scripts/eval.sh \
  --use_clip_critic_once --num_candidates 5 --clip_critic_threshold 32
```

`eval.sh` generates with the EMA weights (`--use_ema`). For this checkpoint the best guidance scale we found is around
7.4 (linear schedule), much higher than for the paper's CC12M models.

### Linear probing

```bash
torchrun --nproc_per_node=4 main_linprobe.py \
  --model mar_base_patch16 --img_size 256 --vae_embed_dim 4 --vae_stride 8 --patch_size 2 --buffer_size 64 \
  --finetune checkpoints/checkpoint-49-ema.pth --use_ema \
  --data_path /path/to/imagenet --output_dir output/linprobe
```

### Loading in Python

```python
import torch
import main  # from the repository root

args = main.get_args_parser().parse_args([
    "--model", "dream_base_txt_conditional", "--img_size", "256", "--vae_embed_dim", "4",
    "--vae_stride", "8", "--patch_size", "2", "--diffloss_d", "3", "--diffloss_w", "1024",
    "--vl_projection", "post_mlp_stablerep", "--ssl_mlp_dim", "1024", "--embed_dim", "256",
])
model = main.build_model(args, main.model_family(args.model))

ckpt = torch.load("checkpoints/checkpoint-49-ema.pth", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model_ema"])
```

## Reference numbers

For sanity checks only. These are not comparable to the paper's numbers:

| Metric | Value | Setup |
|---|---|---|
| ImageNet-1K linear probe, top-1 | 59.3% | EMA weights of the epoch-48 checkpoint (one epoch before this one), `mar_base_patch16`, 90 epochs, batch 256, blr 0.05 |
| FID / IS, no guidance | 19.77 / 15.50 | 50k images, EMA, 64 steps, temperature 1.0, against 256px CC3M images |
| FID, best guidance (CFG 7.4, linear) | 11.37 | same as above |

The FID reference images and captions were drawn from the CC3M data this model was trained on. These FIDs therefore
measure fit to the training distribution, not generalization to held-out prompts.
