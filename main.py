"""Training and text-to-image evaluation for DREAM and the baselines, selected by --model:
  dream_*_txt_conditional     DREAM (and FLUID with --weight_clip_loss 0)
  repa_mar_*_txt_conditional  REPA (encoder features aligned to a frozen DINOv2)
"""
import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from diffusers import AutoencoderKL as StableAIAutoencoderKL
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, T5EncoderModel

import engines.dream
import engines.repa
import util.misc as misc
from dataloaders.cached import CachedFolder_t5_fast, CaptionDataset
from dataloaders.cc12m import return_cc12m_train_dataset
from dataloaders.cc3m import return_cc3m_train_dataset
from dataloaders.clip_tokenizer import SimpleTokenizer
from dataloaders.eval_shards import ShardEvalSet
from dataloaders.transforms import get_image_transform
from models import dream, repa
from util.misc import NativeScalerWithGradNormCount as NativeScaler

DREAM_MODELS = ["dream_base_txt_conditional", "dream_large_txt_conditional", "dream_huge_txt_conditional", "dream_giant_txt_conditional"]
REPA_MODELS = ["repa_mar_base_txt_conditional", "repa_mar_large_txt_conditional", "repa_mar_huge_txt_conditional"]


def get_args_parser():
    parser = argparse.ArgumentParser('DREAM / FLUID / REPA training and evaluation', add_help=False)
    g_training = parser.add_argument_group('training')
    g_training.add_argument('--batch_size', default=16, type=int,
                            help='Batch size per GPU (effective batch size is batch_size * # gpus')
    g_training.add_argument('--epochs', default=400, type=int)

    g_model = parser.add_argument_group('model')
    g_model.add_argument('--model', default='dream_large_txt_conditional', type=str, choices=DREAM_MODELS + REPA_MODELS, metavar='MODEL',
                        help='dream_{base,large,huge,giant}_txt_conditional (DREAM / FLUID) or repa_mar_{base,large,huge}_txt_conditional (REPA)')

    g_vae = parser.add_argument_group('VAE')
    g_vae.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    g_vae.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    g_vae.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride')
    g_vae.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')
    g_vae.add_argument('--autoencoder_type', default='sd', type=str, choices=['sd'],
                        help='VAE to use (only the Stable Diffusion VAE, stabilityai/sd-vae-ft-ema, is supported)')

    g_generation = parser.add_argument_group('generation')
    g_generation.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    g_generation.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    g_generation.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    g_generation.add_argument('--cfg_schedule', default="linear", type=str)
    g_generation.add_argument('--label_drop_prob', default=0.0, type=float)
    g_generation.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    g_generation.add_argument('--evaluate', action='store_true')
    g_generation.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

    g_optimization = parser.add_argument_group('optimization')
    g_optimization.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    g_optimization.add_argument('--grad_checkpointing', action='store_true')
    g_optimization.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    g_optimization.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    g_optimization.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    g_optimization.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    g_optimization.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    g_optimization.add_argument('--ema_rate', default=0.9999, type=float)
    g_optimization.add_argument('--beta1', type=float, default=0.9, metavar='BETA1',
                        help='beta1 for Adam optimizer (default: 0.9)')
    g_optimization.add_argument('--beta2', type=float, default=0.95, metavar='BETA2',
                        help='beta2 for Adam optimizer (default: 0.95)')

    g_masking_and_architecture = parser.add_argument_group('masking and architecture')
    g_masking_and_architecture.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    g_masking_and_architecture.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    g_masking_and_architecture.add_argument('--mask_ratio_mu', type=float, default=1.0,
                        help='Mean mask ratio')
    g_masking_and_architecture.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Std mask ratio')
    g_masking_and_architecture.add_argument('--uniform_masking', action='store_true', help='sample the mask ratio uniformly from [min, max]')

    g_masking_and_architecture.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    g_masking_and_architecture.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    g_masking_and_architecture.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    g_masking_and_architecture.add_argument('--buffer_size', type=int, default=64)
    g_masking_and_architecture.add_argument('--class_num', default=1000, type=int)

    g_masking_warmup = parser.add_argument_group('Masking Warmup (DREAM)')
    g_masking_warmup.add_argument('--variable_masking', action='store_true', help='use variable masking schedule')
    g_masking_warmup.add_argument('--mask_ratio_mu_start', default=None, type=float, help='starting mean mask ratio for variable masking')
    g_masking_warmup.add_argument('--mask_ratio_mu_end', default=None, type=float, help='ending mean mask ratio for variable masking')
    g_masking_warmup.add_argument('--fixed_masking_boundary', action='store_true', help='use fixed boundary for variable masking schedule')
    g_masking_warmup.add_argument('--masking_warmup', default=0, type=int, help='number of epochs before the masking schedule starts')
    g_masking_warmup.add_argument('--masking_warmup_end', default=None, type=int, help='epoch at which the masking schedule reaches mask_ratio_mu_end')
    g_masking_warmup.add_argument('--masking_cooldown', default=0, type=int, help='number of epochs to cooldown the masking schedule')
    g_masking_warmup.add_argument('--warmup_shape', default="linear", type=str, help='shape of the warmup schedule')
    g_masking_warmup.add_argument('--schedule_std', action='store_true', help='use schedule std for variable masking')

    g_diffusion_loss = parser.add_argument_group('diffusion loss')
    g_diffusion_loss.add_argument('--diffloss_d', type=int, default=12)
    g_diffusion_loss.add_argument('--diffloss_w', type=int, default=1536)
    g_diffusion_loss.add_argument('--num_sampling_steps', type=str, default="100")
    g_diffusion_loss.add_argument('--diffusion_batch_mul', type=int, default=1)
    g_diffusion_loss.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    g_contrastive_loss = parser.add_argument_group('contrastive loss (DREAM; --weight_clip_loss 0 gives FLUID)')
    g_contrastive_loss.add_argument('--weight_mar_loss', default=0.0, type=float)
    g_contrastive_loss.add_argument('--weight_clip_loss', default=1.0, type=float)
    g_contrastive_loss.add_argument('--clip_loss_on', default="image-text", type=str, choices=["image-text", "image-tokens-text", "buffer"],
                        help='encoder tokens pooled for the contrastive loss: all, image tokens only, or buffer tokens only')
    g_contrastive_loss.add_argument('--vl_projection', default="identity", type=str,
                        help='vision-language projection type: identity, linear, post_linear, mlp, post_mlp, mlp_stablerep, post_mlp_stablerep')
    g_contrastive_loss.add_argument('--txt_projection', default="linear", type=str, help='text projection type: linear, mlp, stablerep')
    g_contrastive_loss.add_argument('--ssl_mlp_dim', default=2048, type=int)
    g_contrastive_loss.add_argument('--embed_dim', default=None, type=int, help='final embedding dimension for both image and text')
    g_contrastive_loss.add_argument('--min_ratio_for_clip_loss', default=0.0, type=float, help='minimum unmasked ratio to compute CLIP loss')
    g_contrastive_loss.add_argument('--min_masked_ratio_for_mar_loss', default=0.0, type=float, help='minimum masked ratio to compute MAR loss')
    g_contrastive_loss.add_argument('--scale_loss_by_mask', action='store_true', help='scale loss by mask')
    g_contrastive_loss.add_argument('--filter_clip_loss', action='store_true', help='exclude samples below min_ratio_for_clip_loss from the CLIP loss')
    g_contrastive_loss.add_argument('--depth_to_align', default=-1, type=int, help='encoder depth whose features are used for the CLIP loss (-1: last)')

    g_repa = parser.add_argument_group('REPA alignment (--model repa_mar_*, or DREAM with --repa_alignment)')
    g_repa.add_argument('--repa_alignment', action='store_true',
                        help='DREAM: also align encoder block --encoder_depth_proj to frozen DINOv2 features (REPA loss)')
    g_repa.add_argument('--weight_repa_loss', default=1.0, type=float)
    g_repa.add_argument('--dino_v2_size', default='b', type=str, help='DINOv2 ViT size: s, b, l or g')
    g_repa.add_argument('--encoder_depth_proj', default=8, type=int, help='encoder block whose output is aligned to DINOv2')
    g_repa.add_argument('--projector_dim', default=2048, type=int, help='hidden dimension of the REPA projector')
    g_repa.add_argument('--z_dim', default=768, type=int, help='DINOv2 embedding dimension')

    g_text_conditioning = parser.add_argument_group('text conditioning')
    g_text_conditioning.add_argument('--text_encoder', default='t5-large', type=str, choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'], help='Pre-trained text encoder to use')
    g_text_conditioning.add_argument('--text_drop_prob', default=0.0, type=float, help='probability of dropping the caption (for CFG)')
    g_text_conditioning.add_argument('--text_aligner_depth', default=6, type=int, help='Number of layers in the text aligner')
    g_text_conditioning.add_argument('--t5_embed_dim', default=1024, type=int, help='Embedding dimension for T5 text encoder')
    g_text_conditioning.add_argument('--text_max_len', default=128, type=int, help='Maximum length of text input for T5 encoder')

    g_semantically_aligned_decoding = parser.add_argument_group('Semantically Aligned Decoding (DREAM)')
    g_semantically_aligned_decoding.add_argument('--use_clip_critic', action='store_true', help='use CLIP critic to pick among candidates at every step after the threshold')
    g_semantically_aligned_decoding.add_argument('--use_clip_critic_once', action='store_true', help='use CLIP critic once, at the threshold step, to pick one of the candidate trajectories')
    g_semantically_aligned_decoding.add_argument('--num_candidates', default=5, type=int, help='number of candidates to sample for CLIP critic')
    g_semantically_aligned_decoding.add_argument('--clip_critic_threshold', default=32, type=int, help='generation step at which to switch to CLIP critic')

    g_training_data = parser.add_argument_group('training data')
    g_training_data.add_argument('--dataset', default='cc12m', type=str, choices=['cc12m', 'cc3m'], help='training dataset (the paper uses cc12m)')
    g_training_data.add_argument('--cc12m_path', default='./data/cc12m', type=str, help='directory containing cc12m webdataset *.tar shards')
    g_training_data.add_argument('--index_cache_dir', default=None, type=str, help='where to cache the cc12m shard index (default: <cc12m_path>/.index_cache)')
    g_training_data.add_argument('--cc3m_path', default='./data/cc3m', type=str, help='directory containing cc3m webdataset *.tar shards')
    g_training_data.add_argument('--hf_cache_dir', default=None, type=str, help='huggingface datasets cache directory (cc3m)')
    g_training_data.add_argument('--use_cached', action='store_true', help='train from pre-extracted latents and text embeddings')
    g_training_data.add_argument('--cached_path', default='', help='path to cached latents')
    g_training_data.add_argument('--transform_type', type=str, default='default', choices=['default', 'mar', 'clip'])
    g_training_data.add_argument('--debug', action='store_true', help='debug mode (only the first few dataset shards)')

    g_evaluation = parser.add_argument_group('evaluation (prompts and FID reference from --eval_shards_path, unless given as folders / statistics)')
    g_evaluation.add_argument('--eval_shards_path', default=None, type=str,
                        help='held-out webdataset *.tar shards: the first --num_images samples give the prompts and the FID reference images')
    g_evaluation.add_argument('--caption_dataset_path', default=None, type=str, help='instead: folder of caption .txt files to generate images for')
    g_evaluation.add_argument('--fid_path1', default=None, type=str, help='evaluate this folder of already generated images instead of generating')
    g_evaluation.add_argument('--fid_path2', default=None, type=str, help='instead: reference images folder, or .npz FID statistics file')
    g_evaluation.add_argument('--keep_samples', action='store_true', help='keep the generated images after computing FID')
    g_evaluation.add_argument('--use_ema', action='store_true', help='use ema model for evaluation')
    g_evaluation.add_argument('--checkpoint', default="", type=str, help='checkpoint name, used in the output folder name')

    g_checkpointing_and_misc = parser.add_argument_group('checkpointing and misc')
    g_checkpointing_and_misc.add_argument('--output_dir', default='./output', type=str,
                        help='path where to save checkpoints and tensorboard logs')
    g_checkpointing_and_misc.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    g_checkpointing_and_misc.add_argument('--seed', default=1, type=int)
    g_checkpointing_and_misc.add_argument('--resume', default='',
                        help='resume from checkpoint')
    g_checkpointing_and_misc.add_argument('--finetune', action='store_true', help='with --resume, load only the model weights (not optimizer / scheduler state)')
    g_checkpointing_and_misc.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    g_checkpointing_and_misc.add_argument('--num_workers', default=8, type=int)
    g_checkpointing_and_misc.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    g_checkpointing_and_misc.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    g_checkpointing_and_misc.add_argument('--just_print_params', action='store_true', help='just print the number of parameters and exit')

    g_distributed_training = parser.add_argument_group('distributed training')
    g_distributed_training.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    g_distributed_training.add_argument('--local_rank', default=-1, type=int)
    g_distributed_training.add_argument('--dist_on_itp', action='store_true')
    g_distributed_training.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def model_family(name):
    return "repa" if name in REPA_MODELS else "dream"


def build_model(args, family):
    common = dict(
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        mask_ratio_mu=args.mask_ratio_mu,
        mask_ratio_std=args.mask_ratio_std,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
        text_aligner_depth=args.text_aligner_depth,
        t5_embed_dim=args.t5_embed_dim,
        text_max_len=args.text_max_len,
    )
    if family == "repa":
        return repa.__dict__[args.model](
            **common,
            encoder_depth_proj=args.encoder_depth_proj,
            projector_dim=args.projector_dim,
            z_dim=args.z_dim,
        )
    return dream.__dict__[args.model](
        **common,
        clip_loss_on=args.clip_loss_on,
        vl_projection=args.vl_projection,
        txt_projection=args.txt_projection,
        ssl_mlp_dim=args.ssl_mlp_dim,
        embed_dim=args.embed_dim,
        variable_masking=args.variable_masking,
        mask_ratio_mu_start=args.mask_ratio_mu_start,
        mask_ratio_mu_end=args.mask_ratio_mu_end,
        epochs=args.epochs,
        fixed_masking_boundary=args.fixed_masking_boundary,
        masking_warmup=args.masking_warmup,
        masking_warmup_end=args.masking_warmup_end,
        masking_cooldown=args.masking_cooldown,
        warmup_shape=args.warmup_shape,
        schedule_std=args.schedule_std,
        min_ratio_for_clip_loss=args.min_ratio_for_clip_loss,
        min_masked_ratio_for_mar_loss=args.min_masked_ratio_for_mar_loss,
        scale_loss_by_mask=args.scale_loss_by_mask,
        filter_clip_loss=args.filter_clip_loss,
        depth_to_align=args.depth_to_align,
        uniform_masking=args.uniform_masking,
        repa_depth=args.encoder_depth_proj if args.repa_alignment else None,
        repa_projector_dim=args.projector_dim,
        repa_z_dim=args.z_dim,
    )


def load_dino_v2(args, device):
    """Frozen DINOv2 whose patch features are REPA's alignment targets, with position embeddings resized to the input."""
    dino_v2_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{args.dino_v2_size}14')
    del dino_v2_encoder.head
    patch_resolution = 16 * (args.img_size // 256)
    dino_v2_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        dino_v2_encoder.pos_embed.data, [patch_resolution, patch_resolution],
    )
    dino_v2_encoder.head = torch.nn.Identity()
    dino_v2_encoder = dino_v2_encoder.to(device).eval()
    print(f"DINOv2 encoder loaded: {args.dino_v2_size}, embedding dimension {dino_v2_encoder.embed_dim}, "
          f"patch resolution {patch_resolution}x{patch_resolution}")
    if dino_v2_encoder.embed_dim != args.z_dim:
        raise ValueError(f"--z_dim {args.z_dim} must match the DINOv2 embedding dimension {dino_v2_encoder.embed_dim}")

    return dino_v2_encoder


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    family = model_family(args.model)
    engine = engines.repa if family == "repa" else engines.dream
    use_dino = family == "repa" or args.repa_alignment  # frozen DINOv2 targets for the REPA alignment loss

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # --------------------------------------------------------------------------
    # data: training images, or captions to generate images for
    eval_set = None
    if args.evaluate:
        if args.eval_shards_path is not None:
            eval_set = ShardEvalSet(args.eval_shards_path, args.num_images, img_size=args.img_size, index_cache_dir=args.index_cache_dir)
        if args.fid_path1 is not None:
            dataset = []
        elif args.caption_dataset_path is not None:
            dataset = CaptionDataset(text_dir=args.caption_dataset_path)
        elif eval_set is not None:
            dataset = eval_set.captions()
        else:
            raise ValueError("--evaluate needs prompts: --eval_shards_path, --caption_dataset_path, or --fid_path1 (already generated images)")
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        data_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.eval_bsz,
            num_workers=0,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        transform_train, _ = get_image_transform(args.transform_type, image_size=args.img_size)
        if args.use_cached:
            dataset = CachedFolder_t5_fast(args.cached_path, random_horizontal_flip=(args.transform_type == 'mar'))
        elif args.dataset == "cc12m":
            dataset = return_cc12m_train_dataset(transform=transform_train, debug=args.debug,
                                                 data_dir=args.cc12m_path, index_cache_dir=args.index_cache_dir)
        else:
            dataset = return_cc3m_train_dataset(transform=transform_train, debug=args.debug,
                                                data_dir=args.cc3m_path, cache_dir=args.hf_cache_dir)
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    print(dataset)
    print("Sampler = %s" % str(sampler))

    # --------------------------------------------------------------------------
    # frozen VAE, and for REPA alignment the frozen DINOv2 target encoder (loaded before the model, as in the original REPA code)
    vae = StableAIAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for param in vae.parameters():
        param.requires_grad = False
    dino_v2_encoder = load_dino_v2(args, device) if use_dino else None

    # --------------------------------------------------------------------------
    # model
    model = build_model(args, family)

    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    n_params_enc = sum(p.numel() for p in model.encoder_blocks.parameters() if p.requires_grad)
    print("Number of trainable parameters in the encoder: {}M".format(n_params_enc / 1e6))
    if args.just_print_params:
        sys.exit(0)

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # no weight decay on bias, norm layers, and diffloss MLP
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
    print(optimizer)
    loss_scaler = NativeScaler()

    # frozen T5 text encoder (loaded after the model so the model's random init matches the original code for a given seed)
    text_encoder = T5EncoderModel.from_pretrained("google-t5/{}".format(args.text_encoder)).to(device).eval()

    # DREAM loads with strict=False to handle architecture mismatches with older checkpoints
    model_params, ema_params = misc.load_checkpoint(args, model_without_ddp, optimizer, loss_scaler, device, strict=(family == "repa"))

    # --------------------------------------------------------------------------
    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        text_tokenizer = AutoTokenizer.from_pretrained("google-t5/{}".format(args.text_encoder))
        eval_step = max(args.start_epoch - 1, 0)  # epoch of the loaded checkpoint, the x-axis of the logged FID
        eval_kwargs = dict(clip_tokenizer=SimpleTokenizer()) if family == "dream" else {}
        engine.evaluate(model_without_ddp, vae, text_tokenizer, text_encoder, data_loader, ema_params, args, eval_step,
                        log_writer=log_writer, cfg=args.cfg, use_ema=args.use_ema, eval_set=eval_set, **eval_kwargs)
        return

    # --------------------------------------------------------------------------
    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        train_kwargs = dict(dino_v2_encoder=dino_v2_encoder) if use_dino else {}
        engine.train_one_epoch(
            model, vae, text_encoder,
            model_params, ema_params,
            data_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            **train_kwargs,
        )

        # save checkpoint
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")
        if (epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs) and epoch != 0:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name=epoch)
        torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
