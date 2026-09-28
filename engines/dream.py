import math
import sys
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc
from engines.common import add_empty_prompts, dino_v2_targets, generate_and_evaluate, update_ema
from models.vae import DiagonalGaussianDistribution


def train_one_epoch(model, vae, text_encoder,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, dino_v2_encoder=None):
    """dino_v2_encoder: frozen DINOv2 for the optional REPA alignment loss (--repa_alignment), else None."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    empty_text_embedding_t5 = None

    for data_iter_step, curr_batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args.use_cached:
            if dino_v2_encoder is not None:
                if 'samples' not in curr_batch:
                    raise ValueError("REPA alignment needs the images for its DINOv2 targets: build the cache with main_cache.py --save_samples")
                samples = curr_batch['samples'].to(device, non_blocking=True)
            moments = curr_batch['moments'].to(device, non_blocking=True)
            labels = curr_batch['labels'].to(device, non_blocking=True)
            captions = curr_batch['captions'].to(device, non_blocking=True)
            text_embedding_t5 = curr_batch['text_embedding'].to(device, non_blocking=True)

            if empty_text_embedding_t5 is None:
                empty_text_embedding_t5 = curr_batch['empty_caption_embedding_t5'][0:1].to(device, non_blocking=True)
        else:
            samples = curr_batch['image'].to(device, non_blocking=True)
            labels = curr_batch['labels'].to(device, non_blocking=True)
            captions = curr_batch['caption'].to(device, non_blocking=True)
            input_ids = curr_batch['input_ids'].to(device, non_blocking=True)

            if empty_text_embedding_t5 is None:
                empty_input_ids = curr_batch['empty_input_ids'][0:1].to(device, non_blocking=True)
                empty_text_embedding_t5 = text_encoder(input_ids=empty_input_ids).last_hidden_state

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                if args.use_cached:
                    x = DiagonalGaussianDistribution(moments).sample().mul_(0.18215)
                else:
                    x = vae.encode(samples).latent_dist.sample().mul_(0.18215)
                    text_embedding_t5 = text_encoder(input_ids=input_ids).last_hidden_state

                # for cfg, we use empty text embedding
                # make sure dtype/device match
                empty_text_embedding_t5 = empty_text_embedding_t5.to(
                    dtype=text_embedding_t5.dtype, device=text_embedding_t5.device
                )

                target_rep = dino_v2_targets(dino_v2_encoder, samples, args.img_size) if dino_v2_encoder is not None else None

                # randomly drop captions (replace with the empty caption) with probability text_drop_prob
                random_mask = torch.rand(text_embedding_t5.size(0), device=text_embedding_t5.device) < args.text_drop_prob
                mask = random_mask.view(-1, 1, 1)
                text_embedding_t5 = torch.where(
                    mask,
                    empty_text_embedding_t5.expand_as(text_embedding_t5),
                    text_embedding_t5
                )

            # forward
            est_idx = data_iter_step / len(data_loader) + epoch
            loss, loss_clip, loss_repa, mask_info = model(x, captions, labels, text_embedding_t5, est_idx=est_idx, target_rep=target_rep)
            loss_clip_value = loss_clip.item()
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_total = args.weight_mar_loss * loss + args.weight_clip_loss * loss_clip
        if loss_repa is not None:
            loss_total = loss_total + args.weight_repa_loss * loss_repa
        loss_scaler(loss_total, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_clip=loss_clip_value)
        if loss_repa is not None:
            metric_logger.update(loss_repa=loss_repa.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # the MAR loss is zeroed on ranks whose batch is below --min_masked_ratio_for_mar_loss; average over the others
        loss_value_reduce = misc.all_reduce_nonzero_mean(loss_value)
        loss_clip_value_reduce = misc.all_reduce_mean(loss_clip_value)

        mar_mask_reduce = misc.all_reduce_mean(mask_info["mar_mask_ratio"])
        clip_mask_reduce = misc.all_reduce_mean(mask_info["clip_mask_ratio"])
        percentage_batch_with_clip_reduce = misc.all_reduce_mean(mask_info["percentage_batch_with_clip"])
        percentage_batch_with_mar_reduce = misc.all_reduce_mean(mask_info["percentage_batch_with_mar"])
        if loss_repa is not None:  # all ranks take part in the reduction; only rank 0 logs it
            loss_repa_value_reduce = misc.all_reduce_mean(loss_repa.item())

        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_clip', loss_clip_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('mask_ratio_mar', mar_mask_reduce, epoch_1000x)
            log_writer.add_scalar('mask_ratio_clip', clip_mask_reduce, epoch_1000x)
            log_writer.add_scalar('percentage_batch_with_clip', percentage_batch_with_clip_reduce, epoch_1000x)
            log_writer.add_scalar('percentage_batch_with_mar', percentage_batch_with_mar_reduce, epoch_1000x)
            if loss_repa is not None:
                log_writer.add_scalar('train_loss_repa', loss_repa_value_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, text_tokenizer, text_encoder, data_loader,
             ema_params, args, epoch, log_writer=None, cfg=1.0,
             use_ema=True, clip_tokenizer=None, eval_set=None):
    text_encoder.eval()
    device = next(text_encoder.parameters()).device

    def sample_fn(captions):
        bsz = len(captions)
        # encode text, add negative prompt if cfg
        prompts = add_empty_prompts(captions, cfg)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            input_ids = text_tokenizer(prompts, return_tensors="pt", padding="max_length",
                                       max_length=args.text_max_len, truncation=True).input_ids.to(device)
            text_embedding_t5 = text_encoder(input_ids=input_ids).last_hidden_state

            kwargs = dict(bsz=bsz, num_iter=args.num_iter, cfg=cfg, cfg_schedule=args.cfg_schedule,
                          text_embedding=text_embedding_t5, temperature=args.temperature)
            if args.use_clip_critic or args.use_clip_critic_once:
                # Semantically Aligned Decoding: score candidates with the jointly trained CLIP text tower
                critic_kwargs = dict(text=clip_tokenizer(prompts).to(device)[:bsz], num_candidates=args.num_candidates,
                                     clip_critic_threshold=args.clip_critic_threshold)
                if args.use_clip_critic:
                    sampled_tokens = model_without_ddp.sample_tokens_with_clip_critic(**kwargs, **critic_kwargs)
                else:
                    sampled_tokens = model_without_ddp.sample_tokens_with_clip_critic_once(**kwargs, **critic_kwargs)
            else:
                sampled_tokens = model_without_ddp.sample_tokens(**kwargs)
            return vae.decode(sampled_tokens / 0.18215).sample

    generate_and_evaluate(model_without_ddp, data_loader, ema_params, args, epoch, sample_fn,
                          log_writer=log_writer, cfg=cfg, use_ema=use_ema, eval_set=eval_set)
