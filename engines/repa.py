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
            if 'samples' not in curr_batch:
                raise ValueError("REPA needs the images for its DINOv2 targets: build the cache with main_cache.py --save_samples")
            samples = curr_batch['samples'].to(device, non_blocking=True)
            moments = curr_batch['moments'].to(device, non_blocking=True)
            labels = curr_batch['labels'].to(device, non_blocking=True)
            text_embedding_t5 = curr_batch['text_embedding'].to(device, non_blocking=True)

            if empty_text_embedding_t5 is None:
                empty_text_embedding_t5 = curr_batch['empty_caption_embedding_t5'].to(device, non_blocking=True)
                empty_text_embedding_t5 = empty_text_embedding_t5[0:1].to(device, non_blocking=True)

        else:
            samples = curr_batch['image'].to(device, non_blocking=True)
            labels = curr_batch['labels'].to(device, non_blocking=True)
            input_ids = curr_batch['input_ids'].to(device, non_blocking=True)

            if empty_text_embedding_t5 is None:
                empty_input_ids = curr_batch['empty_input_ids'][0:1].to(device, non_blocking=True)
                empty_text_embedding_t5 = text_encoder(input_ids=empty_input_ids).last_hidden_state

        # Use GPU-specific autocast (V100: float16, A100: bfloat16)
        with misc.get_gpu_autocast():
            with torch.no_grad():
                if args.use_cached:
                    posterior = DiagonalGaussianDistribution(moments)
                    x = posterior.sample().mul_(0.18215)

                else:
                    posterior = vae.encode(samples)
                    x = posterior.latent_dist.sample().mul_(0.18215)

                    text_embedding_t5 = text_encoder(input_ids=input_ids).last_hidden_state

                # REPA targets: DINOv2 patch features
                target_rep = dino_v2_targets(dino_v2_encoder, samples, args.img_size)

                # randomly drop captions (replace with the empty caption) with probability text_drop_prob
                random_mask = torch.rand(text_embedding_t5.size(0), device=text_embedding_t5.device) < args.text_drop_prob
                mask = random_mask.view(-1, 1, 1)
                text_embedding_t5 = torch.where(
                    mask,
                    empty_text_embedding_t5.expand_as(text_embedding_t5),
                    text_embedding_t5
                )

            # forward
            loss, loss_repa = model(x, labels, text_embedding_t5, target_rep)
            loss_value = loss.item()
            loss_repa_value = loss_repa.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_total = args.weight_mar_loss * loss + args.weight_repa_loss * loss_repa
        loss_scaler(loss_total, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_repa=loss_repa_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_repa_value_reduce = misc.all_reduce_mean(loss_repa_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_repa', loss_repa_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, text_tokenizer, text_encoder, data_loader,
             ema_params, args, epoch, log_writer=None, cfg=1.0, use_ema=True, eval_set=None):
    text_encoder.eval()
    device = next(text_encoder.parameters()).device

    def sample_fn(captions):
        # encode text, add negative prompt if cfg
        prompts = add_empty_prompts(captions, cfg)
        input_ids = text_tokenizer(prompts, return_tensors="pt", padding="max_length",
                                   max_length=args.text_max_len, truncation=True).input_ids.to(device)
        text_embedding_t5 = text_encoder(input_ids=input_ids).last_hidden_state
        # Use GPU-specific autocast (V100: float16, A100: bfloat16)
        with misc.get_gpu_autocast():
            sampled_tokens = model_without_ddp.sample_tokens(bsz=len(captions), num_iter=args.num_iter, cfg=cfg, cfg_schedule=args.cfg_schedule,
                                                             text_embedding=text_embedding_t5, temperature=args.temperature)
            return vae.decode(sampled_tokens / 0.18215).sample

    generate_and_evaluate(model_without_ddp, data_loader, ema_params, args, epoch, sample_fn,
                          log_writer=log_writer, cfg=cfg, use_ema=use_ema, eval_set=eval_set)
