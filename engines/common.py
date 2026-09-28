import copy
import os
import shutil
import time

import cv2
import numpy as np
import torch
import torch_fidelity
from torchvision.transforms import Normalize

import util.misc as misc


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def preprocess_image(image, img_size):
    """Map images from [-1, 1] to DINOv2's input: ImageNet normalization at 224px (per 256px of input)."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    image = (image + 1) / 2
    image = Normalize(imagenet_mean, imagenet_std)(image)
    image = torch.nn.functional.interpolate(image, 224 * (img_size // 256), mode='bicubic')
    return image


def dino_v2_targets(dino_v2_encoder, samples, img_size):
    """REPA alignment targets: DINOv2 patch features [B, seq_len, D] of images in [-1, 1]."""
    return dino_v2_encoder.forward_features(preprocess_image(samples, img_size))["x_norm_patchtokens"]


def swap_to_ema(model, ema_params):
    """Load the EMA weights into model and return the original state dict (to restore with load_state_dict)."""
    model_state_dict = copy.deepcopy(model.state_dict())
    ema_state_dict = copy.deepcopy(model.state_dict())
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = ema_params[i]
    print("Switch to ema")
    model.load_state_dict(ema_state_dict)
    return model_state_dict


def sample_folder_name(args, cfg, use_ema):
    name = "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}-checkpoint{}".format(
        args.num_iter, args.num_sampling_steps, args.temperature, args.cfg_schedule, cfg, args.num_images, args.checkpoint)
    if use_ema:
        name += "_ema"
    if args.evaluate:
        name += "_evaluate"
    return os.path.join(args.output_dir, name)


def add_empty_prompts(captions, cfg):
    """Append the negative (empty) prompts used by classifier-free guidance; cfg == 0 means unconditional."""
    if cfg == 0.0:
        return [""] * len(captions) * 2
    if cfg != 1.0:
        return list(captions) + [""] * len(captions)
    return list(captions)


def save_images(images, filenames, save_folder, step, num_images):
    """Save a batch of images in [-1, 1] as PNGs, skipping any beyond num_images across all ranks."""
    world_size = misc.get_world_size()
    rank = misc.get_rank()
    images = (images.detach().cpu() + 1) / 2
    for b_id in range(images.size(0)):
        img_id = step * images.size(0) * world_size + rank * images.size(0) + b_id
        if img_id >= num_images:
            break
        gen_img = np.round(np.clip(images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
        gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(save_folder, f"{filenames[b_id]}.png"), gen_img)


def compute_fid(save_folder, args, epoch, cfg, use_ema, log_writer, eval_set=None):
    """
    FID / Inception Score of the images in save_folder against the reference: --fid_path2 (a folder of
    reference images or a torch-fidelity .npz statistics file) if given, otherwise the real images of eval_set
    (a ShardEvalSet), whose statistics torch-fidelity caches after the first run.
    """
    # Disable tokenizer parallelism warnings for FID calculation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    reference_kwargs = {}
    if args.fid_path2 is not None:
        reference = args.fid_path2
        if args.fid_path2.endswith(".npz"):
            reference_kwargs = dict(fid_statistics_file=args.fid_path2)
        else:
            reference_kwargs = dict(input2=args.fid_path2)
    elif eval_set is not None:
        reference = f"{eval_set.num_images} real images from --eval_shards_path"
        reference_kwargs = dict(input2=eval_set.reference_images(), input2_cache_name=eval_set.cache_name)
    else:
        raise ValueError("Specify a FID reference: --eval_shards_path, or --fid_path2 (image folder or .npz statistics)")

    print("Computing FID of", save_folder, "against", reference)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=save_folder,
        **reference_kwargs,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
        batch_size=512,
        num_workers=args.num_workers,
    )
    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']

    postfix = ""
    if use_ema:
        postfix = postfix + "_ema"
    if not cfg == 1.0:
        postfix = postfix + "_cfg{}".format(cfg)
    log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
    log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
    print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
    return fid, inception_score


def generate_and_evaluate(model_without_ddp, data_loader, ema_params, args, epoch, sample_fn,
                          log_writer=None, cfg=1.0, use_ema=True, eval_set=None):
    """
    Generate one image per caption in data_loader (a loader yielding (captions, filenames))
    with sample_fn(captions) -> images in [-1, 1], then compute FID / IS on rank 0.
    If --fid_path1 points to an existing folder, generation is skipped and that folder is evaluated instead.
    """
    if use_ema:
        model_state_dict = swap_to_ema(model_without_ddp, ema_params)
    model_without_ddp.eval()

    generated = args.fid_path1 is None
    if generated:
        batch_size = data_loader.batch_size
        num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
        save_folder = sample_folder_name(args, cfg, use_ema)
        print("Save to:", save_folder)
        if misc.get_rank() == 0:
            os.makedirs(save_folder, exist_ok=True)

        used_time = 0
        gen_img_cnt = 0

        data_loader.sampler.set_epoch(0)
        for i, (captions, filenames) in enumerate(data_loader):
            if i >= num_steps:
                break
            print("Generation step {}/{}".format(i, num_steps))

            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                sampled_images = sample_fn(list(captions)).to(torch.float32)

            # measure speed after the first generation batch
            if i >= 1:
                torch.cuda.synchronize()
                used_time += time.time() - start_time
                gen_img_cnt += len(captions)
                print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

            misc.barrier()
            save_images(sampled_images, filenames, save_folder, i, args.num_images)
            del sampled_images
            torch.cuda.empty_cache()

        misc.barrier()
        time.sleep(10)
    else:
        save_folder = args.fid_path1

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        compute_fid(save_folder, args, epoch, cfg, use_ema, log_writer, eval_set=eval_set)
        if generated and not args.keep_samples:
            shutil.rmtree(save_folder)

    misc.barrier()
    time.sleep(10)
