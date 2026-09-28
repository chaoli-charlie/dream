import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def get_image_transform(transform_type, image_size=256):
    """Returns (transform_train, transform_val) for pre-training."""
    if transform_type == 'default':
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        transform_val = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    elif transform_type == 'mar':
        transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transform_val = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    elif transform_type == 'clip':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.ToTensor(),
                normalize
            ])
        transform_val = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize
            ])
    else:
        raise ValueError(f"Unknown transform type: {transform_type}. Choose from 'default', 'mar', or 'clip'.")

    return transform_train, transform_val


def get_linprobe_transform(aug, image_size=256):
    """Returns (transform_train, transform_val) for linear probing: aug is 'weak' (MAE-style) or 'stablerep'."""
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    if aug == 'weak':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, interpolation=3, scale=(0.2, 1), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        transform_val = transforms.Compose([
            transforms.Resize(292, interpolation=3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize])
    elif aug == 'stablerep':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unknown linear probe augmentation: {aug}")
    return transform_train, transform_val
