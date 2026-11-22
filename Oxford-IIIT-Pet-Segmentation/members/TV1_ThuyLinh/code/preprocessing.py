"""Tiền xử lý & augmentation cho Oxford-IIIT Pet Segmentation (TV1).

Quy ước khóa cứng cho E0-E4:
- Border Policy (a): Pet = {1, 3} -> 1, Background = {2} -> 0
- Resize thẳng về img_size (không padding, méo tỷ lệ nhẹ)
- Ảnh: nội suy bilinear. Mask: BẮT BUỘC nearest-neighbor
- Chuẩn hóa mean/std ImageNet giống nhau cho cả 5 mô hình
- Augmentation CHỈ áp dụng cho train, biến đổi hình học đồng bộ ảnh + mask
"""
import glob
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


def find_raw_dirs(raw_dir="data/raw"):
    """Tìm thư mục images/ và trimaps/ bên dưới raw_dir."""
    img_dirs = glob.glob(os.path.join(raw_dir, "**", "images"), recursive=True)
    tri_dirs = glob.glob(os.path.join(raw_dir, "**", "trimaps"), recursive=True)
    images_dir = img_dirs[0] if img_dirs else os.path.join(raw_dir, "images")
    trimaps_dir = (
        tri_dirs[0] if tri_dirs
        else os.path.join(raw_dir, "annotations", "trimaps")
    )
    return images_dir, trimaps_dir


def load_image(path):
    """Đọc ảnh và luôn chuyển về RGB 3 kênh (trong bộ nhớ, không ghi đè file)."""
    with Image.open(path) as im:
        return im.convert("RGB")


def load_trimap(path):
    """Đọc trimap gốc, giá trị {1, 2, 3}."""
    with Image.open(path) as m:
        return np.array(m)


def trimap_to_binary(trimap):
    """Border Policy (a): nhãn 1 và 3 -> Pet (1); nhãn 2 -> Background (0)."""
    return np.where((trimap == 1) | (trimap == 3), 1, 0).astype(np.uint8)


def resize_pair(image, mask01, img_size):
    """Resize thẳng, không padding. Ảnh: bilinear. Mask: nearest."""
    h, w = img_size
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    mask = Image.fromarray(mask01).resize((w, h), Image.Resampling.NEAREST)
    return image, mask


def augment_pair(image, mask, aug):
    """Augmentation đồng bộ ảnh + mask. Chỉ gọi cho tập train."""
    # Lật ngang
    if random.random() < aug["horizontal_flip_p"]:
        image, mask = TF.hflip(image), TF.hflip(mask)

    # Xoay: cùng một góc cho ảnh và mask; vùng trống của mask gán Background (0)
    angle = random.uniform(-aug["rotation_degrees"], aug["rotation_degrees"])
    image = TF.rotate(
        image, angle, interpolation=InterpolationMode.BILINEAR, fill=0
    )
    mask = TF.rotate(
        mask, angle, interpolation=InterpolationMode.NEAREST,
        fill=aug.get("mask_rotate_fill", 0)
    )

    # Đổi độ sáng / tương phản: chỉ áp dụng cho ảnh, không đụng tới mask
    image = TF.adjust_brightness(
        image, random.uniform(1 - aug["brightness"], 1 + aug["brightness"])
    )
    image = TF.adjust_contrast(
        image, random.uniform(1 - aug["contrast"], 1 + aug["contrast"])
    )
    return image, mask


def to_tensors(image, mask, mean, std):
    """Ảnh -> float32 [3,H,W] đã chuẩn hóa; mask -> int64 [H,W] giá trị {0,1}."""
    img_t = TF.normalize(TF.to_tensor(image), mean=mean, std=std)
    mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
    return img_t, mask_t


def preprocess_pair(image, trimap, config, train=False):
    """Pipeline chuẩn: nhị phân hóa -> resize -> (augment nếu train) -> tensor."""
    mask01 = trimap_to_binary(trimap)
    image, mask = resize_pair(image, mask01, config["img_size"])
    if train:
        image, mask = augment_pair(image, mask, config["augmentation"])
    return to_tensors(image, mask, config["mean"], config["std"])


def denormalize(img_t, mean, std):
    """Đảo chuẩn hóa để hiển thị ảnh."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (img_t * s + m).clamp(0, 1)
