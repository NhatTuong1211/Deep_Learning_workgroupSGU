"""Dataset & DataLoader chuẩn cho E0-E4 (TV1 bàn giao cho TV2).

Dùng trong code của nhóm:
    from dataset import build_dataloaders
    loaders = build_dataloaders(config)      # dict: train / val / test

Chạy sanity check + ghi nhật ký W&B (Run R5_sanity_check) từ terminal:
    python members/TV1_ThuyLinh/code/dataset.py
    python members/TV1_ThuyLinh/code/dataset.py --no-wandb
    python members/TV1_ThuyLinh/code/dataset.py --num-workers 0   # nếu lỗi trên Windows
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .preprocessing import (
        denormalize, find_raw_dirs, load_image, load_trimap, preprocess_pair
    )
except ImportError:
    from preprocessing import (
        denormalize, find_raw_dirs, load_image, load_trimap, preprocess_pair
    )


class PetSegmentationDataset(Dataset):
    """Trả về (image, mask):
    image: float32 [3,H,W] đã chuẩn hóa ImageNet
    mask : int64   [H,W] giá trị {0, 1} (0 = Background, 1 = Pet gồm cả biên)
    """

    def __init__(self, split, config, raw_dir="data/raw",
                 splits_dir="data/splits"):
        assert split in ("train", "val", "test"), split
        self.split = split
        self.config = config
        self.train = split == "train"  # augmentation CHỈ cho train
        self.images_dir, self.trimaps_dir = find_raw_dirs(raw_dir)

        with open(os.path.join(splits_dir, f"{split}.txt"),
                  encoding="utf-8") as f:
            self.names = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        stem = os.path.splitext(name)[0]
        image = load_image(os.path.join(self.images_dir, name))
        trimap = load_trimap(os.path.join(self.trimaps_dir, stem + ".png"))
        return preprocess_pair(image, trimap, self.config, train=self.train)


def seed_worker(worker_id):
    """Cố định seed cho từng worker để augmentation tái lập được."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(config, raw_dir=None, splits_dir=None):
    """Trả về dict {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}."""
    raw_dir = raw_dir or config.get("raw_dir", "data/raw")
    splits_dir = splits_dir or config.get("splits_dir", "data/splits")

    generator = torch.Generator()
    generator.manual_seed(config["seed"])

    loaders = {}
    for split in ("train", "val", "test"):
        ds = PetSegmentationDataset(split, config, raw_dir, splits_dir)
        loaders[split] = DataLoader(
            ds,
            batch_size=config["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["num_workers"],
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


# ======================================================================
# Sanity check + hình ảnh + W&B (chỉ chạy khi gọi bằng terminal)
# ======================================================================

def project_root(start=None):
    """Đi ngược lên đến thư mục có cả data/ và members/ (gốc dự án)."""
    cur = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    while True:
        if (os.path.isdir(os.path.join(cur, "data"))
                and os.path.isdir(os.path.join(cur, "members"))):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


def read_split(splits_dir, name):
    with open(os.path.join(splits_dir, f"{name}.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_sanity(config, raw_dir="data/raw", splits_dir="data/splits"):
    """Kiểm tra 1 batch thực tế của train/val/test. Raise AssertionError nếu sai."""
    loaders = build_dataloaders(config, raw_dir, splits_dir)

    # Augmentation chỉ áp dụng cho train
    assert loaders["train"].dataset.train is True
    assert loaders["val"].dataset.train is False
    assert loaders["test"].dataset.train is False

    H, W = config["img_size"]
    sanity, first_batch = {}, {}

    for split, loader in loaders.items():
        t0 = time.time()
        imgs, masks = next(iter(loader))
        load_time = time.time() - t0

        uniq = sorted(torch.unique(masks).tolist())

        assert imgs.ndim == 4 and tuple(imgs.shape[1:]) == (3, H, W), imgs.shape
        assert tuple(masks.shape) == (imgs.shape[0], H, W), masks.shape
        assert imgs.dtype == torch.float32, imgs.dtype
        assert masks.dtype == torch.int64, masks.dtype
        assert set(uniq) <= {0, 1}, f"Nhãn mask ngoài {{0,1}}: {uniq}"

        sanity[split] = {
            "split": split,
            "n_samples": len(loader.dataset),
            "img_shape": str(tuple(imgs.shape)),
            "img_dtype": str(imgs.dtype),
            "img_min": round(float(imgs.min()), 4),
            "img_max": round(float(imgs.max()), 4),
            "mask_shape": str(tuple(masks.shape)),
            "mask_dtype": str(masks.dtype),
            "mask_unique_labels": str(uniq),
            "batch_load_time_sec": round(load_time, 3),
        }
        first_batch[split] = (imgs, masks)

    sets = {s: set(read_split(splits_dir, s)) for s in ("train", "val", "test")}
    overlap = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }

    return {"sanity": sanity, "first_batch": first_batch,
            "overlap": overlap, "overlap_total": sum(overlap.values())}


def save_pipeline_diagram(path):
    """Sơ đồ khối luồng tiền xử lý (preprocessing_pipeline.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis("off")

    box = dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1B365D")
    box_aug = dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#E65100")

    img_steps = [
        ("Ảnh .jpg", box), ("Chuyển RGB\n(3 kênh)", box),
        ("Resize 256×256\nbilinear", box), ("Augmentation\n(CHỈ train)", box_aug),
        ("ToTensor +\nNormalize\n(ImageNet)", box),
        ("image\nfloat32 [3,H,W]", box),
    ]
    mask_steps = [
        ("Trimap .png\n{1,2,3}", box), ("Nhị phân\n1,3 → 1 ; 2 → 0", box),
        ("Resize 256×256\nnearest", box), ("Augmentation\nđồng bộ ảnh", box_aug),
        ("LongTensor", box), ("mask\nint64 [H,W] {0,1}", box),
    ]
    xs = [1.2, 3.6, 6.0, 8.4, 10.8, 13.4]

    for (label, style), x in zip(img_steps, xs):
        ax.text(x, 3.7, label, ha="center", va="center", fontsize=9, bbox=style)
    for (label, style), x in zip(mask_steps, xs):
        ax.text(x, 1.3, label, ha="center", va="center", fontsize=9, bbox=style)

    for row_y in (3.7, 1.3):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            ax.annotate("", xy=(x1 - 1.0, row_y), xytext=(x0 + 1.0, row_y),
                        arrowprops=dict(arrowstyle="->", color="#1B365D"))

    ax.text(7.5, 4.7, "Luồng ẢNH", ha="center", fontsize=11, weight="bold")
    ax.text(7.5, 2.3, "Luồng MASK (cùng phép biến đổi hình học)",
            ha="center", fontsize=11, weight="bold")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_batch_figure(imgs, masks, config, path, n_show=8):
    """Lưới ảnh sau tiền xử lý + overlay mask. Trả về danh sách (rgb, mask) để log W&B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(n_show, imgs.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5), squeeze=False)
    samples = []

    for i in range(n_show):
        rgb = denormalize(imgs[i], config["mean"], config["std"]
                          ).permute(1, 2, 0).numpy()
        m = masks[i].numpy()
        samples.append((rgb, m))

        axes[0, i].imshow(rgb)
        axes[0, i].axis("off")
        axes[1, i].imshow(rgb)
        axes[1, i].imshow(np.ma.masked_where(m == 0, m), cmap="autumn",
                          alpha=0.5, vmin=0, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_title("Ảnh sau tiền xử lý (train)", loc="left", fontsize=9)
    axes[1, 0].set_title("Overlay mask nhị phân", loc="left", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return samples


def save_preprocessing_examples(config, raw_dir, splits_dir, path, n=4):
    """So sánh ảnh gốc / sau tiền xử lý / 2 lần augmentation (train)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images_dir, trimaps_dir = find_raw_dirs(raw_dir)
    names = read_split(splits_dir, "train")
    picks = random.Random(config["seed"]).sample(names, min(n, len(names)))

    random.seed(config["seed"])
    titles = ["Ảnh gốc", "Sau tiền xử lý (val/test)",
              "Augmentation 1 (train)", "Augmentation 2 (train)"]
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)),
                             squeeze=False)

    for r, name in enumerate(picks):
        stem = os.path.splitext(name)[0]
        image = load_image(os.path.join(images_dir, name))
        tri = load_trimap(os.path.join(trimaps_dir, stem + ".png"))

        axes[r, 0].imshow(image)
        for c, train_flag in enumerate([False, True, True], start=1):
            img_t, m_t = preprocess_pair(image, tri, config, train=train_flag)
            rgb = denormalize(img_t, config["mean"], config["std"]
                              ).permute(1, 2, 0).numpy()
            axes[r, c].imshow(rgb)
            if int(m_t.max()) > 0:
                axes[r, c].contour(m_t.numpy(), levels=[0.5],
                                   colors="red", linewidths=1)
        for c in range(4):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c])

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def log_wandb(result, samples, config, fig_paths, entity, project):
    """W&B Run 05: R5_sanity_check."""
    import pandas as pd
    import wandb

    sanity = result["sanity"]
    tr = sanity["train"]
    sanity_df = pd.DataFrame(sanity.values())

    run = wandb.init(
        project=project,
        entity=entity,
        name="R5_sanity_check",
        job_type="sanity",
        tags=["TV1", "sanity", f"data_version:{config['data_version']}"],
        config={
            **config,
            "log_id": "P005",
            "work_item": "DataLoader Sanity Check (Gate 2)",
            "target_file": "dataset.py",
            "handoff_status": "Ready_for_Gate2",
        },
        notes="Sanity check trên 1 batch thực tế của train/val/test sau khi chia tập.",
        reinit="finish_previous",
    )

    wandb_images = [
        wandb.Image(
            (rgb * 255).astype(np.uint8),
            masks={"ground_truth": {
                "mask_data": m.astype(np.uint8),
                "class_labels": {0: "Background", 1: "Pet"},
            }},
        )
        for rgb, m in samples
    ]

    run.log({
        "n_train": sanity["train"]["n_samples"],
        "n_val": sanity["val"]["n_samples"],
        "n_test": sanity["test"]["n_samples"],
        "overlap_count": result["overlap_total"],
        "img_min": tr["img_min"],
        "img_max": tr["img_max"],
        "batch_load_time_sec": tr["batch_load_time_sec"],
        "sanity_table": wandb.Table(dataframe=sanity_df),
        "batch_overlay": wandb_images,
        "sanity_batch": wandb.Image(fig_paths["batch"]),
        "preprocessing_examples": wandb.Image(fig_paths["examples"]),
        "preprocessing_pipeline": wandb.Image(fig_paths["pipeline"]),
    })

    # Các trường khớp với Sheet Sanity_Check_Log trong TV1_Data_Log.xlsx
    run.summary.update({
        "leakage_check": "PASS" if result["overlap_total"] == 0 else "FAIL",
        "tensor_shape_dtype": (
            f"image {tr['img_shape']} {tr['img_dtype']}; "
            f"mask {tr['mask_shape']} {tr['mask_dtype']}"
        ),
        "min_max_unique_labels": (
            f"img [{tr['img_min']}, {tr['img_max']}]; "
            f"mask labels {tr['mask_unique_labels']}"
        ),
        "check_result": "PASS",
        "decision": "ACCEPT",
        "reason": "Shape, dtype, nhãn mask {0,1} đúng; augmentation chỉ ở train.",
    })

    run.finish()
    print("[W&B] Hoàn thành Run: R5_sanity_check")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Sanity check DataLoader (TV1)")
    ap.add_argument("--config", default="data/splits/split_config.json")
    ap.add_argument("--entity", default=os.environ.get(
        "WANDB_ENTITY", "linhthuynguyen2603-school"))
    ap.add_argument("--project", default="Oxford-IIIT-Pet-Segmentation")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Ghi đè num_workers trong config (đặt 0 nếu lỗi trên Windows)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.chdir(project_root())
    print("Thư mục gốc dự án:", os.getcwd())

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    result = run_sanity(config)

    for row in result["sanity"].values():
        print(row)
    print("Giao giữa các tập:", result["overlap"])
    assert result["overlap_total"] == 0, "Phát hiện giao giữa các tập!"

    fig_dir = "results/figures"
    fig_paths = {
        "batch": os.path.join(fig_dir, "sanity_batch.png"),
        "examples": os.path.join(fig_dir, "preprocessing_examples.png"),
        "pipeline": os.path.join(fig_dir, "preprocessing_pipeline.png"),
    }

    imgs_tr, masks_tr = result["first_batch"]["train"]
    samples = save_batch_figure(imgs_tr, masks_tr, config, fig_paths["batch"])
    save_preprocessing_examples(config, "data/raw", "data/splits",
                                fig_paths["examples"])
    save_pipeline_diagram(fig_paths["pipeline"])
    print("Đã lưu hình:", ", ".join(fig_paths.values()))

    if not args.no_wandb:
        log_wandb(result, samples, config, fig_paths, args.entity, args.project)


if __name__ == "__main__":
    main()
