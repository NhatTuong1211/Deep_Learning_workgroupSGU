from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from inputs.model.cnn_model import build_model
from inputs.model.model_config import ModelConfig


# 1. PATHS
ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "inputs" / "checkpoints" / "best_config.json"
CHECKPOINT_PATH = ROOT_DIR / "inputs" / "checkpoints" / "best_model.pth"

OUTPUT_DIR = ROOT_DIR / "results" / "evaluation" / "convolution_filters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. LOAD CONFIGURATION
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_data = json.load(f)

model_config = ModelConfig.from_dict(config_data["model_config"])

model = build_model(model_config)


# 3. LOAD TRAINED CHECKPOINT
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

# Hỗ trợ một số định dạng checkpoint phổ biến
if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
else:
    raise TypeError("Checkpoint không có định dạng hợp lệ.")

# Xử lý tiền tố module. nếu có
state_dict = {
    key.removeprefix("module."): value
    for key, value in state_dict.items()
}

model.load_state_dict(state_dict, strict=True)
model.eval()

print("TV4 - VISUALIZE CONVOLUTION FILTERS")
print("-" * 60)
print(f"Experiment: {config_data['experiment_id']}")
print(f"Best epoch: {config_data['best_epoch']}")
print("Checkpoint loaded successfully.")


# 4. FIND ALL CONVOLUTION LAYERS
conv_layers = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, nn.Conv2d)
]

print(f"\nTotal Conv2d layers: {len(conv_layers)}")


# 5. VISUALIZE FILTERS
FILTERS_PER_IMAGE = 64
COLS = 8

for layer_index, (layer_name, layer) in enumerate(conv_layers, start=1):

    weights = layer.weight.detach().cpu().numpy()

    # Shape: [out_channels, in_channels, kernel_h, kernel_w]
    out_channels, in_channels, kernel_h, kernel_w = weights.shape

    if in_channels == 1:
        filters = weights[:, 0, :, :]
        filter_type = "Original grayscale kernels"
        cmap = "coolwarm"
        limit = np.max(np.abs(filters))
        vmin = -limit
        vmax = limit
    else:
        filters = np.sqrt(np.mean(weights ** 2, axis=1))
        filter_type = "RMS magnitude across input channels"
        cmap = "viridis"
        vmin = 0
        vmax = np.max(filters)

    if vmax == 0:
        vmax = 1

    # Chia filters thành từng nhóm tối đa 64
    num_pages = int(np.ceil(out_channels / FILTERS_PER_IMAGE))

    for page in range(num_pages):

        start = page * FILTERS_PER_IMAGE
        end = min(start + FILTERS_PER_IMAGE, out_channels)

        page_filters = filters[start:end]
        num_filters = len(page_filters)

        rows = int(np.ceil(num_filters / COLS))

        fig, axes = plt.subplots(
            rows,
            COLS,
            figsize=(16, rows * 2)
        )

        axes = np.array(axes).reshape(-1)

        for local_idx, filter_img in enumerate(page_filters):

            global_idx = start + local_idx
            ax = axes[local_idx]

            ax.imshow(
                filter_img,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(
                f"Filter {global_idx}",
                fontsize=10
            )

            ax.axis("off")

        # Tắt các ô không sử dụng
        for ax in axes[num_filters:]:
            ax.axis("off")

        fig.suptitle(
            f"{layer_name}\n"
            f"{filter_type}\n"
            f"Filters {start} - {end - 1} "
            f"| Shape: {weights.shape}",
            fontsize=14
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        safe_name = layer_name.replace(".", "_")

        output_path = OUTPUT_DIR / (
            f"conv_{layer_index:02d}_{safe_name}"
            f"_part_{page + 1}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"[OK] {layer_name} - "
            f"Part {page + 1}/{num_pages}: "
            f"Filters {start}-{end - 1}"
        )


# 6. SUMMARY
total_images = sum(
    int(np.ceil(layer.out_channels / FILTERS_PER_IMAGE))
    for _, layer in conv_layers
)

print("\nSaved convolution filter visualizations:")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Total Conv2d layers: {len(conv_layers)}")
print(f"Total visualization images: {total_images}")

if len(conv_layers) != 8:
    raise RuntimeError(
        f"Expected 8 Conv2d layers, found {len(conv_layers)}."
    )

print("\nVISUALIZE CONVOLUTION FILTERS: PASSED")