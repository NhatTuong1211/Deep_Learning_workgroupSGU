
from pathlib import Path
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT_DIR / "results" / "evaluation" / "convolution_filters"
OUTPUT_PATH = ROOT_DIR / "results" / "evaluation" / "conv_filters.png"

image_paths = sorted(INPUT_DIR.glob("*.png"))

if not image_paths:
    raise FileNotFoundError("No convolution filter images found.")

n_images = len(image_paths)
cols = 4
rows = math.ceil(n_images / cols)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(20, rows * 5)
)

axes = axes.flatten()

for ax, image_path in zip(axes, image_paths):
    image = mpimg.imread(image_path)

    ax.imshow(image)
    ax.set_title(image_path.stem, fontsize=7)
    ax.axis("off")

for ax in axes[n_images:]:
    ax.axis("off")

fig.suptitle(
    "FER2013CNN - Convolution Filters Visualization",
    fontsize=16
)

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print(f"Number of filter images: {n_images}")
print(f"Saved: {OUTPUT_PATH}")
print("CONVOLUTION FILTERS COMBINATION: PASSED")
