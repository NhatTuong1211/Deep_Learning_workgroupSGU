
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT_DIR / "results" / "evaluation" / "prediction_examples"
OUTPUT_PATH = ROOT_DIR / "results" / "evaluation" / "prediction_examples.png"

correct_path = INPUT_DIR / "correct_predictions.png"
incorrect_path = INPUT_DIR / "incorrect_predictions.png"

correct_image = mpimg.imread(correct_path)
incorrect_image = mpimg.imread(incorrect_path)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

axes[0].imshow(correct_image)
axes[0].set_title("Correct Predictions", fontsize=15)
axes[0].axis("off")

axes[1].imshow(incorrect_image)
axes[1].set_title("Incorrect Predictions", fontsize=15)
axes[1].axis("off")

fig.suptitle(
    "FER2013 - Prediction Examples with Confidence",
    fontsize=18
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUTPUT_PATH}")
print("PREDICTION EXAMPLES COMBINATION: PASSED")
