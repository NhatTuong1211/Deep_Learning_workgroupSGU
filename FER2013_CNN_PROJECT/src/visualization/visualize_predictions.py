from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from inputs.data.dataset import FER2013Dataset
from inputs.data.preprocessing import get_transforms
from inputs.data.label_mapping import CLASS_NAMES


# 1. PATHS AND CONFIGURATION
ROOT_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT_DIR / "inputs" / "data" / "fer2013.csv"
PREDICTIONS_PATH = ROOT_DIR / "results" / "evaluation" / "predictions.npz"

OUTPUT_DIR = ROOT_DIR / "results" / "evaluation" / "prediction_examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CORRECT = 12
NUM_INCORRECT = 12


# 2. LOAD PREDICTIONS
data = np.load(PREDICTIONS_PATH)

y_true = data["y_true"]
y_pred = data["y_pred"]
y_prob = data["y_prob"]

# Confidence of the predicted class
confidence = np.max(y_prob, axis=1)

print("TV4 - VISUALIZE PREDICTIONS")
print("-" * 60)


# 3. LOAD TEST DATASET
_, eval_transform = get_transforms(
    use_aug=False,
    mean=0.5,
    std=0.5
)

test_dataset = FER2013Dataset(
    csv_path=str(CSV_PATH),
    split="test",
    transform=eval_transform
)

print(f"Test dataset: {len(test_dataset)}")
print(f"Predictions: {len(y_true)}")


# 4. CHECK DATA CONSISTENCY
if len(test_dataset) != len(y_true):
    raise ValueError(
        "Số lượng ảnh trong Dataset không khớp predictions.npz."
    )

dataset_labels = test_dataset.labels

if not np.array_equal(dataset_labels, y_true):
    raise ValueError(
        "Thứ tự nhãn Dataset không khớp y_true. "
        "Không thể ghép ảnh với dự đoán một cách an toàn."
    )

print("[OK] Dataset order matches predictions.")


# 5. FIND CORRECT AND INCORRECT PREDICTIONS
correct_indices = np.where(y_true == y_pred)[0]
incorrect_indices = np.where(y_true != y_pred)[0]

print(f"Correct predictions: {len(correct_indices)}")
print(f"Incorrect predictions: {len(incorrect_indices)}")

rng = np.random.default_rng(42)

selected_correct = rng.choice(
    correct_indices,
    size=min(NUM_CORRECT, len(correct_indices)),
    replace=False
)

selected_incorrect = rng.choice(
    incorrect_indices,
    size=min(NUM_INCORRECT, len(incorrect_indices)),
    replace=False
)


# 6. VISUALIZATION FUNCTION
def plot_predictions(indices, title, output_path):

    n = len(indices)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, rows * 3)
    )

    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):

        image, _ = test_dataset[int(idx)]

        # Undo normalization: [-1, 1] -> [0, 1]
        image = image.squeeze().numpy()
        image = np.clip(image * 0.5 + 0.5, 0, 1)

        true_label = CLASS_NAMES[int(y_true[idx])]
        predicted_label = CLASS_NAMES[int(y_pred[idx])]
        confidence_value = confidence[idx] * 100

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label}\n"
            f"Confidence: {confidence_value:.2f}%",
            fontsize=9
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# 7. SAVE VISUALIZATIONS
plot_predictions(
    selected_correct,
    "Correct Predictions - FER2013",
    OUTPUT_DIR / "correct_predictions.png"
)

plot_predictions(
    selected_incorrect,
    "Incorrect Predictions - FER2013",
    OUTPUT_DIR / "incorrect_predictions.png"
)

print("\nSaved results:")
print(f"- {OUTPUT_DIR / 'correct_predictions.png'}")
print(f"- {OUTPUT_DIR / 'incorrect_predictions.png'}")

print("\nVISUALIZATION: PASSED")