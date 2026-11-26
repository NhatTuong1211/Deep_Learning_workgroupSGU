from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# 1. PATHS AND CONFIGURATION
ROOT_DIR = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = ROOT_DIR / "results" / "evaluation" / "predictions.npz"
OUTPUT_DIR = ROOT_DIR / "results" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# 2. LOAD PREDICTIONS
data = np.load(PREDICTIONS_PATH)

y_true = data["y_true"]
y_pred = data["y_pred"]

print("TV4 - CONFUSION MATRIX")
print("-" * 60)
print(f"Total samples: {len(y_true)}")


# 3. COMPUTE CONFUSION MATRIX
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(7)
)

print("\nConfusion Matrix:")
print(cm)


# 4. SAVE RAW CONFUSION MATRIX
cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

csv_path = OUTPUT_DIR / "confusion_matrix.csv"
cm_df.to_csv(csv_path, encoding="utf-8-sig")


# 5. PLOT RAW CONFUSION MATRIX
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Confusion Matrix - FER2013 Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

raw_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(raw_path, dpi=300, bbox_inches="tight")
plt.close()


# 6. PLOT NORMALIZED CONFUSION MATRIX
row_sums = cm.sum(axis=1, keepdims=True)

cm_normalized = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    vmin=0,
    vmax=1
)

plt.title("Normalized Confusion Matrix - FER2013 Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

normalized_path = OUTPUT_DIR / "confusion_matrix_normalized.png"
plt.savefig(normalized_path, dpi=300, bbox_inches="tight")
plt.close()


# 7. SUMMARY
print("\nSaved results:")
print(f"- {csv_path}")
print(f"- {raw_path}")
print(f"- {normalized_path}")

print("\nCONFUSION MATRIX: PASSED")