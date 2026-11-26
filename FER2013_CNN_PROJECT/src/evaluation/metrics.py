import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


# 1. KHAI BAO DUONG DAN
BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    BASE_DIR / "results" / "evaluation" / "predictions.npz"
)

OUTPUT_DIR = BASE_DIR / "results" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUTPUT_DIR / "test_metrics.json"


# 2. KHAI BAO TEN CAC LOP
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

LABELS = list(range(7))


# 3. DOC DU LIEU DU DOAN
print("-" * 60)
print("TV4 - MODEL EVALUATION METRICS")
print("-" * 60)

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay file: {PREDICTIONS_PATH}"
    )

data = np.load(PREDICTIONS_PATH)

y_true = data["y_true"]
y_pred = data["y_pred"]

print("\nDa doc predictions.npz")

print("So mau:", len(y_true))


# 4. TINH ACCURACY
accuracy = accuracy_score(y_true, y_pred)


# 5. TINH PRECISION, RECALL, F1-SCORE
precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0
    )
)


# 6. TINH MACRO AVERAGE
macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )
)


# 7. TINH WEIGHTED AVERAGE
weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0
    )
)


# 8. TAO BAO CAO CHI TIET
report = classification_report(
    y_true,
    y_pred,
    labels=LABELS,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# 9. TAO KET QUA JSON
results = {
    "dataset": "PrivateTest",
    "total_samples": int(len(y_true)),
    "accuracy": float(accuracy),

    "per_class": {},

    "macro_avg": {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1_score": float(macro_f1)
    },

    "weighted_avg": {
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1_score": float(weighted_f1)
    }
}


# 10. LUU CHI SO TUNG LOP
for i, class_name in enumerate(CLASS_NAMES):

    results["per_class"][class_name] = {
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1_score": float(f1[i]),
        "support": int(support[i])
    }


# 11. HIEN THI KET QUA
print("\n" + "-" * 75)
print("CLASSIFICATION METRICS")
print("-" * 75)

print(
    f"{'Emotion':<12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1-score':>12}"
    f"{'Support':>12}"
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<12}"
        f"{precision[i]:>12.4f}"
        f"{recall[i]:>12.4f}"
        f"{f1[i]:>12.4f}"
        f"{support[i]:>12}"
    )


# 12. HIEN THI MACRO VA WEIGHTED
print("\nMACRO AVERAGE")

print("Precision:", round(macro_precision, 4))
print("Recall:", round(macro_recall, 4))
print("F1-score:", round(macro_f1, 4))

print("\nWEIGHTED AVERAGE")

print("Precision:", round(weighted_precision, 4))
print("Recall:", round(weighted_recall, 4))
print("F1-score:", round(weighted_f1, 4))

print("\nAccuracy:", round(accuracy * 100, 2), "%")


# 13. LUU FILE JSON
with open(METRICS_PATH, "w", encoding="utf-8") as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nDa luu ket qua:", METRICS_PATH)


# 14. KIEM TRA
assert len(y_true) == len(y_pred)

assert len(precision) == 7
assert len(recall) == 7
assert len(f1) == 7

assert 0 <= accuracy <= 1

print("\n" + "-" * 60)
print("METRICS EVALUATION: PASSED")
print("-" * 60)