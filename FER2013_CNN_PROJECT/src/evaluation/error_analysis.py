from pathlib import Path
import numpy as np
import json
import csv


# 1. PATHS AND CONFIGURATION
ROOT_DIR = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    ROOT_DIR / "results" / "evaluation" / "predictions.npz"
)

OUTPUT_DIR = ROOT_DIR / "results" / "evaluation" / "error_analysis"
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

NUM_CLASSES = len(CLASS_NAMES)


# 2. LOAD PREDICTIONS
print("TV4 - ERROR ANALYSIS")
print("-" * 60)

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find predictions file: {PREDICTIONS_PATH}"
    )

data = np.load(PREDICTIONS_PATH)

y_true = data["y_true"]
y_pred = data["y_pred"]

if len(y_true) != len(y_pred):
    raise ValueError("y_true and y_pred have different lengths.")

if len(y_true) == 0:
    raise ValueError("Prediction data is empty.")

if not np.isin(y_true, range(NUM_CLASSES)).all():
    raise ValueError("y_true contains invalid labels.")

if not np.isin(y_pred, range(NUM_CLASSES)).all():
    raise ValueError("y_pred contains invalid labels.")

total_samples = len(y_true)

print(f"Total test samples: {total_samples}")


# 3. CALCULATE ERRORS BY CLASS
class_results = []

print("\nERRORS BY CLASS")
print("-" * 75)
print(
    f"{'Emotion':<12}"
    f"{'Total':>10}"
    f"{'Correct':>10}"
    f"{'Errors':>10}"
    f"{'Error Rate':>15}"
)

for class_id, class_name in enumerate(CLASS_NAMES):

    class_mask = (y_true == class_id)

    total = int(np.sum(class_mask))
    correct = int(np.sum(
        (y_true == class_id) & (y_pred == class_id)
    ))
    errors = total - correct

    error_rate = errors / total if total > 0 else 0.0

    class_results.append({
        "emotion": class_name,
        "total": total,
        "correct": correct,
        "errors": errors,
        "error_rate": error_rate
    })

    print(
        f"{class_name:<12}"
        f"{total:>10}"
        f"{correct:>10}"
        f"{errors:>10}"
        f"{error_rate * 100:>14.2f}%"
    )


# 4. CALCULATE CONFUSION PAIRS
confusion_pairs = []

for true_id in range(NUM_CLASSES):

    for pred_id in range(NUM_CLASSES):

        if true_id == pred_id:
            continue

        count = int(np.sum(
            (y_true == true_id) &
            (y_pred == pred_id)
        ))

        if count > 0:
            confusion_pairs.append({
                "true_emotion": CLASS_NAMES[true_id],
                "predicted_emotion": CLASS_NAMES[pred_id],
                "count": count
            })

confusion_pairs.sort(
    key=lambda item: item["count"],
    reverse=True
)

print("\nTOP 10 CONFUSION PAIRS")
print("-" * 65)
print(
    f"{'True Emotion':<15}"
    f"{'Predicted':<15}"
    f"{'Count':>10}"
)

for item in confusion_pairs[:10]:

    print(
        f"{item['true_emotion']:<15}"
        f"{item['predicted_emotion']:<15}"
        f"{item['count']:>10}"
    )


# 5. SUMMARY
total_correct = int(np.sum(y_true == y_pred))
total_errors = total_samples - total_correct

accuracy = total_correct / total_samples
error_rate = total_errors / total_samples

print("\nOVERALL ERROR SUMMARY")
print("-" * 60)
print(f"Total samples : {total_samples}")
print(f"Correct       : {total_correct}")
print(f"Incorrect     : {total_errors}")
print(f"Accuracy      : {accuracy * 100:.2f}%")
print(f"Error rate    : {error_rate * 100:.2f}%")


# 6. SAVE CLASS ERROR ANALYSIS
class_csv_path = OUTPUT_DIR / "class_error_analysis.csv"

with open(
    class_csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "emotion",
            "total",
            "correct",
            "errors",
            "error_rate"
        ]
    )

    writer.writeheader()
    writer.writerows(class_results)


# 7. SAVE CONFUSION PAIRS
pairs_csv_path = OUTPUT_DIR / "confusion_pairs.csv"

with open(
    pairs_csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "true_emotion",
            "predicted_emotion",
            "count"
        ]
    )

    writer.writeheader()
    writer.writerows(confusion_pairs)


# 8. SAVE JSON SUMMARY
summary = {
    "total_samples": total_samples,
    "correct_predictions": total_correct,
    "incorrect_predictions": total_errors,
    "accuracy": accuracy,
    "error_rate": error_rate,
    "class_error_analysis": class_results,
    "top_10_confusion_pairs": confusion_pairs[:10]
}

json_path = OUTPUT_DIR / "error_analysis_summary.json"

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# 9. FINAL CHECK
assert sum(item["total"] for item in class_results) == total_samples
assert sum(item["errors"] for item in class_results) == total_errors

print("\nSaved results:")
print(f"- {class_csv_path}")
print(f"- {pairs_csv_path}")
print(f"- {json_path}")

print("\nERROR ANALYSIS: PASSED")