from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


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

print("TV4 - CLASSIFICATION REPORT")
print("-" * 60)
print(f"Total samples: {len(y_true)}")


# 3. GENERATE REPORT
report = classification_report(
    y_true,
    y_pred,
    labels=range(7),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# 4. CONVERT TO DATAFRAME
rows = []

for class_name in CLASS_NAMES:
    values = report[class_name]

    rows.append({
        "Emotion": class_name,
        "Precision": values["precision"],
        "Recall": values["recall"],
        "F1-score": values["f1-score"],
        "Support": int(values["support"])
    })

for avg_name in ["macro avg", "weighted avg"]:
    values = report[avg_name]

    rows.append({
        "Emotion": avg_name.title(),
        "Precision": values["precision"],
        "Recall": values["recall"],
        "F1-score": values["f1-score"],
        "Support": int(values["support"])
    })

report_df = pd.DataFrame(rows)


# 5. SAVE CSV
output_path = OUTPUT_DIR / "classification_report.csv"

report_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nClassification Report:")
print(report_df.to_string(index=False))

print(f"\nSaved: {output_path}")
print("\nCLASSIFICATION REPORT: PASSED")