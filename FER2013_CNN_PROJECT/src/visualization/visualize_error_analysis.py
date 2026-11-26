from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. PATHS
ROOT_DIR = Path(__file__).resolve().parents[2]

ERROR_DIR = ROOT_DIR / "results" / "evaluation" / "error_analysis"

CLASS_CSV = ERROR_DIR / "class_error_analysis.csv"
PAIRS_CSV = ERROR_DIR / "confusion_pairs.csv"

OUTPUT_DIR = ERROR_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. LOAD DATA
class_df = pd.read_csv(CLASS_CSV)
pairs_df = pd.read_csv(PAIRS_CSV)

print("TV4 - VISUALIZE ERROR ANALYSIS")
print("-" * 60)

# 3. VISUALIZE ERROR RATE BY CLASS
class_df = class_df.sort_values(
    by="error_rate",
    ascending=False
)

plt.figure(figsize=(10, 6))

plt.bar(
    class_df["emotion"],
    class_df["error_rate"] * 100
)

plt.title("Error Rate by Emotion Class")
plt.xlabel("Emotion")
plt.ylabel("Error Rate (%)")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_rate_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 4. VISUALIZE TOP 10 CONFUSION PAIRS
top_pairs = pairs_df.head(10).copy()

top_pairs["pair"] = (
    top_pairs["true_emotion"]
    + " -> "
    + top_pairs["predicted_emotion"]
)

top_pairs = top_pairs.sort_values(
    by="count",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_pairs["pair"],
    top_pairs["count"]
)

plt.title("Top 10 Emotion Confusion Pairs")
plt.xlabel("Number of Misclassifications")
plt.ylabel("True Emotion -> Predicted Emotion")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top10_confusion_pairs.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 5. FINAL CHECK
print("\nSaved visualizations:")
print(OUTPUT_DIR / "error_rate_by_class.png")
print(OUTPUT_DIR / "top10_confusion_pairs.png")

print("\nVISUALIZE ERROR ANALYSIS: PASSED")
