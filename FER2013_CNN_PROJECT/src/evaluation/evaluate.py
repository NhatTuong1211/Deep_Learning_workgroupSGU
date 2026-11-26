import json
from pathlib import Path

import numpy as np
import torch

from inputs.model.cnn_model import build_model
from inputs.data.dataset import create_dataloaders
from inputs.data.label_mapping import CLASS_NAMES


# 1. KHAI BAO DUONG DAN

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    BASE_DIR / "inputs" / "checkpoints" / "best_config.json"
)

CHECKPOINT_PATH = (
    BASE_DIR / "inputs" / "checkpoints" / "best_model.pth"
)

CSV_PATH = (
    BASE_DIR / "inputs" / "data" / "fer2013.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. CHON THIET BI

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TV4 - EVALUATION")
print("=" * 60)

print("Device:", device)


# 3. DOC CAU HINH

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config_data = json.load(file)

model_config = config_data["model_config"]

print("\nExperiment:", config_data["experiment_id"])
print("Model:", model_config["model_name"])


# 4. KHOI TAO MO HINH

model = build_model(model_config)

model = model.to(device)


# 5. NAP CHECKPOINT

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.eval()

print("\nDa nap checkpoint thanh cong!")
print("Checkpoint epoch:", checkpoint["epoch"])


# 6. TAO TEST DATALOADER

dataloaders = create_dataloaders(
    csv_path=str(CSV_PATH),
    batch_size=64,
    num_workers=0,
    use_aug=False,
    seed=42
)

test_loader = dataloaders["test"]

print("\nSo luong anh Test:", len(test_loader.dataset))


# 7. KHOI TAO CAC DANH SACH

all_y_true = []
all_y_pred = []
all_y_prob = []

total_correct = 0
total_samples = 0


# 8. DU DOAN TREN TOAN BO TAP TEST

print("\nDang danh gia mo hinh...")

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        # Dua du lieu len CPU hoac GPU
        images = images.to(device)
        labels = labels.to(device)

        # Mo hinh du doan
        logits = model(images)

        # Chuyen logits thanh xac suat
        probabilities = torch.softmax(logits, dim=1)

        # Lay lop co xac suat cao nhat
        predictions = torch.argmax(logits, dim=1)

        # Luu nhan thuc te
        all_y_true.extend(
            labels.cpu().numpy()
        )

        # Luu nhan du doan
        all_y_pred.extend(
            predictions.cpu().numpy()
        )

        # Luu xac suat cua 7 lop
        all_y_prob.extend(
            probabilities.cpu().numpy()
        )

        # Dem so du doan dung
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        if (batch_index + 1) % 10 == 0:
            print(
                f"Da xu ly: {total_samples}/"
                f"{len(test_loader.dataset)} anh"
            )


# 9. CHUYEN DANH SACH THANH NUMPY ARRAY

y_true = np.array(all_y_true, dtype=np.int64)

y_pred = np.array(all_y_pred, dtype=np.int64)

y_prob = np.array(all_y_prob, dtype=np.float32)


# 10. TINH ACCURACY BAN DAU

accuracy = total_correct / total_samples

print("\n" + "=" * 60)
print("KET QUA DU DOAN")
print("=" * 60)

print("Tong so anh:", total_samples)
print("So du doan dung:", total_correct)
print("So du doan sai:", total_samples - total_correct)

print("Accuracy:", round(accuracy * 100, 2), "%")


# 11. KIEM TRA KICH THUOC CAC MANG

print("\nKich thuoc du lieu:")

print("y_true:", y_true.shape)
print("y_pred:", y_pred.shape)
print("y_prob:", y_prob.shape)


# 12. LUU KET QUA DU DOAN

output_path = OUTPUT_DIR / "predictions.npz"

np.savez_compressed(
    output_path,
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    class_names=np.array(CLASS_NAMES)
)

print("\nDa luu ket qua:", output_path)


# 13. KIEM TRA TINH HOP LE

assert len(y_true) == 3589
assert len(y_pred) == 3589

assert y_prob.shape == (3589, 7)

assert np.allclose(
    y_prob.sum(axis=1),
    1.0,
    atol=1e-5
)

assert np.all((y_pred >= 0) & (y_pred < 7))

print("\n" + "-" * 60)
print("EVALUATION PREDICTION: PASSED")
print("-" * 60)
