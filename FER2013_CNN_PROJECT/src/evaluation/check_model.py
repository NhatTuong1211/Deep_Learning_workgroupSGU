import json
from pathlib import Path

import torch

from inputs.model.cnn_model import build_model
from inputs.data.dataset import create_dataloaders


# 1. KHAI BÁO ĐƯỜNG DẪN

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


# 2. CHỌN THIẾT BỊ

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("KHOI PHUC MO HINH CNN")
print("-" * 60)

print("Device:", device)


# 3. ĐỌC FILE CẤU HÌNH

print("\n[1] Doc best_config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config_data = json.load(file)

model_config = config_data["model_config"]

print("Experiment:", config_data["experiment_id"])
print("Model:", model_config["model_name"])
print("Version:", config_data["model_version"])
print("Best epoch:", config_data["best_epoch"])

print(
    "Validation Accuracy:",
    round(config_data["best_val_accuracy"] * 100, 2),
    "%"
)


# 4. KHỞI TẠO KIẾN TRÚC CNN

print("\n[2] Khoi tao FER2013CNN")

model = build_model(model_config)

model = model.to(device)

print("Khoi tao mo hinh thanh cong!")


# 5. NẠP TRỌNG SỐ TỪ CHECKPOINT

print("\n[3] Nap best_model.pth")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

print("Nap trong so thanh cong!")

print("Checkpoint epoch:", checkpoint["epoch"])


# 6. CHUYỂN SANG CHẾ ĐỘ ĐÁNH GIÁ

model.eval()

print("\n[4] Chuyen mo hinh sang eval mode")

print("Model training:", model.training)


# 7. TẠO TEST DATALOADER

print("\n[5] Tao Test DataLoader")

dataloaders = create_dataloaders(
    csv_path=str(CSV_PATH),
    batch_size=64,
    num_workers=0,
    use_aug=False,
    seed=42
)

test_loader = dataloaders["test"]

print("So anh Test:", len(test_loader.dataset))


# 8. THỬ DỰ ĐOÁN MỘT BATCH

print("\n[6] Thu du doan mot batch")

images, labels = next(iter(test_loader))

images = images.to(device)

with torch.no_grad():

    logits = model(images)

print("Input shape:", images.shape)
print("Output shape:", logits.shape)

print("Output dtype:", logits.dtype)

print("So lop dau ra:", logits.shape[1])

print("Logits co gia tri hop le:",
      torch.isfinite(logits).all().item())


# 9. KIỂM TRA KẾT QUẢ

assert logits.shape == (64, 7), (
    "Kich thuoc output khong dung!"
)

assert not model.training, (
    "Mo hinh chua chuyen sang eval mode!"
)

assert torch.isfinite(logits).all(), (
    "Logits chua gia tri NaN hoac Inf!"
)

assert checkpoint["epoch"] == config_data["best_epoch"], (
    "Epoch trong checkpoint khong khop config!"
)

print("-" * 60)
print("MODEL LOADING CHECK: PASSED")
