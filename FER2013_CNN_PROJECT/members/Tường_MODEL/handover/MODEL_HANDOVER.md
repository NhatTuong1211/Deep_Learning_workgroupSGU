# Tài Liệu Bàn Giao Mô Hình — MODEL

## 1. Tổng quan bàn giao
Module mô hình CNN (`FER2013CNN v1.0`) viết hoàn toàn bằng **PyTorch** cho bài toán nhận diện 7 cảm xúc trên FER2013 (`48×48` grayscale).

- **Tổng số tham số**: `2,356,391` (`100%` trainable; ~`8.99 MB` FP32)
  - Feature Extractor: `1,172,640` (`49.76%`)
  - Classifier (FC): `1,183,751` (`50.24%`)
- **Hợp đồng Tensor**:
  - Input: `torch.float32`, shape `[B, 1, 48, 48]` (khớp đầu ra của TV1: chuẩn hóa `mean=0.5, std=0.5`).
  - Label: `torch.long`, shape `[B]`, giá trị nguyên `0..6`.
  - Output: `torch.float32`, shape `[B, 7]` (**raw logits**, **KHÔNG** có `Softmax`).
  - Loss: `torch.nn.CrossEntropyLoss()`.
  - Prediction: `logits.argmax(dim=1)` (dùng `torch.softmax(logits, dim=1)` chỉ khi cần xác suất hiển thị).

---

## 2. Cấu trúc file bàn giao trong `Tường_MODEL/`

| Đường dẫn trong `Tường_MODEL/` | Đường dẫn đích khi TV5 ghép vào repo chung | Mô tả |
|---|---|---|
| `code/model_config.py` | `src/models/model_config.py` | `ModelConfig`, `get_default_config()` |
| `code/model_blocks.py` | `src/models/model_blocks.py` | `ConvBlock`, `get_activation()` |
| `code/cnn_model.py` | `src/models/cnn_model.py` | `FER2013CNN`, `build_model()` |
| `code/model_summary.py` | `src/models/model_summary.py` | In summary, đếm tham số, lưu file outputs |
| `tests/test_model.py` | `tests/test_model.py` | 10 unit test & smoke test tích hợp TV1 |
| `design/model_architecture.png` | `results/model/model_architecture.png` | Sơ đồ kiến trúc khớp 100% code |
| `design/tensor_shape_table.xlsx` | `members/TV2_MODEL/design/tensor_shape_table.xlsx` | Bảng tensor shape từng layer (Excel) |
| `design/tensor_shape_table.csv` | `members/TV2_MODEL/design/tensor_shape_table.csv` | Bảng tensor shape từng layer (CSV) |
| `design/architecture_notes.md` | `members/TV2_MODEL/design/architecture_notes.md` | Giải thích lý do thiết kế kiến trúc |
| `outputs/model_summary.txt` | `results/model/model_summary.txt` | Bảng summary đầy đủ từ code thật |
| `outputs/parameter_count.txt` | `members/TV2_MODEL/outputs/parameter_count.txt` | Thống kê số tham số & lớp conv cuối |
| `handover/MODEL_HANDOVER.md` | `members/TV2_MODEL/handover/MODEL_HANDOVER.md` | Tài liệu bàn giao này |
| `handover/test_results.txt` | `members/TV2_MODEL/handover/test_results.txt` | Log chạy kiểm thử (`10/10 OK`) |

---

## 3. Hướng dẫn sử dụng cho TV3 (Training) và TV5 (Integration)

### Khởi tạo mô hình mặc định hoặc tùy biến cho thí nghiệm (ví dụ E05 Dropout)
```python
import torch
import torch.nn as nn
from src.models.cnn_model import FER2013CNN, build_model
from src.models.model_config import ModelConfig, get_default_config

# 1) Khởi tạo cấu hình mặc định (Baseline E01)
config = get_default_config()
model = build_model(config)

# 2) Khởi tạo biến thể thí nghiệm (ví dụ E05: thay đổi dropout)
cfg_e05 = get_default_config(dropout_conv=0.2, dropout_fc=0.4)
model_e05 = build_model(cfg_e05)

# 3) Forward + Loss + Backward với batch từ TV1 DataLoader
criterion = nn.CrossEntropyLoss()
logits = model(images)           # images: [B, 1, 48, 48] -> logits: [B, 7]
loss = criterion(logits, labels) # labels: [B] torch.long (0..6)
loss.backward()
```

### Quy ước lưu & nạp Checkpoint (tương thích TV3 → TV4 → TV5)
```python
# Lưu checkpoint (TV3)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "best_epoch": best_epoch,
        "val_accuracy": best_val_acc,
        "val_loss": best_val_loss,
        "label_mapping": {
            0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
            4: "Sad", 5: "Surprise", 6: "Neutral",
        },
    },
    "checkpoints/best_model.pth",
)

# Nạp checkpoint (TV4 / TV5)
ckpt = torch.load("checkpoints/best_model.pth", map_location="cpu", weights_only=False)
model = build_model(ckpt["model_config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

---

## 4. Thông tin trích xuất Convolution Filter cho TV4 (Evaluation & Visualization)

Đề bài yêu cầu **"Visualize the final convolution filters"**. TV4 có thể truy cập trực tiếp bằng helper method có sẵn trong `FER2013CNN`:

```python
last_name, last_conv = model.get_last_conv_layer()
# last_name == "features.block4.conv_layers.conv2"
# last_conv.weight.shape == torch.Size([256, 256, 3, 3])

# Danh sách toàn bộ 8 lớp Conv2d trong mô hình:
all_conv_names = model.get_conv_layer_names()
# (
#   'features.block1.conv_layers.conv1',  # shape [32, 1, 3, 3]  -> dễ nhìn trực tiếp vì 1 kênh vào
#   'features.block1.conv_layers.conv2',  # shape [32, 32, 3, 3]
#   'features.block2.conv_layers.conv1',  # shape [64, 32, 3, 3]
#   'features.block2.conv_layers.conv2',  # shape [64, 64, 3, 3]
#   'features.block3.conv_layers.conv1',  # shape [128, 64, 3, 3]
#   'features.block3.conv_layers.conv2',  # shape [128, 128, 3, 3]
#   'features.block4.conv_layers.conv1',  # shape [256, 128, 3, 3]
#   'features.block4.conv_layers.conv2',  # shape [256, 256, 3, 3] -> lớp Conv cuối cùng
# )
```

---

## 5. Lệnh chạy kiểm thử & sinh lại tài liệu
Từ thư mục gốc dự án:
```bash
# Chạy toàn bộ 10 unit test & smoke test tích hợp
python3 "Tường_MODEL/tests/test_model.py"

# In lại model summary & cập nhật outputs/
python3 "Tường_MODEL/code/model_summary.py"

# Sinh lại sơ đồ kiến trúc và bảng tensor shape (CSV + XLSX)
python3 "Tường_MODEL/design/generate_design_artifacts.py"
```

## 6. Vấn đề tồn tại / Lưu ý tích hợp
- **Môi trường cục bộ**: Code chỉ phụ thuộc `torch` (và `matplotlib` khi vẽ sơ đồ). Tuy nhiên, `tv1_data_handover/src/data/preprocessing.py` cần thư viện **`torchvision`** — TV5 cần ghi rõ `torch` và `torchvision` trong `requirements.txt` để tích hợp đầy đủ DataLoader của TV1.
- **Mapping nhãn**: Đã kiểm thử khớp 100% với `tv1_data_handover/src/data/label_mapping.py` (`0..6`: `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`).
- **Phần việc còn lại của TV2 (Mốc M6)**: Viết báo cáo Chương 3 (`chapter_3_model.docx`) và soạn Slide 9–13 (`TV2_slides_09_13.pptx`) sau khi TV5 bàn giao template slide 16:9.