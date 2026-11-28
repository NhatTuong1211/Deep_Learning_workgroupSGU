# FER2013 Facial Emotion Recognition – CNN (PyTorch)

Dự án nhận diện cảm xúc khuôn mặt bằng mạng CNN trên bộ dữ liệu FER2013.

## Thành viên & Phân công

| Thành viên | Vai trò | Thư mục |
|------------|---------|---------|
| TV1 | Dữ liệu, EDA, tiền xử lý, DataLoader | `members/TV1_DATA/`, `src/data/` |
| TV2 | Thiết kế và cài đặt CNN | `members/TV2_MODEL/`, `src/models/` |
| TV3 | Huấn luyện, validation, thực nghiệm | `members/TV3_TRAINING/`, `src/training/` |
| TV4 | Đánh giá, phân tích lỗi, trực quan hóa | `members/TV4_EVALUATION/`, `src/evaluation/` |
| TV5 | Điều phối, tích hợp, kiểm thử, tổng hợp | `members/TV5_INTEGRATION/`, `main.py`, `src/inference/` |

## Label Mapping (bắt buộc đúng thứ tự)

| Index | Nhãn | Tiếng Việt |
|-------|------|-----------|
| 0 | Angry | Tức giận |
| 1 | Disgust | Ghê tởm |
| 2 | Fear | Sợ hãi |
| 3 | Happy | Vui vẻ |
| 4 | Sad | Buồn |
| 5 | Surprise | Ngạc nhiên |
| 6 | Neutral | Trung tính |

## Cấu trúc thư mục

```
FER2013_CNN_PROJECT/
├── README.md
├── requirements.txt
├── main.py                        ← Entry point (TV5)
├── .gitignore
├── data/
│   ├── raw/FER2013/               ← Đặt fer2013.csv hoặc thư mục ảnh ở đây
│   ├── processed/
│   ├── samples/
│   └── README_data.md
├── src/
│   ├── data/                      ← TV1
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── label_mapping.py
│   │   └── eda.py
│   ├── models/                    ← TV2
│   │   ├── cnn_model.py
│   │   ├── model_blocks.py
│   │   ├── model_config.py
│   │   └── model_summary.py
│   ├── training/                  ← TV3
│   │   ├── train.py
│   │   ├── validate.py
│   │   ├── config.py
│   │   ├── experiments.py
│   │   └── checkpoint.py
│   ├── evaluation/                ← TV4
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   └── classification_report.py
│   ├── visualization/             ← TV4
│   │   ├── plot_history.py
│   │   ├── visualize_filters.py
│   │   └── visualize_predictions.py
│   └── inference/                 ← TV5
│       └── predict.py
├── checkpoints/
│   ├── best_model.pth
│   └── best_config.json
├── experiments/
│   ├── experiment_log.csv
│   └── experiment_summary.csv
├── results/
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_inference.py          ← TV5
├── members/
│   └── TV5_INTEGRATION/
└── docs/
```

## Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/macOS

# Cài thư viện
pip install -r requirements.txt
```

## Đặt dữ liệu

> **Quan trọng**: Dataset FER2013 không được commit lên Git.

1. Tải FER2013 từ [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) hoặc nguồn nhóm đã thống nhất.
2. Đặt file vào:
   - Dạng CSV: `data/raw/FER2013/fer2013.csv`
   - Dạng thư mục ảnh: `data/raw/FER2013/train/`, `data/raw/FER2013/test/`
3. Xem thêm: [`data/README_data.md`](data/README_data.md)

## Cách chạy

### Kiểm tra pipeline nhanh (smoke test)
```bash
python main.py --mode smoke_test
```

### Huấn luyện
```bash
python main.py --mode train --config checkpoints/best_config.json
```

### Đánh giá trên test set
```bash
python main.py --mode eval \
  --checkpoint checkpoints/best_model.pth \
  --config checkpoints/best_config.json
```

### Dự đoán một ảnh
```bash
python main.py --mode predict \
  --image path/to/face.jpg \
  --checkpoint checkpoints/best_model.pth \
  --config checkpoints/best_config.json
```

Hoặc dùng trực tiếp script inference:
```bash
python src/inference/predict.py \
  --image path/to/face.jpg \
  --checkpoint checkpoints/best_model.pth \
  --config checkpoints/best_config.json
```

### Chạy tests
```bash
# Toàn bộ tests
python -m pytest tests/ -v

# Chỉ test inference (TV5)
python -m pytest tests/test_inference.py -v
```

## Yêu cầu kỹ thuật

- **Đầu vào tensor**: `[batch_size, 1, 48, 48]` – float, đã chuẩn hóa
- **Đầu ra logits**: `[batch_size, 7]` – không qua Softmax
- **Loss**: `torch.nn.CrossEntropyLoss()` với logits thô
- **Dự đoán**: `logits.argmax(dim=1)`

## Quy tắc Git

- Mỗi TV làm trên nhánh riêng: `feature/data`, `feature/model`, `feature/training`, `feature/evaluation`, `feature/integration`
- TV5 kiểm tra tích hợp trước khi merge vào `develop`/`main`
- Không commit: dataset lớn, checkpoint `.pth`, file cache, API key

## Liên hệ / Báo lỗi

Tạo issue trên repository hoặc liên hệ TV5 để điều phối.
