# Ghi Chú Thiết Kế Kiến Trúc — TV2 (FER2013CNN v1.0)

## 1. Mục tiêu thiết kế
Xây dựng mạng CNN bằng **PyTorch** cho bài toán nhận diện cảm xúc khuôn mặt trên tập dữ liệu **FER2013**:
- Đầu vào: ảnh grayscale 1 kênh kích thước `48 × 48` (`[B, 1, 48, 48]`, `float32`, đã chuẩn hóa `mean=0.5, std=0.5` theo pipeline TV1).
- Đầu ra: vector **7 logits thô** (`[B, 7]`) tương ứng đúng mapping đề bài:
  `0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral`.

## 2. Lý do lựa chọn kiến trúc 4 ConvBlock (`32 → 64 → 128 → 256`)
1. **Phù hợp với độ phân giải nhỏ `48 × 48`**:
   - Với 4 lần `MaxPool2d(2, 2)`, kích thước không gian giảm theo chuỗi: `48×48 → 24×24 → 12×12 → 6×6 → 3×3`.
   - Không dùng 5 lần pooling như VGG16 (vì `48 → 24 → 12 → 6 → 3 → 1` làm mất thông tin không gian và gây overfitting mạnh như đã ghi nhận trong `demo_report.pdf`).
2. **Xếp chồng 2 lớp `Conv2d(3×3, padding=1)` trong mỗi block**:
   - Hai lớp `3×3` liên tiếp cho vùng tiếp nhận (receptive field) tương đương `5×5` nhưng ít tham số hơn và tăng tính phi tuyến nhờ có thêm hàm kích hoạt `ReLU`.
   - `padding=1, stride=1` giữ nguyên kích thước không gian trong từng block, giúp việc giảm chiều hoàn toàn do `MaxPool2d(2, 2)` kiểm soát.
3. **Ổn định huấn luyện và chống Overfitting**:
   - `BatchNorm2d` đặt ngay sau mỗi `Conv2d` (nên tắt `bias=False` ở `Conv2d` để tránh tham số dư thừa) giúp ổn định phân phối đặc trưng và tăng tốc độ hội tụ.
   - `Dropout2d(0.25)` sau mỗi `MaxPool2d` tắt ngẫu nhiên toàn bộ feature map, hiệu quả hơn dropout thông thường trên bản đồ đặc trưng không gian.
   - `Dropout(0.5)` trước lớp `fc_out` giảm overfitting ở phần Classifier.
   - Không dùng `BatchNorm1d` ở tầng FC để đảm bảo mô hình chạy an toàn ngay cả khi `batch_size = 1` ở chế độ `train()`.
4. **Cân bằng số tham số giữa Feature Extractor và Classifier**:
   - Feature Extractor: `1,172,640` tham số (`49.76%`).
   - Classifier (FC): `1,183,751` tham số (`50.24%`).
   - Tổng cộng: **`2,356,391` tham số** (~`8.99 MB` FP32), đủ năng lực biểu diễn cho FER2013 nhưng vẫn đủ gọn để TV3 chạy nhiều thí nghiệm (`E01–E07`) nhanh chóng.
5. **Không gắn `Softmax` ở cuối `forward()`**:
   - Sử dụng trực tiếp `torch.nn.CrossEntropyLoss()` với raw logits giúp tính toán ổn định số học (kết hợp sẵn `LogSoftmax + NLLLoss`).
   - Khi cần xác suất hiển thị (TV4/TV5), áp dụng `torch.softmax(logits, dim=1)` bên ngoài mô hình.

## 3. Khả năng tùy biến cho Thực nghiệm
Mọi tham số kiến trúc nằm trong `ModelConfig` (`model_config.py`):
- Thí nghiệm **E05 (Dropout)**: chỉ cần đổi `dropout_conv` và `dropout_fc`.
- Thí nghiệm **Activation**: hỗ trợ `relu`, `leaky_relu`, `elu`, `gelu`.
- Thí nghiệm **độ sâu/rộng**: đổi `conv_channels`, `convs_per_block`, `fc_hidden_dim`, `use_batchnorm`.
