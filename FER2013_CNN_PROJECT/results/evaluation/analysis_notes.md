# TV4 – EVALUATION ANALYSIS NOTES

## 1. Thông tin đánh giá

- Experiment: E01
- Model: FER2013CNN
- Model version: v1.0
- Best checkpoint: `best_model.pth`
- Best epoch: 28
- Test samples: 3,589
- Input image: Grayscale, 48 × 48 pixels
- Number of classes: 7

Việc đánh giá được thực hiện trên tập PrivateTest với checkpoint đã được lựa chọn từ validation.

## 2. Overall Evaluation Results

| Metric | Value |
|---|---:|
| Accuracy | 63.95% |
| Precision (Macro) | 59.50% |
| Recall (Macro) | 56.73% |
| F1-score (Macro) | 56.74% |
| Precision (Weighted) | 63.37% |
| Recall (Weighted) | 63.95% |
| F1-score (Weighted) | 62.59% |

Tổng số mẫu kiểm thử là 3,589, trong đó mô hình dự đoán đúng 2,295 mẫu và dự đoán sai 1,294 mẫu.

### Evaluation Averaging Methods

- **Macro average**: Computes the unweighted mean of the metric values across the seven emotion classes.
- **Weighted average**: Computes the mean of the metric values weighted by the number of true samples (support) in each class.

## 3. Error Rate by Emotion Class

| Emotion | Total | Correct | Errors | Error Rate |
|---|---:|---:|---:|---:|
| Angry | 491 | 270 | 221 | 45.01% |
| Disgust | 55 | 13 | 42 | 76.36% |
| Fear | 528 | 137 | 391 | 74.05% |
| Happy | 879 | 779 | 100 | 11.38% |
| Sad | 594 | 323 | 271 | 45.62% |
| Surprise | 416 | 323 | 93 | 22.36% |
| Neutral | 626 | 450 | 176 | 28.12% |

Disgust có tỷ lệ lỗi cao nhất (76.36%), tiếp theo là Fear (74.05%). Ngược lại, Happy có tỷ lệ lỗi thấp nhất (11.38%).

Lưu ý: Disgust chỉ có 55 mẫu trong tập kiểm thử. Số lượng mẫu nhỏ có thể khiến kết quả của lớp này biến động đáng kể.

## 4. Top 10 Confusion Pairs

| True Emotion | Predicted Emotion | Count |
|---|---|---:|
| Sad | Neutral | 147 |
| Fear | Sad | 107 |
| Fear | Angry | 95 |
| Fear | Neutral | 84 |
| Neutral | Sad | 81 |
| Angry | Neutral | 79 |
| Fear | Surprise | 74 |
| Angry | Sad | 72 |
| Sad | Angry | 58 |
| Neutral | Happy | 47 |

Cặp Sad → Neutral có số lượng nhầm lẫn cao nhất với 147 mẫu. Fear xuất hiện trong 4/10 cặp nhầm lẫn phổ biến nhất.

### Confusion Matrix Interpretation

The confusion matrix uses true labels as rows and predicted labels as columns. Diagonal entries represent correct predictions, while off-diagonal entries represent misclassifications.

## 5. Error Analysis

### 5.1. Observations

- Mô hình nhận diện Happy tương đối tốt, với tỷ lệ lỗi 11.38%.
- Disgust và Fear là hai lớp có tỷ lệ lỗi cao nhất.
- Các nhầm lẫn nổi bật xuất hiện giữa Sad và Neutral, Fear và các lớp Sad, Angry, Neutral.
- Fear có số lượng dự đoán sai lớn nhất: 391 mẫu.

### 5.2. Possible Explanations (Hypotheses)

Các nguyên nhân dưới đây là giả thuyết cần được kiểm tra bằng cách quan sát trực tiếp ảnh, không phải kết luận đã được chứng minh:

- Các biểu cảm Sad, Angry và Neutral có thể có đặc điểm khuôn mặt tương đồng ở một số ảnh.
- Fear và Surprise có thể chia sẻ một số biểu hiện như mắt mở rộng hoặc miệng mở.
- Độ phân giải ảnh 48 × 48 có thể làm mất một số chi tiết biểu cảm.
- Sự mất cân bằng số lượng mẫu giữa các lớp có thể ảnh hưởng đến khả năng học đặc trưng của mô hình.

## 6. Evaluation Limitations

- Dữ liệu ảnh grayscale có độ phân giải 48 × 48 nên lượng thông tin hình ảnh bị giới hạn.
- Số lượng mẫu giữa các lớp không đồng đều, đặc biệt Disgust có ít mẫu.
- Nhãn cảm xúc khuôn mặt có tính chủ quan; một số biểu cảm có thể khó phân biệt rõ ràng.
- Các giả thuyết về nguyên nhân nhầm lẫn cần được xác minh thêm bằng việc kiểm tra ảnh dự đoán sai.

## 7. Conclusion

Mô hình FER2013CNN đạt accuracy 63.95% trên tập Test gồm 3,589 ảnh. Mô hình nhận diện tốt nhất lớp Happy, trong khi gặp khó khăn với Disgust và Fear. Các lỗi nổi bật chủ yếu liên quan đến sự nhầm lẫn giữa những lớp cảm xúc có đặc điểm biểu cảm gần nhau, đặc biệt là Sad và Neutral.

Kết quả này cung cấp cơ sở để đánh giá ưu điểm, hạn chế và định hướng cải thiện mô hình trong các bước phát triển tiếp theo.

## 8. Related Result Files

- `test_metrics.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `predictions.npz`
- `error_analysis/class_error_analysis.csv`
- `error_analysis/confusion_pairs.csv`
- `error_analysis/error_analysis_summary.json`
- `error_analysis/visualizations/error_rate_by_class.png`
- `error_analysis/visualizations/top10_confusion_pairs.png`
- `prediction_examples/correct_predictions.png`
- `prediction_examples/incorrect_predictions.png`
- `convolution_filters/`

### Additional Summary Visualizations

- `prediction_examples.png`: Combined visualization of correct and incorrect prediction examples, including ground-truth labels, predicted labels, and confidence scores.
- `conv_filters.png`: Combined visualization of convolutional filters extracted from the trained FER2013CNN model.