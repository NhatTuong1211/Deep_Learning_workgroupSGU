
import torch


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Đánh giá mô hình trên tập Validation trong một Epoch.

    Args:
        model: Mô hình CNN.
        val_loader: DataLoader của tập Validation.
        criterion: Hàm tính Loss.
        device: CPU hoặc CUDA.

    Returns:
        val_loss: Validation Loss trung bình.
        val_accuracy: Validation Accuracy.
    """

    # Chuyển mô hình sang chế độ đánh giá
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Không tính gradient trong quá trình Validation
    with torch.no_grad():

        for images, labels in val_loader:

            # Đưa dữ liệu lên thiết bị đang sử dụng
            images = images.to(device)
            labels = labels.to(device)

            # Forward propagation
            logits = model(images)

            # Tính Cross Entropy Loss
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            # Cộng dồn Loss theo số lượng mẫu
            total_loss += loss.item() * batch_size

            # Dự đoán nhãn
            predictions = torch.argmax(logits, dim=1)

            # Đếm số dự đoán chính xác
            total_correct += (predictions == labels).sum().item()

            total_samples += batch_size

    # Tính kết quả trung bình
    if total_samples == 0:
        raise ValueError("Validation DataLoader không có dữ liệu.")

    val_loss = total_loss / total_samples
    val_accuracy = total_correct / total_samples

    return val_loss, val_accuracy
