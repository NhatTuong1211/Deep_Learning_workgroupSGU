import torch
import torch.nn as nn
import torch.optim as optim
from config import get_device

def train_model(train_loader, val_loader, model, epochs=200, lr=0.01, device=None):
    """
    Vòng lặp huấn luyện mô hình MLP:
    - Sử dụng MSELoss và Adam Optimizer.
    - Kiểm định trên tập Validation sau mỗi epoch.
    - Lưu lại Checkpoint (trọng số tốt nhất) khi val_loss đạt giá trị nhỏ nhất.
    """
    if device is None:
        device = get_device()
        
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # --- 1. Huấn luyện (Train) ---
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # --- 2. Kiểm định (Validation) ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_preds = model(val_X)
                v_loss = criterion(val_preds, val_y)
                epoch_val_loss += v_loss.item() * val_X.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # --- 3. Lưu Checkpoint trọng số tốt nhất ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Giữ bản sao trọng số tốt nhất trên CPU
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Khôi phục lại trạng thái mô hình tốt nhất trước khi trả về
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, train_losses, val_losses

print("Đã tạo thành công file train.py!")
