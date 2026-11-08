import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import get_device

def evaluate_model(model, test_loader, device=None):
    """
    Đánh giá mô hình trên tập Test độc lập.
    Tính toán các chỉ số: MSE, RMSE, MAE.
    """
    if device is None:
        device = get_device()
        
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch_y.numpy())
            
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Tính các chỉ số đánh giá
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }
    
    return metrics, preds

print("Đã tạo thành công file evaluate.py!")
