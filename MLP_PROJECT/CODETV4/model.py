import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    """
    Mạng Perceptron Đa Tầng (MLP) cho bài toán hồi quy hàm Gaussian 2 chiều.
    Phụ trách bởi TV2.
    """
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=1, output_dim=1, activation=nn.ReLU):
        super(MLPRegressor, self).__init__()
        
        layers = []
        # Tầng đầu vào (2 chiều: x1, x2) nối vào tầng ẩn thứ nhất
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        
        # Các tầng ẩn trung gian (phục vụ Thí nghiệm C của TV4)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            
        # Tầng đầu ra (1 chiều: dự đoán giá trị y, không dùng hàm kích hoạt ở output)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

print("Đã tạo thành công file model.py!")
