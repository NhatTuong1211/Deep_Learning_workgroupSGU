"""Cấu hình kiến trúc mô hình CNN cho bài toán nhận diện cảm xúc FER2013."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


VALID_ACTIVATIONS = ("relu", "leaky_relu", "elu", "gelu")


@dataclass
class ModelConfig:
    """Cấu hình siêu tham số kiến trúc cho FER2013CNN.

    Mọi tham số đều có thể điều chỉnh từ config (JSON/dict) để chạy các
    thí nghiệm có kiểm soát (ví dụ E05 Dropout, activation) mà không cần sửa
    code nội bộ của mô hình.
    """

    model_name: str = "FER2013CNN"
    model_version: str = "v1.0"
    in_channels: int = 1
    input_size: Tuple[int, int] = (48, 48)
    num_classes: int = 7

    # Kiến trúc 4 ConvBlock: 48 -> 24 -> 12 -> 6 -> 3
    conv_channels: Tuple[int, ...] = (32, 64, 128, 256)
    convs_per_block: Tuple[int, ...] = (2, 2, 2, 2)
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2
    use_batchnorm: bool = True
    activation: str = "relu"

    # Regularization
    dropout_conv: float = 0.25
    dropout_fc: float = 0.5

    # Classifier (Fully Connected)
    fc_hidden_dim: int = 512

    def __post_init__(self) -> None:
        self.input_size = tuple(int(v) for v in self.input_size)
        self.conv_channels = tuple(int(c) for c in self.conv_channels)
        self.convs_per_block = tuple(int(n) for n in self.convs_per_block)
        self.activation = str(self.activation).lower()

        if self.in_channels <= 0:
            raise ValueError(f"in_channels phải > 0, nhận {self.in_channels}.")
        if len(self.input_size) != 2 or any(s <= 0 for s in self.input_size):
            raise ValueError(f"input_size phải là (H, W) dương, nhận {self.input_size}.")
        if self.num_classes <= 1:
            raise ValueError(f"num_classes phải > 1, nhận {self.num_classes}.")
        if len(self.conv_channels) == 0:
            raise ValueError("conv_channels không được rỗng.")
        if len(self.conv_channels) != len(self.convs_per_block):
            raise ValueError(
                "Độ dài conv_channels và convs_per_block phải bằng nhau: "
                f"{len(self.conv_channels)} vs {len(self.convs_per_block)}."
            )
        if any(c <= 0 for c in self.conv_channels):
            raise ValueError("Mọi số kênh trong conv_channels phải > 0.")
        if any(n <= 0 for n in self.convs_per_block):
            raise ValueError("Mọi giá trị trong convs_per_block phải >= 1.")
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"activation='{self.activation}' không hợp lệ. Chọn trong {VALID_ACTIVATIONS}."
            )
        if not (0.0 <= self.dropout_conv < 1.0):
            raise ValueError(f"dropout_conv phải nằm trong [0, 1), nhận {self.dropout_conv}.")
        if not (0.0 <= self.dropout_fc < 1.0):
            raise ValueError(f"dropout_fc phải nằm trong [0, 1), nhận {self.dropout_fc}.")
        if self.fc_hidden_dim <= 0:
            raise ValueError(f"fc_hidden_dim phải > 0, nhận {self.fc_hidden_dim}.")

        # Kiểm tra kích thước ảnh không bị giảm về 0 sau các lớp MaxPool
        h, w = self.input_size
        divisor = self.pool_size ** len(self.conv_channels)
        if h // divisor < 1 or w // divisor < 1:
            raise ValueError(
                f"Ảnh {self.input_size} quá nhỏ cho {len(self.conv_channels)} tầng "
                f"MaxPool({self.pool_size})."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển cấu hình sang dict chuẩn JSON để TV3 lưu vào best_config.json."""
        data = asdict(self)
        data["input_size"] = list(self.input_size)
        data["conv_channels"] = list(self.conv_channels)
        data["convs_per_block"] = list(self.convs_per_block)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Khởi tạo ModelConfig từ dict (ví dụ đọc từ best_config.json)."""
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def get_default_config(**overrides: Any) -> ModelConfig:
    """Trả về ModelConfig mặc định và cho phép ghi đè nhanh từng tham số."""
    base = ModelConfig().to_dict()
    base.update(overrides)
    return ModelConfig.from_dict(base)