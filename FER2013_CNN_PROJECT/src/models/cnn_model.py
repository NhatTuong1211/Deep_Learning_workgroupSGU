"""Kiến trúc CNN chính bằng PyTorch cho bài toán nhận diện cảm xúc FER2013."""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from .model_blocks import ConvBlock, get_activation
    from .model_config import ModelConfig, get_default_config
except ImportError:
    from model_blocks import ConvBlock, get_activation
    from model_config import ModelConfig, get_default_config


class FER2013CNN(nn.Module):
    """Mạng CNN 4 block phân loại 7 cảm xúc trên ảnh grayscale 48x48.

    Hợp đồng tensor:
        - Input : Float tensor [B, 1, 48, 48] (đã chuẩn hóa).
        - Output: Raw logits [B, 7] theo đúng thứ tự mapping đề bài:
                  0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.
        - Loss  : Dùng trực tiếp với torch.nn.CrossEntropyLoss().
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config: ModelConfig = config if config is not None else get_default_config()

        # Xây dựng phần trích xuất đặc trưng
        self.features = nn.Sequential()
        in_ch = self.config.in_channels
        for idx, (out_ch, n_convs) in enumerate(
            zip(self.config.conv_channels, self.config.convs_per_block), start=1
        ):
            block = ConvBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                num_convs=n_convs,
                kernel_size=self.config.kernel_size,
                stride=self.config.stride,
                padding=self.config.padding,
                pool_size=self.config.pool_size,
                use_batchnorm=self.config.use_batchnorm,
                activation=self.config.activation,
                dropout=self.config.dropout_conv,
            )
            self.features.add_module(f"block{idx}", block)
            in_ch = out_ch

        # Tính tự động kích thước sau khi qua toàn bộ feature extractor
        self.feature_shape: Tuple[int, int, int] = self._infer_feature_shape()
        self.flatten_dim: int = (
            self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]
        )

        # Xây dựng bộ phân loại
        classifier_layers = [
            ("flatten", nn.Flatten(start_dim=1)),
            ("fc1", nn.Linear(self.flatten_dim, self.config.fc_hidden_dim)),
            ("act1", get_activation(self.config.activation)),
        ]
        if self.config.dropout_fc > 0.0:
            classifier_layers.append(("dropout", nn.Dropout(p=self.config.dropout_fc)))
        classifier_layers.append(
            ("fc_out", nn.Linear(self.config.fc_hidden_dim, self.config.num_classes))
        )

        self.classifier = nn.Sequential()
        for name, module in classifier_layers:
            self.classifier.add_module(name, module)

        self._initialize_weights()

    def _infer_feature_shape(self) -> Tuple[int, int, int]:
        """Chạy dummy tensor ở chế độ eval để xác định chính xác (C, H, W) trước Flatten."""
        h, w = self.config.input_size
        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels, h, w)
            out = self.features(dummy)
        self.features.train(was_training)
        return (int(out.shape[1]), int(out.shape[2]), int(out.shape[3]))

    def _initialize_weights(self) -> None:
        """Khởi tạo trọng số Kaiming cho Conv/Linear và chuẩn hóa cho BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_conv_layer_names(self) -> Tuple[str, ...]:
        """Liệt kê tên đầy đủ của tất cả các lớp Conv2d."""
        return tuple(
            name for name, module in self.named_modules() if isinstance(module, nn.Conv2d)
        )

    def get_last_conv_layer(self) -> Tuple[str, nn.Conv2d]:
        """Trả về (tên_module, nn.Conv2d) của lớp convolution cuối cùng."""
        last_name = ""
        last_layer = None
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                last_name = name
                last_layer = module
        assert last_layer is not None
        return last_name, last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, 1, 48, 48] -> logits [B, 7]."""
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def build_model(config: Optional[Union[ModelConfig, Dict[str, Any]]] = None) -> FER2013CNN:
    """Hàm factory khởi tạo FER2013CNN từ ModelConfig hoặc dict."""
    if config is None:
        cfg = get_default_config()
    elif isinstance(config, dict):
        cfg = ModelConfig.from_dict(config)
    elif isinstance(config, ModelConfig):
        cfg = config
    else:
        raise TypeError("config phải là ModelConfig, dict hoặc None.")
    return FER2013CNN(config=cfg)