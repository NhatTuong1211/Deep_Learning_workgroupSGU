"""Các khối building blocks tái sử dụng cho mạng CNN FER2013."""

from typing import Optional
import torch
import torch.nn as nn


def get_activation(name: str = "relu") -> nn.Module:
    """Trả về module hàm kích hoạt theo tên cấu hình."""
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if key == "elu":
        return nn.ELU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Hàm kích hoạt không được hỗ trợ: '{name}'.")


class ConvBlock(nn.Module):
    """Một khối tích chập chuẩn cho ảnh khuôn mặt nhỏ (48x48).

    Cấu trúc:
        [Conv2d -> (BatchNorm2d) -> Activation] x num_convs
        -> MaxPool2d(pool_size)
        -> (Dropout2d(dropout))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
        use_batchnorm: bool = True,
        activation: str = "relu",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_convs = num_convs

        layers = []
        current_in = in_channels
        for conv_idx in range(num_convs):
            # Không cần bias cho Conv2d nếu ngay sau đó là BatchNorm2d
            conv = nn.Conv2d(
                in_channels=current_in,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,
            )
            layers.append((f"conv{conv_idx + 1}", conv))
            if use_batchnorm:
                layers.append((f"bn{conv_idx + 1}", nn.BatchNorm2d(out_channels)))
            layers.append((f"act{conv_idx + 1}", get_activation(activation)))
            current_in = out_channels

        self.conv_layers = nn.Sequential()
        for layer_name, module in layers:
            self.conv_layers.add_module(layer_name, module)

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.dropout: Optional[nn.Module] = (
            nn.Dropout2d(p=dropout) if dropout > 0.0 else None
        )

    def get_last_conv(self) -> nn.Conv2d:
        """Trả về lớp Conv2d cuối cùng bên trong block."""
        last_conv = None
        for module in self.conv_layers.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        assert last_conv is not None
        return last_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x