"""Script in model summary, bảng tensor shape và thống kê số tham số cho FER2013CNN."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

try:
    from .cnn_model import FER2013CNN, build_model
    from .model_config import ModelConfig, get_default_config
except ImportError:
    from cnn_model import FER2013CNN, build_model
    from model_config import ModelConfig, get_default_config


LEAF_LAYER_TYPES = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.GELU,
    nn.MaxPool2d,
    nn.Dropout2d,
    nn.Dropout,
    nn.Flatten,
    nn.Linear,
)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Đếm tổng số tham số, trainable và non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }


def _describe_layer(module: nn.Module) -> str:
    """Mô tả ngắn tham số kiến trúc của một layer (kernel/stride/padding/p)."""
    if isinstance(module, nn.Conv2d):
        return (
            f"in={module.in_channels}, out={module.out_channels}, "
            f"k={module.kernel_size}, s={module.stride}, p={module.padding}, "
            f"bias={module.bias is not None}"
        )
    if isinstance(module, nn.BatchNorm2d):
        return f"num_features={module.num_features}"
    if isinstance(module, nn.MaxPool2d):
        return f"k={module.kernel_size}, s={module.stride}"
    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
        return f"p={module.p}"
    if isinstance(module, nn.Linear):
        return f"in={module.in_features}, out={module.out_features}, bias={module.bias is not None}"
    if isinstance(module, nn.Flatten):
        return f"start_dim={module.start_dim}"
    return "-"


def collect_layer_table(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 48, 48),
) -> List[Dict[str, object]]:
    """Chạy forward hook trên dummy tensor để ghi lại shape và số tham số từng layer."""
    records: List[Dict[str, object]] = []
    hooks = []

    def register_hook(name: str, module: nn.Module) -> None:
        def hook(_mod: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            in_shape = list(inp[0].shape) if inp else []
            out_shape = list(out.shape) if isinstance(out, torch.Tensor) else []
            in_shape_str = "[" + ", ".join(["B"] + [str(d) for d in in_shape[1:]]) + "]"
            out_shape_str = "[" + ", ".join(["B"] + [str(d) for d in out_shape[1:]]) + "]"
            params = sum(p.numel() for p in _mod.parameters(recurse=False))
            trainable = sum(
                p.numel() for p in _mod.parameters(recurse=False) if p.requires_grad
            )
            records.append(
                {
                    "layer_name": name,
                    "layer_type": _mod.__class__.__name__,
                    "config": _describe_layer(_mod),
                    "input_shape": in_shape_str,
                    "output_shape": out_shape_str,
                    "params": params,
                    "trainable_params": trainable,
                }
            )

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if isinstance(module, LEAF_LAYER_TYPES):
            register_hook(name, module)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_size)
        _ = model(dummy)
    model.train(was_training)

    for h in hooks:
        h.remove()

    return records


def format_model_summary(
    model: FER2013CNN,
    input_size: Tuple[int, int, int, int] = (1, 1, 48, 48),
) -> str:
    """Tạo chuỗi bảng summary chi tiết cho model."""
    rows = collect_layer_table(model, input_size=input_size)
    counts = count_parameters(model)
    last_conv_name, last_conv = model.get_last_conv_layer()

    feature_params = sum(p.numel() for p in model.features.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    sep = "=" * 116
    dash = "-" * 116
    lines = [
        sep,
        f"MODEL SUMMARY: {model.config.model_name} ({model.config.model_version})",
        f"Input shape : [B, {input_size[1]}, {input_size[2]}, {input_size[3]}] (float32, normalized)",
        f"Output shape: [B, {model.config.num_classes}] (raw logits, NO Softmax)",
        f"Last Conv2d : {last_conv_name} -> weight shape {tuple(last_conv.weight.shape)}",
        sep,
        f"{'Layer Name':<30} {'Type':<14} {'Input Shape':<18} {'Output Shape':<18} {'Param #':>12}  {'Details'}",
        dash,
    ]

    for r in rows:
        lines.append(
            f"{str(r['layer_name']):<30} "
            f"{str(r['layer_type']):<14} "
            f"{str(r['input_shape']):<18} "
            f"{str(r['output_shape']):<18} "
            f"{int(r['params']):>12,}  "
            f"{str(r['config'])}"
        )

    param_size_mb = counts["total_params"] * 4 / (1024 ** 2)
    lines.extend(
        [
            sep,
            f"Feature Extractor params : {feature_params:>12,} ({feature_params / counts['total_params'] * 100:.2f}%)",
            f"Classifier (FC) params   : {classifier_params:>12,} ({classifier_params / counts['total_params'] * 100:.2f}%)",
            dash,
            f"Total params             : {counts['total_params']:>12,}",
            f"Trainable params         : {counts['trainable_params']:>12,}",
            f"Non-trainable params     : {counts['non_trainable_params']:>12,}",
            f"Estimated weights (FP32) : {param_size_mb:>11.2f} MB",
            sep,
        ]
    )
    return "\n".join(lines)


def _resolve_default_output_dirs() -> Tuple[Path, Path]:
    here = Path(__file__).resolve()
    if here.parent.name == "models" and here.parent.parent.name == "src":
        project_root = here.parent.parent.parent
        summary_dir = project_root / "results" / "model"
        member_out_dir = project_root / "members" / "Tường_MODEL" / "outputs"
        return summary_dir, member_out_dir

    local_out = here.parent.parent / "outputs"
    return local_out, local_out


def save_summary_artifacts(
    model: Optional[FER2013CNN] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Lưu model_summary.txt."""
    if model is None:
        model = build_model(get_default_config())

    if output_dir is None:
        summary_dir, member_out_dir = _resolve_default_output_dirs()
    else:
        summary_dir = Path(output_dir)
        member_out_dir = Path(output_dir)

    summary_dir.mkdir(parents=True, exist_ok=True)
    member_out_dir.mkdir(parents=True, exist_ok=True)

    summary_text = format_model_summary(model)
    summary_path = summary_dir / "model_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    if member_out_dir != summary_dir:
        (member_out_dir / "model_summary.txt").write_text(summary_text + "\n", encoding="utf-8")

    counts = count_parameters(model)
    feature_params = sum(p.numel() for p in model.features.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    last_conv_name, last_conv = model.get_last_conv_layer()

    param_lines = [
        f"model_name: {model.config.model_name}",
        f"model_version: {model.config.model_version}",
        f"total_params: {counts['total_params']}",
        f"trainable_params: {counts['trainable_params']}",
        f"non_trainable_params: {counts['non_trainable_params']}",
        f"feature_extractor_params: {feature_params}",
        f"classifier_params: {classifier_params}",
        f"flatten_dim: {model.flatten_dim}",
        f"last_conv_layer: {last_conv_name}",
        f"last_conv_weight_shape: {list(last_conv.weight.shape)}",
    ]
    param_path = member_out_dir / "parameter_count.txt"
    param_path.write_text("\n".join(param_lines) + "\n", encoding="utf-8")
    return summary_path, param_path


if __name__ == "__main__":
    default_model = build_model(get_default_config())
    print(format_model_summary(default_model))
    s_path, p_path = save_summary_artifacts(default_model)
    print(f"\nSaved summary to: {s_path}")
    print(f"Saved parameter count to: {p_path}")