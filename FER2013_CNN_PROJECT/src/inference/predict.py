"""
predict.py – Inference: nhận ảnh, áp dụng preprocessing, load checkpoint, trả nhãn và xác suất
Chịu trách nhiệm: TV5 – INTEGRATION
Vị trí: src/inference/predict.py

Cách dùng độc lập:
  python src/inference/predict.py --image path/to/face.jpg \
         --checkpoint checkpoints/best_model.pth \
         --config checkpoints/best_config.json
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.data.label_mapping import EMOTION_LABELS
except ImportError:
    EMOTION_LABELS = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    }


def _load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load model từ checkpoint. Trả model đã ở eval mode."""
    try:
        from src.models.cnn_model import build_model
    except ImportError as exc:
        raise ImportError(
            "Không tìm thấy src.models.cnn_model. "
            "Đảm bảo TV2 đã cung cấp cnn_model.py."
        ) from exc

    raw_model_cfg = config.get("model_config", config) if isinstance(config, dict) else {}
    model = build_model(raw_model_cfg if raw_model_cfg else None).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError("Định dạng checkpoint không hợp lệ.")

    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _preprocess_image(image_path: str, config: dict) -> torch.Tensor:
    """Đọc ảnh, chuyển grayscale, resize 48x48, normalize, trả tensor [1, 1, 48, 48]."""
    from PIL import Image

    data_cfg = config.get("data_config", config.get("preprocessing", config))
    mean = float(data_cfg.get("mean", config.get("normalize_mean", 0.5)))
    std = float(data_cfg.get("std", config.get("normalize_std", 0.5)))

    try:
        from src.data.preprocessing import get_eval_transform, preprocess_image_01

        img = Image.open(image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        tensor_01 = preprocess_image_01(arr, img_size=48)
        transform = get_eval_transform(mean=mean, std=std)
        tensor = transform(tensor_01)
    except ImportError:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ])
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img)

    return tensor.unsqueeze(0)

def predict_image(
    image_path: str,
    checkpoint_path: str,
    config: dict | None = None,
    return_probs: bool = True,
) -> dict:
    """
    Dự đoán cảm xúc từ một ảnh khuôn mặt.
    """
    if config is None:
        config = {}

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(checkpoint_path, config, device)
    input_tensor = _preprocess_image(image_path, config).to(device)

    with torch.no_grad():
        logits = model(input_tensor)                        # [1, 7]
        probs = F.softmax(logits, dim=1).squeeze(0)         # [7]
        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())

    result = {
        "class_idx": class_idx,
        "label": EMOTION_LABELS[class_idx],
        "confidence": confidence,
    }

    if return_probs:
        result["probabilities"] = {
            EMOTION_LABELS[i]: float(probs[i].item())
            for i in range(7)
        }

    return result


def predict_batch(
    image_paths: list[str],
    checkpoint_path: str,
    config: dict | None = None,
) -> list[dict]:
    """
    Dự đoán batch nhiều ảnh. Trả list dict kết quả đúng thứ tự đầu vào.
    """
    if config is None:
        config = {}

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, config, device)

    results: list[dict] = [{} for _ in range(len(image_paths))]
    batch_size = int(config.get("batch_size", 32))

    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        tensors = []
        valid_global_indices = []

        for offset, path in enumerate(chunk):
            global_idx = start + offset
            if not os.path.isfile(path):
                results[global_idx] = {"error": f"Không tìm thấy: {path}"}
                continue
            try:
                t = _preprocess_image(path, config)
                tensors.append(t)
                valid_global_indices.append(global_idx)
            except Exception as e:
                results[global_idx] = {"error": str(e)}

        if tensors:
            batch_tensor = torch.cat(tensors, dim=0).to(device)
            with torch.no_grad():
                logits = model(batch_tensor)
                probs_all = F.softmax(logits, dim=1)

            for g_idx, probs in zip(valid_global_indices, probs_all):
                class_idx = int(probs.argmax().item())
                results[g_idx] = {
                    "class_idx": class_idx,
                    "label": EMOTION_LABELS[class_idx],
                    "confidence": float(probs[class_idx].item()),
                    "probabilities": {
                        EMOTION_LABELS[k]: float(probs[k].item())
                        for k in range(7)
                    },
                }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="FER2013 – Predict emotion from image")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Đường dẫn checkpoint (.pth)")
    parser.add_argument("--config", default="checkpoints/best_config.json",
                        help="Đường dẫn config JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    result = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        config=config,
    )

    print(f"\n{'='*50}")
    print(f"  Ảnh     : {args.image}")
    print(f"  Dự đoán : {result['label']} (index={result['class_idx']})")
    print(f"  Xác suất: {result['confidence']:.2%}")
    if "probabilities" in result:
        print("\n  Phân phối xác suất:")
        for name, prob in sorted(result["probabilities"].items(),
                                  key=lambda x: -x[1]):
            bar = "█" * int(prob * 25)
            print(f"    {name:<10} {prob:.4f}  {bar}")
    print("=" * 50)