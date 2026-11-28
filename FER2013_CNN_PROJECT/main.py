"""
main.py – Entry point / Pipeline tích hợp toàn dự án FER2013 CNN
Chịu trách nhiệm: TV5 – INTEGRATION

Cách chạy:
  python main.py --mode train   --config checkpoints/best_config.json
  python main.py --mode eval    --config checkpoints/best_config.json --checkpoint checkpoints/best_model.pth
  python main.py --mode predict --image path/to/face.jpg --checkpoint checkpoints/best_model.pth --config checkpoints/best_config.json
  python main.py --mode smoke_test
"""

import argparse
import json
import os
import sys
import time
import datetime

# ---------------------------------------------------------------------------
# Đảm bảo import từ project root
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_config(config_path: str) -> dict:
    """Đọc config JSON từ file. Trả dict rỗng nếu không tìm thấy."""
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_smoke_test():
    """Kiểm thử end-to-end nhanh: data → model → loss → backward → inference."""
    print("=" * 60)
    print("SMOKE TEST – kiểm tra pipeline cơ bản")
    print("=" * 60)

    import torch

    # 1. Kiểm tra import DataLoader
    try:
        from src.data.dataset import create_dataloaders
        print("[OK] src.data.dataset – import thành công")
    except ImportError as e:
        print(f"[FAIL] src.data.dataset – {e}")
        return False

    # 2. Kiểm tra import Model
    try:
        from src.models.cnn_model import FER2013CNN
        print("[OK] src.models.cnn_model – import thành công")
    except ImportError as e:
        print(f"[FAIL] src.models.cnn_model – {e}")
        return False

    # 3. Kiểm tra forward pass với dummy data
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FER2013CNN().to(device)
        dummy_input = torch.randn(4, 1, 48, 48).to(device)
        dummy_labels = torch.randint(0, 7, (4,)).to(device)

        model.train()
        logits = model(dummy_input)
        assert logits.shape == (4, 7), f"Shape sai: {logits.shape}"

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, dummy_labels)
        loss.backward()

        print(f"[OK] Forward/backward – loss={loss.item():.4f}, device={device}")
    except Exception as e:
        print(f"[FAIL] Forward/backward – {e}")
        return False

    # 4. Kiểm tra inference module
    try:
        from src.inference.predict import predict_image
        print("[OK] src.inference.predict – import thành công")
    except ImportError as e:
        print(f"[FAIL] src.inference.predict – {e}")
        return False

    print("-" * 60)
    print("SMOKE TEST PASSED")
    return True


def run_training(config: dict):
    """Gọi training loop từ TV3."""
    print("=" * 60)
    print("MODE: TRAINING")
    print("=" * 60)
    try:
        from src.training.train import run_training_loop
    except ImportError as e:
        print(f"[ERROR] Không import được src.training.train: {e}")
        sys.exit(1)

    start = time.time()
    run_training_loop(config)
    elapsed = time.time() - start
    print(f"\nTraining hoàn thành. Thời gian: {datetime.timedelta(seconds=int(elapsed))}")


def run_evaluation(config: dict, checkpoint_path: str):
    """Gọi chuỗi script đánh giá từ TV4."""
    import subprocess

    print("=" * 60)
    print("MODE: EVALUATION")
    print("=" * 60)

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing_pp}" if existing_pp else ROOT

    eval_scripts = [
        os.path.join(ROOT, "src", "evaluation", "evaluate.py"),
        os.path.join(ROOT, "src", "evaluation", "metrics.py"),
        os.path.join(ROOT, "src", "evaluation", "classification_report.py"),
        os.path.join(ROOT, "src", "evaluation", "confusion_matrix.py"),
    ]

    for script_path in eval_scripts:
        if os.path.isfile(script_path):
            print(f"\n>>> Đang chạy: {os.path.relpath(script_path, ROOT)}")
            result = subprocess.run([sys.executable, script_path], cwd=ROOT, env=env)
            if result.returncode != 0:
                print(f"[ERROR] Lỗi khi chạy {os.path.basename(script_path)}")
                sys.exit(result.returncode)

def run_inference(image_path: str, checkpoint_path: str, config: dict):
    """Dự đoán một ảnh đơn."""
    print("=" * 60)
    print("MODE: PREDICT")
    print("=" * 60)
    try:
        from src.inference.predict import predict_image
    except ImportError as e:
        print(f"[ERROR] Không import được src.inference.predict: {e}")
        sys.exit(1)

    result = predict_image(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        config=config,
    )
    print(f"\nẢnh      : {image_path}")
    print(f"Dự đoán  : {result['label']} (index={result['class_idx']})")
    print(f"Xác suất : {result['confidence']:.2%}")
    if "probabilities" in result:
        print("\nPhân phối xác suất:")
        for name, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"  {name:<10} {prob:.4f}  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FER2013 CNN – Pipeline tích hợp (TV5)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "predict", "smoke_test"],
        default="smoke_test",
        help="Chế độ chạy:\n"
             "  train      – huấn luyện model\n"
             "  eval       – đánh giá trên test set\n"
             "  predict    – dự đoán một ảnh\n"
             "  smoke_test – kiểm tra pipeline nhanh (mặc định)",
    )
    parser.add_argument("--config", default="checkpoints/best_config.json",
                        help="Đường dẫn file config JSON")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Đường dẫn checkpoint .pth")
    parser.add_argument("--image", default=None,
                        help="Đường dẫn ảnh để dự đoán (dùng với --mode predict)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print("  FER2013 Facial Emotion Recognition – CNN")
    print(f"  Mode    : {args.mode}")
    print(f"  Config  : {args.config}")
    print(f"  Time    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if args.mode == "smoke_test":
        success = run_smoke_test()
        sys.exit(0 if success else 1)

    elif args.mode == "train":
        run_training(config)

    elif args.mode == "eval":
        if not args.checkpoint:
            print("[ERROR] Cần truyền --checkpoint để đánh giá")
            sys.exit(1)
        run_evaluation(config, args.checkpoint)

    elif args.mode == "predict":
        if not args.image:
            print("[ERROR] Cần truyền --image để dự đoán")
            sys.exit(1)
        run_inference(args.image, args.checkpoint, config)


if __name__ == "__main__":
    main()
