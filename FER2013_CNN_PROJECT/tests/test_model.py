"""Unit tests & Data-Model integration smoke tests cho FER2013CNN."""

import csv
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
# Tự động nhận diện đang chạy ở repo chính (<root>/tests/) hay ở thư mục riêng (Tường_MODEL/tests/)
if (BASE_DIR / "src" / "models").exists():
    PROJECT_ROOT = BASE_DIR
    MODEL_SRC_DIR = PROJECT_ROOT / "src" / "models"
else:
    PROJECT_ROOT = BASE_DIR.parent
    MODEL_SRC_DIR = BASE_DIR / "code"

TV1_HANDOVER_DIR = PROJECT_ROOT / "tv1_data_handover"

for p in (PROJECT_ROOT, MODEL_SRC_DIR, TV1_HANDOVER_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from src.models.cnn_model import FER2013CNN, build_model
    from src.models.model_config import ModelConfig, get_default_config
    from src.models.model_summary import count_parameters
except ImportError:
    from cnn_model import FER2013CNN, build_model
    from model_config import ModelConfig, get_default_config
    from model_summary import count_parameters


class TestFER2013CNN(unittest.TestCase):
    """Kiểm thử toàn diện kiến trúc, shape, gradient, config và checkpoint."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.config = get_default_config()
        self.model = build_model(self.config)

    def test_output_shape_multiple_batch_sizes(self) -> None:
        """Kiểm tra đầu ra [B, 7] với nhiều batch size ở cả train() và eval()."""
        for mode_train in (True, False):
            self.model.train(mode_train)
            for batch_size in (1, 4, 16, 64):
                x = torch.randn(batch_size, 1, 48, 48, dtype=torch.float32)
                logits = self.model(x)
                self.assertEqual(tuple(logits.shape), (batch_size, 7))
                self.assertEqual(logits.dtype, torch.float32)
                self.assertFalse(torch.isnan(logits).any().item())
                self.assertFalse(torch.isinf(logits).any().item())

    def test_raw_logits_no_softmax(self) -> None:
        """Xác nhận mô hình trả logits thô (không gắn Softmax trước CrossEntropyLoss)."""
        self.model.eval()
        for module in self.model.modules():
            self.assertNotIsInstance(module, (nn.Softmax, nn.LogSoftmax))

        x = torch.randn(8, 1, 48, 48)
        with torch.no_grad():
            logits = self.model(x)
        row_sums = logits.sum(dim=1)
        # Nếu là xác suất softmax thì mọi row_sum đều ~ 1.0 và mọi giá trị >= 0
        is_all_prob_like = (
            torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
            and bool((logits >= 0).all().item())
        )
        self.assertFalse(is_all_prob_like)

    def test_forward_loss_backward_and_gradients(self) -> None:
        """Kiểm tra CrossEntropyLoss, backward và gradient trên mọi tham số."""
        self.model.train()
        x = torch.randn(16, 1, 48, 48)
        y = torch.randint(0, 7, (16,), dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        self.model.zero_grad()
        logits = self.model(x)
        loss = criterion(logits, y)

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Thiếu gradient tại {name}")
                self.assertTrue(
                    torch.isfinite(param.grad).all().item(),
                    f"Gradient chứa NaN/Inf tại {name}",
                )

    def test_config_variants_and_json_roundtrip(self) -> None:
        """Kiểm tra cấu hình linh hoạt cho thực nghiệm (dropout, activation, JSON)."""
        cfg = get_default_config(
            dropout_conv=0.2,
            dropout_fc=0.4,
            activation="leaky_relu",
            conv_channels=(32, 64, 128),
            convs_per_block=(2, 2, 2),
        )
        as_json = json.dumps(cfg.to_dict())
        restored = ModelConfig.from_dict(json.loads(as_json))
        self.assertEqual(cfg, restored)

        variant_model = build_model(restored)
        out = variant_model(torch.randn(4, 1, 48, 48))
        self.assertEqual(tuple(out.shape), (4, 7))

    def test_invalid_config_raises(self) -> None:
        """Kiểm tra validation của ModelConfig khi tham số sai."""
        with self.assertRaises(ValueError):
            get_default_config(activation="invalid_act")
        with self.assertRaises(ValueError):
            get_default_config(dropout_fc=1.0)
        with self.assertRaises(ValueError):
            get_default_config(conv_channels=(32, 64), convs_per_block=(2, 2, 2))

    def test_checkpoint_save_load_compatibility(self) -> None:
        """Kiểm tra lưu/nạp checkpoint (state_dict + config)."""
        self.model.eval()
        x = torch.randn(4, 1, 48, 48)
        with torch.no_grad():
            expected_logits = self.model(x)

        buffer = io.BytesIO()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.config.to_dict(),
            },
            buffer,
        )
        buffer.seek(0)
        ckpt = torch.load(buffer, map_location="cpu", weights_only=False)

        reloaded = build_model(ckpt["model_config"])
        reloaded.load_state_dict(ckpt["model_state_dict"])
        reloaded.eval()
        with torch.no_grad():
            actual_logits = reloaded(x)

        self.assertTrue(torch.allclose(expected_logits, actual_logits, atol=1e-6))

    def test_last_conv_layer_for_tv4_visualization(self) -> None:
        """Xác nhận có thể trích xuất lớp Conv2d cuối cùng để vẽ filter."""
        conv_names = self.model.get_conv_layer_names()
        self.assertEqual(len(conv_names), 8)
        last_name, last_conv = self.model.get_last_conv_layer()
        self.assertEqual(last_name, "features.block4.conv_layers.conv2")
        self.assertIsInstance(last_conv, nn.Conv2d)
        self.assertEqual(tuple(last_conv.weight.shape), (256, 256, 3, 3))

    def test_parameter_count_positive_and_consistent(self) -> None:
        """Xác nhận số tham số > 0 và trainable == total."""
        counts = count_parameters(self.model)
        self.assertGreater(counts["total_params"], 0)
        self.assertEqual(counts["total_params"], counts["trainable_params"])
        self.assertEqual(counts["non_trainable_params"], 0)


class TestTV1TV2Integration(unittest.TestCase):
    """Smoke test tích hợp giữa pipeline (mapping + dữ liệu FER2013) và model."""

    def test_label_mapping_matches_tv1(self) -> None:
        """Xác nhận số lớp và thứ tự nhãn khớp 100% với TV1 và đề bài."""
        try:
            from src.data.label_mapping import CLASS_NAMES, NUM_CLASSES
        except ImportError:
            self.skipTest("Chưa có module src.data.label_mapping của TV1 trong repo.")

        expected = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.assertEqual(NUM_CLASSES, 7)
        self.assertEqual(CLASS_NAMES, expected)

        model = build_model(get_default_config(num_classes=NUM_CLASSES))
        self.assertEqual(model.config.num_classes, NUM_CLASSES)

    def test_real_fer2013_batch_forward_backward(self) -> None:
        """Chạy 1 batch ảnh thật từ FER2013 qua FER2013CNN (forward + loss + backward)."""
        candidate_csvs = [
            PROJECT_ROOT / "data" / "raw" / "FER2013" / "fer2013.csv",
            PROJECT_ROOT
            / "challenges-in-representation-learning-facial-expression-recognition-challenge"
            / "fer2013"
            / "fer2013.csv",
        ]
        csv_path = next((p for p in candidate_csvs if p.exists()), None)
        if csv_path is None:
            self.skipTest("Không tìm thấy file fer2013.csv để chạy smoke test dữ liệu thật.")

        # Ưu tiên dùng trực tiếp create_dataloaders của TV1 nếu môi trường có torchvision
        used_tv1_loader = False
        try:
            from src.data.dataset import create_dataloaders

            loaders = create_dataloaders(
                csv_path=str(csv_path), batch_size=16, num_workers=0, use_aug=False
            )
            images, labels = next(iter(loaders["train"]))
            used_tv1_loader = True
        except ImportError:
            # Fallback khi máy cục bộ chưa cài torchvision: đọc 16 dòng thật theo đúng chuẩn TV1
            imgs, lbls = [], []
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    px = np.asarray(row["pixels"].strip().split(), dtype=np.float32)
                    img = ((np.clip(px, 0.0, 255.0) / 255.0) - 0.5) / 0.5
                    imgs.append(img.reshape(1, 48, 48))
                    lbls.append(int(row["emotion"]))
                    if len(imgs) == 16:
                        break
            images = torch.from_numpy(np.stack(imgs, axis=0)).float()
            labels = torch.tensor(lbls, dtype=torch.long)

        self.assertEqual(tuple(images.shape), (16, 1, 48, 48))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)
        self.assertTrue(bool(((labels >= 0) & (labels <= 6)).all().item()))

        model = build_model(get_default_config())
        model.train()
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        self.assertEqual(tuple(logits.shape), (16, 7))
        self.assertTrue(torch.isfinite(loss).item())
        print(
            f"\n[Integration Smoke Test] real FER2013 batch OK "
            f"(via_tv1_dataloader={used_tv1_loader}, loss={loss.item():.4f})"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
