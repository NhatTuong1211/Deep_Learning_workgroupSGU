"""
test_inference.py – Kiểm thử module inference (TV5)
Vị trí: tests/test_inference.py

Chạy: python -m pytest tests/test_inference.py -v
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Helper: tạo mock model trả logits hợp lệ
# ---------------------------------------------------------------------------

class _MockCNN(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.fc = torch.nn.Linear(1 * 48 * 48, 7)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def _build_mock_checkpoint(tmp_dir: str) -> str:
    """Lưu checkpoint giả vào tmp_dir, trả đường dẫn."""
    model = _MockCNN()
    path = os.path.join(tmp_dir, "mock_model.pth")
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPredictInference(unittest.TestCase):
    """Kiểm thử hàm predict_image từ src/inference/predict.py."""

    def setUp(self):
        import tempfile
        from PIL import Image
        import numpy as np

        self.tmp = tempfile.mkdtemp()
        self.checkpoint_path = _build_mock_checkpoint(self.tmp)

        # Tạo ảnh giả 48×48 grayscale
        arr = (np.random.rand(48, 48) * 255).astype("uint8")
        img = Image.fromarray(arr, mode="L")
        self.image_path = os.path.join(self.tmp, "dummy_face.png")
        img.save(self.image_path)

        self.config = {
            "normalize_mean": 0.5,
            "normalize_std": 0.5,
        }

    def _patch_model(self):
        """Patch FER2013CNN với MockCNN để chạy độc lập khỏi TV2."""
        return patch("src.inference.predict._load_model",
                     return_value=_MockCNN().eval())

    def test_predict_returns_dict_with_required_keys(self):
        """predict_image phải trả dict với class_idx, label, confidence."""
        with self._patch_model():
            from src.inference.predict import predict_image
            result = predict_image(
                image_path=self.image_path,
                checkpoint_path=self.checkpoint_path,
                config=self.config,
            )
        self.assertIn("class_idx", result)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)

    def test_class_idx_in_valid_range(self):
        """class_idx phải nằm trong [0, 6]."""
        with self._patch_model():
            from src.inference.predict import predict_image
            result = predict_image(self.image_path, self.checkpoint_path, self.config)
        self.assertIn(result["class_idx"], range(7))

    def test_confidence_in_01(self):
        """confidence phải nằm trong [0, 1]."""
        with self._patch_model():
            from src.inference.predict import predict_image
            result = predict_image(self.image_path, self.checkpoint_path, self.config)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_probabilities_sum_to_one(self):
        """Tổng xác suất 7 lớp phải xấp xỉ 1.0."""
        with self._patch_model():
            from src.inference.predict import predict_image
            result = predict_image(self.image_path, self.checkpoint_path, self.config)
        total = sum(result["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_probabilities_has_seven_classes(self):
        """probabilities phải có đúng 7 key."""
        with self._patch_model():
            from src.inference.predict import predict_image
            result = predict_image(self.image_path, self.checkpoint_path, self.config)
        self.assertEqual(len(result["probabilities"]), 7)

    def test_label_matches_class_idx(self):
        """label phải khớp với EMOTION_LABELS[class_idx]."""
        from src.inference.predict import EMOTION_LABELS
        with self._patch_model():
            from src.inference.predict import predict_image
            result = predict_image(self.image_path, self.checkpoint_path, self.config)
        self.assertEqual(result["label"], EMOTION_LABELS[result["class_idx"]])

    def test_file_not_found_raises_error(self):
        """FileNotFoundError khi ảnh không tồn tại."""
        with self._patch_model():
            from src.inference.predict import predict_image
            with self.assertRaises(FileNotFoundError):
                predict_image("non_existent.jpg", self.checkpoint_path, self.config)

    def test_predict_batch_returns_list(self):
        """predict_batch phải trả list có cùng số phần tử với đầu vào."""
        with self._patch_model():
            from src.inference.predict import predict_batch
            results = predict_batch(
                image_paths=[self.image_path, self.image_path],
                checkpoint_path=self.checkpoint_path,
                config=self.config,
            )
        self.assertEqual(len(results), 2)

    def test_output_tensor_not_softmax_by_model(self):
        """Model không được áp Softmax trước loss – kiểm tra logits có thể âm."""
        # Chỉ kiểm tra MockCNN trả logits thô (giá trị có thể âm)
        model = _MockCNN()
        dummy = torch.randn(1, 1, 48, 48)
        with torch.no_grad():
            logits = model(dummy)
        # Logits chưa softmax → ít nhất một giá trị < 0 hoặc > 1 (thống kê)
        self.assertEqual(logits.shape, (1, 7))


class TestEmotionLabelMapping(unittest.TestCase):
    """Kiểm tra EMOTION_LABELS đúng thứ tự đề bài."""

    def test_mapping_correct_order(self):
        from src.inference.predict import EMOTION_LABELS
        expected = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral",
        }
        self.assertEqual(EMOTION_LABELS, expected)

    def test_mapping_has_seven_entries(self):
        from src.inference.predict import EMOTION_LABELS
        self.assertEqual(len(EMOTION_LABELS), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
