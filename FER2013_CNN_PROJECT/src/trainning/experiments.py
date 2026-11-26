from pathlib import Path
from datetime import datetime
import csv
import json


# ============================================================
# 1. DUONG DAN THU MUC
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
LOG_PATH = EXPERIMENT_DIR / "experiment_log.csv"
SUMMARY_PATH = EXPERIMENT_DIR / "experiment_summary.csv"


# ============================================================
# 2. CAC COT CUA EXPERIMENT LOG
# ============================================================

EXPERIMENT_FIELDS = [
    "Experiment_ID",
    "date_time",
    "owner",
    "status",
    "git_commit",
    "dataset_version",
    "split",
    "seed",
    "model_version",
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs",
    "dropout",
    "activation",
    "augmentation",
    "scheduler",
    "weight_decay",
    "best_epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "device",
    "runtime",
    "checkpoint_path",
    "config_path",
    "log_path",
    "observation",
    "next_action"
]


# ============================================================
# 3. CHUYEN DOI DU LIEU SANG DINH DANG JSON
# ============================================================

def make_json_serializable(value):
    """Chuyển các kiểu dữ liệu đặc biệt thành kiểu JSON."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if hasattr(value, "to_dict"):
        return make_json_serializable(value.to_dict())

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


# ============================================================
# 4. TAO THU MUC RIENG CHO TUNG THI NGHIEM
# ============================================================

def create_experiment_directory(experiment_id):
    """
    Tạo thư mục riêng cho từng thí nghiệm.

    experiments/
        E02/
            checkpoints/
            logs/
    """

    experiment_id = str(experiment_id).strip()

    if not experiment_id:
        raise ValueError("Experiment_ID không được để trống.")

    if (
        Path(experiment_id).name != experiment_id
        or experiment_id in (".", "..")
        or "/" in experiment_id
        or "\\" in experiment_id
    ):
        raise ValueError("Experiment_ID không hợp lệ.")

    experiment_path = EXPERIMENT_DIR / experiment_id

    experiment_path.mkdir(parents=True, exist_ok=True)

    (experiment_path / "checkpoints").mkdir(exist_ok=True)
    (experiment_path / "logs").mkdir(exist_ok=True)

    return experiment_path


# ============================================================
# 5. LUU CAU HINH THI NGHIEM
# ============================================================

def save_experiment_config(experiment_id, config):
    """Lưu cấu hình của thí nghiệm thành file JSON."""

    experiment_path = create_experiment_directory(experiment_id)
    config_path = experiment_path / "config.json"

    serializable_config = make_json_serializable(config)

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            serializable_config,
            file,
            indent=4,
            ensure_ascii=False
        )

    return config_path


# ============================================================
# 6. TAO CAC DUONG DAN CUA THI NGHIEM
# ============================================================

def get_experiment_paths(experiment_id):
    """Trả về các đường dẫn riêng của từng thí nghiệm."""

    experiment_path = create_experiment_directory(experiment_id)

    return {
        "experiment_dir": experiment_path,
        "checkpoint_path": (
            experiment_path / "checkpoints" / "best_model.pth"
        ),
        "config_path": experiment_path / "config.json",
        "log_path": (
            experiment_path / "logs" / "training_history.csv"
        )
    }


# ============================================================
# 7. KHOI TAO EXPERIMENT LOG
# ============================================================

def initialize_experiment_log():
    """Tạo experiment_log.csv nếu chưa tồn tại."""

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    if not LOG_PATH.exists():
        with open(
            LOG_PATH,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=EXPERIMENT_FIELDS
            )

            writer.writeheader()


# ============================================================
# 8. GHI KET QUA THI NGHIEM
# ============================================================

def log_experiment(experiment_info):
    """Ghi thông tin một lần thực nghiệm vào CSV."""

    initialize_experiment_log()

    row = {
        field: experiment_info.get(field, "")
        for field in EXPERIMENT_FIELDS
    }

    if not row["date_time"]:
        row["date_time"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

    with open(
        LOG_PATH,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=EXPERIMENT_FIELDS
        )

        writer.writerow(row)


# ============================================================
# 9. DOC CAC KET QUA THI NGHIEM
# ============================================================

def read_experiment_log():
    """Đọc toàn bộ dữ liệu trong experiment_log.csv."""

    if not LOG_PATH.exists():
        return []

    with open(
        LOG_PATH,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)


# ============================================================
# 10. TAO BANG TONG HOP THI NGHIEM
# ============================================================

def save_experiment_summary(experiment_rows=None):
    """Tạo experiment_summary.csv từ nhật ký thực nghiệm."""

    summary_fields = [
        "Experiment_ID",
        "optimizer",
        "learning_rate",
        "batch_size",
        "dropout",
        "augmentation",
        "scheduler",
        "best_epoch",
        "train_accuracy",
        "val_accuracy",
        "train_loss",
        "val_loss",
        "runtime",
        "observation"
    ]

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    if experiment_rows is None:
        experiment_rows = read_experiment_log()

    with open(
        SUMMARY_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields
        )

        writer.writeheader()

        for row in experiment_rows:
            writer.writerow({
                field: row.get(field, "")
                for field in summary_fields
            })

    return SUMMARY_PATH