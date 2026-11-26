from pathlib import Path
import torch

# 1. DUONG DAN DU AN
# Thu muc goc cua du an
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Duong dan dataset
CSV_PATH = PROJECT_ROOT / "data" / "fer2013.csv"

# Thu muc luu checkpoint
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Thu muc luu ket qua training
RESULTS_DIR = PROJECT_ROOT / "results" / "training"

# 2. CAU HINH DU LIEU
BATCH_SIZE = 64
NUM_WORKERS = 0
USE_AUGMENTATION = True

# 3. CAU HINH HUAN LUYEN
EPOCHS = 30

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

SEED = 42


# 4. LOSS FUNCTION
LOSS_NAME = "CrossEntropyLoss"

# 5. OPTIMIZER
OPTIMIZER_NAME = "Adam"

# 6. LEARNING RATE SCHEDULER
USE_SCHEDULER = True

SCHEDULER_PATIENCE = 3

SCHEDULER_FACTOR = 0.5

# 7. CHECKPOINT
SAVE_BEST_ONLY = True

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"


# 8. THIET BI
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 9. HAM LAY TOAN BO CONFIG
def get_training_config():

    config = {
        "csv_path": str(CSV_PATH),

        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "use_aug": USE_AUGMENTATION,

        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "seed": SEED,

        "loss_name": LOSS_NAME,
        "optimizer_name": OPTIMIZER_NAME,

        "use_scheduler": USE_SCHEDULER,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_factor": SCHEDULER_FACTOR,

        "save_best_only": SAVE_BEST_ONLY,
        "best_model_path": str(BEST_MODEL_PATH),

        "device": DEVICE
    }

    return config
