from pathlib import Path
import numpy as np

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
import numpy as np
MU = np.array([0.0, 0.0])
C = np.array([[0.3, -0.5],
              [-0.5, 2.0]])

# --- Mien sinh x = (x1, x2) ---
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)

# --- Kich thuoc cac tap ---
N_TRAIN_POOL = 2000  
N_VAL = 200          
N_TEST = 100          
SEED_TRAIN = 42
SEED_VAL = 43
SEED_TEST = 44

A_VALUES = list(range(200, 2001, 200))

TRAIN_SEEDS = [0, 1, 2]
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer=None):
    import torch
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in (None, "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if prefer in (None, "mps") and mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
