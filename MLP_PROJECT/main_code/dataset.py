import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg
from data import load_split, get_train_subset
class GaussianDataset(Dataset):
    """Chuyen (x, y) numpy -> tensor float32.  x: (N, 2),  y: (N, 1) de khop MSELoss."""

    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        assert len(self.x) == len(self.y), "x va y khac so mau"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
def get_dataloader(split, batch_size, train_size=None, shuffle=None, seed=0, num_workers=0):
    """Tra ve DataLoader cho 'train' | 'val' | 'test'.

    batch_size : BAT BUOC truyen vao (thuoc cau hinh baseline do TV2 cong bo).
    train_size : chi dung cho split='train' -- lay A mau dau (thi nghiem A). None = dung het.
    shuffle    : mac dinh True voi train, False voi val/test.
    seed       : seed xao tron batch => cung seed thi cung thu tu batch (tai lap).
    """
    if split == "train":
        x, y = get_train_subset(train_size) if train_size else load_split("train")
    elif split in ("val", "test"):
        if train_size is not None:
            raise ValueError("train_size chi dung cho split='train'")
        x, y = load_split(split)
    else:
        raise ValueError("split phai la 'train', 'val' hoac 'test'")

    if shuffle is None:
        shuffle = (split == "train")
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(GaussianDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      generator=gen, num_workers=num_workers)
def get_all_loaders(batch_size, train_size=None, seed=0):
    """(train_loader, val_loader, test_loader). Nho: test CHI dung danh gia cuoi."""
    return (get_dataloader("train", batch_size, train_size=train_size, seed=seed),
            get_dataloader("val", batch_size),
            get_dataloader("test", batch_size))
