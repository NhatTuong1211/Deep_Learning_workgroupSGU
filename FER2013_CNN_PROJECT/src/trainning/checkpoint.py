"""Luu va nap checkpoint PyTorch an toan cho mo hinh FER2013 CNN."""

from pathlib import Path
import os
import torch


CHECKPOINT_FORMAT_VERSION = 1


def _validate_checkpoint_payload(checkpoint):
    """Kiem tra cac truong bat buoc cua checkpoint sau khi doc lai."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint khong phai la dict hop le.")

    required_keys = {
        "epoch",
        "model_state_dict",
        "best_val_accuracy",
        "config",
    }
    missing = required_keys.difference(checkpoint.keys())
    if missing:
        raise ValueError(
            "Checkpoint thieu cac truong bat buoc: "
            + ", ".join(sorted(missing))
        )

    if not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError("model_state_dict trong checkpoint khong hop le.")

    if not checkpoint["model_state_dict"]:
        raise ValueError("model_state_dict dang rong.")

    return checkpoint


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_accuracy,
    config,
    checkpoint_path,
):
    """
    Luu checkpoint theo cach an toan:
    1. Ghi vao file tam cung thu muc.
    2. Doc lai file tam bang torch.load de kiem tra.
    3. Chi thay the checkpoint dich sau khi file tam hop le.

    Cach nay han che de lai file checkpoint dang ghi do bi ngat dot ngot.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")

    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "best_val_accuracy": float(best_val_accuracy),
        "config": config,
    }

    try:
        # Ghi file tam, khong ghi truc tiep len checkpoint dang ton tai.
        torch.save(checkpoint, temp_path)

        # Doc lai de xac nhan file vua ghi co the duoc PyTorch nap.
        try:
            saved = torch.load(
                temp_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Tuong thich voi cac phien ban PyTorch chua co weights_only.
            saved = torch.load(temp_path, map_location="cpu")

        _validate_checkpoint_payload(saved)

        # Xac nhan state_dict trong file tam co cung tap key voi model hien tai.
        expected_keys = set(model.state_dict().keys())
        saved_keys = set(saved["model_state_dict"].keys())
        if expected_keys != saved_keys:
            raise ValueError(
                "State dict trong checkpoint khong khop voi model dang luu."
            )

        # Thay the nguyen tu: file dich cu van con nguyen neu buoc tren that bai.
        os.replace(temp_path, checkpoint_path)

    except Exception:
        # Don dep file tam neu co loi; khong xoa checkpoint dich cu.
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise

    print(f"Checkpoint saved and verified: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Nap checkpoint vao model.

    Neu truyen optimizer va checkpoint co optimizer_state_dict thi nap ca
    trang thai optimizer de tiep tuc huan luyen. Tuong thich voi checkpoint
    cu khong co format_version.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    _validate_checkpoint_payload(checkpoint)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    model.to(device)

    return {
        "model": model,
        "optimizer": optimizer,
        "epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["best_val_accuracy"],
        "config": checkpoint["config"],
    }
