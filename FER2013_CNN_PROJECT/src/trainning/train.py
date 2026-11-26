
"""Training baseline E01 va controlled experiments E02-E07 cho FER2013."""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.config import get_training_config, PROJECT_ROOT
from src.training.validate import validate_one_epoch
from src.training.checkpoint import save_checkpoint
from src.training.experiments import (
    get_experiment_paths,
    save_experiment_config,
    log_experiment,
    save_experiment_summary,
    make_json_serializable,
)
from src.data.dataset import create_dataloaders
from src.models.cnn_model import build_model
from src.models.model_config import get_default_config


SCREENING_EPOCHS = 5

EXPERIMENT_CONFIGS = {
    "E02": {"optimizer_name": "SGD", "momentum": 0.9},
    "E03": {"learning_rate": 0.0005},
    "E04": {"batch_size": 32},
    "E05": {"dropout_conv": 0.35, "dropout_fc": 0.6},
    "E06": {"use_aug": False},
    "E07": {"use_scheduler": False},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader khong co du lieu.")

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def save_training_history(history, history_path):
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = history_path.with_name(history_path.name + ".tmp")

    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
    ]

    with open(temp_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    temp_path.replace(history_path)


def main(experiment_id):
    if experiment_id != "E01" and experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(
            "Experiment khong hop le. Chon E01 den E07."
        )

    print("=" * 60)
    print(f"START TRAINING: {experiment_id}")
    print("=" * 60)

    start_time = time.time()

    # 1. Lay cau hinh baseline
    config = get_training_config()

    # 2. Ap dung thay doi rieng cho tung experiment
    changes = EXPERIMENT_CONFIGS.get(experiment_id, {})
    config.update(changes)

    # E01: 30 epoch; E02-E07: 5 epoch sang loc
    if experiment_id == "E01":
        config["epochs"] = 30
    else:
        config["epochs"] = SCREENING_EPOCHS

    seed = int(config["seed"])
    set_seed(seed)

    device = config["device"]

    print(f"Device: {device}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Optimizer: {config['optimizer_name']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Augmentation: {config['use_aug']}")
    print(f"Scheduler: {config['use_scheduler']}")

    # 3. Tao DataLoader
    loaders = create_dataloaders(
        csv_path=config["csv_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_aug=config["use_aug"],
        seed=seed,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Khong su dung loaders['test'] trong qua trinh training.

    # 4. Tao model
    model_overrides = {
        key: value
        for key, value in changes.items()
        if key in ("dropout_conv", "dropout_fc")
    }

    model_config = get_default_config(**model_overrides)
    model = build_model(model_config).to(device)

    # 5. Loss function
    criterion = nn.CrossEntropyLoss()

    # 6. Optimizer
    optimizer_name = config["optimizer_name"].lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Optimizer khong ho tro: {optimizer_name}")

    # 7. Scheduler
    scheduler = None

    if config["use_scheduler"]:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
        )

    # 8. Xac dinh duong dan
    paths = get_experiment_paths(experiment_id)

    if experiment_id == "E01":
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_model.pth"
        history_path = (
            PROJECT_ROOT / "results" / "training" / "training_history.csv"
        )
    else:
        checkpoint_path = Path(paths["checkpoint_path"])
        history_path = Path(paths["log_path"])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # 9. Luu cau hinh thuc nghiem
    config["experiment_id"] = experiment_id
    config["model_config"] = model_config.to_dict()
    config["checkpoint_path"] = str(checkpoint_path)
    config["device"] = str(device)

    save_experiment_config(
        experiment_id,
        make_json_serializable(config),
    )

    # 10. Training loop
    history = []
    best_val_accuracy = -1.0
    best_epoch = 0

    print("\nBAT DAU HUAN LUYEN\n")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
        }

        history.append(row)
        save_training_history(history, history_path)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch}/{config['epochs']}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy * 100:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy * 100:.2f}% "
            f"LR: {current_lr:.6f} "
            f"Time: {epoch_time:.1f}s",
            flush=True,
        )

        # Chi chon checkpoint theo Validation Accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                config=config,
                checkpoint_path=checkpoint_path,
            )

            print(
                f"  -> BEST CHECKPOINT SAVED: epoch {best_epoch}",
                flush=True,
            )

    # 11. Luu tong ket
    runtime = time.time() - start_time
    final_row = history[-1]

    experiment_info = {
        "Experiment_ID": experiment_id,
        "date_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "owner": "TV3",
        "status": "completed",
        "git_commit": "not_recorded",
        "dataset_version": "FER2013",
        "split": "Train/Validation",
        "seed": seed,
        "model_version": "v1.0",
        "optimizer": config["optimizer_name"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "dropout": (
            f"conv={model_config.dropout_conv}, "
            f"fc={model_config.dropout_fc}"
        ),
        "activation": model_config.activation,
        "augmentation": config["use_aug"],
        "scheduler": config["use_scheduler"],
        "weight_decay": config["weight_decay"],
        "best_epoch": best_epoch,
        "train_loss": final_row["train_loss"],
        "train_accuracy": final_row["train_accuracy"],
        "val_loss": min(history, key=lambda x: x["val_loss"])["val_loss"],
        "val_accuracy": best_val_accuracy,
        "device": str(device),
        "runtime": runtime,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(paths["config_path"]),
        "log_path": str(history_path),
        "observation": (
            f"Best validation accuracy at epoch {best_epoch}"
        ),
        "next_action": "Review experiment results",
    }

    log_experiment(experiment_info)
    save_experiment_summary()

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETED: {experiment_id}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Runtime: {runtime / 60:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FER2013 CNN experiments"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["E01", "E02", "E03", "E04", "E05", "E06", "E07"],
        help="Experiment ID",
    )

    args = parser.parse_args()
    main(args.experiment)
