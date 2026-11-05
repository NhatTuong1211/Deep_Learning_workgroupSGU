"""
Thực nghiệm A (TV3): Khảo sát ảnh hưởng của số mẫu huấn luyện (A) đến hiệu năng mạng MLP.
- Chạy 10 mức số mẫu train: A = 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000.
- Lặp lại trên 3 random seeds (TRAIN_SEEDS = [0, 1, 2]) theo khuyến nghị của đề để đảm bảo tính khách quan.
- Giữ cố định cấu hình chuẩn (baseline) của TV2:
    + Kiến trúc MLP: input_dim=2, hidden_dim=32, num_layers=2, output_dim=1, activation=ReLU
    + Huấn luyện: batch_size=64, epochs=200, lr=0.01, optimizer=Adam, loss=MSELoss
    + Chọn mô hình tốt nhất theo Validation Loss nhỏ nhất (checkpointing).
- Xuất kết quả:
    + results_A.csv (bảng tổng hợp trung bình ± độ lệch chuẩn theo từng mức A)
    + results_A_runs.csv (log chi tiết từng lần chạy cho cả 30 lượt chạy)
    + loss_history_A.csv (lịch sử train/val loss theo epoch)
    + config_A.json (cấu hình thực nghiệm A)
    + checkpoints/best_model_A.pt (trọng số mô hình tốt nhất)
    + Bộ biểu đồ phân tích trong thư mục figures/
"""
import os
import sys
import time
import json
from pathlib import Path

# 1. Cấu hình đường dẫn gốc (CHỈ DÙNG ĐOẠN NÀY, KHÔNG DÙNG os.chdir)
SCRIPT_DIR = Path(__file__).resolve().parent
project_root = SCRIPT_DIR.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 2. Import thư viện bên ngoài
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 3. Import các module hệ thống từ main_code
from main_code import config as cfg
from main_code.data import gaussian_2d, load_split
from main_code.dataset import get_all_loaders, get_dataloader
from main_code.model import MLPRegressor
from main_code.train import train_model
from main_code.evaluate import evaluate_model

# ==============================================================================
# 1. CẤU HÌNH CHUẨN (BASELINE) CHO THỰC NGHIỆM A
# ==============================================================================
BASELINE_CONFIG = {
    "experiment": "Experiment A - Training Sample Size (A)",
    "input_dim": 2,
    "hidden_dim": 32,
    "num_layers": 2,
    "output_dim": 1,
    "activation": "ReLU",
    "batch_size": 64,
    "epochs": 200,
    "lr": 0.01,
    "optimizer": "Adam",
    "loss_function": "MSELoss",
    "stopping_criterion": "Chạy cố định 200 epochs, lưu và khôi phục checkpoint có Validation MSE nhỏ nhất",
    "checkpoint_metric": "val_MSE (best epoch)",
    "A_values": cfg.A_VALUES,
    "seeds": cfg.TRAIN_SEEDS,
    "val_size": cfg.N_VAL,
    "test_size": cfg.N_TEST,
}


def run_experiment_a():
    device = cfg.get_device()
    print("=" * 72)
    print("THỰC NGHIỆM A: KHẢO SÁT ẢNH HƯỞNG CỦA SỐ MẪU HUẤN LUYỆN (A)")
    print("=" * 72)
    print(f"Thiết bị chạy (Device) : {device}")
    print(f"Các mức số mẫu A       : {cfg.A_VALUES}")
    print(f"Các random seeds       : {cfg.TRAIN_SEEDS}")
    print(f"Cấu hình MLP Baseline  : {BASELINE_CONFIG['num_layers']} tầng ẩn x "
          f"{BASELINE_CONFIG['hidden_dim']} nơ-ron ({BASELINE_CONFIG['activation']})")
    print(f"Siêu tham số huấn luyện: batch_size={BASELINE_CONFIG['batch_size']}, "
          f"epochs={BASELINE_CONFIG['epochs']}, lr={BASELINE_CONFIG['lr']}, "
          f"optimizer={BASELINE_CONFIG['optimizer']}")
    print(f"Điều kiện dừng         : {BASELINE_CONFIG['stopping_criterion']}")
    print("=" * 72)

    # Lưu file cấu hình thực nghiệm A
    config_path = SCRIPT_DIR / "config_A.json"
    config_path.write_text(json.dumps(BASELINE_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")

    run_records = []
    loss_records = []
    best_overall_val = float("inf")
    best_overall_state = None
    best_overall_info = {}

    # Lưu mô hình của seed đầu tiên cho A=200 và A=2000 để vẽ biểu đồ trực quan bề mặt xấp xỉ
    saved_models_for_plot = {}

    for a in cfg.A_VALUES:
        for seed in cfg.TRAIN_SEEDS:
            # 1. Thiết lập seed tái lập cho PyTorch, NumPy, Random
            cfg.set_seed(seed)

            # 2. Tạo DataLoader (lấy A mẫu đầu tiên của train.csv, xáo trộn batch theo seed)
            train_loader, val_loader, test_loader = get_all_loaders(
                batch_size=BASELINE_CONFIG["batch_size"],
                train_size=a,
                seed=seed,
            )
            # Tạo loader đánh giá trên tập train (không shuffle để tính chính xác sai số)
            train_eval_loader = get_dataloader(
                "train",
                batch_size=BASELINE_CONFIG["batch_size"],
                train_size=a,
                shuffle=False,
                seed=seed,
            )

            # 3. Khởi tạo mô hình MLP của TV2
            model = MLPRegressor(
                input_dim=BASELINE_CONFIG["input_dim"],
                hidden_dim=BASELINE_CONFIG["hidden_dim"],
                num_layers=BASELINE_CONFIG["num_layers"],
                output_dim=BASELINE_CONFIG["output_dim"],
                activation=nn.ReLU,
            )

            # 4. Huấn luyện mô hình thông qua hàm train_model của TV2
            t0 = time.perf_counter()
            trained_model, train_losses, val_losses = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                epochs=BASELINE_CONFIG["epochs"],
                lr=BASELINE_CONFIG["lr"],
                device=device,
            )
            elapsed = time.perf_counter() - t0

            best_epoch = int(np.argmin(val_losses)) + 1
            best_val_loss = float(np.min(val_losses))

            # 5. Đánh giá mô hình (đã nạp checkpoint tốt nhất theo val_loss) trên Train, Val, Test
            train_metrics, _ = evaluate_model(trained_model, train_eval_loader, device=device)
            val_metrics, _ = evaluate_model(trained_model, val_loader, device=device)
            test_metrics, _ = evaluate_model(trained_model, test_loader, device=device)

            # Ghi nhận lịch sử loss theo từng epoch
            for ep_idx, (tr_l, va_l) in enumerate(zip(train_losses, val_losses), start=1):
                loss_records.append({
                    "A": a,
                    "seed": seed,
                    "epoch": ep_idx,
                    "train_loss": tr_l,
                    "val_loss": va_l,
                })

            record = {
                "A": a,
                "seed": seed,
                "hidden_dim": BASELINE_CONFIG["hidden_dim"],
                "num_layers": BASELINE_CONFIG["num_layers"],
                "activation": BASELINE_CONFIG["activation"],
                "batch_size": BASELINE_CONFIG["batch_size"],
                "epochs": BASELINE_CONFIG["epochs"],
                "lr": BASELINE_CONFIG["lr"],
                "optimizer": BASELINE_CONFIG["optimizer"],
                "best_epoch": best_epoch,
                "train_time_sec": elapsed,
                "train_MSE": train_metrics["MSE"],
                "train_RMSE": train_metrics["RMSE"],
                "train_MAE": train_metrics["MAE"],
                "val_MSE": val_metrics["MSE"],
                "val_RMSE": val_metrics["RMSE"],
                "val_MAE": val_metrics["MAE"],
                "test_MSE": test_metrics["MSE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_MAE": test_metrics["MAE"],
            }
            run_records.append(record)

            if seed == cfg.TRAIN_SEEDS[0] and a in (200, 1000, 2000):
                saved_models_for_plot[a] = {
                    k: v.cpu().clone() for k, v in trained_model.state_dict().items()
                }

            if val_metrics["MSE"] < best_overall_val:
                best_overall_val = val_metrics["MSE"]
                best_overall_state = {
                    k: v.cpu().clone() for k, v in trained_model.state_dict().items()
                }
                best_overall_info = record.copy()

            print(
                f"[A={a:4d} | seed={seed}] "
                f"best_ep={best_epoch:3d} | "
                f"Train MSE={train_metrics['MSE']:.6f} | "
                f"Val MSE={val_metrics['MSE']:.6f} | "
                f"Test MSE={test_metrics['MSE']:.6f}, "
                f"RMSE={test_metrics['RMSE']:.6f}, "
                f"MAE={test_metrics['MAE']:.6f} "
                f"({elapsed:.2f}s)"
            )

    # Lưu log chi tiết từng lần chạy (30 runs) và lịch sử loss
    df_runs = pd.DataFrame(run_records)
    runs_csv_path = SCRIPT_DIR / "results_A_runs.csv"
    df_runs.to_csv(runs_csv_path, index=False, float_format="%.8f")

    df_loss = pd.DataFrame(loss_records)
    loss_csv_path = SCRIPT_DIR / "loss_history_A.csv"
    df_loss.to_csv(loss_csv_path, index=False, float_format="%.8f")

    # Lưu checkpoint tốt nhất toàn cục của Thực nghiệm A
    ckpt_dir = SCRIPT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if best_overall_state is not None:
        torch.save(
            {
                "config": BASELINE_CONFIG,
                "best_run_info": best_overall_info,
                "model_state_dict": best_overall_state,
            },
            ckpt_dir / "best_model_A.pt",
        )
    if saved_models_for_plot:
        torch.save(saved_models_for_plot, ckpt_dir / "models_A_plot.pt")

    # Tổng hợp bảng kết quả theo từng mức A (Trung bình & Độ lệch chuẩn qua 3 seeds)
    summary_rows = []
    for a, grp in df_runs.groupby("A", as_index=False):
        row = {
            "A": int(a),
            "n_seeds": len(grp),
            "train_MSE_mean": grp["train_MSE"].mean(),
            "train_MSE_std": grp["train_MSE"].std(ddof=1),
            "val_MSE_mean": grp["val_MSE"].mean(),
            "val_MSE_std": grp["val_MSE"].std(ddof=1),
            "test_MSE_mean": grp["test_MSE"].mean(),
            "test_MSE_std": grp["test_MSE"].std(ddof=1),
            "test_RMSE_mean": grp["test_RMSE"].mean(),
            "test_RMSE_std": grp["test_RMSE"].std(ddof=1),
            "test_MAE_mean": grp["test_MAE"].mean(),
            "test_MAE_std": grp["test_MAE"].std(ddof=1),
            "best_epoch_mean": grp["best_epoch"].mean(),
            "train_time_sec_mean": grp["train_time_sec"].mean(),
            "test_MSE_formatted": f"{grp['test_MSE'].mean():.6f} ± {grp['test_MSE'].std(ddof=1):.6f}",
            "test_RMSE_formatted": f"{grp['test_RMSE'].mean():.6f} ± {grp['test_RMSE'].std(ddof=1):.6f}",
            "test_MAE_formatted": f"{grp['test_MAE'].mean():.6f} ± {grp['test_MAE'].std(ddof=1):.6f}",
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    results_csv_path = SCRIPT_DIR / "results_A.csv"
    df_summary.to_csv(results_csv_path, index=False, float_format="%.8f")

    print("\n" + "=" * 72)
    print("BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM A (TRUNG BÌNH ± ĐỘ LỆCH CHUẨN QUA 3 SEEDS)")
    print("=" * 72)
    display_cols = [
        "A",
        "train_MSE_mean",
        "val_MSE_mean",
        "test_MSE_formatted",
        "test_RMSE_formatted",
        "test_MAE_formatted",
    ]
    print(df_summary[display_cols].to_string(index=False))
    print("=" * 72)
    print(f"Đã lưu kết quả tổng hợp tại : {results_csv_path}")
    print(f"Đã lưu kết quả chi tiết tại : {runs_csv_path}")
    print(f"Đã lưu lịch sử loss tại     : {loss_csv_path}")
    print(f"Đã lưu checkpoint tốt nhất  : {ckpt_dir / 'best_model_A.pt'}")

    return df_summary, df_runs, df_loss, saved_models_for_plot


# ==============================================================================
# 2. VẼ ĐỒ THỊ PHÂN TÍCH CHO THỰC NGHIỆM A
# ==============================================================================
def plot_experiment_a(df_summary, df_runs, df_loss, saved_models_for_plot):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 200,
    })

    a_vals = df_summary["A"].to_numpy()

    # ------------------------------------------------------------------
    # Hình A1: Đồ thị sai số trên tập Test (MSE, RMSE, MAE) theo số mẫu A
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    metrics_info = [
        ("test_MSE", "Test MSE", "tab:blue", "Sai số bình phương trung bình (MSE)"),
        ("test_RMSE", "Test RMSE", "tab:red", "Căn bậc hai sai số bình phương (RMSE)"),
        ("test_MAE", "Test MAE", "tab:green", "Sai số tuyệt đối trung bình (MAE)"),
    ]

    for ax, (col_prefix, label, color, title) in zip(axes, metrics_info):
        mean_v = df_summary[f"{col_prefix}_mean"].to_numpy()
        std_v = df_summary[f"{col_prefix}_std"].to_numpy()

        ax.plot(a_vals, mean_v, marker="o", linewidth=2, color=color, label=f"{label} (Mean)")
        ax.fill_between(
            a_vals,
            np.maximum(0, mean_v - std_v),
            mean_v + std_v,
            color=color,
            alpha=0.2,
            label="±1 Std (3 seeds)",
        )

        # Chọn mức A tốt nhất dựa trên Validation MSE trung bình (không dùng Test để chọn cấu hình)
        best_idx = int(np.argmin(df_summary["val_MSE_mean"].to_numpy()))
        ax.scatter(
            [a_vals[best_idx]],
            [mean_v[best_idx]],
            color="gold",
            edgecolor="black",
            s=90,
            zorder=5,
            label=f"Tốt nhất theo Val (A={a_vals[best_idx]})",
        )

        ax.set_title(title)
        ax.set_xlabel("Số mẫu huấn luyện A")
        ax.set_ylabel(label)
        ax.set_xticks(a_vals)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("Thực nghiệm A: Ảnh hưởng của số mẫu huấn luyện (A) đến sai số trên tập Test", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = cfg.FIG_DIR / "fig_A1_error_vs_samples.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {p1}")

    # ------------------------------------------------------------------
    # Hình A2: So sánh MSE trên Train / Validation / Test theo A
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5))

    splits_plot = [
        ("train_MSE", "Train MSE", "tab:green", "s", "--"),
        ("val_MSE", "Validation MSE", "tab:orange", "^", "-."),
        ("test_MSE", "Test MSE", "tab:red", "o", "-"),
    ]

    for prefix, label, color, marker, ls in splits_plot:
        m = df_summary[f"{prefix}_mean"].to_numpy()
        s = df_summary[f"{prefix}_std"].to_numpy()
        ax1.plot(a_vals, m, marker=marker, linestyle=ls, linewidth=2, color=color, label=label)
        ax1.fill_between(a_vals, np.maximum(1e-7, m - s), m + s, color=color, alpha=0.15)

        ax2.plot(a_vals, m, marker=marker, linestyle=ls, linewidth=2, color=color, label=label)
        ax2.fill_between(a_vals, np.maximum(1e-6, m - s), m + s, color=color, alpha=0.15)

    ax1.set_title("So sánh MSE trên Train / Val / Test (Thang tuyến tính)")
    ax1.set_xlabel("Số mẫu huấn luyện A")
    ax1.set_ylabel("MSE")
    ax1.set_xticks(a_vals)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_yscale("log")
    ax2.set_title("So sánh MSE trên Train / Val / Test (Thang Log)")
    ax2.set_xlabel("Số mẫu huấn luyện A")
    ax2.set_ylabel("MSE (log scale)")
    ax2.set_xticks(a_vals)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    fig.suptitle("Thực nghiệm A: Khoảng cách tổng quát hóa (Generalization Gap) theo số mẫu A", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = cfg.FIG_DIR / "fig_A2_train_val_test_mse.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {p2}")

    # ------------------------------------------------------------------
    # Hình A3: Đường cong hội tụ Loss theo Epoch ở các mức A đại diện
    # ------------------------------------------------------------------
    rep_A = [200, 600, 1200, 2000]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3), sharey=True)
    for ax, a_val in zip(axes, rep_A):
        sub = df_loss[df_loss["A"] == a_val]
        grp = sub.groupby("epoch", as_index=False).agg(
            tr_mean=("train_loss", "mean"),
            tr_min=("train_loss", "min"),
            tr_max=("train_loss", "max"),
            va_mean=("val_loss", "mean"),
            va_min=("val_loss", "min"),
            va_max=("val_loss", "max"),
        )
        ep = grp["epoch"].to_numpy()
        ax.plot(ep, grp["tr_mean"], label="Train Loss (Mean)", color="tab:green", linewidth=1.6)
        ax.fill_between(ep, grp["tr_min"], grp["tr_max"], color="tab:green", alpha=0.15)
        ax.plot(ep, grp["va_mean"], label="Val Loss (Mean)", color="tab:orange", linewidth=1.6)
        ax.fill_between(ep, grp["va_min"], grp["va_max"], color="tab:orange", alpha=0.2, label="Dải Min–Max (3 seeds)")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 3e-2)
        ax.set_title(f"A = {a_val} mẫu")
        ax.set_xlabel("Epoch")
        if a_val == rep_A[0]:
            ax.set_ylabel("MSE Loss (log scale)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Thực nghiệm A: Quá trình hội tụ Train & Validation Loss theo Epoch (Trung bình 3 seeds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p3 = cfg.FIG_DIR / "fig_A3_learning_curves.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {p3}")

    # ------------------------------------------------------------------
    # Hình A4: So sánh bề mặt xấp xỉ và Scatter y_true vs y_pred (A=200 vs A=2000)
    # ------------------------------------------------------------------
    if 200 in saved_models_for_plot and 2000 in saved_models_for_plot:
        g1 = np.linspace(*cfg.X1_RANGE, 80)
        g2 = np.linspace(*cfg.X2_RANGE, 80)
        G1, G2 = np.meshgrid(g1, g2)
        grid_pts = np.column_stack([G1.ravel(), G2.ravel()])
        Z_true = gaussian_2d(grid_pts).reshape(G1.shape)

        x_test, y_test = load_split("test")
        x_test_tensor = torch.as_tensor(x_test.copy(), dtype=torch.float32)
        grid_tensor = torch.as_tensor(grid_pts.copy(), dtype=torch.float32)

        preds_surface = {}
        preds_test = {}
        for a_key in (200, 2000):
            m = MLPRegressor(
                input_dim=BASELINE_CONFIG["input_dim"],
                hidden_dim=BASELINE_CONFIG["hidden_dim"],
                num_layers=BASELINE_CONFIG["num_layers"],
                output_dim=BASELINE_CONFIG["output_dim"],
                activation=nn.ReLU,
            )
            m.load_state_dict(saved_models_for_plot[a_key])
            m.eval()
            with torch.no_grad():
                preds_surface[a_key] = m(grid_tensor).numpy().reshape(G1.shape)
                preds_test[a_key] = m(x_test_tensor).numpy().ravel()

        fig = plt.figure(figsize=(18.5, 4.8))

        # Cột 1: Bề mặt hàm Gaussian thật
        ax1 = fig.add_subplot(1, 4, 1, projection="3d")
        ax1.plot_surface(G1, G2, Z_true, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
        ax1.set_title("Bề mặt Gaussian thật $f(x_1, x_2)$", pad=10)
        ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.set_zlabel("$y$")
        ax1.set_zlim(0, 0.28)

        # Cột 2: Bề mặt xấp xỉ với A = 200
        ax2 = fig.add_subplot(1, 4, 2, projection="3d")
        ax2.plot_surface(G1, G2, preds_surface[200], cmap="plasma", linewidth=0, antialiased=True, alpha=0.9)
        ax2.set_title("Bề mặt MLP xấp xỉ (A = 200)", pad=10)
        ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$"); ax2.set_zlabel("$\\hat{y}$")
        ax2.set_zlim(0, 0.28)

        # Cột 3: Bề mặt xấp xỉ với A = 2000
        ax3 = fig.add_subplot(1, 4, 3, projection="3d")
        ax3.plot_surface(G1, G2, preds_surface[2000], cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
        ax3.set_title("Bề mặt MLP xấp xỉ (A = 2000)", pad=10)
        ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$"); ax3.set_zlabel("$\\hat{y}$")
        ax3.set_zlim(0, 0.28)

        # Cột 4: Tương quan y_true vs y_pred trên tập Test
        ax4 = fig.add_subplot(1, 4, 4)
        max_y = max(y_test.max(), preds_test[200].max(), preds_test[2000].max()) * 1.05
        ax4.plot([0, max_y], [0, max_y], "k--", linewidth=1.2, label="Lý tưởng ($\\hat{y} = y$)")
        ax4.scatter(y_test, preds_test[200], color="tab:orange", alpha=0.7, s=22, label="A = 200 (seed=0)")
        ax4.scatter(y_test, preds_test[2000], color="tab:blue", alpha=0.8, s=22, marker="^", label="A = 2000 (seed=0)")
        ax4.set_title("Tương quan $y$ thật và $\\hat{y}$ dự đoán (Test)")
        ax4.set_xlabel("Giá trị thật $y$")
        ax4.set_ylabel("Giá trị dự đoán $\\hat{y}$")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper left", fontsize=8.5)

        fig.suptitle("Thực nghiệm A: Minh họa chất lượng xấp xỉ hàm khi ít mẫu (A=200) và nhiều mẫu (A=2000)", fontsize=13, y=1.02)
        fig.subplots_adjust(left=0.03, right=0.98, wspace=0.34, top=0.86, bottom=0.12)
        p4 = cfg.FIG_DIR / "fig_A4_pred_vs_true_surface.png"
        fig.savefig(p4, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Đã lưu biểu đồ: {p4}")


def main():
    if "--plot-only" in sys.argv and (SCRIPT_DIR / "results_A.csv").exists():
        df_summary = pd.read_csv(SCRIPT_DIR / "results_A.csv")
        df_runs = pd.read_csv(SCRIPT_DIR / "results_A_runs.csv")
        df_loss = pd.read_csv(SCRIPT_DIR / "loss_history_A.csv")
        plot_ckpt = SCRIPT_DIR / "checkpoints" / "models_A_plot.pt"
        if plot_ckpt.exists():
            saved_models_for_plot = torch.load(plot_ckpt, map_location="cpu")
        else:
            device = cfg.get_device()
            saved_models_for_plot = {}
            for a_val in (200, 2000):
                cfg.set_seed(cfg.TRAIN_SEEDS[0])
                tr_l, va_l, _ = get_all_loaders(BASELINE_CONFIG["batch_size"], train_size=a_val, seed=cfg.TRAIN_SEEDS[0])
                m = MLPRegressor(
                    input_dim=BASELINE_CONFIG["input_dim"],
                    hidden_dim=BASELINE_CONFIG["hidden_dim"],
                    num_layers=BASELINE_CONFIG["num_layers"],
                    output_dim=BASELINE_CONFIG["output_dim"],
                    activation=nn.ReLU,
                )
                m, _, _ = train_model(tr_l, va_l, m, epochs=BASELINE_CONFIG["epochs"], lr=BASELINE_CONFIG["lr"], device=device)
                saved_models_for_plot[a_val] = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            (SCRIPT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(saved_models_for_plot, plot_ckpt)
        plot_experiment_a(df_summary, df_runs, df_loss, saved_models_for_plot)
        return

    df_summary, df_runs, df_loss, saved_models_for_plot = run_experiment_a()
    plot_experiment_a(df_summary, df_runs, df_loss, saved_models_for_plot)


if __name__ == "__main__":
    main()
