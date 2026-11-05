"""Truc quan hoa du lieu (TV1): be mat that, scatter 3D, phan bo."""
import matplotlib
matplotlib.use("Agg")           # chay duoc ca khi khong co man hinh
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from data import gaussian_2d, load_split

PEAK = float(gaussian_2d(cfg.MU)[0])   # f(mu): gia tri lon nhat cua ham

def _save(fig, name):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Da luu", cfg.FIG_DIR / name)


def _grid(n=100):
    g1 = np.linspace(*cfg.X1_RANGE, n)
    g2 = np.linspace(*cfg.X2_RANGE, n)
    G1, G2 = np.meshgrid(g1, g2)
    Z = gaussian_2d(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
    return G1, G2, Z

def plot_true_surface():
    G1, G2, Z = _grid()
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(G1, G2, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$y = f(x_1, x_2)$")
    ax.set_title("Be mat Gaussian that")
    fig.colorbar(surf, shrink=0.6, pad=0.1, label="f(x)")
    _save(fig, "fig1_true_surface.png")

def plot_scatter3d():
    fig = plt.figure(figsize=(16, 5))
    for i, (name, color, label) in enumerate([("train", "tab:green", "Train (2000)"),
                                              ("val", "tab:orange", "Validation (200)"),
                                              ("test", "tab:red", "Test (100)")], start=1):
        x, y = load_split(name)
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], y, c=color, s=6 if name == "train" else 12, label=label)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("y")
        ax.set_zlim(0, 1.05 * PEAK)   # chung 1 truc z cho 3 tap, tinh tu dinh that cua f
        ax.set_title(label); ax.legend(loc="upper left")
    _save(fig, "fig2_scatter3d_splits.png")

def plot_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cols = {"train": "tab:green", "val": "tab:orange", "test": "tab:red"}
    for name, c in cols.items():
        x, y = load_split(name)
        for ax, v in zip(axes, (x[:, 0], x[:, 1], y)):
            ax.hist(v, bins=25, density=True, alpha=0.45, color=c, label=name)
    for ax, t, xl in zip(axes, ("Phan bo $x_1$", "Phan bo $x_2$", "Phan bo nhan $y$"), ("$x_1$", "$x_2$", "y")):
        ax.set_title(t); ax.set_xlabel(xl); ax.set_ylabel("Mat do"); ax.legend()
    _save(fig, "fig3_distributions.png")

def plot_contour_with_points():
    G1, G2, Z = _grid()
    x, _ = load_split("train")
    xt, _ = load_split("test")
    fig, ax = plt.subplots(figsize=(5.5, 8))
    cs = ax.contourf(G1, G2, Z, levels=20, cmap="viridis")
    fig.colorbar(cs, label="f(x)")
    ax.scatter(x[:, 0], x[:, 1], s=3, c="white", alpha=0.5, label="train")
    ax.scatter(xt[:, 0], xt[:, 1], s=18, c="red", edgecolor="k", linewidth=0.4, label="test")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Duong muc f(x) va vi tri mau")
    ax.legend(loc="upper right")
    _save(fig, "fig4_contour_samples.png")

def main():
    plot_true_surface()
    plot_scatter3d()
    plot_distributions()
    plot_contour_with_points()


if __name__ == "__main__":
    main()
