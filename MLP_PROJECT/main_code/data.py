"""Sinh, luu, nap va kiem tra du lieu Gaussian 2 chieu (TV1)."""
import json
import numpy as np
import pandas as pd

import config as cfg
def gaussian_2d(x, mu=cfg.MU, cov=cfg.C):
    """Tinh f(x). x co shape (N, 2) hoac (2,). Tra ve mang shape (N,)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    d = x - mu                                   # (N, 2)
    cov_inv = np.linalg.inv(cov)
    quad = np.einsum("ni,ij,nj->n", d, cov_inv, d)   # (x-mu)^T C^-1 (x-mu) tung mau
    norm = 2.0 * np.pi * np.sqrt(np.linalg.det(cov))
    return np.exp(-0.5 * quad) / norm


# ----------------------------------------------------------------------
# 2. Sinh du lieu: x lay DEU trong hinh chu nhat, nhan y = f(x)
# ----------------------------------------------------------------------
def generate_dataset(n_samples, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(cfg.X1_RANGE[0], cfg.X1_RANGE[1], n_samples)
    x2 = rng.uniform(cfg.X2_RANGE[0], cfg.X2_RANGE[1], n_samples)  # mien rieng cua x2!
    x = np.column_stack([x1, x2])
    y = gaussian_2d(x)
    return x, y


# ----------------------------------------------------------------------
# 3. Luu / nap CSV
# ----------------------------------------------------------------------
def save_csv(x, y, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})
    df.to_csv(path, index=False)  # giu day du do chinh xac float64


def load_csv(path):
    df = pd.read_csv(path)
    return df[["x1", "x2"]].to_numpy(dtype=np.float64), df["y"].to_numpy(dtype=np.float64)


def load_split(name):
    """name in {'train', 'val', 'test'} -> (x, y). TV2-TV5 chi can ham nay."""
    return load_csv(cfg.DATA_DIR / f"{name}.csv")


def get_train_subset(a):
    """Lay A mau DAU TIEN cua train pool => cac tap con long nhau, so sanh cong bang."""
    x, y = load_split("train")
    if a > len(x):
        raise ValueError(f"A={a} vuot qua kich thuoc train pool {len(x)}")
    return x[:a], y[:a]


# ----------------------------------------------------------------------
# 4. Kiem tra du lieu
# ----------------------------------------------------------------------
def validate_splits():
    """Kiem tra mien, nhan, NaN, doi chieu scipy, va ro ri giua cac tap."""
    from scipy.stats import multivariate_normal

    splits = {s: load_split(s) for s in ("train", "val", "test")}
    sizes = {"train": cfg.N_TRAIN_POOL, "val": cfg.N_VAL, "test": cfg.N_TEST}

    assert np.all(np.linalg.eigvalsh(cfg.C) > 0), "C phai xac dinh duong"
    for name, (x, y) in splits.items():
        assert len(x) == sizes[name], f"{name}: sai so mau"
        assert not (np.isnan(x).any() or np.isnan(y).any()), f"{name}: co NaN"
        assert x[:, 0].min() >= cfg.X1_RANGE[0] and x[:, 0].max() <= cfg.X1_RANGE[1], f"{name}: x1 ngoai mien"
        assert x[:, 1].min() >= cfg.X2_RANGE[0] and x[:, 1].max() <= cfg.X2_RANGE[1], f"{name}: x2 ngoai mien"
        assert y.min() >= 0.0 and y.max() <= 1.0, f"{name}: y ngoai [0,1]"
        ref = multivariate_normal(mean=cfg.MU, cov=cfg.C).pdf(x)
        assert np.allclose(y, ref, atol=1e-12), f"{name}: khac scipy"

    def rows(x):
        return {tuple(r) for r in np.round(x, 10)}
    r_tr, r_va, r_te = (rows(splits[s][0]) for s in ("train", "val", "test"))
    assert not (r_tr & r_va) and not (r_tr & r_te) and not (r_va & r_te), "Ro ri: trung mau giua cac tap"
    print("[OK] Kiem tra du lieu: dung mien, dung nhan (khop scipy), khong NaN, khong trung mau giua cac tap.")


def describe_splits():
    rows = []
    for s in ("train", "val", "test"):
        x, y = load_split(s)
        for col, v in (("x1", x[:, 0]), ("x2", x[:, 1]), ("y", y)):
            rows.append({"tap": s, "bien": col, "n": len(v), "mean": v.mean(), "std": v.std(ddof=1),
                         "min": v.min(), "median": np.median(v), "max": v.max()})
    df = pd.DataFrame(rows)
    out = cfg.DATA_DIR / "data_statistics.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    return df


# ----------------------------------------------------------------------
# 5. Chay: python data.py  -> sinh toan bo du lieu, metadata, thong ke
# ----------------------------------------------------------------------
def main():
    plan = [("train", cfg.N_TRAIN_POOL, cfg.SEED_TRAIN),
            ("val", cfg.N_VAL, cfg.SEED_VAL),
            ("test", cfg.N_TEST, cfg.SEED_TEST)]
    for name, n, seed in plan:
        x, y = generate_dataset(n, seed)
        save_csv(x, y, cfg.DATA_DIR / f"{name}.csv")
        print(f"Da luu {name}.csv: {n} mau, seed={seed}")

    meta = {
        "mu": cfg.MU.tolist(), "C": cfg.C.tolist(),
        "det_C": float(np.linalg.det(cfg.C)),
        "peak_f": float(gaussian_2d(cfg.MU)[0]),
        "x1_range": cfg.X1_RANGE, "x2_range": cfg.X2_RANGE,
        "sampling": "x1, x2 doc lap, phan phoi deu trong hinh chu nhat; y = f(x1, x2)",
        "splits": {n: {"n": k, "seed": s, "file": f"{n}.csv"} for n, k, s in plan},
        "A_values": cfg.A_VALUES,
        "note": "Thi nghiem A dung A mau DAU TIEN cua train.csv (tap con long nhau).",
    }
    (cfg.DATA_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    validate_splits()
    print(describe_splits().to_string(index=False))


if __name__ == "__main__":
    main()
