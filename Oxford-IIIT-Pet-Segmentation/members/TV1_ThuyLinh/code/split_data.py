"""Chia tập train/val/test (stratified theo 37 giống) và kiểm tra rò rỉ dữ liệu.

Chạy từ terminal (ở bất kỳ thư mục nào bên trong dự án):
    python members/TV1_ThuyLinh/code/split_data.py
    python members/TV1_ThuyLinh/code/split_data.py --no-wandb
    python members/TV1_ThuyLinh/code/split_data.py --entity linhthuynguyen2603

Đầu vào : data/raw/, data/splits/split_config.json, data/splits/excluded.txt
Đầu ra  : data/splits/train.txt, val.txt, test.txt (+ cập nhật excluded.txt
          nếu phát hiện thêm ảnh trùng nội dung) và W&B Run R4_split_preprocess.
"""
import argparse
import hashlib
import json
import os
import sys

from PIL import Image
from sklearn.model_selection import train_test_split

try:
    from .preprocessing import find_raw_dirs
except ImportError:
    from preprocessing import find_raw_dirs


def project_root(start=None):
    """Đi ngược lên đến thư mục có cả data/ và members/ (gốc dự án)."""
    cur = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    while True:
        if (os.path.isdir(os.path.join(cur, "data"))
                and os.path.isdir(os.path.join(cur, "members"))):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


def breed_of(name):
    """Abyssinian_12.jpg -> Abyssinian ; american_bulldog_5.jpg -> american_bulldog"""
    return os.path.splitext(name)[0].rsplit("_", 1)[0]


def list_valid_images(raw_dir="data/raw", excluded_path=None):
    """Ảnh hợp lệ = có đủ cặp ảnh-trimap và không nằm trong excluded.txt."""
    images_dir, trimaps_dir = find_raw_dirs(raw_dir)

    excluded = read_excluded(excluded_path)

    names = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith(".jpg") and not f.startswith("._")
    )
    valid = [
        n for n in names
        if n not in excluded
        and os.path.exists(
            os.path.join(trimaps_dir, os.path.splitext(n)[0] + ".png")
        )
    ]
    return valid, images_dir


def read_excluded(excluded_path):
    if excluded_path and os.path.exists(excluded_path):
        with open(excluded_path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def _content_hash(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return hashlib.md5(im.tobytes() + str(im.size).encode()).hexdigest()


def compute_hashes(names, images_dir):
    return {n: _content_hash(os.path.join(images_dir, n)) for n in names}


def drop_content_duplicates(valid, hashes):
    """Giữ ảnh đứng trước (theo thứ tự tên), loại các bản trùng nội dung."""
    seen, kept, dups = {}, [], []
    for n in valid:
        h = hashes[n]
        if h in seen:
            dups.append({
                "file": n,
                "duplicate_of": seen[h],
                "same_breed": breed_of(n) == breed_of(seen[h]),
            })
        else:
            seen[h] = n
            kept.append(n)
    return kept, dups


def split_stratified(names, ratios, seed):
    """Chia 70/15/15 theo tỷ lệ trong config, stratify theo giống, seed cố định."""
    y = [breed_of(n) for n in names]
    rest = ratios["val"] + ratios["test"]

    train, tmp, _, y_tmp = train_test_split(
        names, y, test_size=rest, stratify=y, random_state=seed
    )
    val, test = train_test_split(
        tmp, test_size=ratios["test"] / rest, stratify=y_tmp, random_state=seed
    )
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def check_leakage(splits, n_valid, hashes=None):
    """Giao của từng cặp tập, mỗi giống có mặt ở cả 3 tập,
    và (nếu có hashes) trùng nội dung ảnh giữa các tập."""
    sets = {k: set(v) for k, v in splits.items()}

    rep = {
        "overlap_train_val": len(sets["train"] & sets["val"]),
        "overlap_train_test": len(sets["train"] & sets["test"]),
        "overlap_val_test": len(sets["val"] & sets["test"]),
        "total_matches_valid": sum(len(v) for v in splits.values()) == n_valid,
    }

    breeds = {k: {breed_of(n) for n in v} for k, v in splits.items()}
    rep["n_breeds_total"] = len(set.union(*breeds.values()))
    rep["n_breeds_in_all_splits"] = len(set.intersection(*breeds.values()))

    if hashes is not None:
        hs = {k: {hashes[n] for n in v} for k, v in splits.items()}
        rep["content_overlap_train_val"] = len(hs["train"] & hs["val"])
        rep["content_overlap_train_test"] = len(hs["train"] & hs["test"])
        rep["content_overlap_val_test"] = len(hs["val"] & hs["test"])

    zero_keys = [
        k for k in rep if k.startswith(("overlap_", "content_overlap_"))
    ]
    rep["leakage_pass"] = bool(
        all(rep[k] == 0 for k in zero_keys)
        and rep["total_matches_valid"]
        and rep["n_breeds_in_all_splits"] == rep["n_breeds_total"]
    )
    return rep


def failed_checks(report):
    """Trả về dict các kiểm tra thất bại (rỗng nếu đạt)."""
    failed = {
        k: v for k, v in report.items()
        if k.startswith(("overlap_", "content_overlap_")) and v != 0
    }
    if report["n_breeds_in_all_splits"] != report["n_breeds_total"]:
        failed["breeds_in_all_splits"] = (
            f"{report['n_breeds_in_all_splits']}/{report['n_breeds_total']}"
        )
    if not report["total_matches_valid"]:
        failed["total_matches_valid"] = False
    return failed


def run_split(config, raw_dir="data/raw",
              excluded_path="data/splits/excluded.txt",
              splits_dir="data/splits", check_content=True, dedupe=True):
    """Chia tập, ghi train.txt / val.txt / test.txt và trả về báo cáo kiểm tra."""
    valid, images_dir = list_valid_images(raw_dir, excluded_path)

    hashes = compute_hashes(valid, images_dir) if (check_content or dedupe) else {}

    extra_dups = []
    if dedupe:
        valid, extra_dups = drop_content_duplicates(valid, hashes)
        if extra_dups and excluded_path:
            excluded = read_excluded(excluded_path) | {d["file"] for d in extra_dups}
            with open(excluded_path, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(excluded)))

    splits = split_stratified(valid, config["split_ratio"], config["seed"])

    os.makedirs(splits_dir, exist_ok=True)
    for name, items in splits.items():
        with open(os.path.join(splits_dir, f"{name}.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(items))

    report = check_leakage(
        splits, len(valid), hashes if check_content else None
    )
    report.update({
        "n_valid": len(valid),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "extra_duplicates": extra_dups,
    })

    all_breeds = sorted({breed_of(n) for n in valid})
    report["breed_counts"] = [
        {
            "breed": b,
            "train": sum(breed_of(n) == b for n in splits["train"]),
            "val": sum(breed_of(n) == b for n in splits["val"]),
            "test": sum(breed_of(n) == b for n in splits["test"]),
        }
        for b in all_breeds
    ]
    return report


def log_wandb(report, config, entity, project, splits_dir="data/splits"):
    """W&B Run 04: R4_split_preprocess."""
    import pandas as pd
    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        name="R4_split_preprocess",
        job_type="split",
        tags=["TV1", "split", "preprocess",
              f"data_version:{config['data_version']}"],
        config={
            **config,
            "log_id": "P004",
            "work_item": "Split & Preprocessing",
            "target_file": "split_data.py, preprocessing.py",
            "handoff_status": "Pending_Gate2",
            "mask_interpolation": "nearest",
            "image_interpolation": "bilinear",
            "resize_mode": "direct_no_padding",
        },
        notes=(
            "Chia lại 70/15/15 stratified theo 37 giống, seed=42, không dùng "
            "split chính thức vì chỉ so sánh nội bộ E0-E4 và E0, E3 học từ đầu "
            "cần nhiều dữ liệu train."
        ),
        reinit="finish_previous",
    )

    overlap_total = (report["overlap_train_val"]
                     + report["overlap_train_test"]
                     + report["overlap_val_test"])

    run.log({
        "n_valid": report["n_valid"],
        "n_train": report["n_train"],
        "n_val": report["n_val"],
        "n_test": report["n_test"],
        "overlap_count": overlap_total,
        "overlap_train_val": report["overlap_train_val"],
        "overlap_train_test": report["overlap_train_test"],
        "overlap_val_test": report["overlap_val_test"],
        "content_overlap_total": (
            report.get("content_overlap_train_val", 0)
            + report.get("content_overlap_train_test", 0)
            + report.get("content_overlap_val_test", 0)
        ),
        "extra_content_duplicates_removed": len(report["extra_duplicates"]),
        "n_breeds_in_all_splits": report["n_breeds_in_all_splits"],
        "breed_split_table": wandb.Table(
            dataframe=pd.DataFrame(report["breed_counts"])
        ),
    })

    if report["extra_duplicates"]:
        run.log({
            "extra_duplicates_table": wandb.Table(
                dataframe=pd.DataFrame(report["extra_duplicates"]).astype(str)
            )
        })

    run.summary.update({
        "leakage_check": "PASS" if report["leakage_pass"] else "FAIL",
        "decision": "ACCEPT",
        "reason": ("Giao của các tập bằng rỗng, mỗi giống có mặt ở cả 3 tập, "
                   "không trùng nội dung ảnh giữa các tập."),
    })

    # Chỉ 3 file danh sách nhẹ (+ file cấu hình seed/split)
    artifact = wandb.Artifact(
        name="oxford-pet-splits",
        type="dataset",
        description="Danh sách chia tập train/val/test (stratified 37 giống, seed=42)",
    )
    for fname in ["train.txt", "val.txt", "test.txt", "split_config.json"]:
        artifact.add_file(os.path.join(splits_dir, fname))
    run.log_artifact(artifact)

    run.finish()
    print("[W&B] Hoàn thành Run: R4_split_preprocess")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Chia tập + kiểm tra rò rỉ (TV1)")
    ap.add_argument("--config", default="data/splits/split_config.json")
    ap.add_argument("--entity", default=os.environ.get(
        "WANDB_ENTITY", "linhthuynguyen2603-school"))
    ap.add_argument("--project", default="Oxford-IIIT-Pet-Segmentation")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Không ghi nhật ký lên W&B")
    args = ap.parse_args()

    os.chdir(project_root())
    print("Thư mục gốc dự án:", os.getcwd())

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    report = run_split(config)

    brief = {k: v for k, v in report.items()
             if k not in ("breed_counts", "extra_duplicates")}
    print(json.dumps(brief, indent=2, ensure_ascii=False))

    dups = report["extra_duplicates"]
    print(f"Ảnh trùng nội dung đã loại thêm: {len(dups)}")
    for d in dups[:20]:
        print("  ", d)
    if any(not d["same_breed"] for d in dups):
        print("CẢNH BÁO: có ảnh trùng nhưng khác giống (xung đột nhãn), nên xem tay.")

    failed = failed_checks(report)
    if failed:
        print("KIỂM TRA RÒ RỈ THẤT BẠI:", failed)
        sys.exit(1)

    print(f"[Bước 7] Train/Val/Test = {report['n_train']}/{report['n_val']}/"
          f"{report['n_test']} (tổng {report['n_valid']}), Overlap = 0, "
          f"{report['n_breeds_in_all_splits']}/{report['n_breeds_total']} "
          f"giống có mặt ở cả 3 tập")

    if not args.no_wandb:
        log_wandb(report, config, args.entity, args.project)


if __name__ == "__main__":
    main()
