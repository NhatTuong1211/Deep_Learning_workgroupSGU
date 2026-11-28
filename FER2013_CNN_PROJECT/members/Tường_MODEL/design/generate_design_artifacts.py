"""Sinh bảng tensor shape (CSV + XLSX) và sơ đồ kiến trúc model_architecture.png từ code thật."""

import csv
import sys
import zipfile
from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve()
TV2_DIR = HERE.parent.parent
# Tự động nhận diện đang chạy trong <project_root>/members/TV2_MODEL/design/ hay Tường_MODEL/design/
if TV2_DIR.parent.name == "members" and (TV2_DIR.parent.parent / "src" / "models").exists():
    PROJECT_ROOT = TV2_DIR.parent.parent
    RESULTS_MODEL_DIR = PROJECT_ROOT / "results" / "model"
    CODE_DIR = PROJECT_ROOT / "src" / "models"
else:
    PROJECT_ROOT = TV2_DIR
    RESULTS_MODEL_DIR = None
    CODE_DIR = TV2_DIR / "code"

for p in (PROJECT_ROOT, CODE_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from src.models.cnn_model import build_model
    from src.models.model_config import get_default_config
    from src.models.model_summary import collect_layer_table, count_parameters, save_summary_artifacts
except ImportError:
    from cnn_model import build_model
    from model_config import get_default_config
    from model_summary import collect_layer_table, count_parameters, save_summary_artifacts


def _col_letter(idx: int) -> str:
    """Chuyển chỉ số cột 0-based sang ký tự Excel (A, B, ...)."""
    res = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        res = chr(65 + rem) + res
    return res


def write_minimal_xlsx(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    """Ghi file .xlsx chuẩn OpenXML bằng thư viện chuẩn Python."""
    all_rows = [headers] + rows
    sheet_rows_xml = []
    for r_idx, row in enumerate(all_rows, start=1):
        cells_xml = []
        for c_idx, val in enumerate(row):
            ref = f"{_col_letter(c_idx)}{r_idx}"
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                cells_xml.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                text = escape(str(val))
                cells_xml.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows_xml.append(f'<row r="{r_idx}">{"".join(cells_xml)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows_xml)}</sheetData>'
        "</worksheet>"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="TensorShapes" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )

    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def export_tensor_tables(rows: List[Dict[str, object]], design_dir: Path) -> None:
    """Xuất bảng tensor shape ra cả CSV và XLSX."""
    headers = [
        "STT",
        "Layer Name",
        "Layer Type",
        "Input Shape",
        "Output Shape",
        "Parameters",
        "Trainable Parameters",
        "Layer Details",
    ]
    data_rows = []
    for idx, r in enumerate(rows, start=1):
        data_rows.append(
            [
                idx,
                r["layer_name"],
                r["layer_type"],
                r["input_shape"],
                r["output_shape"],
                int(r["params"]),
                int(r["trainable_params"]),
                r["config"],
            ]
        )

    csv_path = design_dir / "tensor_shape_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data_rows)

    xlsx_path = design_dir / "tensor_shape_table.xlsx"
    write_minimal_xlsx(xlsx_path, headers, data_rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")


def draw_architecture_diagram(model, design_dir: Path) -> Path:
    """Vẽ sơ đồ kiến trúc FER2013CNN khớp 100% với code."""
    counts = count_parameters(model)
    cfg = model.config
    feat_params = sum(p.numel() for p in model.features.parameters())
    clf_params = sum(p.numel() for p in model.classifier.parameters())
    last_conv_name, _ = model.get_last_conv_layer()

    fig, ax = plt.subplots(figsize=(18, 7.2), dpi=220)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Tiêu đề
    ax.text(
        9.0,
        6.82,
        f"FER2013CNN ({cfg.model_version}) — PyTorch Architecture & Tensor Flow",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color="#1B2A4A",
    )
    ax.text(
        9.0,
        6.40,
        f"Total Parameters: {counts['total_params']:,}  |  "
        f"Feature Extractor: {feat_params:,}  |  Classifier: {clf_params:,}  |  "
        f"Last Conv2d: {last_conv_name}",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#4A5568",
    )

    # Khung nhóm Feature Extractor & Classifier
    feat_group = FancyBboxPatch(
        (2.45, 1.35),
        10.25,
        4.65,
        boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor="#F0F6FC",
        edgecolor="#90B4CE",
        linewidth=1.3,
        linestyle="--",
    )
    ax.add_patch(feat_group)
    ax.text(
        7.57,
        5.80,
        "Feature Extractor (4 ConvBlocks — 8 Conv2d layers)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#1D4E89",
    )

    clf_group = FancyBboxPatch(
        (13.00, 1.35),
        4.75,
        4.65,
        boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor="#FFF8EE",
        edgecolor="#E6A96B",
        linewidth=1.3,
        linestyle="--",
    )
    ax.add_patch(clf_group)
    ax.text(
        15.37,
        5.80,
        "Classifier (FC Head — Raw Logits)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#9C4A1A",
    )

    # Định nghĩa các hộp khối
    block_params = [
        sum(p.numel() for p in getattr(model.features, f"block{i}").parameters())
        for i in range(1, len(cfg.conv_channels) + 1)
    ]

    boxes = [
        {
            "x": 0.25,
            "w": 1.85,
            "title": "Input Image",
            "body": "FER2013\nGrayscale\n48 × 48 × 1\nNorm(0.5, 0.5)",
            "shape": "[B, 1, 48, 48]",
            "footer": "float32",
            "fc": "#E2E8F0",
            "ec": "#475569",
            "tc": "#1E293B",
        },
        {
            "x": 2.65,
            "w": 2.20,
            "title": "Block 1 (32)",
            "body": "2× Conv2d(3×3, p=1)\n+ BatchNorm2d\n+ ReLU\nMaxPool2d(2×2)\nDropout2d(0.25)",
            "shape": "[B, 32, 24, 24]",
            "footer": f"{block_params[0]:,} params",
            "fc": "#DBEAFE",
            "ec": "#2563EB",
            "tc": "#1E3A8A",
        },
        {
            "x": 5.20,
            "w": 2.20,
            "title": "Block 2 (64)",
            "body": "2× Conv2d(3×3, p=1)\n+ BatchNorm2d\n+ ReLU\nMaxPool2d(2×2)\nDropout2d(0.25)",
            "shape": "[B, 64, 12, 12]",
            "footer": f"{block_params[1]:,} params",
            "fc": "#DBEAFE",
            "ec": "#2563EB",
            "tc": "#1E3A8A",
        },
        {
            "x": 7.75,
            "w": 2.20,
            "title": "Block 3 (128)",
            "body": "2× Conv2d(3×3, p=1)\n+ BatchNorm2d\n+ ReLU\nMaxPool2d(2×2)\nDropout2d(0.25)",
            "shape": "[B, 128, 6, 6]",
            "footer": f"{block_params[2]:,} params",
            "fc": "#BFDBFE",
            "ec": "#1D4ED8",
            "tc": "#1E3A8A",
        },
        {
            "x": 10.30,
            "w": 2.20,
            "title": "Block 4 (256)",
            "body": "2× Conv2d(3×3, p=1)\n+ BatchNorm2d\n+ ReLU\nMaxPool2d(2×2)\nDropout2d(0.25)",
            "shape": "[B, 256, 3, 3]",
            "footer": f"{block_params[3]:,} params",
            "fc": "#93C5FD",
            "ec": "#1E40AF",
            "tc": "#1E3A8A",
        },
        {
            "x": 13.20,
            "w": 2.10,
            "title": "Flatten + FC1",
            "body": f"Flatten ({model.flatten_dim})\nLinear({model.flatten_dim} → {cfg.fc_hidden_dim})\nReLU\nDropout({cfg.dropout_fc})",
            "shape": f"[B, {cfg.fc_hidden_dim}]",
            "footer": f"{sum(p.numel() for p in model.classifier.fc1.parameters()):,} params",
            "fc": "#FED7AA",
            "ec": "#C2410C",
            "tc": "#7C2D12",
        },
        {
            "x": 15.65,
            "w": 1.95,
            "title": "Output Logits",
            "body": f"Linear({cfg.fc_hidden_dim} → {cfg.num_classes})\n\nNo Softmax\n(CrossEntropyLoss)",
            "shape": f"[B, {cfg.num_classes}]",
            "footer": f"{sum(p.numel() for p in model.classifier.fc_out.parameters()):,} params",
            "fc": "#D1FAE5",
            "ec": "#047857",
            "tc": "#064E3B",
        },
    ]

    y_bottom = 2.05
    h_box = 3.35

    for idx, b in enumerate(boxes):
        rect = FancyBboxPatch(
            (b["x"], y_bottom),
            b["w"],
            h_box,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=b["fc"],
            edgecolor=b["ec"],
            linewidth=1.6,
        )
        ax.add_patch(rect)

        cx = b["x"] + b["w"] / 2
        ax.text(
            cx,
            y_bottom + h_box - 0.32,
            b["title"],
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=b["tc"],
        )
        ax.text(
            cx,
            y_bottom + h_box / 2 - 0.05,
            b["body"],
            ha="center",
            va="center",
            fontsize=8.3,
            color="#1F2937",
            linespacing=1.45,
        )
        ax.text(
            cx,
            y_bottom + 0.25,
            b["footer"],
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=b["tc"],
        )

        # Tensor shape bên dưới mỗi khối
        ax.text(
            cx,
            y_bottom - 0.34,
            b["shape"],
            ha="center",
            va="center",
            fontsize=8.5,
            fontfamily="monospace",
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#CBD5E1"),
        )

        # Mũi tên sang khối tiếp theo
        if idx < len(boxes) - 1:
            x_start = b["x"] + b["w"] + 0.06
            x_end = boxes[idx + 1]["x"] - 0.06
            y_mid = y_bottom + h_box / 2
            ax.annotate(
                "",
                xy=(x_end, y_mid),
                xytext=(x_start, y_mid),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.8),
            )

    # Chú thích mapping 7 lớp ở đáy ảnh
    ax.text(
        9.0,
        0.55,
        "7 Emotion Classes Mapping (Assignment Standard):  "
        "0: Angry  |  1: Disgust  |  2: Fear  |  3: Happy  |  4: Sad  |  5: Surprise  |  6: Neutral",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    out_path = design_dir / "model_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    model = build_model(get_default_config())
    design_dir = TV2_DIR / "design"
    design_dir.mkdir(parents=True, exist_ok=True)

    if RESULTS_MODEL_DIR is not None:
        save_summary_artifacts(model)  # Tự lưu vào results/model/ và members/TV2_MODEL/outputs/
    else:
        save_summary_artifacts(model, TV2_DIR / "outputs")

    rows = collect_layer_table(model)
    export_tensor_tables(rows, design_dir)
    arch_path = draw_architecture_diagram(model, design_dir)

    if RESULTS_MODEL_DIR is not None:
        import shutil
        RESULTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dest_arch = RESULTS_MODEL_DIR / "model_architecture.png"
        shutil.copy2(arch_path, dest_arch)
        print(f"Copied architecture diagram to: {dest_arch}")
