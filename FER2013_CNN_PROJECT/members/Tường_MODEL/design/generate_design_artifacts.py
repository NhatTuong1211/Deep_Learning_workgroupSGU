"""Sinh bảng tensor shape (CSV + XLSX) và sơ đồ kiến trúc model_architecture.png (trắng-đen 400 DPI) từ code thật."""

import csv
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    from src.models.model_summary import collect_layer_table, save_summary_artifacts
except ImportError:
    from cnn_model import build_model
    from model_config import get_default_config
    from model_summary import collect_layer_table, save_summary_artifacts


def _col_letter(idx: int) -> str:
    """Chuyển chỉ số cột 0-based sang ký tự Excel (A, B, ...)."""
    res = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        res = chr(65 + rem) + res
    return res


def write_minimal_xlsx(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    """Ghi file .xlsx chuẩn OpenXML bằng thư viện chuẩn Python (không cần openpyxl)."""
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
    """Vẽ sơ đồ kiến trúc FER2013CNN nền trắng, khung đen, chữ đen (400 DPI) khớp 100% code."""
    cfg = model.config
    h, w = cfg.input_size

    boxes: List[Tuple[str, str]] = [("Input", f"[B, {cfg.in_channels}, {h}, {w}]")]
    cur_h, cur_w = h, w
    for idx, out_ch in enumerate(cfg.conv_channels, start=1):
        cur_h //= cfg.pool_size
        cur_w //= cfg.pool_size
        boxes.append((f"Block {idx} ({out_ch})", f"[B, {out_ch}, {cur_h}, {cur_w}]"))

    boxes.append(("Flatten + FC1", f"{model.flatten_dim} → {cfg.fc_hidden_dim}"))
    boxes.append(("Raw Logits", f"[B, {cfg.num_classes}]"))

    n = len(boxes)
    box_w = 2.15
    box_h = 1.15
    gap = 0.45
    margin_x = 0.25
    margin_y = 0.25

    total_w = 2 * margin_x + n * box_w + (n - 1) * gap
    total_h = 2 * margin_y + box_h

    fig, ax = plt.subplots(figsize=(total_w, total_h), dpi=400)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (title, sub) in enumerate(boxes):
        x = margin_x + i * (box_w + gap)
        y = margin_y
        rect = Rectangle(
            (x, y),
            box_w,
            box_h,
            facecolor="#FFFFFF",
            edgecolor="#000000",
            linewidth=1.6,
        )
        ax.add_patch(rect)

        cx = x + box_w / 2
        cy = y + box_h / 2
        ax.text(
            cx,
            cy + 0.16,
            title,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#000000",
            family="DejaVu Sans",
        )
        ax.text(
            cx,
            cy - 0.20,
            sub,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="normal",
            color="#000000",
            family="DejaVu Sans",
        )

        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap - 0.04, cy),
                xytext=(x + box_w + 0.04, cy),
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.22,head_length=0.32",
                    color="#000000",
                    lw=1.6,
                ),
            )

    out_path = design_dir / "model_architecture.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.12, facecolor="white")
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
        RESULTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dest_arch = RESULTS_MODEL_DIR / "model_architecture.png"
        shutil.copy2(arch_path, dest_arch)
        print(f"Copied architecture diagram to: {dest_arch}")
