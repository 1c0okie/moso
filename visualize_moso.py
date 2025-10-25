#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_hcm_pretty.py
- Đọc moso_clean.csv (đã chuẩn hoá)
- Bộ biểu đồ đẹp/dễ đọc (đơn vị chuẩn; giá = TỶ VND):
  1) Hist + KDE giá [tỷ VND]
  2) Scatter Giá ↔ Diện tích [tỷ VND, m²]
  3) Top quận theo số lượng tin
  4) Giá trung bình theo quận (top) [tỷ VND]
  5) Scatter Giá ↔ Số ngày từ đăng [tỷ VND, ngày]
  6) Hist tỷ lệ sử dụng/đất
  7) Heatmap 2D Giá ↔ Số ngày (tỷ, ngày)
  8) Heatmap 2D Giá ↔ Diện tích sử dụng (tỷ, m²)
- Xuất PNG + 1 PDF tổng hợp
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

# ---------- Styling ----------
def set_theme(dark=False):
    base_palette = "bright" if not dark else "dark"
    sns.set_theme(
        context="notebook",
        style=("whitegrid" if not dark else "darkgrid"),
        palette=base_palette,
        font_scale=1.12,
    )
    plt.rcParams.update({
        "figure.figsize": (9.8, 5.8),
        "figure.dpi": 130,
        "axes.edgecolor": "#222",
        "axes.linewidth": 1.1,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "savefig.bbox": "tight",
    })

# Màu nổi bật
PALETTE = sns.color_palette("tab10")      # rực, tương phản cao
BAR_GRAD = sns.color_palette("viridis", 9)
CM_HEAT  = "YlOrRd"                      # heatmap 2D: vùng mật độ cao sáng

# ---------- Formatters ----------
fmt_int  = FuncFormatter(lambda x, p: f"{int(x):,}".replace(",", "."))
fmt_bill = FuncFormatter(lambda x, p: f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))  # 1.234,56

def safe_num(df, col):
    return pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")

# ---------- Helpers for 2D heatmap ----------
def _format_interval_labels(edges, decimals=1):
    fmt = lambda v: f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return [f"[{fmt(edges[i])},{fmt(edges[i+1])}]" for i in range(len(edges)-1)]

def _edges_from_percentile(arr: np.ndarray, bins: int, qlow=1.0, qhigh=99.0):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.linspace(0, 1, bins+1)
    lo, hi = np.percentile(arr, [qlow, qhigh])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = arr.min(), (arr.max() if arr.max() > arr.min() else arr.min() + 1.0)
    return np.linspace(lo, hi, bins+1)

def plot_heatmap_2d(x, y, bins, qlow, qhigh, title, xlabel, ylabel, out_png, pdf=None,
                    cmap="inferno", xtick_dec=1, ytick_dec=1):
    xedges = _edges_from_percentile(x, bins, qlow, qhigh)
    yedges = _edges_from_percentile(y, bins, qlow, qhigh)
    H, xe, ye = np.histogram2d(x[np.isfinite(x)], y[np.isfinite(y)], bins=[xedges, yedges])

    fig = plt.figure(figsize=(9.5, 7.2))
    ax = plt.gca()
    mesh = ax.pcolormesh(xe, ye, H.T, cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Tần suất (Frequency)", fontsize=11)

    xcenters = (xe[:-1] + xe[1:]) / 2
    ycenters = (ye[:-1] + ye[1:]) / 2
    ax.set_xticks(xcenters); ax.set_yticks(ycenters)
    ax.set_xticklabels(_format_interval_labels(xe, xtick_dec), rotation=45, ha="right")
    ax.set_yticklabels(_format_interval_labels(ye, ytick_dec))

    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, pad=14, weight="bold")

    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    if pdf: pdf.savefig(fig)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Visualize HCM real estate (pretty charts).")
    ap.add_argument("--input", default="moso_clean.csv", help="CSV đã chuẩn hoá")
    ap.add_argument("--outdir", default="viz_pretty", help="Thư mục xuất ảnh")
    ap.add_argument("--top", type=int, default=10, help="Top N quận hiển thị cho bar chart")
    ap.add_argument("--dark", action="store_true", help="Bật chế độ dark theme")

    # Heatmap 2D #8 (Giá ↔ Diện tích)
    ap.add_argument("--x2d", default="Diện tích sử dụng")
    ap.add_argument("--y2d", default="Giá (VND)")
    ap.add_argument("--x2d-div", type=float, default=1.0)
    ap.add_argument("--y2d-div", type=float, default=1e9)   # mặc định chia 1e9 -> tỷ VND
    ap.add_argument("--bins2d", type=int, default=20)

    # Heatmap 2D #7 (Giá ↔ Số ngày)
    ap.add_argument("--x7", default="Số ngày từ đăng")
    ap.add_argument("--y7", default="Giá (VND)")
    ap.add_argument("--x7-div", type=float, default=1.0)
    ap.add_argument("--y7-div", type=float, default=1e9)    # mặc định chia 1e9 -> tỷ VND
    ap.add_argument("--bins7", type=int, default=20)

    # Cắt biên percentile dùng chung
    ap.add_argument("--clip-low", type=float, default=1.0)
    ap.add_argument("--clip-high", type=float, default=99.0)
    args = ap.parse_args()

    set_theme(dark=args.dark)

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Không tìm thấy file: {inp}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp)

    # ----- Chuẩn hoá dữ liệu -----
    price_vnd = safe_num(df, "Giá (VND)")
    price_bil = price_vnd / 1e9
    area_use  = safe_num(df, "Diện tích sử dụng")
    area_land = safe_num(df, "Diện tích đất")
    age_days  = safe_num(df, "Số ngày từ đăng")
    usage_ratio = (area_use / area_land).replace([np.inf, -np.inf], np.nan)
    district = df.get("Quận/Huyện/TP", pd.Series(dtype=object)).astype("string")

    pdf_path = outdir / "visual_report_pretty.pdf"
    with PdfPages(pdf_path) as pdf:
        # 1) Phân bố giá (Hist + KDE) — fix seaborn >=0.13
        fig, ax = plt.subplots()
        sns.histplot(price_bil.dropna(), bins=50, kde=True, ax=ax, color=PALETTE[0])
        for line in ax.lines:              # chỉnh tay đường KDE
            line.set_linewidth(2)
            line.set_color(PALETTE[3])
        ax.set_title("Phân bố giá bất động sản (tỷ VND)", weight="bold")
        ax.set_xlabel("Giá (tỷ VND)")
        ax.set_ylabel("Số bản ghi")
        ax.xaxis.set_major_formatter(fmt_bill)
        fig.savefig(outdir / "01_price_hist_kde.png", dpi=200)
        pdf.savefig(fig); plt.close(fig)

        # 2) Scatter Giá ↔ Diện tích
        fig, ax = plt.subplots()
        sns.scatterplot(x=area_use, y=price_bil, s=22, alpha=0.65, edgecolor="none",
                        ax=ax, color=PALETTE[1])
        ax.set_title("Giá theo Diện tích sử dụng", weight="bold")
        ax.set_xlabel("Diện tích sử dụng (m²)")
        ax.set_ylabel("Giá (tỷ VND)")
        ax.yaxis.set_major_formatter(fmt_bill)
        fig.savefig(outdir / "02_scatter_area_price.png", dpi=200)
        pdf.savefig(fig); plt.close(fig)

        # 3) Top quận theo số lượng tin (hết FutureWarning & palette cycling)
        vc = district.dropna().value_counts().head(args.top)
        vc_df = vc.rename("count").reset_index()
        vc_df.columns = ["district", "count"]
        colors3 = sns.color_palette("viridis", n_colors=len(vc_df))

        fig, ax = plt.subplots()
        sns.barplot(
            data=vc_df, x="count", y="district",
            hue="district", palette=colors3, legend=False, ax=ax
        )
        ax.set_title(f"Top {args.top} quận có nhiều tin đăng nhất", weight="bold")
        ax.set_xlabel("Số tin đăng (bản ghi)")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(fmt_int)
        for i, v in enumerate(vc_df["count"].values):
            ax.text(v, i, f" {int(v):,}".replace(",", "."), va="center", weight="semibold")
        fig.savefig(outdir / "03_top_district_count.png", dpi=200)
        pdf.savefig(fig); plt.close(fig)

        # 4) Giá trung bình theo quận (Top) [tỷ VND] (hết FutureWarning & palette cycling)
        mean_price_by_d = (df.groupby("Quận/Huyện/TP", dropna=True)["Giá (VND)"]
                           .mean().sort_values(ascending=False) / 1e9)
        s = mean_price_by_d.head(args.top).sort_values()
        s_df = s.rename("price").reset_index()
        s_df.columns = ["district", "price"]
        colors4 = sns.color_palette("viridis", n_colors=len(s_df))

        fig, ax = plt.subplots()
        sns.barplot(
            data=s_df, x="price", y="district",
            hue="district", palette=colors4, legend=False, ax=ax
        )
        ax.set_title(f"Top {args.top} quận có Giá trung bình cao nhất", weight="bold")
        ax.set_xlabel("Giá trung bình (tỷ VND)")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(fmt_bill)
        for i, v in enumerate(s_df["price"].values):
            ax.text(v, i, f" {v:.1f}", va="center", weight="semibold")
        fig.savefig(outdir / "04_top_district_mean_price.png", dpi=200)
        pdf.savefig(fig); plt.close(fig)

        # 5) Scatter Giá ↔ Số ngày
        fig, ax = plt.subplots()
        sns.scatterplot(x=age_days, y=price_bil, s=20, alpha=0.6, edgecolor="none",
                        color=PALETTE[2], ax=ax)
        ax.set_title("Giá theo độ cũ của tin", weight="bold")
        ax.set_xlabel("Số ngày từ đăng (ngày)")
        ax.set_ylabel("Giá (tỷ VND)")
        ax.xaxis.set_major_formatter(fmt_int)
        ax.yaxis.set_major_formatter(fmt_bill)
        fig.savefig(outdir / "05_scatter_age_price.png", dpi=200)
        pdf.savefig(fig); plt.close(fig)

        # 6) Hist Tỷ lệ sử dụng/đất
        fig, ax = plt.subplots()
        sns.histplot(usage_ratio.dropna(), bins=50, ax=ax, color=PALETTE[4])
        ax.set_title("Tỷ lệ Diện tích sử dụng / Diện tích đất", weight="bold")
        ax.set_xlabel("Tỷ lệ sử dụng")
        ax.set_ylabel("Số bản ghi")
        fig.savefig(outdir / "06_hist_usage_ratio.png", dpi=200)
        pdf.savefig(fig); plt.close(fig)

        # 7) Heatmap 2D: Giá ↔ Số ngày (tỷ, ngày)
        x7 = safe_num(df, args.x7).to_numpy(dtype=float) / (args.x7_div or 1.0)
        y7 = safe_num(df, args.y7).to_numpy(dtype=float) / (args.y7_div or 1.0)
        mask7 = np.isfinite(x7) & np.isfinite(y7)
        x7, y7 = x7[mask7], y7[mask7]
        if x7.size > 0 and y7.size > 0:
            plot_heatmap_2d(
                x7, y7, args.bins7, args.clip_low, args.clip_high,
                "Heatmap 2D: Giá (tỷ VND) ↔ Số ngày từ đăng",
                "Số ngày từ đăng (ngày)", "Giá (tỷ VND)",
                outdir / "07_heatmap_2d_age_price.png", pdf, cmap=CM_HEAT
            )

        # 8) Heatmap 2D: Giá ↔ Diện tích (tỷ, m²)
        x2 = safe_num(df, args.x2d).to_numpy(dtype=float) / (args.x2d_div or 1.0)
        y2 = safe_num(df, args.y2d).to_numpy(dtype=float) / (args.y2d_div or 1.0)
        mask2 = np.isfinite(x2) & np.isfinite(y2)
        x2, y2 = x2[mask2], y2[mask2]
        if x2.size > 0 and y2.size > 0:
            plot_heatmap_2d(
                x2, y2, args.bins2d, args.clip_low, args.clip_high,
                "Heatmap 2D: Giá (tỷ VND) ↔ Diện tích sử dụng (m²)",
                "Diện tích sử dụng (m²)", "Giá (tỷ VND)",
                outdir / "08_heatmap_2d_area_price.png", pdf, cmap=CM_HEAT
            )

    print(f"✅ Đã xuất biểu đồ PNG vào: {outdir}")
    print(f"📄 PDF tổng hợp: {pdf_path}")

if __name__ == "__main__":
    main()
