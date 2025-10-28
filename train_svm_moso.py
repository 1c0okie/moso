#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_svm_moso.py
- Đọc moso_clean.csv
- Tiền xử lý (impute missing, scale, one-hot quận)
- (Tuỳ chọn) Biến đổi log1p cho mục tiêu
- Huấn luyện SVR (RBF), đánh giá và trực quan
- Lưu pipeline mô hình để dự đoán tin mới

Chạy:
python train_svm_moso.py --input moso_clean.csv --outdir model_out --log-target
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from matplotlib.backends.backend_pdf import PdfPages


# ----------------------- Utils -----------------------
def fmt_vnd_to_bil(x):
    """Chuyển VND -> tỷ VND cho hiển thị."""
    return x / 1e9


def inverse_fmt_bil_to_vnd(x_bil):
    """Chuyển tỷ VND -> VND (ít dùng)."""
    return x_bil * 1e9


def describe_metrics(y_true_vnd, y_pred_vnd):
    mae = mean_absolute_error(y_true_vnd, y_pred_vnd)
    rmse = np.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))
    r2 = r2_score(y_true_vnd, y_pred_vnd)
    return {
        "MAE_VND": mae,
        "RMSE_VND": rmse,
        "R2": r2,
        "MAE_bil": fmt_vnd_to_bil(mae),
        "RMSE_bil": fmt_vnd_to_bil(rmse),
    }


def plot_parity(ax, y_true_vnd, y_pred_vnd, title="Thật vs Dự đoán (tỷ VND)"):
    yt = fmt_vnd_to_bil(np.asarray(y_true_vnd))
    yp = fmt_vnd_to_bil(np.asarray(y_pred_vnd))
    ax.scatter(yt, yp, s=14, alpha=0.6)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Giá thật (tỷ VND)")
    ax.set_ylabel("Giá dự đoán (tỷ VND)")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def plot_residual_hist(ax, y_true_vnd, y_pred_vnd, bins=50, title="Phân bố sai số (tỷ VND)"):
    resid_bil = fmt_vnd_to_bil(np.asarray(y_pred_vnd) - np.asarray(y_true_vnd))
    ax.hist(resid_bil, bins=bins)
    ax.set_xlabel("Sai số = Dự đoán - Thật (tỷ VND)")
    ax.set_ylabel("Số mẫu")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def plot_residual_scatter(ax, y_pred_vnd, y_true_vnd, title="Sai số theo Giá dự đoán"):
    resid_bil = fmt_vnd_to_bil(np.asarray(y_pred_vnd) - np.asarray(y_true_vnd))
    ypred_bil = fmt_vnd_to_bil(np.asarray(y_pred_vnd))
    ax.scatter(ypred_bil, resid_bil, s=12, alpha=0.6)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Giá dự đoán (tỷ VND)")
    ax.set_ylabel("Sai số (tỷ VND)")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def plot_error_by_district(ax, df_test, y_true_vnd, y_pred_vnd, top=15, agg="mae"):
    """Vẽ lỗi theo Quận/Huyện/TP (Top N theo số mẫu)."""
    tmp = df_test.copy()
    tmp["__y_true"] = y_true_vnd
    tmp["__y_pred"] = y_pred_vnd
    tmp["__abs_err_bil"] = np.abs(tmp["__y_pred"] - tmp["__y_true"]) / 1e9

    # lấy top quận theo số mẫu trong test
    vc = tmp["Quận/Huyện/TP"].astype(str).value_counts().head(top)
    keep = vc.index
    sub = tmp[tmp["Quận/Huyện/TP"].astype(str).isin(keep)].copy()

    if agg == "mae":
        s = sub.groupby("Quận/Huyện/TP")["__abs_err_bil"].mean().sort_values(ascending=False)
        ax.set_title(f"MAE theo quận (tỷ VND) - Top {top}")
        ax.set_xlabel("MAE (tỷ VND)")
    else:
        # RMSE
        s = sub.groupby("Quận/Huyện/TP").apply(
            lambda g: np.sqrt(np.mean(((g["__y_pred"]-g["__y_true"]) / 1e9) ** 2))
        ).sort_values(ascending=False)
        ax.set_title(f"RMSE theo quận (tỷ VND) - Top {top}")
        ax.set_xlabel("RMSE (tỷ VND)")

    ax.barh(s.index, s.values)
    ax.invert_yaxis()
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Train SVR for moso_clean.csv")
    ap.add_argument("--input", default="moso_clean.csv", help="CSV đã chuẩn hoá (clean)")
    ap.add_argument("--outdir", default="model_out", help="Thư mục xuất mô hình & báo cáo")
    ap.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ test split")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--log-target", action="store_true", help="Dùng log1p cho mục tiêu (khuyến nghị)")
    ap.add_argument("--top-district", type=int, default=15, help="Top quận hiển thị trong biểu đồ lỗi")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Không tìm thấy file: {inp}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Đọc dữ liệu
    df = pd.read_csv(inp)

    # 2) Chọn features & target
    feat_num = ["Diện tích sử dụng", "Diện tích đất", "Số ngày từ đăng"]
    # tạo thêm đặc trưng tỉ lệ (nếu có đủ dữ liệu)
    if "Diện tích sử dụng" in df.columns and "Diện tích đất" in df.columns:
        ratio = (pd.to_numeric(df["Diện tích sử dụng"], errors="coerce") /
                 pd.to_numeric(df["Diện tích đất"], errors="coerce"))
        df["Tỷ lệ sử dụng"] = ratio.replace([np.inf, -np.inf], np.nan)
        feat_num.append("Tỷ lệ sử dụng")

    feat_cat = ["Quận/Huyện/TP"]  # có thể thêm "Phường/Xã/TT" nếu muốn
    target_col = "Giá (VND)"

    # 3) Lọc cột tồn tại và tạo X, y
    for c in feat_num:
        if c not in df.columns:
            raise SystemExit(f"Thiếu cột số: {c}")
    for c in feat_cat:
        if c not in df.columns:
            raise SystemExit(f"Thiếu cột phân loại: {c}")
    if target_col not in df.columns:
        raise SystemExit(f"Thiếu cột mục tiêu: {target_col}")

    X = df[feat_num + feat_cat].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")

    # 4) Tách tập train/test (drop mẫu thiếu y)
    valid_mask = np.isfinite(y)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 5) Tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), feat_num),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), feat_cat),
        ],
        remainder="drop"
    )

    # 6) Mô hình SVR (RBF)
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")

    # 7) Pipeline + (tuỳ chọn) biến đổi mục tiêu
    base_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("svr", svr)
    ])

    if args.log_target:
        # log1p/exp1p ổn định đuôi phân phối (giá BĐS lệch phải)
        model = TransformedTargetRegressor(
            regressor=base_pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )
    else:
        model = base_pipeline

    # 8) Huấn luyện
    model.fit(X_train, y_train)

    # 9) Dự đoán & đánh giá (trên VND)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics_train = describe_metrics(y_train, y_pred_train)
    metrics_test = describe_metrics(y_test, y_pred_test)

    # 10) Lưu báo cáo số
    metrics_txt = outdir / "svm_metrics.txt"
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write("=== SVR (RBF) on moso_clean ===\n")
        f.write(f"Input: {inp.name}\n")
        f.write(f"Samples: train={len(X_train)}, test={len(X_test)}\n")
        f.write(f"log-target: {args.log_target}\n\n")
        f.write("Train:\n")
        f.write(f"  MAE  : {metrics_train['MAE_VND']:,.0f} VND (~{metrics_train['MAE_bil']:.3f} tỷ)\n")
        f.write(f"  RMSE : {metrics_train['RMSE_VND']:,.0f} VND (~{metrics_train['RMSE_bil']:.3f} tỷ)\n")
        f.write(f"  R^2  : {metrics_train['R2']:.4f}\n\n")
        f.write("Test:\n")
        f.write(f"  MAE  : {metrics_test['MAE_VND']:,.0f} VND (~{metrics_test['MAE_bil']:.3f} tỷ)\n")
        f.write(f"  RMSE : {metrics_test['RMSE_VND']:,.0f} VND (~{metrics_test['RMSE_bil']:.3f} tỷ)\n")
        f.write(f"  R^2  : {metrics_test['R2']:.4f}\n")
    print(f"📄 Metrics: {metrics_txt}")

    # 11) Lưu dự đoán chi tiết
    pred_df = X_test.copy()
    pred_df["Giá_thật(VND)"] = y_test.values
    pred_df["Giá_dự_đoán(VND)"] = y_pred_test
    pred_df["Sai_số(VND)"] = pred_df["Giá_dự_đoán(VND)"] - pred_df["Giá_thật(VND)"]
    pred_df["Giá_thật(tỷ)"] = fmt_vnd_to_bil(pred_df["Giá_thật(VND)"].values)
    pred_df["Giá_dự_đoán(tỷ)"] = fmt_vnd_to_bil(pred_df["Giá_dự_đoán(VND)"].values)
    pred_df["Sai_số(tỷ)"] = fmt_vnd_to_bil(pred_df["Sai_số(VND)"].values)

    pred_csv = outdir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"🧾 Predictions: {pred_csv}")

    # 12) Trực quan & PDF
    pdf_path = outdir / "svm_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Parity plot
        fig, ax = plt.subplots(figsize=(7.2, 6.6))
        plot_parity(ax, y_test, y_pred_test, title="Thật vs Dự đoán (tập Test)")
        fig.savefig(outdir / "plot_parity.png", dpi=180)
        pdf.savefig(fig); plt.close(fig)

        # Residual histogram
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        plot_residual_hist(ax, y_test, y_pred_test, bins=50)
        fig.savefig(outdir / "plot_residual_hist.png", dpi=180)
        pdf.savefig(fig); plt.close(fig)

        # Residuals vs Predicted
        fig, ax = plt.subplots(figsize=(7.8, 6.0))
        plot_residual_scatter(ax, y_pred_test, y_test)
        fig.savefig(outdir / "plot_residual_vs_pred.png", dpi=180)
        pdf.savefig(fig); plt.close(fig)

        # Error by district (MAE)
        if "Quận/Huyện/TP" in pred_df.columns:
            fig, ax = plt.subplots(figsize=(7.2, 7.8))
            # map lại data test để có quận tương ứng
            test_with_district = X_test.copy()
            test_with_district["Quận/Huyện/TP"] = test_with_district["Quận/Huyện/TP"].astype(str)
            plot_error_by_district(
                ax, test_with_district, y_test.values, y_pred_test,
                top=args.top_district, agg="mae"
            )
            fig.savefig(outdir / "plot_error_by_district.png", dpi=180)
            pdf.savefig(fig); plt.close(fig)

    print(f"📑 PDF: {pdf_path}")

    # 13) Lưu pipeline mô hình (để dự đoán tin mới)
    model_path = outdir / "svm_price_pipeline.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Saved model pipeline: {model_path}")

    # 14) Gợi ý cách dùng mô hình đã lưu
    howto_path = outdir / "README_predict_new.txt"
    with open(howto_path, "w", encoding="utf-8") as f:
        f.write(
            "Dùng mô hình đã lưu để dự đoán tin mới:\n\n"
            "import joblib\n"
            "import pandas as pd\n\n"
            "model = joblib.load('svm_price_pipeline.joblib')\n"
            "# Ví dụ 1 tin mới (chú ý đúng tên cột):\n"
            "new_df = pd.DataFrame([\n"
            "  {\n"
            "    'Diện tích sử dụng': 75,\n"
            "    'Diện tích đất': 80,\n"
            "    'Số ngày từ đăng': 10,\n"
            "    'Tỷ lệ sử dụng': 75/80,\n"
            "    'Quận/Huyện/TP': 'Quận 1'\n"
            "  }\n"
            "])\n"
            "pred_vnd = model.predict(new_df)\n"
            "pred_bil = pred_vnd / 1e9\n"
            "print('Giá dự đoán (VND):', float(pred_vnd[0]))\n"
            "print('Giá dự đoán (tỷ):', float(pred_bil[0]))\n"
        )
    print(f"ℹ️ How-to: {howto_path}")


if __name__ == "__main__":
    main()
