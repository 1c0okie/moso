#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
svm_predict_moso.py
- Đọc và huấn luyện SVM từ moso_clean.csv
- Lưu model & scaler
- Cho phép nhập dữ liệu mới và dự đoán giá (tỷ VND)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


MODEL_PATH = Path("svm_moso_model.joblib")
SCALER_PATH = Path("svm_moso_scaler.joblib")


def train_and_save_model(csv_path="moso_clean.csv"):
    df = pd.read_csv(csv_path)
    # Lấy các cột chính
    X = df[["Diện tích sử dụng", "Diện tích đất", "Số ngày từ đăng"]]
    y = df["Giá (VND)"] / 1e9  # tỷ VND

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # pipeline gồm chuẩn hóa + SVR
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=50, epsilon=0.2, gamma="scale"))
    ])

    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("📊 Đánh giá mô hình SVM:")
    print(f"  R² Score : {r2:.3f}")
    print(f"  MAE      : {mae:.3f} tỷ VND")
    print(f"  RMSE     : {rmse:.3f} tỷ VND")

    # Vẽ scatter so sánh thực tế & dự đoán
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, s=20, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Giá thực tế (tỷ VND)")
    plt.ylabel("Giá dự đoán (tỷ VND)")
    plt.title("SVM - So sánh Giá dự đoán vs Thực tế")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("svm_pred_scatter.png", dpi=180)
    plt.show()

    # Lưu mô hình
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Đã lưu mô hình tại: {MODEL_PATH}")

def load_and_predict():
    if not MODEL_PATH.exists():
        print("❌ Chưa có mô hình. Hãy train trước bằng cách chạy lại file.")
        return

    model = joblib.load(MODEL_PATH)

    print("\n=== Dự đoán giá BĐS mới ===")
    try:
        area_use = float(input("Diện tích sử dụng (m²): "))
        area_land = float(input("Diện tích đất (m²): "))
        age_days = float(input("Số ngày từ đăng: "))
    except ValueError:
        print("⚠️ Dữ liệu nhập không hợp lệ.")
        return

    # ✅ Chuyển thành DataFrame có tên cột
    X_new = pd.DataFrame(
        [[area_use, area_land, age_days]],
        columns=["Diện tích sử dụng", "Diện tích đất", "Số ngày từ đăng"]
    )

    y_pred = model.predict(X_new)[0]
    print(f"💰 Giá dự đoán: {y_pred:.2f} tỷ VND")



def main():
    print("=== MÔ HÌNH SVM CHO MOSO ===")
    if not MODEL_PATH.exists():
        print("🔧 Chưa có mô hình — tiến hành huấn luyện...")
        train_and_save_model()
    else:
        print("✅ Đã có mô hình — dùng để dự đoán.")
    load_and_predict()


if __name__ == "__main__":
    main()
