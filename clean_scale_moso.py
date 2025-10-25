#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean + Scale (HCM only) + short report
- IQR -> NaN (giữ nguyên số dòng)
- Impute missing (median/mean/none) cho cột số (mặc định: median)
- Chuẩn hoá Z-score & MinMax
- Chuẩn hoá địa chỉ TP.HCM
"""

import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ====== Helpers số ======
def to_float(x):
    s = str(x).strip() if x is not None else ""
    if s == "" or s.lower() in {"nan","null","none"}: return np.nan
    s = re.sub(r"(m2|m²|㎡)", "", s, flags=re.I).replace(" ", "")
    if re.search(r"\d+,\d+$", s): s = s.replace(",", ".")
    s = re.sub(r"(?<=\d)[\.,](?=\d{3}(\D|$))", "", s)  # 1.234.567 -> 1234567
    try: return float(s)
    except: return np.nan

def parse_price_to_vnd(v):
    s = str(v).lower().strip() if v is not None else ""
    if s == "" or "thỏa thuận" in s or "thoa thuan" in s: return np.nan
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(t[yỷ]|tỷ|ty)", s)
    if m:
        n = float(m.group(1).replace(",", "."))
        val = n * 1_000_000_000
        tail = re.search(r"ty\s*([0-9]{2,3})\b", s)
        if tail: val += float(tail.group(1)) * 1_000_000
        return val
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(triệu|tr|million)", s)
    if m: return float(m.group(1).replace(",", ".")) * 1_000_000
    n = to_float(s)
    if n and n <= 200 and "đ" not in s and "vnd" not in s: return n * 1_000_000_000
    return n or np.nan

def parse_age_days(x):
    s = str(x).strip() if x is not None else ""
    if s == "": return np.nan
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt): return np.nan
    return (pd.Timestamp.today().normalize() - pd.Timestamp(dt)).days

def iqr_mask(series, k=1.5):
    s = pd.to_numeric(series, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return s, (np.nan, np.nan), 0
    lo, hi = q1 - k*iqr, q3 + k*iqr
    before = s.isna().sum()
    s = s.mask((s < lo) | (s > hi))
    after = s.isna().sum()
    return s, (lo, hi), int(after - before)

def scale_keep_nan(col, scaler_cls):
    s = pd.to_numeric(col, errors="coerce")
    mask = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if mask.any():
        sc = scaler_cls().fit(s[mask].to_numpy().reshape(-1,1))
        out.loc[mask] = sc.transform(s[mask].to_numpy().reshape(-1,1)).ravel()
    return out

def fnum(x):
    return "nan" if pd.isna(x) else f"{float(x):.1f}"

# ====== Helpers địa chỉ (chỉ TP.HCM) ======
HCM_PAT = re.compile(
    r"(tp\.?\s*h(?:ồ)?\s*chí\s*minh|thành\s*phố\s*h(?:ồ)?\s*chí\s*minh|hcmc?|s[àa]i\s*g[òo]n)",
    flags=re.I
)

def normalize_addr_hcm(s):
    """Đưa về dạng đồng nhất, bỏ token tỉnh/thành khác; HCM được cố định."""
    if s is None: return ""
    t = re.sub(r"[;/|]+", ",", str(s).strip())
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r",\s*,+", ", ", t)
    return t

def norm_quan_token(x: str) -> str:
    """Chuẩn Quận/Huyện/TP Thủ Đức, tránh 'Quận uận 3'."""
    if not x: return ""
    s = x.strip()
    if re.search(r"(?i)\b(tp\.?\s*)?th(ủ|u)\s*đ(ứ|u)c\b", s): return "TP Thủ Đức"
    s = re.sub(r"(?i)\b(?:h\.?|huyen)\s+(?=\S)", "Huyện ", s)
    s = re.sub(r"(?i)(?<!\w)q\.?\s*(?=(\d{1,2})\b)", "Quận ", s)
    s = re.sub(r"(?i)\bquan\s+(\d{1,2})\b", r"Quận \1", s)
    s = re.sub(r"(?i)\bquận\s+(\d{1,2})\b", r"Quận \1", s)
    s = re.sub(r"(?i)\bquận\s+quận\s+(\d{1,2})\b", r"Quận \1", s)
    return s.strip()

def norm_phuong_token(x: str) -> str:
    """Chuẩn Phường/Xã/Thị trấn (P.6 -> Phường 6, TT. -> Thị trấn, x./xa -> Xã)."""
    if not x: return ""
    s = x.strip()
    s = re.sub(r"(?i)\btt\.?\b", "Thị trấn", s)
    s = re.sub(r"(?i)(?<!\w)p\.?\s*(\d{1,2})\b", r"Phường \1", s)
    s = re.sub(r"(?i)\bphuong\s+(\d{1,2})\b", r"Phường \1", s)
    s = re.sub(r"(?i)\bphường\s+(\d{1,2})\b", r"Phường \1", s)
    s = re.sub(r"(?i)\bx\.\s*(\S+)", r"Xã \1", s)
    s = re.sub(r"(?i)\bxa\b", "Xã", s)
    return s.strip()

def split_hcm(addr):
    """
    Tách từ phải sang trái cho dữ liệu HCM:
      ... , Phường/Xã/TT , (Quận|Huyện|TP Thủ Đức) , [HCM]
    Còn lại bên trái -> Đường/Số nhà. Luôn cố định 'Hồ Chí Minh'.
    """
    norm = normalize_addr_hcm(addr)
    if norm == "":
        return ("Hồ Chí Minh", "", "", "", "")

    tokens = [t.strip() for t in norm.split(",") if t.strip()]

    # Loại token tỉnh/thành khác; bỏ token HCM vì đã cố định
    filtered = []
    for t in tokens:
        if HCM_PAT.search(t):  # biến thể HCM
            continue
        if re.search(r"(?i)\btỉnh|thành\s*phố|tp\.", t) and not HCM_PAT.search(t):
            continue
        filtered.append(t)
    tokens = filtered

    tinh = "Hồ Chí Minh"
    quan = phuong = duong = ""

    # Bắt quận/huyện/TP Thủ Đức từ cuối
    for i in range(len(tokens) - 1, -1, -1):
        cand = norm_quan_token(tokens[i])
        if re.search(r"^(Quận\s+\d{1,2}|Huyện\s+\S+|TP Thủ Đức)$", cand, flags=re.I):
            quan = cand; tokens.pop(i); break

    # Bắt phường/xã/thị trấn từ cuối
    for i in range(len(tokens) - 1, -1, -1):
        cand = norm_phuong_token(tokens[i])
        if re.search(r"^(Phường\s+\S+|Xã\s+\S+|Thị trấn\s+\S*|Phường\s+\d{1,2})$", cand, flags=re.I):
            phuong = cand; tokens.pop(i); break

    if tokens: duong = ", ".join(tokens)

    return (tinh, (quan or "").strip(), (phuong or "").strip(), (duong or "").strip(), norm)

# ====== Main ======
def main():
    ap = argparse.ArgumentParser(description="Clean + Scale (HCM only) + short report")
    ap.add_argument("--input", default="moso_api_data.csv")
    ap.add_argument("--out-csv", default="moso_clean.csv")
    ap.add_argument("--out-report", default="moso_clean_report.txt")
    ap.add_argument("--iqr", type=float, default=1.5)
    ap.add_argument("--impute", choices=["none", "median", "mean"], default="median",
                    help="Điền giá trị thiếu cho cột số sau IQR (mặc định: median)")
    ap.add_argument("--col-price", default="Giá")
    ap.add_argument("--col-dtsud", default="Diện tích sử dụng")
    ap.add_argument("--col-dtdat", default="Diện tích đất")
    ap.add_argument("--col-date",  default="Ngày đăng")
    ap.add_argument("--col-addr",  default="Địa chỉ")
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists(): raise SystemExit(f"Không tìm thấy file: {p}")

    df = pd.read_csv(p)
    n0 = len(df)
    print(f"📂 {n0} dòng | File: {p.name}")

    # ==== Chuẩn hoá số ====
    df["Giá (VND)"]          = df.get(args.col_price, pd.Series(dtype=object)).apply(parse_price_to_vnd)
    df["Diện tích sử dụng"]  = df.get(args.col_dtsud, pd.Series(dtype=object)).apply(to_float)
    df["Diện tích đất"]      = df.get(args.col_dtdat, pd.Series(dtype=object)).apply(to_float)
    df["Số ngày từ đăng"]    = df.get(args.col_date,  pd.Series(dtype=object)).apply(parse_age_days)

    cols = ["Giá (VND)","Diện tích sử dụng","Diện tích đất","Số ngày từ đăng"]
    stats = {}
    print("🧹 IQR → NaN:")
    for c in cols:
        s,(lo,hi),masked = iqr_mask(df[c], k=args.iqr)
        df[c] = s
        stats[c] = dict(masked=masked, nan=int(s.isna().sum()), lo=lo, hi=hi,
                        med=s.median(skipna=True), mn=s.min(skipna=True),
                        mx=s.max(skipna=True), mean=s.mean(skipna=True))
        print(f"  • {c:<18} masked={masked:<4} NaN={stats[c]['nan']:<5} "
              f"Clip=[{fnum(lo)}, {fnum(hi)}] Med={fnum(stats[c]['med'])}")

    # ==== Điền giá trị thiếu cho cột số (chuẩn cho SVM) ====
    if args.impute != "none":
        print(f"🩹 Impute missing: {args.impute}")
        for c in cols:
            before_nan = int(df[c].isna().sum())
            if args.impute == "median":
                fill_val = df[c].median(skipna=True)
            else:  # mean
                fill_val = df[c].mean(skipna=True)
            df[c] = df[c].fillna(fill_val)
            after_nan = int(df[c].isna().sum())
            print(f"  • {c:<18} filled {before_nan - after_nan} NaN -> {args.impute}={fnum(fill_val)}")

    # ==== Scale ====
    print("⚙️ Scale:")
    for c in cols:
        df[f"{c}_std"] = scale_keep_nan(df[c], StandardScaler)
        df[f"{c}_mm"]  = scale_keep_nan(df[c], MinMaxScaler)
        print(f"  • {c} -> _std, _mm")

    # ==== Địa chỉ HCM ====
    print("📍 Địa chỉ:")
    addr = df.get(args.col_addr, pd.Series(dtype=object))
    parsed = addr.apply(split_hcm)
    df["Tỉnh/TP"]        = parsed.apply(lambda x: x[0])
    df["Quận/Huyện/TP"]  = parsed.apply(lambda x: x[1] or np.nan)
    df["Phường/Xã/TT"]   = parsed.apply(lambda x: x[2] or np.nan)
    df["Đường/Số nhà"]   = parsed.apply(lambda x: x[3] or np.nan)
    df["Địa chỉ (chuẩn)"] = parsed.apply(lambda x: x[4] or np.nan)

    top_quan   = df["Quận/Huyện/TP"].dropna().astype(str).value_counts().head(5)
    top_phuong = df["Phường/Xã/TT"].dropna().astype(str).value_counts().head(5)
    print(f"  • Bản ghi có địa chỉ chuẩn: {int(df['Địa chỉ (chuẩn)'].notna().sum())}/{n0}")
    if not top_quan.empty:
        print("  • Top Quận/Huyện/TP:")
        for k,v in top_quan.items(): print(f"     - {k}: {v}")
    if not top_phuong.empty:
        print("  • Top Phường/Xã/TT:")
        for k,v in top_phuong.items(): print(f"     - {k}: {v}")

    # ==== Lưu & Báo cáo ====
    out_csv = Path(args.out_csv); df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ CSV: {out_csv}")

    lines = [
        f"Input : {p.name}",
        f"Output: {out_csv.name}",
        f"Tổng dòng: {n0}",
        f"IQR: {args.iqr}",
        f"Imputation: {args.impute}",
        "",
        "I. Số liệu sau IQR:",
    ]
    for c,st in stats.items():
        lines.append(
            f"- {c} | masked={st['masked']} NaN={st['nan']} "
            f"| Clip=[{fnum(st['lo'])}, {fnum(st['hi'])}] "
            f"| Min/Med/Mean/Max={fnum(st['mn'])}/{fnum(st['med'])}/{fnum(st['mean'])}/{fnum(st['mx'])}"
        )
    lines += ["", "II. Địa chỉ (TP. Hồ Chí Minh):",
              f"- Có địa chỉ chuẩn: {int(df['Địa chỉ (chuẩn)'].notna().sum())}/{n0}",
              "- Top Quận/Huyện/TP:"] + \
             ([f"  + {k}: {v}" for k,v in top_quan.items()] or ["  (trống)"]) + \
             ["- Top Phường/Xã/TT:"] + \
             ([f"  + {k}: {v}" for k,v in top_phuong.items()] or ["  (trống)"])

    out_report = Path(args.out_report)
    with open(out_report, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    print(f"📄 Report: {out_report}")

if __name__ == "__main__":
    main()
