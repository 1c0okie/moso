# crawl_moso_api.py — Thu thập toàn bộ tin đang bán từ MOSO.vn API (kèm Địa chỉ)
import json
import csv
import time
import requests
from copy import deepcopy
from typing import Any, Dict, List

# ===================== CẤU HÌNH =====================
API_URL = "https://moso.vn/api"            # API nội bộ gọi khi duyệt web
REQUEST_JSON_PATH = "request.json"         # Payload JSON lấy từ DevTools
OUTFILE = "moso_api_data.csv"              # CSV xuất ra

FIELDS = [
    "URL", "Giá", "Diện tích sử dụng", "Diện tích đất",
    "Phòng ngủ", "Phòng tắm", "Giấy tờ pháp lý", "Ngày đăng", "Địa chỉ"
]

# ===================== HTTP CLIENT ==================
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://moso.vn",
    "Referer": "https://moso.vn/",
    "X-Requested-With": "XMLHttpRequest",
})

# ===================== ALIAS TRƯỜNG =================
ALIAS = {
    "price":  ["priceText", "price"],
    "usable": ["pArea", "usableArea"],
    "land":   ["pLandArea", "landArea"],
    "bed":    ["pNumberOfBedrooms", "bedrooms"],
    "bath":   ["pNumberOfBathrooms", "bathrooms"],
    "legal":  ["pCertificateType", "publicCertificate"],
    "date":   ["publishedAt", "_createdAt"],
}

# ===================== HÀM HỖ TRỢ ===================
def api_post(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Gửi POST request tới MOSO API và trả về JSON."""
    resp = SESSION.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def format_date_iso_to_ddmmyyyy(iso_str: str) -> str:
    """Chuyển yyyy-mm-dd... → dd/mm/yyyy (nếu parse được)."""
    try:
        y, m, d = map(int, iso_str[:10].split("-"))
        return f"{d:02d}/{m:02d}/{y}"
    except Exception:
        return iso_str or ""

def format_price_display(v: Any) -> str:
    """Định dạng giá hiển thị nhanh: Triệu/Tỷ (không chuyển đổi logic dữ liệu)."""
    try:
        val = float(v)
    except Exception:
        return str(v)
    if val >= 1e9:
        return f"{val/1e9:.1f}".rstrip("0.") + " Tỷ"
    if val >= 1e6:
        return f"{val/1e6:.0f} Triệu"
    return str(val)

def build_item_url(item: Dict[str, Any]) -> str:
    """Trả về URL chi tiết tin đăng."""
    page = item.get("@page") or item.get("page") or {}
    url = page.get("canonicalUrl") or page.get("url") or ""
    if not url and item.get("_id"):
        url = f"/ban-{item['_id']}"
    return ("https://moso.vn" + url) if (url and not url.startswith("http")) else (url or "")

def is_valid_value(v: Any) -> bool:
    """True nếu khác None/rỗng/0."""
    return v not in (None, "", 0, 0.0)

def first_nonempty(values: List[Any]) -> Any:
    """Trả về phần tử có giá trị meaningful đầu tiên."""
    for v in values:
        if is_valid_value(v):
            return v
    return ""

def find_first_value(obj: Any, keys: List[str]) -> Any:
    """Tìm đệ quy giá trị theo danh sách khóa; trả về giá trị đầu tiên tìm được."""
    found: List[Any] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keys and is_valid_value(v):
                    found.append(v)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return first_nonempty(found)

def extract_address(item: Dict[str, Any]) -> str:
    """
    Lấy địa chỉ thô (không chuẩn hoá). Ưu tiên:
      - pAddress.full / newPropertyAddress.full
      - _pAddress / _newPropertyAddress (string)
      - pDetailedAddress / address (string)
      - Nếu object không có 'full': ghép streetName, ward, district, provinceCity (nếu có)
    """
    def to_full(addr_obj: Any) -> str | None:
        if isinstance(addr_obj, dict):
            if addr_obj.get("full"):
                return str(addr_obj["full"])
            parts = [addr_obj.get(k) for k in ("streetName", "ward", "district", "provinceCity") if addr_obj.get(k)]
            if parts:
                return ", ".join(map(str, parts))
        elif isinstance(addr_obj, str):
            return addr_obj
        return None

    candidates: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("pAddress", "newPropertyAddress"):
                    s = to_full(v)
                    if s: candidates.append(s)
                elif k in ("_pAddress", "_newPropertyAddress", "pDetailedAddress", "address"):
                    s = to_full(v)
                    if s: candidates.append(s)
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(item)

    # loại trùng, trả về đầu tiên
    seen: set[str] = set()
    for s in candidates:
        ss = str(s).strip()
        if ss and ss not in seen:
            seen.add(ss)
            return ss
    return ""

def build_row(item: Dict[str, Any]) -> Dict[str, Any]:
    """Chuyển 1 item JSON thành 1 dòng dữ liệu theo FIELDS."""
    legal = find_first_value(item, ALIAS["legal"])
    if isinstance(legal, bool):
        legal = "Có giấy tờ" if legal else ""

    return {
        "URL": build_item_url(item),
        "Giá": format_price_display(find_first_value(item, ALIAS["price"])),
        "Diện tích sử dụng": find_first_value(item, ALIAS["usable"]),
        "Diện tích đất": find_first_value(item, ALIAS["land"]),
        "Phòng ngủ": find_first_value(item, ALIAS["bed"]),
        "Phòng tắm": find_first_value(item, ALIAS["bath"]),
        "Giấy tờ pháp lý": legal,
        "Ngày đăng": format_date_iso_to_ddmmyyyy(str(find_first_value(item, ALIAS["date"]))),
        "Địa chỉ": extract_address(item),
    }

# ===================== CHƯƠNG TRÌNH CHÍNH =====================
def main() -> None:
    # 1) Đọc payload mẫu từ DevTools
    with open(REQUEST_JSON_PATH, encoding="utf-8") as f:
        base = json.load(f)

    payload = deepcopy(base)
    limit = base.get("options", {}).get("limit", 100)
    offset = 0
    rows: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    # 2) Phân trang bằng offset
    while True:
        payload.setdefault("options", {})
        payload["options"]["offset"] = offset

        data = api_post(payload)
        models = data.get("models", [])
        if not models:
            print(f"[{offset}] Không có dữ liệu, dừng.")
            break

        added = 0
        for item in models:
            url = build_item_url(item)
            if not url or url in seen_urls:
                continue
            rows.append(build_row(item))
            seen_urls.add(url)
            added += 1

        print(f"[{offset}] +{added} (tổng {len(rows)})")

        total = data.get("count") or 0
        if added == 0:
            break
        if total and len(rows) >= total:
            break

        offset += limit
        time.sleep(0.2)

    # 3) Ghi CSV
    with open(OUTFILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Ghi {len(rows)} dòng vào {OUTFILE}")

# ===================== ENTRYPOINT ========== ===========
if __name__ == "__main__":
    main()
