Dùng mô hình đã lưu để dự đoán tin mới:

import joblib
import pandas as pd

model = joblib.load('svm_price_pipeline.joblib')
# Ví dụ 1 tin mới (chú ý đúng tên cột):
new_df = pd.DataFrame([
  {
    'Diện tích sử dụng': 75,
    'Diện tích đất': 80,
    'Số ngày từ đăng': 10,
    'Tỷ lệ sử dụng': 75/80,
    'Quận/Huyện/TP': 'Quận 1'
  }
])
pred_vnd = model.predict(new_df)
pred_bil = pred_vnd / 1e9
print('Giá dự đoán (VND):', float(pred_vnd[0]))
print('Giá dự đoán (tỷ):', float(pred_bil[0]))
