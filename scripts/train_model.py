import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

DATA_PATH = "data/processed/test.csv"
MODEL_PATH = "models/oulad_lstm.h5"
PRED_PATH = "data/processed/ou_pred.csv"

# 1️⃣ Nạp dữ liệu
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["diem_tb", "so_bai_nop", "tong_click"], how="any")

X = df[["tong_click", "so_bai_nop", "diem_tb", "ti_le_hoan_thanh", "so_tuan_hoat_dong"]].values
y = df["label"].values if "label" in df.columns else np.zeros(len(df))

# 2️⃣ Chuẩn hóa và reshape cho LSTM
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # (samples, timesteps=1, features=5)

# 3️⃣ Mô hình LSTM chuẩn
model = Sequential([
    LSTM(64, input_shape=(1, 5)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_seq, y, epochs=8, batch_size=16, verbose=1)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print("✅ Mô hình LSTM 3D đã lưu thành công:", MODEL_PATH)

# 4️⃣ Dự đoán
y_prob = model.predict(X_seq).ravel()
y_pred = (y_prob > 0.5).astype(int)
pred_df = pd.DataFrame({
    "id_student": df["id_student"],
    "predicted_label": y_pred,
    "prob_pass": y_prob
})
pred_df.to_csv(PRED_PATH, index=False)
print("✅ Lưu kết quả dự đoán:", PRED_PATH)
