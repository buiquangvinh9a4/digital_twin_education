# scripts/train_lstm.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

PROCESSED = "data/processed"
MODELS = "models"
os.makedirs(MODELS, exist_ok=True)

train = pd.read_csv(f"{PROCESSED}/train.csv")
test  = pd.read_csv(f"{PROCESSED}/test.csv")

# Lấy danh sách cột tuần
feat_cols = [c for c in train.columns if c.startswith('w') and c.endswith('_clicks')]

def to_sequences(df):
    X = df[feat_cols].values.astype(np.float32)
    # reshape -> (samples, timesteps, features)
    # mỗi timestep = 1 đặc trưng (sum_click chuẩn hoá), nên features=1, timesteps=len(feat_cols)
    X = X.reshape((X.shape[0], len(feat_cols), 1))
    y = df['label'].values.astype(np.float32)
    ids = df['id_student'].values
    return ids, X, y

train_ids, X_train, y_train = to_sequences(train)
test_ids,  X_test,  y_test  = to_sequences(test)

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(f"{MODELS}/lstm_best.h5", save_best_only=True, monitor='val_loss')
]

hist = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

"""Đánh giá nhanh với thêm AUC-PR/AUC-ROC"""
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
try:
    auc_pr = average_precision_score(y_test, y_prob)
except Exception:
    auc_pr = float('nan')
try:
    auc_roc = roc_auc_score(y_test, y_prob)
except Exception:
    auc_roc = float('nan')

print(f"Test ACC: {acc:.4f} | F1: {f1:.4f} | AUC-PR: {auc_pr:.4f} | AUC-ROC: {auc_roc:.4f}")

# Lưu model cuối
model.save(f"{MODELS}/lstm_final.h5")

# Lưu dự báo để dashboard dùng
pred_out = pd.DataFrame({
    'id_student': test_ids,
    'predicted_label': y_pred,
    'prob_pass': y_prob
})
pred_out.to_csv(f"{PROCESSED}/ou_pred.csv", index=False)
print("Saved:", f"{PROCESSED}/ou_pred.csv")
