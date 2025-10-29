import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

PROCESSED = "data/processed"
MODELS = "models"
os.makedirs(MODELS, exist_ok=True)

train = pd.read_csv(f"{PROCESSED}/train.csv")
test = pd.read_csv(f"{PROCESSED}/test.csv")

# Use flattened weekly clicks + extra features if exist
click_cols = [c for c in train.columns if c.startswith('w') and c.endswith('_clicks')]
extra_cols = [c for c in ["tong_click","so_bai_nop","diem_tb","ti_le_hoan_thanh","so_tuan_hoat_dong"] if c in train.columns]
feat_cols = click_cols + extra_cols

def to_features(df):
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df["label"].values.astype(int) if "label" in df.columns else None
    ids = df["id_student"].values
    return ids, X, y

train_ids, X_train, y_train = to_features(train)
test_ids, X_test, _ = to_features(test)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42
)
gb.fit(X_tr, y_tr)

y_prob = gb.predict_proba(X_val)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

try:
    auc_pr = average_precision_score(y_val, y_prob)
except Exception:
    auc_pr = float('nan')
try:
    auc_roc = roc_auc_score(y_val, y_prob)
except Exception:
    auc_roc = float('nan')
f1 = f1_score(y_val, y_pred)
print(f"GB Val => AUC-PR={auc_pr:.3f} | AUC-ROC={auc_roc:.3f} | F1={f1:.3f}")

joblib.dump(gb, os.path.join(MODELS, "baseline_gb.joblib"))

y_prob_test = gb.predict_proba(X_test)[:,1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

out = pd.DataFrame({
    "id_student": test_ids,
    "predicted_label": y_pred_test,
    "prob_pass": y_prob_test
})
out.to_csv(os.path.join(PROCESSED, "ou_pred_baseline.csv"), index=False)
print("Saved:", os.path.join(PROCESSED, "ou_pred_baseline.csv"))




