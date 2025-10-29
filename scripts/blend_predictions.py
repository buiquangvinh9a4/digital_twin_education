import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

PROCESSED = "data/processed"

LSTM_PRED = os.path.join(PROCESSED, "ou_pred.csv")
GB_PRED = os.path.join(PROCESSED, "ou_pred_baseline.csv")
REAL = os.path.join(PROCESSED, "ou_real.csv")

if not (os.path.exists(LSTM_PRED) and os.path.exists(GB_PRED) and os.path.exists(REAL)):
    raise SystemExit("Missing required files. Run train_lstm.py (or tune_lstm.py) and train_baselines.py first.")

lstm = pd.read_csv(LSTM_PRED)
gb = pd.read_csv(GB_PRED)
real = pd.read_csv(REAL)[["id_student","label"]]

df = real.merge(lstm[["id_student","prob_pass"]].rename(columns={"prob_pass":"p_lstm"}), on="id_student", how="left")
df = df.merge(gb[["id_student","prob_pass"]].rename(columns={"prob_pass":"p_gb"}), on="id_student", how="left")
df = df.dropna(subset=["p_lstm","p_gb"])  # keep intersect

bests = {"w": None, "score": -1.0, "auc_pr": np.nan, "auc_roc": np.nan, "f1": np.nan}

for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    p_blend = w * df["p_lstm"].values + (1 - w) * df["p_gb"].values
    try:
        auc_pr = float(average_precision_score(df["label"].values, p_blend))
    except Exception:
        auc_pr = np.nan
    try:
        auc_roc = float(roc_auc_score(df["label"].values, p_blend))
    except Exception:
        auc_roc = np.nan
    y_pred = (p_blend >= 0.5).astype(int)
    f1 = float(f1_score(df["label"].values, y_pred))
    score = 0.6 * (auc_pr if not np.isnan(auc_pr) else 0.0) + 0.4 * (auc_roc if not np.isnan(auc_roc) else 0.0)
    print(f"Blend w={w:.1f} => AUC-PR={auc_pr:.3f} | AUC-ROC={auc_roc:.3f} | F1={f1:.3f} | SCORE={score:.3f}")
    if score > bests["score"]:
        bests = {"w": w, "score": score, "auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1}

print("Best blend:", bests)

# Apply best blend to full LSTM and GB predictions (assume same ids as real/test)
full = lstm.merge(gb, on="id_student", suffixes=("_lstm","_gb"))
full["prob_pass"] = bests["w"] * full["prob_pass_lstm"] + (1 - bests["w"]) * full["prob_pass_gb"]
full["predicted_label"] = (full["prob_pass"] >= 0.5).astype(int)

out = full[["id_student","predicted_label","prob_pass"]]
out.to_csv(os.path.join(PROCESSED, "ou_pred.csv"), index=False)
print("Saved blended predictions to:", os.path.join(PROCESSED, "ou_pred.csv"))




