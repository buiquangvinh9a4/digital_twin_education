import os
import itertools
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

PROCESSED = "data/processed"
MODELS = "models"
os.makedirs(MODELS, exist_ok=True)

train = pd.read_csv(f"{PROCESSED}/train.csv")
test = pd.read_csv(f"{PROCESSED}/test.csv")

feat_cols = [c for c in train.columns if c.startswith("w") and c.endswith("_clicks")]

def to_sequences(df):
    X = df[feat_cols].values.astype(np.float32)
    X = X.reshape((X.shape[0], len(feat_cols), 1))
    y = df["label"].values.astype(np.float32)
    ids = df["id_student"].values
    return ids, X, y

train_ids, X_train, y_train = to_sequences(train)
test_ids, X_test, y_test = to_sequences(test)

# Validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

pos = (y_tr == 1).sum()
neg = (y_tr == 0).sum()
class_weight = None
if pos > 0 and neg > 0:
    # inverse frequency
    total = pos + neg
    class_weight = {0: total / (2.0 * neg), 1: total / (2.0 * pos)}

def build_model(units=64, layers=1, dropout=0.3, bidir=False, lr=1e-3):
    m = Sequential()
    if layers == 1:
        layer = LSTM(units, input_shape=(X_tr.shape[1], X_tr.shape[2]))
        if bidir:
            layer = Bidirectional(layer)
        m.add(layer)
        if dropout > 0:
            m.add(Dropout(dropout))
    else:
        # multi-layer LSTM (return_sequences for all but last)
        for i in range(layers - 1):
            layer = LSTM(units, return_sequences=True, input_shape=(X_tr.shape[1], X_tr.shape[2]) if i == 0 else None)
            if bidir:
                layer = Bidirectional(layer)
            m.add(layer)
            if dropout > 0:
                m.add(Dropout(dropout))
        layer_last = LSTM(units)
        if bidir:
            layer_last = Bidirectional(layer_last)
        m.add(layer_last)
        if dropout > 0:
            m.add(Dropout(dropout))

    m.add(Dense(1, activation='sigmoid'))
    m.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
    return m

search_space = {
    "units": [32, 64, 128],
    "layers": [1, 2],
    "dropout": [0.2, 0.4],
    "bidir": [False, True],
    "lr": [1e-3, 5e-4],
    "batch": [64, 128]
}

def iter_configs(space):
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        cfg = dict(zip(keys, values))
        yield cfg

def evaluate_on_val(model):
    y_prob = model.predict(X_val, verbose=0).ravel()
    try:
        auc_pr = float(average_precision_score(y_val, y_prob))
    except Exception:
        auc_pr = np.nan
    try:
        auc_roc = float(roc_auc_score(y_val, y_prob))
    except Exception:
        auc_roc = np.nan
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = float(f1_score(y_val, y_pred))
    return auc_pr, auc_roc, f1

best_score = -1.0
best_cfg = None
best_model = None

early = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

for cfg in iter_configs(search_space):
    try:
        model = build_model(cfg["units"], cfg["layers"], cfg["dropout"], cfg["bidir"], cfg["lr"])
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=cfg["batch"],
            callbacks=[early],
            verbose=0,
            class_weight=class_weight
        )
        auc_pr, auc_roc, f1 = evaluate_on_val(model)
        # Multi-objective: weighted combination (favor PR in imbalanced; still care ROC)
        score = 0.6 * (auc_pr if not np.isnan(auc_pr) else 0.0) + 0.4 * (auc_roc if not np.isnan(auc_roc) else 0.0)
        print(f"CFG {cfg} => AUC-PR={auc_pr:.3f} | AUC-ROC={auc_roc:.3f} | F1={f1:.3f} | SCORE={score:.3f}")
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_model = model
    except Exception as e:
        print("Skip config due to error:", cfg, e)

if best_model is None:
    raise SystemExit("No model trained successfully. Check data and dependencies.")

print("Best config:", best_cfg, "with AUC-PR=", best_score)
best_path = os.path.join(MODELS, "lstm_best.h5")
best_model.save(best_path)
print("Saved best model:", best_path)

# Evaluate on test and write predictions
y_prob_test = best_model.predict(X_test, verbose=0).ravel()
y_pred_test = (y_prob_test >= 0.5).astype(int)

out = pd.DataFrame({
    "id_student": test_ids,
    "predicted_label": y_pred_test,
    "prob_pass": y_prob_test
})
out.to_csv(os.path.join(PROCESSED, "ou_pred.csv"), index=False)
print("Saved:", os.path.join(PROCESSED, "ou_pred.csv"))


