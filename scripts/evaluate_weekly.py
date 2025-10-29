import os
import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
try:
    import joblib
except Exception:
    joblib = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


PROCESSED_DIR = "data/processed"
SIM_DIR = "data/simulations"
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
REAL_PATH = os.path.join(PROCESSED_DIR, "ou_real.csv")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler_minmax.csv")
MODEL_CKPT = "models/lstm_final.h5"
CALIBRATOR_PATH = "models/prob_calibrator.joblib"
# Write evaluation to a separate file to avoid schema conflict with simulator metrics
METRICS_EVAL = os.path.join(SIM_DIR, "metrics_eval.csv")


def load_click_scaler():
    if not os.path.exists(SCALER_PATH):
        return None
    sc = pd.read_csv(SCALER_PATH)
    return sc


def scale_partial_sequence(df: pd.DataFrame, t: int, max_weeks: int, scaler_df: pd.DataFrame) -> np.ndarray:
    """
    Build sequence up to week t and zero-pad future weeks, applying per-column min-max scaling.
    Returns array of shape (n_samples, time_steps, 1)
    """
    week_cols = [f"w{w}_clicks" for w in range(max_weeks)]
    X = df[week_cols].copy()
    # zero future weeks
    for w in range(t + 1, max_weeks):
        col = f"w{w}_clicks"
        if col in X.columns:
            X[col] = 0.0

    # min-max per column
    if scaler_df is not None:
        mins = dict(zip(scaler_df["col"], scaler_df["min"]))
        maxs = dict(zip(scaler_df["col"], scaler_df["max"]))
        for c in X.columns:
            if c in mins:
                denom = (maxs[c] - mins[c]) if (maxs[c] - mins[c]) != 0 else 1.0
                X[c] = (X[c] - mins[c]) / denom
            else:
                # fallback
                col_min = float(X[c].min())
                col_max = float(X[c].max())
                denom = (col_max - col_min) if (col_max - col_min) != 0 else 1.0
                X[c] = (X[c] - col_min) / denom

    arr = X.values.astype(np.float32)
    # reshape: (samples, timesteps, 1)
    arr = arr.reshape((arr.shape[0], max_weeks, 1))
    return arr


def calibrate_probs(y_prob: np.ndarray, y_true: np.ndarray, method: str = "platt"):
    if method == "platt":
        # Logistic regression on prob as single feature
        X = y_prob.reshape(-1, 1)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y_true)
        return clf
    elif method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_prob, y_true)
        return ir
    else:
        return None


def apply_calibrator(calibrator, y_prob: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return y_prob
    # support both LogisticRegression and IsotonicRegression
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    else:
        return calibrator.predict(y_prob)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, strategy: str = "f1", precision_floor: float = 0.8, fp_cost: float = 2.0):
    # grid search over thresholds
    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr, best_f1 = 0.5, -1.0
    chosen = 0.5
    if strategy == "f1":
        for thr in thresholds:
            y_hat = (y_prob >= thr).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        chosen = best_thr
    elif strategy == "precision_floor":
        # choose highest recall while keeping precision >= floor
        best_recall = -1.0
        for thr in thresholds:
            y_hat = (y_prob >= thr).astype(int)
            p, r, _, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
            if p >= precision_floor and r > best_recall:
                best_recall, chosen = r, thr
        if best_recall < 0:
            chosen = 0.5
    elif strategy == "cost":
        # cost-sensitive: U = TP*u+ - FP*u-, where u+ is benefit of TP, u- is cost of FP
        best_u, chosen = -1e9, 0.5
        for thr in thresholds:
            y_hat = (y_prob >= thr).astype(int)
            tp = int(((y_hat == 1) & (y_true == 1)).sum())
            fp = int(((y_hat == 1) & (y_true == 0)).sum())
            u = tp * 1.0 - fp * fp_cost  # fp_cost > 1 penalizes FP more
            if u > best_u:
                best_u, chosen = u, thr
    return chosen


def compute_weekly_metrics(
    df_test: pd.DataFrame,
    df_real: pd.DataFrame,
    model,
    scaler_df: pd.DataFrame,
    max_weeks: int,
    calibration: str = None,
    threshold_strategy: str = "f1",
    precision_floor: float = 0.8,
    fp_cost: float = 2.0,
    mode: str = "pass",  # "pass" or "risk"
    write_log: bool = True,
    output_file: str = None,
):
    os.makedirs(SIM_DIR, exist_ok=True)
    if output_file is None:
        output_file = METRICS_EVAL
    header = not os.path.exists(output_file)

    # align labels to test ids and sanitize
    y_true_full = (
        df_real.set_index("id_student")["label"].reindex(df_test["id_student"]).astype("float").values
    )
    # replace NaNs (unknown labels) with 0 by default to allow metrics; real setup should drop unknowns
    y_true_full = np.nan_to_num(y_true_full, nan=0.0).astype(int)
    
    # Mode "risk": invert labels (label=1 is pass, so risk = 1-pass = 1-label)
    if mode == "risk":
        y_true_full = 1 - y_true_full  # Now 1 = risk, 0 = safe

    for t in range(max_weeks):
        X_seq = scale_partial_sequence(df_test, t, max_weeks, scaler_df)
        # if model is missing, use a simple proxy: mean of observed weeks
        if model is None:
            obs = X_seq[:, : (t + 1), 0]
            y_prob = obs.mean(axis=1)
            # normalize to [0,1]
            mmin, mmax = float(y_prob.min()), float(y_prob.max())
            denom = (mmax - mmin) if (mmax - mmin) != 0 else 1.0
            y_prob = (y_prob - mmin) / denom
        else:
            y_prob = model.predict(X_seq, verbose=0).ravel()
        
        # Mode "risk": invert probabilities (prob_pass -> prob_risk = 1 - prob_pass)
        if mode == "risk":
            y_prob = 1.0 - y_prob

        # optional calibration using a small split of current week
        calibrator = None
        if calibration in ("platt", "isotonic"):
            # Only calibrate if both classes present and enough samples
            unique, counts = np.unique(y_true_full, return_counts=True)
            if len(unique) >= 2 and min(counts) >= 5:
                try:
                    X_cal, X_app, y_cal, y_app = train_test_split(
                        y_prob, y_true_full, test_size=0.7, random_state=42, stratify=y_true_full
                    )
                    calibrator = calibrate_probs(np.asarray(X_cal), np.asarray(y_cal), method=calibration)
                    if joblib is not None:
                        try:
                            joblib.dump(calibrator, CALIBRATOR_PATH)
                        except Exception:
                            pass
                    y_prob = apply_calibrator(calibrator, y_prob)
                except Exception:
                    calibrator = None
                    # fall back to uncalibrated probs

        # select threshold
        try:
            theta_t = select_threshold(y_true_full, y_prob, strategy=threshold_strategy, precision_floor=precision_floor, fp_cost=fp_cost)
        except Exception:
            theta_t = 0.5
        y_hat = (y_prob >= theta_t).astype(int)

        # metrics
        try:
            auc_roc = float(roc_auc_score(y_true_full, y_prob))
        except Exception:
            auc_roc = np.nan
        try:
            auc_pr = float(average_precision_score(y_true_full, y_prob))
        except Exception:
            auc_pr = np.nan
        try:
            brier = float(brier_score_loss(y_true_full, y_prob))
        except Exception:
            brier = np.nan

        p, r, f1, _ = precision_recall_fscore_support(y_true_full, y_hat, average="binary", zero_division=0)
        tp = int(((y_hat == 1) & (y_true_full == 1)).sum())
        fp = int(((y_hat == 1) & (y_true_full == 0)).sum())
        warn = int((y_hat == 1).sum())

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "week": int(t),
            "AUC_PR": auc_pr,
            "AUC_ROC": auc_roc,
            "F1": float(f1),
            "Precision": float(p),
            "Recall": float(r),
            "Brier": brier,
            "theta_t": float(theta_t),
            "warn_count": warn,
            "TP": tp,
            "FP": fp,
        }

        if write_log:
            try:
                pd.DataFrame([row]).to_csv(output_file, mode="a", index=False, header=header)
                header = False
            except Exception:
                pass

        print(
            f"[t={t:02d}] AUC-PR={auc_pr:.3f} | AUC-ROC={auc_roc:.3f} | F1={f1:.3f} | Brier={brier:.3f} | theta={theta_t:.2f} | #warn={warn} (#TP={tp}, #FP={fp})"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate weekly metrics and log to metrics_log.csv")
    parser.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="none")
    parser.add_argument("--threshold", choices=["f1", "precision_floor", "cost"], default="f1")
    parser.add_argument("--precision_floor", type=float, default=0.8)
    parser.add_argument("--fp_cost", type=float, default=2.0, help="Cost of FP relative to TP (for cost-sensitive threshold)")
    parser.add_argument("--mode", choices=["pass", "risk"], default="pass", help="pass: predict pass probability; risk: predict risk probability (inverted)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path (default: metrics_eval.csv)")
    args = parser.parse_args()

    if not os.path.exists(TEST_PATH) or not os.path.exists(REAL_PATH):
        raise SystemExit("Missing processed files. Run scripts/etl_prepare.py and scripts/train_lstm.py first.")

    df_test = pd.read_csv(TEST_PATH)
    df_real = pd.read_csv(REAL_PATH)
    max_weeks = len([c for c in df_test.columns if c.startswith("w") and c.endswith("_clicks")])

    # load model if available
    model = None
    if load_model is not None and os.path.exists(MODEL_CKPT):
        try:
            model = load_model(MODEL_CKPT)
            print(f"Loaded model: {MODEL_CKPT}")
        except Exception:
            model = None

    scaler_df = load_click_scaler()
    calibration = None if args.calibration == "none" else args.calibration

    compute_weekly_metrics(
        df_test=df_test,
        df_real=df_real,
        model=model,
        scaler_df=scaler_df,
        max_weeks=max_weeks,
        calibration=calibration,
        threshold_strategy=args.threshold,
        precision_floor=args.precision_floor,
        fp_cost=args.fp_cost,
        mode=args.mode,
        write_log=True,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()


