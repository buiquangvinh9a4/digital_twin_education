#!/usr/bin/env python3
"""
So sánh đánh giá giữa hai phương án:
1. Mode "pass": Giữ nguyên prob_pass, điều chỉnh ngưỡng với cost-sensitive/precision_floor
2. Mode "risk": Đảo logic prob_risk = 1 - prob_pass, đánh giá với ngưỡng động
"""
import os
import subprocess
import pandas as pd
import numpy as np

SIM_DIR = "data/simulations"
METRICS_PASS_F1 = os.path.join(SIM_DIR, "metrics_pass_f1.csv")
METRICS_PASS_COST = os.path.join(SIM_DIR, "metrics_pass_cost.csv")
METRICS_PASS_PREC = os.path.join(SIM_DIR, "metrics_pass_prec.csv")
METRICS_RISK_F1 = os.path.join(SIM_DIR, "metrics_risk_f1.csv")
METRICS_RISK_COST = os.path.join(SIM_DIR, "metrics_risk_cost.csv")

def run_evaluation(mode, threshold, output_file, fp_cost=2.0, precision_floor=0.75):
    """Chạy đánh giá với cấu hình cụ thể"""
    cmd = [
        "python", "scripts/evaluate_weekly.py",
        "--mode", mode,
        "--threshold", threshold,
        "--output", output_file,
        "--calibration", "none"
    ]
    if threshold == "cost":
        cmd.extend(["--fp_cost", str(fp_cost)])
    elif threshold == "precision_floor":
        cmd.extend(["--precision_floor", str(precision_floor)])
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)

def summarize_results(file_path, label):
    """Tóm tắt kết quả từ file CSV"""
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    latest = df.tail(1).iloc[0] if len(df) > 0 else None
    if latest is None:
        return None
    
    return {
        "label": label,
        "week": int(latest["week"]),
        "auc_pr": latest.get("AUC_PR", np.nan),
        "auc_roc": latest.get("AUC_ROC", np.nan),
        "f1": latest.get("F1", np.nan),
        "precision": latest.get("Precision", np.nan),
        "recall": latest.get("Recall", np.nan),
        "brier": latest.get("Brier", np.nan),
        "theta_t": latest.get("theta_t", np.nan),
        "tp": int(latest.get("TP", 0)),
        "fp": int(latest.get("FP", 0)),
        "warn_count": int(latest.get("warn_count", 0)),
    }

def main():
    print("So sánh đánh giá giữa Mode 'pass' và Mode 'risk'\n")
    print("Phương án A: Mode 'pass' với các chiến lược ngưỡng khác nhau")
    print("Phương án B: Mode 'risk' (đảo logic) với ngưỡng động\n")
    
    # Phương án A: Mode "pass" với các chiến lược
    print("\n>>> PHƯƠNG ÁN A: Mode 'pass' <<<")
    run_evaluation("pass", "f1", METRICS_PASS_F1)
    run_evaluation("pass", "cost", METRICS_PASS_COST, fp_cost=2.5)
    run_evaluation("pass", "precision_floor", METRICS_PASS_PREC, precision_floor=0.75)
    
    # Phương án B: Mode "risk" với các chiến lược
    print("\n>>> PHƯƠNG ÁN B: Mode 'risk' (đảo logic) <<<")
    run_evaluation("risk", "f1", METRICS_RISK_F1)
    run_evaluation("risk", "cost", METRICS_RISK_COST, fp_cost=2.5)
    
    # Tóm tắt và so sánh
    print("\n" + "="*80)
    print("TÓM TẮT VÀ SO SÁNH KẾT QUẢ (Tuần cuối - Week 19)")
    print("="*80)
    
    results = []
    results.append(summarize_results(METRICS_PASS_F1, "Pass + F1"))
    results.append(summarize_results(METRICS_PASS_COST, "Pass + Cost (FP=2.5x)"))
    results.append(summarize_results(METRICS_PASS_PREC, "Pass + Precision≥0.75"))
    results.append(summarize_results(METRICS_RISK_F1, "Risk + F1"))
    results.append(summarize_results(METRICS_RISK_COST, "Risk + Cost (FP=2.5x)"))
    
    # Loại bỏ None
    results = [r for r in results if r is not None]
    
    if not results:
        print("❌ Không có kết quả để so sánh. Kiểm tra lại các file metrics.")
        return
    
    # In bảng so sánh
    print("\n" + "-"*80)
    print(f"{'Phương án':<25} | {'AUC-PR':<8} | {'AUC-ROC':<8} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'FP':<6} | {'TP':<6}")
    print("-"*80)
    for r in results:
        print(
            f"{r['label']:<25} | "
            f"{r['auc_pr']:>7.3f} | "
            f"{r['auc_roc']:>7.3f} | "
            f"{r['precision']:>5.3f} | "
            f"{r['recall']:>5.3f} | "
            f"{r['f1']:>5.3f} | "
            f"{r['fp']:>5} | "
            f"{r['tp']:>5}"
        )
    print("-"*80)
    
    # Tìm phương án tốt nhất theo từng tiêu chí
    print("\n>>> PHƯƠNG ÁN TỐT NHẤT THEO TỪNG TIÊU CHÍ <<<")
    best_auc_pr = max(results, key=lambda x: x['auc_pr'])
    best_precision = max(results, key=lambda x: x['precision'])
    best_f1 = max(results, key=lambda x: x['f1'])
    lowest_fp = min(results, key=lambda x: x['fp'])
    
    print(f"AUC-PR cao nhất: {best_auc_pr['label']} ({best_auc_pr['auc_pr']:.3f})")
    print(f"Precision cao nhất: {best_precision['label']} ({best_precision['precision']:.3f}, FP={best_precision['fp']})")
    print(f"F1 cao nhất: {best_f1['label']} ({best_f1['f1']:.3f})")
    print(f"FP thấp nhất: {lowest_fp['label']} (FP={lowest_fp['fp']}, Precision={lowest_fp['precision']:.3f})")
    
    print("\n✅ So sánh hoàn tất. Xem chi tiết trong data/simulations/metrics_*.csv")

if __name__ == "__main__":
    main()

