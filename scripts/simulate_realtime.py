import os
import time
import random
import json
from datetime import datetime
import pandas as pd

PROCESSED_TEST = "data/processed/test.csv"
STATUS_DIR = "data/simulations"
STATUS_FILE = os.path.join(STATUS_DIR, "status.json")
METRICS_LOG = os.path.join(STATUS_DIR, "metrics_log.csv")

SCENARIOS = {
    "baseline": {"click_factor": 0.0, "submit_delta": 0, "score_delta": 0.0, "completion_delta": 0.0, "active_weeks_delta": 0},
    "exam_season": {"click_factor": 0.25, "submit_delta": +1, "score_delta": +2.5, "completion_delta": +0.05, "active_weeks_delta": +1},
    "holiday": {"click_factor": -0.3, "submit_delta": -1, "score_delta": -2.0, "completion_delta": -0.08, "active_weeks_delta": -1},
    "early_intervention": {"click_factor": +0.15, "submit_delta": +1, "score_delta": +1.5, "completion_delta": +0.1, "active_weeks_delta": +1},
    "late_dropout": {"click_factor": -0.5, "submit_delta": -2, "score_delta": -5.0, "completion_delta": -0.2, "active_weeks_delta": -2}
}

def clamp(value, min_v, max_v):
    return max(min_v, min(max_v, value))

def apply_scenario(df: pd.DataFrame, scenario: dict, sample_frac: float = 0.2) -> pd.DataFrame:
    if df.empty:
        return df

    df_mod = df.copy()

    # chọn một phân đoạn sinh viên để tạo cảm giác realtime từng phần
    mask = df_mod.sample(frac=sample_frac, random_state=None).index

    # cập nhật chỉ số tổng hợp (mức/features không tuần)
    if "tong_click" in df_mod.columns:
        df_mod.loc[mask, "tong_click"] = (df_mod.loc[mask, "tong_click"] * (1.0 + scenario["click_factor"]))
        df_mod.loc[mask, "tong_click"] = df_mod.loc[mask, "tong_click"].clip(lower=0)

    for col, scale in [("so_bai_nop", 10), ("diem_tb", 100), ("ti_le_hoan_thanh", 1.0), ("so_tuan_hoat_dong", 20)]:
        if col in df_mod.columns:
            delta_key = f"{col}_delta"
            # map delta keys
            mapping = {
                "so_bai_nop_delta": scenario.get("submit_delta", 0),
                "diem_tb_delta": scenario.get("score_delta", 0.0),
                "ti_le_hoan_thanh_delta": scenario.get("completion_delta", 0.0),
                "so_tuan_hoat_dong_delta": scenario.get("active_weeks_delta", 0),
            }
            delta = mapping.get(delta_key, 0)
            df_mod.loc[mask, col] = df_mod.loc[mask, col] + delta
            if col in ("so_bai_nop", "so_tuan_hoat_dong"):
                df_mod.loc[mask, col] = df_mod.loc[mask, col].clip(lower=0, upper=scale)
            elif col == "diem_tb":
                df_mod.loc[mask, col] = df_mod.loc[mask, col].clip(lower=0, upper=100)
            elif col == "ti_le_hoan_thanh":
                df_mod.loc[mask, col] = df_mod.loc[mask, col].clip(lower=0.0, upper=1.0)

    # cập nhật các cột weekly clicks (w0_clicks..w19_clicks) để twin nhất quán
    week_cols = [c for c in df_mod.columns if c.startswith("w") and c.endswith("_clicks")]
    if week_cols:
        # factor theo scenario và thêm nhiễu nhỏ
        factor = 1.0 + scenario["click_factor"]
        noise = lambda n: (1.0 + (random.uniform(-0.03, 0.03)))
        df_mod.loc[mask, week_cols] = df_mod.loc[mask, week_cols].apply(
            lambda col: (col * factor).clip(lower=0.0)
        )
        # thêm chút nhiễu để tạo cảm giác realtime
        df_mod.loc[mask, week_cols] = df_mod.loc[mask, week_cols] * noise(1)
        df_mod.loc[mask, week_cols] = df_mod.loc[mask, week_cols].clip(lower=0.0, upper=1.0)

    return df_mod

def main():
    if not os.path.exists(PROCESSED_TEST):
        print("❌ Missing:", PROCESSED_TEST)
        return

    print("🔁 Realtime simulator started. Press Ctrl+C to stop.")
    scenario_keys = list(SCENARIOS.keys())
    current_idx = 0

    # vòng lặp mô phỏng
    while True:
        try:
            df = pd.read_csv(PROCESSED_TEST)
        except Exception as e:
            print("⚠️ Read error:", e)
            time.sleep(2)
            continue

        scenario_name = scenario_keys[current_idx]
        scenario = SCENARIOS[scenario_name]

        # tỉ lệ sinh viên bị tác động thay đổi theo kịch bản
        frac = 0.15 if scenario_name in ("holiday", "late_dropout") else 0.3
        df_new = apply_scenario(df, scenario, sample_frac=frac)

        # ghi đè an toàn (atomic write)
        tmp_path = PROCESSED_TEST + ".tmp"
        try:
            df_new.to_csv(tmp_path, index=False)
            os.replace(tmp_path, PROCESSED_TEST)
            print(f"✅ Applied scenario: {scenario_name} | rows={len(df_new)} | frac={frac}")
            # ghi status & metrics
            os.makedirs(STATUS_DIR, exist_ok=True)
            status = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scenario": scenario_name,
                "affected_fraction": frac,
                "rows": int(len(df_new)),
                "test_csv_mtime": os.path.getmtime(PROCESSED_TEST)
            }
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f)

            # tính vài metric gọn cho biểu đồ thời gian
            metrics = {
                "timestamp": status["timestamp"],
                "scenario": scenario_name,
                "mean_click": float(df_new["tong_click"].mean() if "tong_click" in df_new.columns else 0.0),
                "mean_score": float(df_new["diem_tb"].mean() if "diem_tb" in df_new.columns else 0.0),
                "mean_completion": float(df_new["ti_le_hoan_thanh"].mean() if "ti_le_hoan_thanh" in df_new.columns else 0.0),
                "mean_submits": float(df_new["so_bai_nop"].mean() if "so_bai_nop" in df_new.columns else 0.0),
                "mean_weeks": float(df_new["so_tuan_hoat_dong"].mean() if "so_tuan_hoat_dong" in df_new.columns else 0.0),
            }
            header = not os.path.exists(METRICS_LOG)
            pd.DataFrame([metrics]).to_csv(METRICS_LOG, mode="a", index=False, header=header)
        except Exception as e:
            print("❌ Write error:", e)

        # chuyển kịch bản sau mỗi chu kỳ
        current_idx = (current_idx + 1) % len(scenario_keys)

        # ngủ ngắn để Streamlit/ watcher bắt kịp
        time.sleep(5)

if __name__ == "__main__":
    main()


