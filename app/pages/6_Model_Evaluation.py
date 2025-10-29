import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Đánh giá mô hình", layout="wide")
st.title("🧪 Đánh giá mô hình — Digital Twin")

METRICS_EVAL = "data/simulations/metrics_eval.csv"
METRICS_LOG = "data/simulations/metrics_log.csv"
EVAL_GUIDE = "EVALUATION.md"

@st.cache_data(ttl=5)
def load_metrics():
    path = METRICS_EVAL if os.path.exists(METRICS_EVAL) else METRICS_LOG
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    # chuẩn hoá cột
    rename_map = {
        "AUC_PR": "auc_pr",
        "AUC_ROC": "auc_roc",
        "F1": "f1",
        "Precision": "precision",
        "Recall": "recall",
        "Brier": "brier",
        "theta_t": "theta_t",
        "warn_count": "warn_count",
        "TP": "TP",
        "FP": "FP",
        "week": "week",
        "timestamp": "timestamp",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # Chuẩn hoá cột week; nếu thiếu, tạo chỉ số tuần giả dựa trên thứ tự thời gian
    if "week" in df.columns:
        with pd.option_context('mode.chained_assignment', None):
            try:
                df["week"] = df["week"].astype(int)
            except Exception:
                # nếu week chứa NaN, điền tăng dần
                df["week"] = pd.Series(range(len(df)))
    else:
        df["week"] = pd.Series(range(len(df)))

    # Sắp xếp linh hoạt
    if "timestamp" in df.columns:
        return df.sort_values(["week", "timestamp"]) if not df.empty else df
    else:
        return df.sort_values(["week"]) if not df.empty else df

df = load_metrics()

if df.empty:
    st.warning("⚠️ Chưa có metrics_log.csv để hiển thị. Xem hướng dẫn trong EVALUATION.md để tạo.")
    if os.path.exists(EVAL_GUIDE):
        st.link_button("📘 Mở hướng dẫn đánh giá (EVALUATION.md)", EVAL_GUIDE)
    st.stop()

# Sidebar controls
st.sidebar.header("⚙️ Tuỳ chọn")
agg = st.sidebar.selectbox("Gộp theo", ["week", "timestamp"], index=0)

# Head KPIs for latest week
latest_week = int(df["week"].max()) if "week" in df.columns and not df.empty else None
df_latest = df[df["week"] == latest_week] if latest_week is not None else df.tail(1)
row_latest = df_latest.tail(1).iloc[0] if not df_latest.empty else {}

has_eval_cols = any(c in df.columns for c in ["auc_pr","auc_roc","f1","brier"]) 

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tuần", latest_week if latest_week is not None else "-" )
col2.metric("AUC-PR", f"{row_latest.get('auc_pr', float('nan')):.3f}" if has_eval_cols and 'auc_pr' in df.columns else "-")
col3.metric("AUC-ROC", f"{row_latest.get('auc_roc', float('nan')):.3f}" if has_eval_cols and 'auc_roc' in df.columns else "-")
col4.metric("F1", f"{row_latest.get('f1', float('nan')):.3f}" if has_eval_cols and 'f1' in df.columns else "-")
col5.metric("Brier", f"{row_latest.get('brier', float('nan')):.3f}" if has_eval_cols and 'brier' in df.columns else "-")

st.markdown("---")

# Line charts across weeks
charts_all = [
    ("AUC-PR theo tuần", "auc_pr"),
    ("AUC-ROC theo tuần", "auc_roc"),
    ("F1 theo tuần", "f1"),
    ("Brier score theo tuần (càng thấp càng tốt)", "brier"),
]
charts = [(t, c) for (t, c) in charts_all if c in df.columns]

if charts:
    c1, c2 = st.columns(2)
    for i, (title, col) in enumerate(charts):
        fig = px.line(df, x="week", y=col, markers=True, title=title)
        fig.update_layout(template="plotly_white")
        (c1 if i % 2 == 0 else c2).plotly_chart(fig, use_container_width=True)
else:
    st.info("Chưa có cột đánh giá (AUC-PR/ROC, F1, Brier). Hãy chạy: python scripts/evaluate_weekly.py --calibration none --threshold f1")

st.markdown("---")
st.subheader("📏 Ngưỡng cảnh báo và số lượng cảnh báo")
fig_thr = go.Figure()
fig_thr.add_trace(go.Scatter(x=df["week"], y=df["theta_t"], mode="lines+markers", name="θ_t", line=dict(color="orange")))
fig_thr.add_trace(go.Bar(x=df["week"], y=df["warn_count"], name="#warn", marker_color="rgba(99, 110, 250, 0.6)", yaxis="y2"))
fig_thr.update_layout(
    template="plotly_white",
    yaxis=dict(title="θ_t (threshold)"),
    yaxis2=dict(title="#warn", overlaying="y", side="right"),
    title="Ngưỡng θ_t và #warn theo tuần"
)
st.plotly_chart(fig_thr, use_container_width=True)

st.markdown("---")
st.subheader("🧮 TP/FP theo tuần")
df_tf = df.groupby("week").agg({"TP": "max", "FP": "max"}).reset_index()
fig_tf = go.Figure()
fig_tf.add_trace(go.Bar(x=df_tf["week"], y=df_tf["TP"], name="TP", marker_color="#2ca02c"))
fig_tf.add_trace(go.Bar(x=df_tf["week"], y=df_tf["FP"], name="FP", marker_color="#d62728"))
fig_tf.update_layout(barmode="group", template="plotly_white")
st.plotly_chart(fig_tf, use_container_width=True)

st.markdown("---")
st.subheader("📋 Bảng chi tiết (mới nhất theo tuần)")
df_week_last = df.sort_values("timestamp").groupby("week").tail(1)
show_cols = [c for c in ["timestamp","week","auc_pr","auc_roc","precision","recall","f1","brier","theta_t","warn_count","TP","FP"] if c in df_week_last.columns]
st.dataframe(df_week_last[show_cols], use_container_width=True)

st.markdown("---")
st.info("Hướng dẫn đánh giá chi tiết (quy trình, calibration, chọn ngưỡng) nằm trong EVALUATION.md.")
if os.path.exists(EVAL_GUIDE):
    st.link_button("📘 Mở EVALUATION.md", EVAL_GUIDE)


