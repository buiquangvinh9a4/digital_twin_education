import os
import pandas as pd
import streamlit as st
import plotly.express as px
try:
    from app.utils import read_status
except ModuleNotFoundError:
    import sys
    import os as _os
    _ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from app.utils import read_status

st.set_page_config(page_title="Live Monitor", layout="wide")
st.title("📡 Live Monitor — Dòng dữ liệu thời gian thực (giả lập)")

METRICS_LOG = "data/simulations/metrics_log.csv"
METRICS_EVAL = "data/simulations/metrics_eval.csv"

status = read_status()
if status:
    st.success(f"🟢 Live scenario: {status.get('scenario','?')} @ {status.get('timestamp','--')}")
else:
    st.warning("⚠️ Live simulator chưa chạy hoặc chưa ghi status.json")

@st.cache_data(ttl=5)
def load_metrics():
    # Ưu tiên đọc file đánh giá nếu có, để thêm được #warn, TP/FP, θ_t
    path = METRICS_LOG
    if os.path.exists(METRICS_EVAL):
        path = METRICS_EVAL
    if os.path.exists(path):
        try:
            # Hỗ trợ file cũ có dòng lẫn schema
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            return df
        except Exception:
            try:
                df = pd.read_csv(path)
                return df
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()

df = load_metrics()
if df.empty:
    st.info("Chưa có dữ liệu metrics. Hãy chạy scripts/simulate_realtime.py và/hoặc scripts/evaluate_weekly.py")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
if "mean_click" in df.columns:
    col1.metric("Mean click", f"{df['mean_click'].iloc[-1]:.1f}")
if "mean_score" in df.columns:
    col2.metric("Mean score", f"{df['mean_score'].iloc[-1]:.1f}")
if "mean_completion" in df.columns:
    col3.metric("Mean completion", f"{df['mean_completion'].iloc[-1]*100:.1f}%")
if "warn_count" in df.columns:
    col4.metric("# Cảnh báo", int(df['warn_count'].iloc[-1]))

st.markdown("---")

if all(c in df.columns for c in ["timestamp","mean_click","mean_submits","mean_weeks"]):
    fig1 = px.line(df, x="timestamp", y=["mean_click","mean_submits","mean_weeks"], title="Hoạt động & tuần học (trung bình)")
    st.plotly_chart(fig1, use_container_width=True)

if all(c in df.columns for c in ["timestamp","mean_score","mean_completion"]):
    fig2 = px.line(df, x="timestamp", y=["mean_score","mean_completion"], title="Điểm TB & Hoàn thành (trung bình)")
    st.plotly_chart(fig2, use_container_width=True)

if all(c in df.columns for c in ["mean_click","mean_score"]):
    color_arg = "scenario" if "scenario" in df.columns else None
    fig3 = px.scatter(df, x="mean_click", y="mean_score", color=color_arg, title="Tương quan mean_click vs mean_score theo scenario")
    st.plotly_chart(fig3, use_container_width=True)

# Nếu có metrics đánh giá, hiển thị thêm biểu đồ ngưỡng và TP/FP theo thời gian (timestamp)
if all(c in df.columns for c in ["timestamp","theta_t","warn_count"]):
    st.markdown("---")
    fig4 = px.line(df, x="timestamp", y=["theta_t"], title="Ngưỡng cảnh báo θ_t theo thời gian")
    st.plotly_chart(fig4, use_container_width=True)

if all(c in df.columns for c in ["timestamp","TP","FP"]):
    fig5 = px.bar(df, x="timestamp", y=["TP","FP"], title="TP/FP theo thời gian", barmode="group")
    st.plotly_chart(fig5, use_container_width=True)

# Tuỳ chọn làm mới tự động (mặc định 5 giây)
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Tự động làm mới mỗi 5s", False)
if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='5'>", unsafe_allow_html=True)


