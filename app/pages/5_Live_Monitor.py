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

status = read_status()
if status:
    st.success(f"🟢 Live scenario: {status.get('scenario','?')} @ {status.get('timestamp','--')}")
else:
    st.warning("⚠️ Live simulator chưa chạy hoặc chưa ghi status.json")

@st.cache_data(ttl=5)
def load_metrics():
    if os.path.exists(METRICS_LOG):
        df = pd.read_csv(METRICS_LOG)
        return df
    return pd.DataFrame()

df = load_metrics()
if df.empty:
    st.info("Chưa có dữ liệu metrics. Hãy chạy scripts/simulate_realtime.py")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Mean click", f"{df['mean_click'].iloc[-1]:.1f}")
col2.metric("Mean score", f"{df['mean_score'].iloc[-1]:.1f}")
col3.metric("Mean completion", f"{df['mean_completion'].iloc[-1]*100:.1f}%")

st.markdown("---")

fig1 = px.line(df, x="timestamp", y=["mean_click","mean_submits","mean_weeks"], title="Hoạt động & tuần học (trung bình)")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(df, x="timestamp", y=["mean_score","mean_completion"], title="Điểm TB & Hoàn thành (trung bình)")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(df, x="mean_click", y="mean_score", color="scenario", title="Tương quan mean_click vs mean_score theo scenario")
st.plotly_chart(fig3, use_container_width=True)


