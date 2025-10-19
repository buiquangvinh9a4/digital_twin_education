import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# =============================
# ⚙️ CẤU HÌNH TRANG
# =============================
st.set_page_config(page_title="Tổng quan lớp học", layout="wide")
st.title("🏫 Tổng quan lớp học — Digital Twin (Song song Thực & Ảo)")

TEST_PATH = "data/processed/test.csv"
REAL_PATH = "data/processed/ou_real.csv"
PRED_PATH = "data/processed/ou_pred.csv"

# =============================
# 1️⃣ NẠP DỮ LIỆU
# =============================
@st.cache_data
def load_data():
    if not os.path.exists(TEST_PATH):
        st.error("❌ Không tìm thấy test.csv. Hãy chạy lại ETL.")
        return pd.DataFrame()

    test = pd.read_csv(TEST_PATH)
    real = pd.read_csv(REAL_PATH) if os.path.exists(REAL_PATH) else pd.DataFrame()
    pred = pd.read_csv(PRED_PATH) if os.path.exists(PRED_PATH) else pd.DataFrame()

    # Merge nhãn thật và dự đoán
    if "label" not in test.columns and not real.empty and "label" in real.columns:
        test = test.merge(real[["id_student", "label"]], on="id_student", how="left")
    if not pred.empty and "predicted_label" in pred.columns:
        test = test.merge(pred[["id_student", "predicted_label", "prob_pass"]], on="id_student", how="left")

    for c in ["tong_click", "so_bai_nop", "diem_tb", "ti_le_hoan_thanh", "so_tuan_hoat_dong"]:
        if c not in test.columns:
            test[c] = np.nan
    return test

df = load_data()
if df.empty:
    st.stop()

# =============================
# 2️⃣ CẤU HÌNH NGƯỜI DÙNG
# =============================
st.sidebar.header("⚙️ Tùy chỉnh hiển thị")

metrics_map = {
    "composite_index": "Chỉ số tổng hợp (Learning Index)",
    "tong_click": "Tổng lượt truy cập",
    "so_bai_nop": "Số bài đã nộp",
    "diem_tb": "Điểm trung bình",
    "ti_le_hoan_thanh": "Tỷ lệ hoàn thành (%)",
    "so_tuan_hoat_dong": "Số tuần hoạt động"
}

selected_metric = st.sidebar.selectbox(
    "Chọn chỉ số hiển thị:",
    list(metrics_map.keys()),
    format_func=lambda x: metrics_map[x]
)

chart_type = st.sidebar.selectbox("Loại biểu đồ:", ["Histogram", "Boxplot", "Scatter"])
show_twin_delta = st.sidebar.checkbox("Hiển thị Twin Delta (Radar so sánh)", True)

# =============================
# 3️⃣ MÔ PHỎNG TÁC ĐỘNG
# =============================
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Mô phỏng tác động trung bình lớp")

click_factor = st.sidebar.slider("📚 Tăng/Giảm lượt click (%)", -50, 50, 0)
submit_factor = st.sidebar.slider("📝 Tăng/Giảm số bài nộp (%)", -50, 50, 0)
score_factor = st.sidebar.slider("💯 Tăng/Giảm điểm TB (%)", -50, 50, 0)
completion_factor = st.sidebar.slider("📈 Tăng/Giảm tỷ lệ hoàn thành (%)", -50, 50, 0)
week_factor = st.sidebar.slider("🗓️ Tăng/Giảm số tuần hoạt động (%)", -50, 50, 0)

# =============================
# 4️⃣ HÀM TÍNH CHỈ SỐ TỔNG HỢP
# =============================
def compute_learning_index(df_input):
    df_calc = df_input.copy()
    for col in ["tong_click", "so_bai_nop", "diem_tb", "ti_le_hoan_thanh", "so_tuan_hoat_dong"]:
        if col not in df_calc.columns:
            df_calc[col] = 0
    df_calc["composite_index"] = (
        0.3 * df_calc["diem_tb"].fillna(0) / 100 +
        0.25 * df_calc["ti_le_hoan_thanh"].fillna(0) +
        0.2 * df_calc["so_bai_nop"].fillna(0) / 10 +
        0.15 * df_calc["tong_click"].fillna(0) / 20000 +
        0.1 * df_calc["so_tuan_hoat_dong"].fillna(0) / 20
    ) * 100
    return df_calc

# Áp dụng cho cả bản thực & mô phỏng
df = compute_learning_index(df)
df_sim = compute_learning_index(df)
df_sim["tong_click"] *= (1 + click_factor / 100)
df_sim["so_bai_nop"] *= (1 + submit_factor / 100)
df_sim["diem_tb"] *= (1 + score_factor / 100)
df_sim["ti_le_hoan_thanh"] *= (1 + completion_factor / 100)
df_sim["so_tuan_hoat_dong"] *= (1 + week_factor / 100)
df_sim = compute_learning_index(df_sim)

# =============================
# 5️⃣ HAI CỘT SONG SONG (THỰC & ẢO)
# =============================
colL, colR = st.columns(2)

# --- CỘT TRÁI: PHYSICAL TWIN ---
with colL:
    st.header("🧩 Physical Class Twin — Dữ liệu thực tế")
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Số SV", len(df))
    col2.metric("🎯 Tỷ lệ đạt (thực)", f"{(df['label'].mean()*100):.1f}%")
    col3.metric("💯 Điểm TB", f"{df['diem_tb'].mean():.1f}")

    if chart_type == "Histogram":
        fig1 = px.histogram(df, x=selected_metric, color="label",
                            title=f"Phân bố {metrics_map[selected_metric]} (Thực tế)",
                            labels={"label": "Kết quả"}, nbins=30)
    elif chart_type == "Boxplot":
        fig1 = px.box(df, x="label", y=selected_metric, color="label",
                      title=f"Phân bố {metrics_map[selected_metric]} (Thực tế)")
    else:
        fig1 = px.scatter(df, x="diem_tb", y=selected_metric, color="label",
                          title=f"Tương quan Điểm TB và {metrics_map[selected_metric]} (Thực tế)")
    st.plotly_chart(fig1, use_container_width=True)

# --- CỘT PHẢI: DIGITAL TWIN ---
with colR:
    st.header("🤖 Digital Class Twin — Mô phỏng dự đoán")
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 SV mô phỏng", len(df_sim))
    sim_rate = (df["label"].mean() * 100) + (score_factor + completion_factor) / 2
    delta = abs(sim_rate - df["label"].mean() * 100)
    col2.metric("🎯 Tỷ lệ đạt (mô phỏng)", f"{sim_rate:.1f}%")
    col3.metric("⚙️ Sai lệch twin (%)", f"{delta:.1f}%")

    if chart_type == "Histogram":
        fig2 = px.histogram(df_sim, x=selected_metric, color="label",
                            title=f"Phân bố {metrics_map[selected_metric]} (Mô phỏng)",
                            labels={"label": "Kết quả"}, nbins=30)
    elif chart_type == "Boxplot":
        fig2 = px.box(df_sim, x="label", y=selected_metric, color="label",
                      title=f"Phân bố {metrics_map[selected_metric]} (Mô phỏng)")
    else:
        fig2 = px.scatter(df_sim, x="diem_tb", y=selected_metric, color="label",
                          title=f"Tương quan Điểm TB và {metrics_map[selected_metric]} (Mô phỏng)")
    st.plotly_chart(fig2, use_container_width=True)

# =============================
# 6️⃣ BIỂU ĐỒ SO SÁNH CHUNG (ĐIỂM CHUNG)
# =============================
st.markdown("---")
st.subheader(f"📊 So sánh {metrics_map[selected_metric]} giữa Thực tế và Mô phỏng (cùng thang điểm)")

if "label" in df.columns:
    real_group = df.groupby("label")[selected_metric].mean().reset_index()
    sim_group = df_sim.groupby("label")[selected_metric].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=real_group["label"], y=real_group[selected_metric],
        mode="lines+markers", name="Thực tế",
        line=dict(color="blue", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=sim_group["label"], y=sim_group[selected_metric],
        mode="lines+markers", name="Mô phỏng",
        line=dict(color="red", dash="dash", width=2)
    ))

    # Đường chênh lệch
    for i in range(len(real_group)):
        fig.add_shape(
            type="line",
            x0=real_group["label"].iloc[i], y0=real_group[selected_metric].iloc[i],
            x1=sim_group["label"].iloc[i], y1=sim_group[selected_metric].iloc[i],
            line=dict(color="gray", dash="dot")
        )

    fig.update_layout(
        title=f"So sánh {metrics_map[selected_metric]} theo nhóm (0=Không đạt, 1=Đạt)",
        xaxis_title="Nhóm kết quả",
        yaxis_title="Điểm (chuẩn hóa 0–100)",
        legend_title="Nguồn dữ liệu",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================
# 7️⃣ RADAR — TWIN DELTA
# =============================
if show_twin_delta:
    st.markdown("---")
    st.subheader("📈 Twin Delta — So sánh trung bình lớp (Thực vs Ảo)")
    metrics = ["tong_click", "so_bai_nop", "diem_tb", "ti_le_hoan_thanh", "so_tuan_hoat_dong"]
    real_avg = [df[m].mean() for m in metrics]
    sim_avg = [df_sim[m].mean() for m in metrics]

    radar_df = pd.DataFrame({
        "Chỉ số": [metrics_map[m] for m in metrics],
        "Thực tế": real_avg,
        "Mô phỏng": sim_avg
    })

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=radar_df["Thực tế"], theta=radar_df["Chỉ số"],
                                        fill="toself", name="Thực tế"))
    fig_radar.add_trace(go.Scatterpolar(r=radar_df["Mô phỏng"], theta=radar_df["Chỉ số"],
                                        fill="toself", name="Mô phỏng",
                                        line=dict(dash="dash")))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# =============================
# 8️⃣ GHI CHÚ
# =============================
st.caption("""
💡 Trang này hiển thị song song hai Twin (Thực & Ảo).  
Bạn có thể chọn **“Chỉ số tổng hợp (Learning Index)”** để xem điểm học tập tổng thể (0–100),  
và quan sát **đường nét đứt** thể hiện chênh lệch giữa hai mô hình.
""")
