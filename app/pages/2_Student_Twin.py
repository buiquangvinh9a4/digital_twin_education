import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os, csv
from tensorflow.keras.models import load_model
from scipy.special import expit

# =============================
# ⚙️ Cấu hình trang
# =============================
st.set_page_config(page_title="Digital Twin Sinh viên", layout="wide")
st.title("🧑‍🎓 Digital Twin Sinh viên — So sánh Thực & Ảo")

# =============================
# 0️⃣ Load mô hình (nếu có)
# =============================
MODEL_PATH = "models/oulad_lstm.h5"
use_real_model = os.path.exists(MODEL_PATH)
if use_real_model:
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Đã tải mô hình LSTM thật")
else:
    st.sidebar.warning("⚠️ Chưa có mô hình thật. Dùng mô phỏng logistic giả lập")

# =============================
# 1️⃣ Load dữ liệu
# =============================
TEST_PATH = "data/processed/test.csv"
REAL_PATH = "data/processed/ou_real.csv"

@st.cache_data
def load_data():
    test = pd.read_csv(TEST_PATH)
    real = pd.read_csv(REAL_PATH) if os.path.exists(REAL_PATH) else pd.DataFrame()
    if "label" not in test.columns and not real.empty and "label" in real.columns:
        test = test.merge(real[["id_student", "label"]], on="id_student", how="left")
    if "label" not in test.columns:
        test["label"] = 0
    return test

df = load_data()
if df.empty:
    st.error("❌ Không tìm thấy dữ liệu test.csv — hãy chạy lại ETL trước.")
    st.stop()

# =============================
# 2️⃣ Chọn sinh viên
# =============================
student_ids = df["id_student"].unique()
selected_student = st.sidebar.selectbox("🎓 Chọn mã sinh viên:", student_ids)
student_real = df[df["id_student"] == selected_student].iloc[0]

# =============================
# 3️⃣ Bố cục hai cột song song
# =============================
colL, colR = st.columns(2)

# ---------------------------------------------------------------------
# 🧩 PHYSICAL TWIN — DỮ LIỆU THẬT
# ---------------------------------------------------------------------
with colL:
    st.header("🧩 PHYSICAL TWIN — Dữ liệu thực tế (LMS)")
    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Tổng lượt click", int(student_real.get("tong_click", 0)))
    col2.metric("📝 Số bài nộp", int(student_real.get("so_bai_nop", 0)))
    col3.metric("💯 Điểm TB", f"{student_real.get('diem_tb', 0):.1f}")

    col4, col5 = st.columns(2)
    col4.metric("📈 Hoàn thành", f"{student_real.get('ti_le_hoan_thanh', 0)*100:.1f}%")
    col5.metric("🗓️ Số tuần hoạt động", int(student_real.get("so_tuan_hoat_dong", 0)))

    # ---- Biểu đồ hoạt động theo tuần ----
    st.subheader("📊 Hoạt động học tập theo tuần")

    week_cols = [c for c in df.columns if c.startswith("w") and c.endswith("_clicks")]
    if week_cols:
        clicks = student_real[week_cols].values
        week_range = list(range(1, len(week_cols) + 1))

        fig_click = px.line(
            x=week_range, y=clicks, markers=True,
            title="Lượt click theo tuần",
            labels={"x": "Tuần học", "y": "Số lượt click"},
        )
        st.plotly_chart(fig_click, use_container_width=True)

        # Heatmap phân bố hoạt động
        fig_heat = go.Figure(data=go.Heatmap(
            z=[clicks],
            x=[f"Tuần {i}" for i in week_range],
            y=["Hoạt động"],
            colorscale="Blues"
        ))
        fig_heat.update_layout(title="Phân bố click theo tuần (Heatmap)")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("⚠️ Chưa có dữ liệu tuần (w0_clicks...).")

# ---------------------------------------------------------------------
# 🤖 DIGITAL TWIN — MÔ PHỎNG AI
# ---------------------------------------------------------------------
with colR:
    st.header("🤖 DIGITAL TWIN — Mô phỏng hành vi học tập")
    st.caption("Điều chỉnh các tham số dưới đây để xem mô hình dự đoán thay đổi ra sao:")

    # ==== Nhập hành vi mô phỏng ====
    input_clicks = st.slider("📚 Tổng lượt truy cập (clicks)", 0, 20000, int(student_real.get("tong_click", 0)))
    input_submits = st.slider("📝 Số bài nộp", 0, 10, int(student_real.get("so_bai_nop", 0)))
    input_avg_score = st.slider("💯 Điểm trung bình", 0, 100, int(student_real.get("diem_tb", 0)))
    input_completion = st.slider("📈 Tỷ lệ hoàn thành (%)", 0, 100, int(student_real.get("ti_le_hoan_thanh", 0) * 100))
    input_active_weeks = st.slider("🗓️ Số tuần hoạt động", 0, 20, int(student_real.get("so_tuan_hoat_dong", 0)))

    simulate_button = st.button("🧪 Chạy mô phỏng Digital Twin")

    if simulate_button:
        X_fake = np.array([[
            input_clicks / 20000,
            input_submits / 10,
            input_avg_score / 100,
            input_completion / 100,
            input_active_weeks / 20
        ]], dtype="float32")

        # ---- Dự đoán kết quả ----
        if use_real_model:
            # reshape (1, timesteps, features)
            X_seq = np.repeat(X_fake[:, np.newaxis, :], 1, axis=1)  # (1,1,5)
            y_prob = model.predict(X_seq, verbose=0).ravel()[0]
        else:
            score = (
                0.3 * (input_avg_score / 100)
                + 0.25 * (input_completion / 100)
                + 0.2 * (input_submits / 10)
                + 0.15 * min(input_clicks / 20000, 1)
                + 0.1 * (input_active_weeks / 20)
            )
            y_prob = expit(3.5 * (score - 0.5))

        y_label = int(y_prob > 0.5)
        result_text = "🎓 ĐẠT" if y_label else "⚠️ KHÔNG ĐẠT"
        st.metric("Kết quả mô phỏng", result_text, f"{y_prob*100:.1f}%")

        # -----------------------------------------------------------------
        # 📊 TWIN DELTA — So sánh Thực & Ảo
        # -----------------------------------------------------------------
        st.subheader("📊 Twin Delta — So sánh chỉ số Thực và Ảo")

        real_vals = np.array([
            student_real.get("tong_click", 0)/20000,
            student_real.get("so_bai_nop", 0)/10,
            student_real.get("diem_tb", 0)/100,
            student_real.get("ti_le_hoan_thanh", 0),
            student_real.get("so_tuan_hoat_dong", 0)/20
        ])
        sim_vals = np.array([
            input_clicks/20000,
            input_submits/10,
            input_avg_score/100,
            input_completion/100,
            input_active_weeks/20
        ])

        delta_df = pd.DataFrame({
            "Chỉ số": ["Tổng click", "Số bài nộp", "Điểm TB", "Hoàn thành", "Tuần hoạt động"],
            "Thực tế": real_vals * 100,
            "Mô phỏng": sim_vals * 100,
            "Sai lệch (%)": (sim_vals - real_vals) * 100
        })
        st.dataframe(delta_df, use_container_width=True)

        # Radar chart
        categories = delta_df["Chỉ số"].tolist()
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=delta_df["Thực tế"], theta=categories, fill="toself", name="Thực tế"))
        fig_radar.add_trace(go.Scatterpolar(r=delta_df["Mô phỏng"], theta=categories, fill="toself", name="Mô phỏng"))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

        # -----------------------------------------------------------------
        # 📈 Biểu đồ tuần — song song thực & mô phỏng
        # -----------------------------------------------------------------
        st.subheader("📈 Biểu đồ tuần: Thực tế vs Mô phỏng (so sánh trực quan)")
        if week_cols:
            clicks_real = student_real[week_cols].values
            clicks_sim = clicks_real * (input_clicks / max(student_real.get("tong_click", 1), 1))

            fig_week = go.Figure()
            fig_week.add_trace(go.Scatter(x=week_range, y=clicks_real, mode='lines+markers', name='Thực tế'))
            fig_week.add_trace(go.Scatter(x=week_range, y=clicks_sim, mode='lines+markers', name='Mô phỏng', line=dict(dash='dash')))
            fig_week.update_layout(title="So sánh lượt click theo tuần (Thực vs Ảo)",
                                   xaxis_title="Tuần học", yaxis_title="Lượt click")
            st.plotly_chart(fig_week, use_container_width=True)

        # -----------------------------------------------------------------
        # 💾 Lưu lịch sử mô phỏng
        # -----------------------------------------------------------------
        os.makedirs("data/simulations", exist_ok=True)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id_student": selected_student,
            "so_bai_nop": input_submits,
            "diem_tb": input_avg_score,
            "ti_le_hoan_thanh": input_completion,
            "tong_click": input_clicks,
            "so_tuan_hoat_dong": input_active_weeks,
            "prob_pass": round(y_prob, 4),
        }
        with open("data/simulations/history.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

st.caption("🔁 Trang này mô phỏng Digital Twin cho từng sinh viên, hiển thị dữ liệu thật (Physical) và mô phỏng (Digital), kèm biểu đồ tuần, radar và lưu lịch sử mô phỏng.")
