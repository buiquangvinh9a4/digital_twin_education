import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# =============================
# ⚙️ CẤU HÌNH TRANG
# =============================
st.set_page_config(page_title="Cảnh báo sớm học tập", layout="wide")
st.title("🚨 Trung tâm cảnh báo sớm — Digital Twin trong giáo dục")

TEST_PATH = "data/processed/test.csv"
PRED_PATH = "data/processed/ou_pred.csv"

# =============================
# 1️⃣ NẠP DỮ LIỆU
# =============================
@st.cache_data
def load_data():
    if not os.path.exists(TEST_PATH):
        st.error("❌ Không tìm thấy test.csv. Hãy chạy lại ETL.")
        return pd.DataFrame()
    df = pd.read_csv(TEST_PATH)
    if os.path.exists(PRED_PATH):
        pred = pd.read_csv(PRED_PATH)
        df = df.merge(pred[["id_student", "predicted_label", "prob_pass"]], on="id_student", how="left")
    else:
        st.warning("⚠️ Chưa có dữ liệu mô phỏng (ou_pred.csv), sẽ chỉ hiển thị phần thực tế.")
        df["predicted_label"] = np.nan
        df["prob_pass"] = np.nan

    # Bổ sung cột rỗng nếu thiếu
    for c in ["tong_click", "so_bai_nop", "diem_tb", "ti_le_hoan_thanh", "so_tuan_hoat_dong", "label"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

df = load_data()
if df.empty:
    st.stop()

# =============================
# 2️⃣ HÀM TÍNH CHỈ SỐ TỔNG HỢP
# =============================
def compute_learning_index(df_input):
    df_calc = df_input.copy()
    df_calc["composite_index"] = (
        0.3 * df_calc["diem_tb"].fillna(0) / 100 +
        0.25 * df_calc["ti_le_hoan_thanh"].fillna(0) +
        0.2 * df_calc["so_bai_nop"].fillna(0) / 10 +
        0.15 * df_calc["tong_click"].fillna(0) / 20000 +
        0.1 * df_calc["so_tuan_hoat_dong"].fillna(0) / 20
    ) * 100
    return df_calc

df = compute_learning_index(df)

# =============================
# 3️⃣ CÁC THAM SỐ MÔ PHỎNG
# =============================
st.sidebar.header("⚙️ Điều chỉnh tham số mô phỏng")

click_factor = st.sidebar.slider("📚 Tăng/Giảm lượt click (%)", -50, 50, 0)
submit_factor = st.sidebar.slider("📝 Tăng/Giảm số bài nộp (%)", -50, 50, 0)
score_factor = st.sidebar.slider("💯 Tăng/Giảm điểm TB (%)", -50, 50, 0)
completion_factor = st.sidebar.slider("📈 Tăng/Giảm tỷ lệ hoàn thành (%)", -50, 50, 0)
week_factor = st.sidebar.slider("🗓️ Tăng/Giảm số tuần hoạt động (%)", -50, 50, 0)

threshold = st.sidebar.slider("🚨 Ngưỡng cảnh báo (Learning Index)", 0, 100, 50)
show_download = st.sidebar.checkbox("Hiển thị nút tải danh sách cảnh báo", True)

# Tạo bản sao mô phỏng
df_sim = df.copy()
df_sim["tong_click"] *= (1 + click_factor / 100)
df_sim["so_bai_nop"] *= (1 + submit_factor / 100)
df_sim["diem_tb"] *= (1 + score_factor / 100)
df_sim["ti_le_hoan_thanh"] *= (1 + completion_factor / 100)
df_sim["so_tuan_hoat_dong"] *= (1 + week_factor / 100)
df_sim = compute_learning_index(df_sim)

# =============================
# 4️⃣ XÁC ĐỊNH NGUY CƠ TRƯỢT
# =============================
df_sim["at_risk"] = (df_sim["composite_index"] < threshold) | (df_sim["prob_pass"] < 0.5)
at_risk_students = df_sim[df_sim["at_risk"] == True].copy()

st.subheader("📋 Danh sách sinh viên có nguy cơ trượt học phần")

if len(at_risk_students) == 0:
    st.success("🎓 Không có sinh viên nào trong vùng nguy cơ trượt học phần theo mô phỏng hiện tại.")
else:
    st.error(f"⚠️ Có {len(at_risk_students)} sinh viên đang trong vùng nguy cơ!")
    cols_show = [
        "id_student", "diem_tb", "ti_le_hoan_thanh", "so_bai_nop",
        "tong_click", "so_tuan_hoat_dong", "composite_index", "prob_pass"
    ]
    st.dataframe(at_risk_students[cols_show].sort_values("composite_index", ascending=True), use_container_width=True)

    # Nút tải CSV
    if show_download:
        csv = at_risk_students.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Tải danh sách cảnh báo", csv, "at_risk_students.csv", "text/csv")

# =============================
# 5️⃣ BIỂU ĐỒ PHÂN BỐ & RADAR
# =============================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Phân bố chỉ số học tập (Learning Index)")
    fig_hist = px.histogram(
        df_sim, x="composite_index", nbins=30,
        color=df_sim["at_risk"].map({True: "Cảnh báo", False: "An toàn"}),
        title="Phân bố Learning Index sau khi điều chỉnh tham số",
        labels={"composite_index": "Learning Index"}
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Ngưỡng cảnh báo")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("📈 Radar so sánh trung bình lớp (Thực vs Mô phỏng)")
    metrics = ["tong_click", "so_bai_nop", "diem_tb", "ti_le_hoan_thanh", "so_tuan_hoat_dong"]
    real_avg = [df[m].mean() for m in metrics]
    sim_avg = [df_sim[m].mean() for m in metrics]
    radar_df = pd.DataFrame({
        "Chỉ số": ["Click", "Nộp bài", "Điểm TB", "Hoàn thành", "Tuần hoạt động"],
        "Thực tế": real_avg,
        "Mô phỏng": sim_avg
    })
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=radar_df["Thực tế"], theta=radar_df["Chỉ số"],
                                        fill="toself", name="Thực tế"))
    fig_radar.add_trace(go.Scatterpolar(r=radar_df["Mô phỏng"], theta=radar_df["Chỉ số"],
                                        fill="toself", name="Mô phỏng", line=dict(dash="dash")))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# =============================
# 6️⃣ THỐNG KÊ TỔNG QUAN
# =============================
st.markdown("---")
st.subheader("📊 Thống kê tổng quan sau điều chỉnh")

colA, colB, colC = st.columns(3)
colA.metric("🎯 Trung bình Learning Index", f"{df_sim['composite_index'].mean():.1f}")
colB.metric("🚨 Tỷ lệ nguy cơ trượt (%)", f"{(len(at_risk_students)/len(df_sim)*100):.1f}%")
colC.metric("📘 Tỷ lệ đạt mô phỏng (%)", f"{(1 - len(at_risk_students)/len(df_sim))*100:.1f}%")


# =============================
# 8️⃣ GIẢI THÍCH & GỢI Ý CAN THIỆP
# =============================
st.markdown("---")
st.subheader("🧭 Giải thích chi tiết & gợi ý can thiệp cho sinh viên")

if len(at_risk_students) > 0:
    selected_student = st.selectbox(
        "Chọn sinh viên cần xem chi tiết:",
        at_risk_students["id_student"].unique(),
        key="explain_student"
    )

    stu = at_risk_students[at_risk_students["id_student"] == selected_student].iloc[0]
    st.markdown(f"### 👩‍🎓 Mã sinh viên: `{selected_student}`")

    # --- So sánh cá nhân với trung bình lớp ---
    metrics = ["tong_click", "so_bai_nop", "diem_tb", "ti_le_hoan_thanh", "so_tuan_hoat_dong"]
    labels = ["Tổng click", "Số bài nộp", "Điểm TB", "Tỷ lệ hoàn thành", "Tuần hoạt động"]
    weights = [0.15, 0.2, 0.3, 0.25, 0.1]

    class_avg = df_sim[metrics].mean()
    diff = (stu[metrics] - class_avg) / class_avg * 100

    explain_df = pd.DataFrame({
        "Chỉ số": labels,
        "Giá trị cá nhân": [stu[m] for m in metrics],
        "Trung bình lớp": class_avg.values,
        "Sai lệch (%)": diff.values,
        "Trọng số": weights
    })

    explain_df["Tác động"] = explain_df["Sai lệch (%)"] * explain_df["Trọng số"]
    explain_df = explain_df.sort_values("Tác động")

    st.markdown("#### 📊 So sánh cá nhân với trung bình lớp")
    st.dataframe(explain_df.style.format({
        "Giá trị cá nhân": "{:.2f}",
        "Trung bình lớp": "{:.2f}",
        "Sai lệch (%)": "{:+.1f}",
        "Trọng số": "{:.2f}",
        "Tác động": "{:+.2f}"
    }), use_container_width=True)

    # --- Biểu đồ ---
    fig_bar = px.bar(
        explain_df, x="Chỉ số", y="Tác động", color="Tác động",
        color_continuous_scale="RdYlGn",
        title="Đóng góp của từng yếu tố tới nguy cơ trượt (âm = tiêu cực)"
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="Ảnh hưởng (theo trọng số)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Gợi ý can thiệp ---
    st.markdown("#### 💡 Gợi ý cải thiện")

    weakest = explain_df.iloc[0]["Chỉ số"]
    suggestions = {
        "Điểm TB": "Cần ôn tập lại nội dung học phần, làm thêm bài luyện tập hoặc nhận hỗ trợ từ giảng viên.",
        "Tỷ lệ hoàn thành": "Hoàn thành các bài tập và hoạt động còn thiếu để cải thiện tiến độ học tập.",
        "Số bài nộp": "Cần đảm bảo nộp đầy đủ các bài tập, dự án đúng hạn.",
        "Tổng click": "Tăng cường truy cập hệ thống, xem thêm tài liệu và tham gia diễn đàn.",
        "Tuần hoạt động": "Duy trì tham gia đều đặn hàng tuần, tránh gián đoạn quá lâu."
    }
    st.info(f"🔎 Yếu tố ảnh hưởng lớn nhất: **{weakest}**\n\n🧩 Gợi ý: {suggestions.get(weakest, 'Hãy cải thiện hoạt động học tập tổng thể.')}")
else:
    st.info("🎓 Hiện chưa có sinh viên nào thuộc vùng cảnh báo để hiển thị chi tiết.")


# =============================
# 7️⃣ GHI CHÚ
# =============================
st.caption("""
🧭 Trang này là **Trung tâm cảnh báo sớm** của mô hình Digital Twin.  
Hệ thống cho phép điều chỉnh các tham số học tập (số bài nộp, lượt truy cập, điểm trung bình...)  
để **dự đoán và phát hiện sớm sinh viên có khả năng trượt học phần**,  
giúp giảng viên can thiệp kịp thời.
""")
