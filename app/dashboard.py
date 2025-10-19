# app/dashboard.py
import streamlit as st

st.set_page_config(
    page_title="Digital Twin Lớp học — OULAD",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Mô hình Song Sinh Số (Digital Twin) — OULAD Learning Analytics")
st.markdown("""
Xin chào 👋  
Đây là hệ thống mô phỏng kết quả học tập dựa trên dữ liệu **OULAD** (Open University Learning Analytics Dataset).

**Chọn trang bên trái để bắt đầu:**
- **Tổng quan lớp học:** Quan sát hành vi và kết quả của toàn lớp  
- **Sinh viên cụ thể:** Xem và mô phỏng hành vi học tập của một sinh viên  
- **Danh sách mô phỏng:** Lưu và xem lại các kết quả mô phỏng đã chạy
""")

st.sidebar.success("➡️ Hãy chọn 1 trang ở thanh bên trái để bắt đầu.")
