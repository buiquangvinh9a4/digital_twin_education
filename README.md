# 🎓 Digital Twin for Learning Analytics (OULAD Dataset)

Dự án mô phỏng hệ thống **Digital Twin trong giáo dục** dựa trên dữ liệu OULAD,  
giúp giảng viên quan sát, dự đoán và can thiệp sớm hành vi học tập của sinh viên.

---

## 🚀 Cấu trúc hệ thống

| Trang | Mô tả |
|-------|-------|
| **1️⃣ Class Overview** | Song song hiển thị dữ liệu *Physical Twin* (thực tế) và *Digital Twin* (mô phỏng) |
| **2️⃣ Student Twin** | Phân tích và mô phỏng hành vi học tập của từng sinh viên |
| **3️⃣ Model Training** | Huấn luyện mô hình LSTM dự đoán kết quả học tập (nếu kích hoạt) |
| **4️⃣ Early Warning Center** | Cảnh báo sớm sinh viên có nguy cơ trượt học phần, kèm giải thích & gợi ý cải thiện |

---

## 🧩 Chạy ứng dụng

```bash
# Bật môi trường ảo
source .venv/bin/activate

# Cài gói phụ thuộc
pip install -r requirements.txt

# Chạy ứng dụng Streamlit
streamlit run app/Home.py
