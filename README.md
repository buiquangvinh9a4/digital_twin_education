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
| **5️⃣ Live Monitor** | Giám sát thời gian thực (giả lập) các chỉ số lớp và scenario |

---

## 🧩 Chạy ứng dụng

```bash
# Bật môi trường ảo
source .venv/bin/activate

# Cài gói phụ thuộc
pip install -r requirements.txt

# Chạy ứng dụng Streamlit
streamlit run app/dashboard.py

```
---

## 🔴 Demo realtime (giả lập Digital Twin)

Trong môi trường bài tập lớn, bạn có thể mô phỏng dữ liệu realtime dựa trên `data/processed/test.csv` để dashboard tự cập nhật:

```bash
# 1) Chuẩn bị dữ liệu processed (nếu chưa có)
python scripts/etl_prepare.py

# 2) Bật watcher để tái huấn luyện/dự đoán khi test.csv đổi
python scripts/update_twin.py

# 3) Chạy simulator realtime (luân phiên kịch bản: baseline, exam_season, holiday...)
python scripts/simulate_realtime.py

# 4) Mở dashboard (auto-refresh mỗi 5s)
streamlit run app/dashboard.py

# Trang mới: Live Monitor
# Vào trang "5️⃣ Live Monitor" để xem đồ thị chuỗi thời gian theo kịch bản
```

Các kịch bản mô phỏng thay đổi chỉ số: `exam_season`, `holiday`, `early_intervention`, `late_dropout` (tăng/giảm clicks, số bài nộp, điểm TB, tỉ lệ hoàn thành, số tuần hoạt động) áp dụng ngẫu nhiên cho một phần sinh viên mỗi chu kỳ ~5s.

---

## 📄 Báo cáo Digital Twin
- Xem báo cáo tổng quan về kiến trúc và giá trị của Digital Twin: [REPORT.md](./REPORT.md)
