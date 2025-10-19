import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW = "data/raw"
PROCESSED = "data/processed"
os.makedirs(PROCESSED, exist_ok=True)

# =============================
# 1️⃣ NẠP DỮ LIỆU GỐC
# =============================
info = pd.read_csv(f"{RAW}/studentInfo.csv")
vle = pd.read_csv(f"{RAW}/studentVle.csv")           # id_student, id_site, date, sum_click
ass = pd.read_csv(f"{RAW}/studentAssessment.csv")    # điểm từng assessment
ass_meta = pd.read_csv(f"{RAW}/assessments.csv")     # id_assessment, date, weight
courses = pd.read_csv(f"{RAW}/courses.csv")

# =============================
# 2️⃣ CHỌN KHÓA HỌC THÍ NGHIỆM
# =============================
top_course = info.groupby(['code_module','code_presentation']).size().sort_values(ascending=False).index[0]
mod, pres = top_course
print(f"🔍 Đang xử lý khóa học: {mod}-{pres}")

info = info[(info['code_module']==mod) & (info['code_presentation']==pres)]
pass_labels = {'Pass', 'Distinction'}
info['label'] = info['final_result'].apply(lambda x: 1 if x in pass_labels else 0)

# =============================
# 3️⃣ TIỀN XỬ LÝ LOG HOẠT ĐỘNG (VLE)
# =============================
vle = vle[vle['id_student'].isin(info['id_student'])].copy()
min_date = vle['date'].min()
vle['day_shift'] = vle['date'] - min_date
vle['week_idx'] = (vle['day_shift'] // 7).astype(int)

weekly = vle.groupby(['id_student','week_idx'], as_index=False)['sum_click'].sum()

MAX_WEEKS = 20

def build_matrix(df):
    students = df['id_student'].unique()
    mat = np.zeros((len(students), MAX_WEEKS), dtype=np.float32)
    for i, sid in enumerate(students):
        sub = df[df['id_student']==sid]
        for _, r in sub.iterrows():
            w = int(r['week_idx'])
            if 0 <= w < MAX_WEEKS:
                mat[i, w] = r['sum_click']
    return students, mat

student_ids, X_clicks = build_matrix(weekly)

X_df = pd.DataFrame(X_clicks, columns=[f"w{w}_clicks" for w in range(MAX_WEEKS)])
X_df.insert(0, 'id_student', student_ids)

# =============================
# 4️⃣ CÁC CHỈ SỐ PHỤ TRỢ KHÁC
# =============================

# --- Số bài đã nộp & điểm trung bình ---
ass = ass[ass['id_student'].isin(info['id_student'])].copy()
ass_count = ass.groupby('id_student')['score'].count().reset_index().rename(columns={'score':'so_bai_nop'})
ass_mean  = ass.groupby('id_student')['score'].mean().reset_index().rename(columns={'score':'diem_tb'})

# --- Tổng số bài cần nộp theo khoá ---
total_ass = ass_meta.shape[0]
ass_count['tong_bai'] = total_ass
ass_count['ti_le_hoan_thanh'] = ass_count['so_bai_nop'] / ass_count['tong_bai']

# --- Tổng clicks toàn khóa ---
total_click = vle.groupby('id_student')['sum_click'].sum().reset_index().rename(columns={'sum_click':'tong_click'})

# --- Số tuần hoạt động ---
active_weeks = weekly[weekly['sum_click']>0].groupby('id_student')['week_idx'].nunique().reset_index().rename(columns={'week_idx':'so_tuan_hoat_dong'})

# --- Gộp tất cả ---
extra_features = ass_count.merge(ass_mean, on='id_student', how='outer')
extra_features = extra_features.merge(total_click, on='id_student', how='outer')
extra_features = extra_features.merge(active_weeks, on='id_student', how='outer')
extra_features = extra_features.fillna(0)

# =============================
# 5️⃣ GỘP & CHUẨN HÓA
# =============================
y_df = info[['id_student','label','final_result']].drop_duplicates()
dataset = X_df.merge(y_df, on='id_student', how='inner')
dataset = dataset.merge(extra_features, on='id_student', how='left')

# Chuẩn hóa clicks (0-1)
click_cols = [c for c in dataset.columns if c.startswith('w') and c.endswith('_clicks')]
mins = dataset[click_cols].min()
maxs = dataset[click_cols].max().replace(0, 1)
dataset[click_cols] = (dataset[click_cols] - mins) / (maxs - mins)

# Lưu scaler
scaler = pd.DataFrame({'col': click_cols, 'min': mins.values, 'max': maxs.values})
scaler.to_csv(f"{PROCESSED}/scaler_minmax.csv", index=False)

# =============================
# 6️⃣ CHIA TẬP TRAIN/TEST
# =============================
train_ids, test_ids = train_test_split(
    dataset['id_student'], test_size=0.2, random_state=42, stratify=dataset['label']
)

train = dataset[dataset['id_student'].isin(train_ids)].reset_index(drop=True)
test  = dataset[dataset['id_student'].isin(test_ids)].reset_index(drop=True)

train.to_csv(f"{PROCESSED}/train.csv", index=False)
test.to_csv(f"{PROCESSED}/test.csv", index=False)

# Lưu bản “thực tế” cho dashboard
real_out = info[['id_student','final_result','label']]
real_out = real_out[real_out['id_student'].isin(test_ids)].drop_duplicates()
real_out.to_csv(f"{PROCESSED}/ou_real.csv", index=False)

# =============================
# 7️⃣ THỐNG KÊ KẾT QUẢ
# =============================
print("✅ ETL hoàn tất.")
print(f"Khóa học: {mod}-{pres}")
print(f"Sinh viên huấn luyện: {len(train)}, kiểm thử: {len(test)}")
print("Các chỉ số sẵn sàng:")
print(" → Tổng lượt truy cập (theo tuần)")
print(" → Số bài đã nộp (so_bai_nop)")
print(" → Điểm trung bình (diem_tb)")
print(" → Tỷ lệ hoàn thành (%) (ti_le_hoan_thanh)")
print(" → Tổng clicks toàn khóa (tong_click)")
print(" → Số tuần hoạt động (so_tuan_hoat_dong)")
