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
# 2️⃣ SỬ DỤNG TOÀN BỘ CÁC KHÓA HỌC (ALL COURSES)
# =============================
info['id_student'] = info['id_student'].astype(int)
vle['id_student'] = vle['id_student'].astype(int)

pass_labels = {'Pass', 'Distinction'}
info['label'] = info['final_result'].apply(lambda x: 1 if x in pass_labels else 0)

# Tạo khoá nhận dạng duy nhất theo sinh viên-khoá học, giữ nguyên id_student số nguyên để tiện join
info['student_uid'] = (
    info['id_student'].astype(str) + '_' + info['code_module'].astype(str) + '_' + info['code_presentation'].astype(str)
)

# =============================
# 3️⃣ TIỀN XỬ LÝ LOG HOẠT ĐỘNG (VLE) — THEO TỪNG KHÓA HỌC
# =============================
vle = vle[vle['id_student'].isin(info['id_student'])].copy()
# Gắn thông tin khoá học vào VLE theo id_student (join trực tiếp)
vle = vle.merge(
    info[['id_student','code_module','code_presentation','student_uid']].drop_duplicates(),
    on='id_student', how='left', suffixes=("", "_info")
)

# Chuẩn hoá tên cột sau merge (trường hợp pandas tạo hậu tố)
if 'code_module' not in vle.columns:
    for cand in ['code_module_info','code_module_x','code_module_y']:
        if cand in vle.columns:
            vle.rename(columns={cand:'code_module'}, inplace=True)
            break
if 'code_presentation' not in vle.columns:
    for cand in ['code_presentation_info','code_presentation_x','code_presentation_y']:
        if cand in vle.columns:
            vle.rename(columns={cand:'code_presentation'}, inplace=True)
            break
if ('code_module' not in vle.columns) or ('code_presentation' not in vle.columns):
    # Gợi ý debug nhanh
    missing = [c for c in ['code_module','code_presentation'] if c not in vle.columns]
    raise RuntimeError(f"Merge failed: missing columns {missing}. Check studentInfo.csv headers and id_student types.")

# Tính min_date theo từng (module, presentation) để chuẩn hoá week_idx
course_min = vle.groupby(['code_module','code_presentation'])['date'].min().reset_index().rename(columns={'date':'min_date'})
vle = vle.merge(course_min, on=['code_module','code_presentation'], how='left')
vle['day_shift'] = vle['date'] - vle['min_date']
vle['week_idx'] = (vle['day_shift'] // 7).astype(int)

weekly = vle.groupby(['student_uid','week_idx'], as_index=False)['sum_click'].sum()

MAX_WEEKS = 20

def build_matrix(df):
    students = df['student_uid'].unique()
    mat = np.zeros((len(students), MAX_WEEKS), dtype=np.float32)
    for i, sid in enumerate(students):
        sub = df[df['student_uid']==sid]
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

# --- Số bài đã nộp & điểm trung bình --- theo từng khoá học
ass = ass[ass['id_student'].isin(info['id_student'])].copy()
# Gắn khoá học cho bài đánh giá qua studentInfo (join trực tiếp)
ass = ass.merge(
    info[['id_student','code_module','code_presentation','student_uid']].drop_duplicates(),
    on='id_student', how='left'
)

ass_count = ass.groupby('student_uid')['score'].count().reset_index().rename(columns={'score':'so_bai_nop'})
ass_mean  = ass.groupby('student_uid')['score'].mean().reset_index().rename(columns={'score':'diem_tb'})

# --- Tổng số bài cần nộp theo khoá ---
ass_meta_count = ass_meta.groupby(['code_module','code_presentation']).size().reset_index(name='tong_bai')
course_to_total = ass_meta_count.set_index(['code_module','code_presentation'])['tong_bai']
info_course_total = info[['student_uid','code_module','code_presentation']].copy()
info_course_total['tong_bai'] = info_course_total.apply(lambda r: course_to_total.get((r['code_module'], r['code_presentation']), np.nan), axis=1)

# --- Tổng clicks toàn khóa ---
total_click = vle.groupby('student_uid')['sum_click'].sum().reset_index().rename(columns={'sum_click':'tong_click'})

# --- Số tuần hoạt động ---
active_weeks = weekly[weekly['sum_click']>0].groupby('student_uid')['week_idx'].nunique().reset_index().rename(columns={'week_idx':'so_tuan_hoat_dong'})

# --- Gộp tất cả ---
extra_features = ass_count.merge(ass_mean, on='student_uid', how='outer')
extra_features = extra_features.merge(total_click, on='student_uid', how='outer')
extra_features = extra_features.merge(active_weeks, on='student_uid', how='outer')
extra_features = extra_features.merge(info_course_total[['student_uid','tong_bai']], on='student_uid', how='left')
extra_features = extra_features.fillna(0)

# =============================
# 5️⃣ GỘP & CHUẨN HÓA
# =============================
y_df = info[['student_uid','label','final_result']].drop_duplicates().rename(columns={'student_uid':'id_student'})
dataset = X_df.merge(y_df, on='id_student', how='inner')
dataset = dataset.merge(extra_features.rename(columns={'student_uid':'id_student'}), on='id_student', how='left')

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
real_out = y_df[y_df['id_student'].isin(test_ids)].drop_duplicates()
real_out.to_csv(f"{PROCESSED}/ou_real.csv", index=False)

# =============================
# 7️⃣ THỐNG KÊ KẾT QUẢ
# =============================
print("✅ ETL hoàn tất (ALL COURSES).")
print(f"Sinh viên-khóa học huấn luyện: {len(train)}, kiểm thử: {len(test)}")
print("Các chỉ số sẵn sàng:")
print(" → Tổng lượt truy cập (theo tuần)")
print(" → Số bài đã nộp (so_bai_nop)")
print(" → Điểm trung bình (diem_tb)")
print(" → Tỷ lệ hoàn thành (%) (ti_le_hoan_thanh)")
print(" → Tổng clicks toàn khóa (tong_click)")
print(" → Số tuần hoạt động (so_tuan_hoat_dong)")
