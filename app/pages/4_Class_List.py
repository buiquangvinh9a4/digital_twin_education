# app/pages/3_Class_List.py
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Danh sách mô phỏng", layout="wide")
st.title("📋 Danh sách các mô phỏng đã lưu")

HISTORY_FILE = "data/simulations/history.csv"

if os.path.exists(HISTORY_FILE):
    df = pd.read_csv(HISTORY_FILE)
    st.dataframe(df, use_container_width=True)

    st.download_button("📥 Tải toàn bộ dữ liệu CSV", df.to_csv(index=False).encode("utf-8"), "class_simulations.csv")
else:
    st.warning("⚠️ Chưa có mô phỏng nào được lưu. Hãy vào trang **Sinh viên cụ thể** để tạo trước.")
