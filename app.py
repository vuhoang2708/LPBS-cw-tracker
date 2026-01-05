import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# ==========================================
# 1. CONFIG & SYSTEM SETTINGS
# ==========================================
st.set_page_config(page_title="LPBank CW Tracker", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;}
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; border-bottom: 2px solid #4CAF50; }
    .uploaded-file { border: 1px dashed #4CAF50; padding: 10px; border-radius: 5px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER
# ==========================================
class DataManager:
    @staticmethod
    def get_default_master_data():
        return pd.DataFrame([
            {"Mã CW": "CHPG2316", "Mã CS": "HPG", "Tỷ lệ CĐ": 2, "Giá thực hiện": 28000, "Ngày đáo hạn": "2026-06-01", "Trạng thái": "Listed"},
            {"Mã CW": "CMWG2305", "Mã CS": "MWG", "Tỷ lệ CĐ": 5, "Giá thực hiện": 45000, "Ngày đáo hạn": "2026-12-31", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CVHM2322", "Mã CS": "VHM", "Tỷ lệ CĐ": 4, "Giá thực hiện": 42000, "Ngày đáo hạn": "2026-08-15", "Trạng thái": "Listed"},
        ])

    @staticmethod
    def get_realtime_price(symbol):
        base_prices = {
            "HPG": 28500, "MWG": 48200, "VHM": 41800, "STB": 30500, "VNM": 66000,
            "FPT": 95000, "MBB": 18500, "TCB": 33000, "VPB": 19200, "MSN": 62000,
            "VIB": 21500, "SHB": 11200, "ACB": 24500
        }
        noise = np.random.uniform(0.99, 1.01)
        return base_prices.get(symbol, 20000) * noise

# ==========================================
# 3. LOGIC LAYER
# ==========================================
class FinancialEngine:
    @staticmethod
    def calc_intrinsic_value(price_underlying, price_exercise, ratio):
        try:
            p_u = float(price_underlying)
            p_e = float(price_exercise)
            r = float(ratio)
            if r == 0: return 0
            return max((p_u - p_e) / r, 0)
        except:
            return 0

    @staticmethod
    def calc_bep(price_exercise, price_cost, ratio):
        try:
            p_e = float(price_exercise)
            p_c = float(price_cost)
            r = float(ratio)
            return p_e + (p_c * r)
        except:
            return 0

# ==========================================
# 4. UI PRESENTATION
# ==========================================
def main():
    st.title("📈 LPBank Invest - CW Tracker & Simulator")
    # --- CẬP NHẬT VERSION TIMESTAMP ---
    st.caption("System Architect: AI Guardian | Version: 4.3 | Build: 14:55 05/01/2026 (Fix Data Parsing)")

    # --- SIDEBAR: IMPORT & CONFIG ---
    with st.sidebar:
        st.header("📂 Dữ liệu Nguồn (Master Data)")
        
        uploaded_file = st.file_uploader("Upload danh sách CW (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                master_df = pd.read_csv(uploaded_file)
                master_df.columns = master_df.columns.str.strip()
                
                # === DATA CLEANING (REGEX) ===
                numeric_cols = ["Giá thực hiện", "Tỷ lệ CĐ"]
                for col in numeric_cols:
                    if col in master_df.columns:
                        # Chỉ giữ lại số và dấu chấm
                        master_df[col] = master_df[col].astype(str).apply(lambda x: re.sub(r'[^\d.]', '', x))
                        master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
                # =============================

                st.success(f"✅ Đã tải {len(master_df)} mã CW từ file.")
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
                master_df = DataManager.get_default_master_data()
        else:
            st.info("Đang dùng dữ liệu mẫu. Hãy upload file CSV để cập nhật.")
            master_df = DataManager.get_default_master_data()

        st.divider()
        st.header
