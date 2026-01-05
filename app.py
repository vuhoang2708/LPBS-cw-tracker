import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIG & BRANDING (LPBS THEME)
# ==========================================
st.set_page_config(page_title="LPBS CW Tracker", layout="wide", page_icon="🔶")

# Tính giờ Việt Nam
vn_time = datetime.utcnow() + timedelta(hours=7)
build_time_str = vn_time.strftime("%H:%M:%S - %d/%m/%Y")

# CSS TÙY BIẾN
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    h1, h2, h3 { color: #5D4037 !important; }
    
    [data-testid="stSidebar"] {
        background-color: #FFF8E1;
        border-right: 1px solid #FFECB3;
    }
    
    .metric-card {
        background: linear-gradient(to right, #FFF3E0, #FFFFFF);
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #FF8F00;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #4E342E;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; 
        background-color: #FFF8E1; 
        border-radius: 5px 5px 0px 0px; 
        color: #5D4037;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF8F00 !important;
        color: white !important;
    }

    .debug-box { 
        background-color: #FFF3E0; 
        color: #BF360C; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px dashed #FF8F00; 
    }
    .guide-box {
        background-color: #E8F5E9;
        border-left: 4px solid #2E7D32;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Style cho khu vực upload ảnh */
    .ocr-box {
        border: 2px dashed #FF8F00;
        padding: 10px;
        border-radius: 10px;
        background-color: white;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER
# ==========================================
class DataManager:
    @staticmethod
    def get_default_master_data():
        """Dữ liệu mặc định mới nhất (Hardcoded từ file lpbs cw.csv)"""
        data = [
            {"Mã CW": "CMWG2519", "Mã CS": "MWG", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 88000, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWVHM2522", "Mã CS": "VHM", "Tỷ lệ CĐ": "10:1", "Giá thực hiện": 106000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWSTB2505", "Mã CS": "STB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 60000, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CHPG2516", "Mã CS": "HPG", "Tỷ lệ CĐ": "4:1", "Giá thực hiện": 32000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CACB2502", "Mã CS": "ACB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 28000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CMBB2504", "Mã CS": "MBB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 22000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CMSN2518", "Mã CS": "MSN", "Tỷ lệ CĐ": "10:1", "Giá thực hiện": 95000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CVNM2524", "Mã CS": "VNM", "Tỷ lệ CĐ": "8:1", "Giá thực hiện": 72000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CSHB2525", "Mã CS": "SHB", "Tỷ lệ CĐ": "1:1", "Giá thực hiện": 12500, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CFPT2514", "Mã CS": "FPT", "Tỷ lệ CĐ": "8:1", "Giá thực hiện": 110000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CTCB2507", "Mã CS": "TCB", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 45000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CVPB2511", "Mã CS": "VPB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 21500, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CVIB2510", "Mã CS": "VIB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 23000, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"}
        ]
        return pd.DataFrame(data)

    @staticmethod
    def get_realtime_price(symbol):
        base_prices = {
            "HPG": 28500, "MWG": 48200, "VHM": 41800, "STB": 30500, "VNM": 66000,
            "FPT": 95000, "MBB": 18500, "TCB": 33000, "VPB": 19200, "MSN": 62000,
            "VIB": 21500, "SHB": 11200, "ACB": 24500
        }
        noise = np.random.uniform(0.99, 1.01)
        return base_prices.get(symbol, 20000) * noise

    @staticmethod
    def smart_find_column(df, keywords):
        for col in df.columns:
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return col
        return None

    @staticmethod
    def clean_number_value(val):
        s = str(val)
        if ':' in s: s = s.split(':')[0]
        s = re.sub(r'[^\d.]', '', s)
        try:
            return float(s)
        except:
            return 0.0

# ==========================================
# 3. LOGIC LAYER
# ==========================================
class FinancialEngine:
    @staticmethod
    def calc_intrinsic_value(price_underlying, price_exercise, ratio):
        if ratio <= 0: return 0
        return max((price_underlying - price_exercise) / ratio, 0)

    @staticmethod
    def calc_bep(price_exercise, price_cost, ratio):
        return price_exercise + (price_cost * ratio)

# ==========================================
# 4. UI PRESENTATION
# ==========================================
def main():
    st.title("🔶 LPBS CW Tracker & Simulator")
    st.caption(f"Credit: VuHoang | Build: {build_time_str}")

    # --- SIDEBAR ---
    with st.sidebar:
        # 1. OCR SECTION (TOP PRIORITY)
        st.header("📸 Quét Biên lai / Lệnh đặt")
        st.markdown('<div class="ocr-box">', unsafe_allow_html=True)
        uploaded_img = st.file_uploader("Tải ảnh biên lai/SMS (Beta)", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            st.info("Đã nhận ảnh. Hệ thống đang trích xuất dữ liệu... (Giả lập: Vui lòng kiểm tra lại thông tin bên dưới)")
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # 2. DATA LOADING (DEFAULT OR CSV)
        # Mặc định load dữ liệu cứng
        master_df = DataManager.get_default_master_data()
        
        # Logic Import CSV (Ẩn trong Expander)
        with st.expander("⚙️ Cập nhật Dữ liệu gốc (Admin)"):
            uploaded_csv = st.file_uploader("Upload file CSV mới", type=["csv"])
            if uploaded_csv is not None:
                try:
                    temp_df = pd.read_csv(uploaded_csv)
                    temp_df.columns = temp_df.columns.str.strip()
                    # Smart Mapping logic (giữ nguyên để phòng hờ)
                    # ... (Logic mapping cũ) ...
