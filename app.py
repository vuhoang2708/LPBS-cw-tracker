import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import google.generativeai as genai
import json
from json import JSONDecoder
from datetime import datetime, timedelta
from PIL import Image

# ==========================================
# 1. CONFIG & BRANDING
# ==========================================
st.set_page_config(page_title="LPBS CW Tracker & Simulator", layout="wide", page_icon="🔶")

vn_time = datetime.utcnow() + timedelta(hours=7)
build_time_str = vn_time.strftime("%H:%M:%S - %d/%m/%Y")

# --- SECURITY: SECRETS ONLY ---
if "GEMINI_API_KEY" in st.secrets:
    SYSTEM_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    SYSTEM_API_KEY = None 

st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    h1, h2, h3 { color: #5D4037 !important; font-family: 'Segoe UI', sans-serif; }
    
    [data-testid="stSidebar"] {
        background-color: #FFF8E1;
        border-right: 1px solid #FFECB3;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #FFF; border-radius: 6px; 
        color: #666; font-weight: 600; border: 1px solid #EEE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF8F00 !important; color: white !important; border-color: #FF8F00;
    }
    
    .stRadio [role="radiogroup"] {
        background-color: #FFF; padding: 10px; border-radius: 8px;
        border: 1px solid #EEE; justify-content: center;
    }

    .metric-card {
        background: white; padding: 20px; border-radius: 12px; 
        border: 1px solid #EEE; border-left: 5px solid #FF8F00;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); color: #4E342E; margin-bottom: 15px;
    }
    
    .cw-profile-box {
        background-color: #E3F2FD; border: 1px solid #90CAF9;
        border-radius: 10px; padding: 15px; margin-bottom: 20px; color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER
# ==========================================
class DataManager:
    @staticmethod
    def get_default_master_data():
        data = [
            {"Mã CW": "CWMWG2519", "Mã CS": "MWG", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 88000, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWVHM2522", "Mã CS": "VHM", "Tỷ lệ CĐ": "10:1", "Giá thực hiện": 106000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWSTB2505", "Mã CS": "STB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 60000, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWHPG2516", "Mã CS": "HPG", "Tỷ lệ CĐ": "4:1", "Giá thực hiện": 32000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWACB2502", "Mã CS": "ACB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 28000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWMBB2504", "Mã CS": "MBB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 22000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWMSN2518", "Mã CS": "MSN", "Tỷ lệ CĐ": "10:1", "Giá thực hiện": 95000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWVNM2524", "Mã CS": "VNM", "Tỷ lệ CĐ": "8:1", "Giá thực hiện": 72000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWSHB2525", "Mã CS": "SHB", "Tỷ lệ CĐ": "1:1", "Giá thực hiện": 12500, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWFPT2514", "Mã CS": "FPT", "Tỷ lệ CĐ": "8:1", "Giá thực hiện": 110000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWTCB2507", "Mã CS": "TCB", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 45000, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWVPB2511", "Mã CS": "VPB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 21500, "Ngày đáo hạn": "2026-12-28", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CWVIB2510", "Mã CS": "VIB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 23000, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"}
        ]
        return pd.DataFrame(data)

    @staticmethod
    def get_realtime_price_simulated(symbol):
        base_prices = {"HPG":28500,"MWG":48200,"VHM":41800,"STB":30500,"VNM":66000,"FPT":95000,"MBB":18500,"TCB":33000,"VPB":19200,"MSN":62000,"VIB":21500,"SHB":11200,"ACB":24500}
        noise = np.random.uniform(0.99, 1.01)
        return base_prices.get(symbol, 20000) * noise

    @staticmethod
    def clean_number_value(val):
        s = str(val)
        if ':' in s: s = s.split(':')[0]
        s = re.sub(r'[^\d.]', '', s)
        try: return float(s)
        except: return 0.0

    @staticmethod
    def calc_days_to_maturity(date_str):
        try:
            mat_date = pd.to_datetime(date_str)
            now = datetime.utcnow() + timedelta(hours=7)
