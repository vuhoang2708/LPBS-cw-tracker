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

# --- SECURITY: Ưu tiên Secrets, Fallback sang Hardcode (User Request) ---
if "GEMINI_API_KEY" in st.secrets:
    SYSTEM_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    # Key dự phòng để test nhanh (theo yêu cầu)
    SYSTEM_API_KEY = "AIzaSyD2K9Chj6mRACW-yyrPG8egSLdXDKzjA3M"

st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    h1, h2, h3 { color: #5D4037 !important; font-family: 'Segoe UI', sans-serif; }
    
    [data-testid="stSidebar"] {
        background-color: #FFF8E1;
        border-right: 1px solid #FFECB3;
    }
    
    /* UX: Tùy chỉnh Tab & Radio */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #FFF; border-radius: 6px; 
        color: #666; font-weight: 600; border: 1px solid #EEE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF8F00 !important; color: white !important; border-color: #FF8F00;
    }
    
    /* Radio Button nằm ngang đẹp hơn */
    .stRadio [role="radiogroup"] {
        background-color: #FFF;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #EEE;
        justify-content: center;
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
            {"Mã CW": "CMWG2519", "Mã CS": "MWG", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 88000, "Ngày đáo hạn": "2026-06-29", "Trạng thái": "Pre-listing"},
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
            delta = mat_date - now
            return delta.days
        except: return 0

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

    @staticmethod
    def get_moneyness(price_underlying, price_exercise):
        if price_underlying > price_exercise: return "ITM (Có lời)", "green"
        elif price_underlying < price_exercise: return "OTM (Chưa lời)", "red"
        else: return "ATM (Ngang giá)", "orange"

# ==========================================
# 4. AI SERVICE LAYER
# ==========================================
def process_image_with_gemini(image, api_key, mode="ALL"):
    genai.configure(api_key=api_key)
    generation_config = {"temperature": 0.0}
    priority_models = ['gemini-3-flash-preview', 'gemini-2.0-flash-exp'] 
    
    if mode == "BUY_ORDER":
        task_desc = "Trích xuất thông tin LỆNH MUA / BIÊN LAI. Tập trung vào: Mã, Số lượng và Giá vốn (Giá khớp)."
    elif mode == "MARKET_BOARD":
        task_desc = "Trích xuất thông tin BẢNG GIÁ / CAFEF. Tập trung vào: Mã và Giá thị trường (Cột Last/Current)."
    else:
        task_desc = "Trích xuất dữ liệu tài chính."

    prompt = f"""
    Bạn là một trợ lý tài chính (OCR). Nhiệm vụ: {task_desc}
    
    Các trường cần tìm:
    1. Mã chứng khoán (Symbol): Ưu tiên tìm mã CW (ví dụ CWVHM). Nếu là Bảng giá (chỉ có VHM), lấy mã đó.
    2. Số lượng (Qty): Khối lượng mua (Nếu là Bảng giá -> null).
    3. Giá vốn (Price): Giá khớp lệnh/Giá mua (Nếu là Bảng giá -> null).
    4. Giá thị trường (Market Price): Giá hiện tại trên bảng điện (Nếu là Biên lai mua -> null).

    Trả về JSON: 
    {{"symbol": "XXX", "qty": 1000, "price": 50000, "market_price": 52000}}
    """
    
    errors_log = [] 

    for model_name in priority_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image], generation_config=generation_config)
            text = response.text.strip()
            
            start_idx = text.find('{')
            if start_idx != -1:
                json_data, _ = JSONDecoder().raw_decode(text[start_idx:])
                json_data['_processed_by'] = model_name 
                return json_data
            else:
                errors_log.append(f"{model_name}: No JSON found.")
                continue
        except Exception as e:
            errors_log.append(f"{model_name}: {str(e)}")
            continue 
            
    return {"error": "Thất bại. Log lỗi:\n" + "\n".join(errors_log)}

def auto_map_symbol_and_rerun(ocr_result, master_df):
    if not ocr_result: return
    det_sym = str(ocr_result.get('symbol', '')).upper().strip()
    if det_sym:
        mask_exact = master_df['Mã CW'] == det_sym
        mask_contains = master_df['Mã CW'].str.contains(det_sym) | master_df['Mã CS'].str.contains(det_sym)
        core_sym = re.sub(r'[^A-Z]', '', det_sym).replace("CW", "").replace("CV", "")
        if len(core_sym) >= 3:
            mask_core = master_df['Mã CS'].str.contains(core_sym)
        else:
            mask_core = mask_contains

        found_index = -1
        if mask_exact.any(): found_index = master_df.index[mask_exact].tolist()[0]
        elif mask_core.any(): found_index = master_df.index[mask_core].tolist()[0]
        
        if found_index != -1:
            st.session_state['user_index'] = found_index
            st.toast(f"✅ Đã nhận diện: {master_df.iloc[found_index]['Mã CW']}")
            st.rerun() # Refresh UI immediately

# ==========================================
# 5. HELPER
# ==========================================
def render_metric_card(label, value, sub="", color="black"):
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.9em; color:#666; margin-bottom: 5px;">{label}</div>
        <div style="font-size:1.6em; font-weight:bold; color:{color};">{value}</div>
        <div style="font-size:0.85em; color:#888; margin-top: 5px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def render_cw_profile(cw_code, und_code, exercise_price, ratio, maturity_date, days_left):
    st.markdown(f"""
    <div class="cw-profile-box">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h3 style="margin:0; color:#0277BD;">{cw_code} (Cơ sở: {und_code})</h3>
                <small>Ngày đáo hạn: <b>{maturity_date}</b></small>
            </div>
            <div style="text-align:right;">
                 <div>Còn lại: <b>{days_left} ngày</b></div>
                 <small>Tỷ lệ CĐ: <b>{ratio}:1</b> | Giá thực hiện: <b>{exercise_price:,.0f}</b></small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. MAIN APP
# ==========================================
def main():
    st.title("🔶 LPBS CW Tracker & Simulator")
    st.caption(f"System: V12.2 | Build: {build_time_str} | UX Redesign (Radio)")

    # Init Session
    if 'ocr_result' not in st.session_state: st.session_state['ocr_result'] = None
    if 'user_qty' not in st.session_state: st.session_state['user_qty'] = 1000.0
    if 'user_price' not in st.session_state: st.session_state['user_price'] = 1000.0
    if 'user_index' not in st.session_state: st.session_state['user_index'] = 0

    master_df = DataManager.get_default_master_data()
    if "Giá thực hiện" in master_df.columns:
        master_df["Giá thực hiện"] = master_df["Giá thực hiện"].apply(DataManager.clean_number_value)
        master_df["Tỷ lệ CĐ"] = master_df["Tỷ lệ CĐ"].apply(DataManager.clean_number_value)

    # --- SIDEBAR ---
    with st.sidebar:
        if not SYSTEM_API_KEY:
            st.warning("⚠️ Chưa có API Key.")
            active_key = st.text_input("Nhập Key:", type="password")
        else:
            active_key = SYSTEM_API_KEY
            
        st.header("📸 QUY TRÌNH ĐẦU TƯ")
        has_position = (st.session_state['user_qty'] > 0) and (st.session_state['user_price'] > 0)
        
        tab_buy, tab_market = st.tabs(["1️⃣ VỊ THẾ", "2️⃣ GIÁ & P/L"])

        # --- TAB 1: VỊ THẾ ---
        with tab_buy:
            # RADIO: CHỌN CHẾ ĐỘ
            mode_t1 = st.radio("Chế độ nhập:", ["📸 Quét OCR", "✍️ Nhập Tay"], horizontal=True, label_visibility="collapsed")

            if mode_t1 == "📸 Quét OCR":
                st.info("Quét Lệnh Mua / SMS")
                uploaded_buy = st.file_uploader("Chọn ảnh", type=["png", "jpg", "jpeg"], key="u_buy", label_visibility="collapsed")
                
                if uploaded_buy and active_key:
                    if st.button("🚀 Phân Tích", use_container_width=True):
                        with st.spinner("Đang xử lý..."):
                            image = Image.open(uploaded_buy)
                            result = process_image_with_gemini(image, active_key, mode="BUY_ORDER")
                            if "error" in result: st.error(result['error'])
                            else:
                                st.session_state['ocr_result'] = result
                                if result.get('price'):
                                    raw_p = float(result['price'])
                                    if raw_p < 1000 and raw_p > 0: raw_p *= 1000
                                    st.session_state['user_price'] = raw_p
                                if result.get('qty'): st.session_state['user_qty'] = float(result['qty'])
                                auto_map_symbol_and_rerun(result, master_df)
            
            else: # Mode Nhập Tay
                st.info("Nhập liệu thủ công")
                cw_list = master_df["Mã CW"].unique()
                selected_cw = st.selectbox("Chọn Mã CW", cw_list, index=int(st.session_state.get('user_index', 0)))
                
                qty = st.number_input("Số lượng", value=st.session_state['user_qty'], step=100.0)
                cost_price = st.number_input("Giá vốn (VND)", value=st.session_state['user_price'], step=50.0)
                
                # Cập nhật Session ngay khi nhập tay
                st.session_state['user_qty'] = qty
                st.session_state['user_price'] = cost_price
                
                # Cập nhật index nếu user chọn tay
                new_index = list(cw_list).index(selected_cw)
                if new_index != st.session_state['user_index']:
                    st.session_state['user_index'] = new_index
                    st.rerun()

        # --- TAB 2: GIÁ THỊ TRƯỜNG ---
        with tab_market:
            if not has_position:
                st.error("⛔ CHƯA CÓ VỊ THẾ")
                st.caption("Vui lòng nhập xong Tab 1.")
            else:
                # RADIO: CHỌN CHẾ ĐỘ TAB 2
                mode_t2 = st.radio("Nguồn giá:", ["📸 Quét Bảng", "✍️ Nhập Giá"], horizontal=True, label_visibility="collapsed")
                
                # Logic xác định CW đang chọn để lấy giá giả lập
                idx = int(st.session_state.get('user_index', 0))
                cw_row = master_df.iloc[idx]
                und_code = str(cw_row.get("Mã CS", "UNKNOWN"))
                
                # Key cho session giá
                manual_key = f"manual_price_{und_code}"
                if manual_key not in st.session_state:
                    st.session_state[manual_key] = float(DataManager.get_realtime_price_simulated(und_code))

                if mode_t2 == "📸 Quét Bảng":
                    st.info(f"Cập nhật giá cho **{cw_row['Mã CW']}**")
                    uploaded_mkt = st.file_uploader("Chọn ảnh bảng giá", type=["png", "jpg", "jpeg"], key="u_mkt", label_visibility="collapsed")
                    
                    if uploaded_mkt and active_key:
                        if st.button("🚀 Cập Nhật Giá", use_container_width=True):
                            with st.spinner("Đang đọc giá..."):
                                image = Image.open(uploaded_mkt)
                                result = process_image_with_gemini(image, active_key, mode="MARKET_BOARD")
                                if "error" in result: st.error(result['error'])
                                else:
                                    if result.get('market_price'):
                                        raw_mp = float(result['market_price'])
                                        if raw_mp < 1000 and raw_mp > 0: raw_mp *= 1000 
                                        # Cập nhật vào Session giá thủ công
                                        st.session_state[manual_key] = raw_mp
                                        st.toast(f"✅ Giá mới: {raw_mp:,.0f}")
                                        st.rerun() # Refresh để chart ăn theo
                                    else:
                                        st.warning("Không tìm thấy giá.")
                else:
                    # Mode Nhập Giá Tay (Chuyển từ Main vào Sidebar)
                    st.info(f"Chỉnh giá thị trường cho **{und_code}**")
                    new_price = st.number_input("Giá hiện tại (VND):", value=float(st.session_state[manual_key]), step=100.0, format="%.0f")
                    st.session_state[manual_key] = new_price
                    if st.button("🔄 Reset giá giả lập", use_container_width=True):
                        st.session_state[manual_key] = float(DataManager.get_realtime_price_simulated(und_code))
                        st.rerun()

    # --- MAIN DISPLAY ---
    # Lấy thông tin CW đang chọn
    idx = int(st.session_state.get('user_index', 0))
    selected_cw = master_df.iloc[idx]["Mã CW"]
    cw_info = master_df.iloc[idx]
    
    val_exercise = float(cw_info.get("Giá thực hiện", 0))
    val_ratio = float(cw_info.get("Tỷ lệ CĐ", 0))
    val_underlying_code = str(cw_info.get("Mã CS", "UNKNOWN"))
    val_maturity_date = str(cw_info.get("Ngày đáo hạn", ""))
    
    days_left = DataManager.calc_days_to_maturity(val_maturity_date)
    render_cw_profile(selected_cw, val_underlying_code, val_exercise, val_ratio, val_maturity_date, days_left)
    
    # Lấy giá hiện tại từ Session (đã được cập nhật bởi Sidebar Tab 2)
    manual_key = f"manual_price_{val_underlying_code}"
    # Fallback nếu chưa có (lần đầu load)
    if manual_key not in st.session_state:
        st.session_state[manual_key] = float(DataManager.get_realtime_price_simulated(val_underlying_code))
    
    current_real_price = st.session_state[manual_key]

    if has_position:
        engine = FinancialEngine()
        bep = engine.calc_bep(val_exercise, st
