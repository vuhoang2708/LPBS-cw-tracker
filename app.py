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
import uuid

# ==========================================
# 1. CONFIG & BRANDING
# ==========================================
st.set_page_config(page_title="LPBS CW Portfolio Master", layout="wide", page_icon="💎")

vn_time = datetime.utcnow() + timedelta(hours=7)
build_time_str = vn_time.strftime("%H:%M:%S - %d/%m/%Y")

# --- SECURITY ---
if "GEMINI_API_KEY" in st.secrets:
    SYSTEM_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    SYSTEM_API_KEY = None 

st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    h1, h2, h3 { color: #1A237E !important; font-family: 'Segoe UI', sans-serif; }
    
    [data-testid="stSidebar"] {
        background-color: #E8EAF6;
        border-right: 1px solid #C5CAE9;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #FFF; border-radius: 6px; 
        color: #5C6BC0; font-weight: 600; border: 1px solid #E8EAF6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3949AB !important; color: white !important; border-color: #3949AB;
    }

    .report-card {
        background: white; padding: 20px; border-radius: 12px; 
        border: 1px solid #E0E0E0; border-top: 5px solid #3949AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #263238; margin-bottom: 10px;
    }
    .report-value { font-size: 1.8em; font-weight: bold; margin: 5px 0; }
    .report-label { font-size: 0.9em; color: #78909C; text-transform: uppercase; letter-spacing: 0.5px; }
    
    .profit { color: #2E7D32; }
    .loss { color: #C62828; }
    
    .debug-box { background-color: #263238; color: #ECEFF1; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER (PORTFOLIO UPGRADE)
# ==========================================
class DataManager:
    @staticmethod
    def get_default_master_data():
        data = [
            {"Mã CW": "CMWG2519", "Mã CS": "MWG", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 88000, "Ngày đáo hạn": "2026-06-29"},
            {"Mã CW": "CWVHM2522", "Mã CS": "VHM", "Tỷ lệ CĐ": "10:1", "Giá thực hiện": 106000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWSTB2505", "Mã CS": "STB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 60000, "Ngày đáo hạn": "2026-06-29"},
            {"Mã CW": "CWHPG2516", "Mã CS": "HPG", "Tỷ lệ CĐ": "4:1", "Giá thực hiện": 32000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWACB2502", "Mã CS": "ACB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 28000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWMBB2504", "Mã CS": "MBB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 22000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWMSN2518", "Mã CS": "MSN", "Tỷ lệ CĐ": "10:1", "Giá thực hiện": 95000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWVNM2524", "Mã CS": "VNM", "Tỷ lệ CĐ": "8:1", "Giá thực hiện": 72000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWSHB2525", "Mã CS": "SHB", "Tỷ lệ CĐ": "1:1", "Giá thực hiện": 12500, "Ngày đáo hạn": "2026-06-29"},
            {"Mã CW": "CWFPT2514", "Mã CS": "FPT", "Tỷ lệ CĐ": "8:1", "Giá thực hiện": 110000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWTCB2507", "Mã CS": "TCB", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 45000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWVPB2511", "Mã CS": "VPB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 21500, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWVIB2510", "Mã CS": "VIB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 23000, "Ngày đáo hạn": "2026-06-29"}
        ]
        return pd.DataFrame(data)

    @staticmethod
    def get_realtime_price_simulated(symbol):
        # Giá giả lập để test report
        base_prices = {"HPG":28500,"MWG":48200,"VHM":41800,"STB":30500,"VNM":66000,"FPT":95000,"MBB":18500,"TCB":33000,"VPB":19200,"MSN":62000,"VIB":21500,"SHB":11200,"ACB":24500}
        noise = np.random.uniform(0.98, 1.02)
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
        if price_underlying > price_exercise: return "ITM", "green"
        elif price_underlying < price_exercise: return "OTM", "red"
        else: return "ATM", "orange"

# ==========================================
# 4. AI SERVICE LAYER (V14.2 - FIXED AUTO MAP)
# ==========================================
def process_image_with_gemini(image, api_key, mode="ALL"):
    genai.configure(api_key=api_key)
    generation_config = {"temperature": 0.0}
    
    priority_models = [
        'gemini-3-flash-preview', 
        'gemini-2.0-flash-exp',    
        'gemini-1.5-flash'         
    ]
    
    if mode == "BUY_ORDER":
        task_desc = "Trích xuất thông tin LỆNH MUA / BIÊN LAI."
    elif mode == "MARKET_BOARD":
        task_desc = "Trích xuất thông tin BẢNG GIÁ."
    else:
        task_desc = "Trích xuất dữ liệu tài chính."

    prompt = f"""
    Bạn là một trợ lý tài chính (OCR). Nhiệm vụ: {task_desc}
    
    Các trường cần tìm:
    1. Mã chứng khoán (Symbol): Tìm mã Chứng quyền (C...) hoặc mã Cơ sở.
    2. Số lượng (Qty): Khối lượng mua.
    3. Giá vốn (Price): Giá khớp lệnh/đơn giá.
    4. Tổng tiền (Total Amount): Tổng giá trị giao dịch (nếu có).
    5. Giá thị trường (Market Price): Giá hiện tại trên bảng điện.

    Trả về JSON (chỉ số): 
    {{"symbol": "XXX", "qty": 1000, "price": 2168, "total_amount": 65040000, "market_price": 52000}}
    """
    
    errors_log = [] 

    for model_name in priority_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image], generation_config=generation_config)
            text = response.text.strip()
            
            # Glass Box Debug Logic
            start_idx = text.find('{')
            if start_idx != -1:
                try:
                    json_data, _ = JSONDecoder().raw_decode(text[start_idx:])
                    json_data['_meta_model'] = model_name
                    json_data['_meta_raw_text'] = text
                    json_data['_meta_logs'] = errors_log
                    return json_data
                except Exception as e:
                    errors_log.append(f"{model_name} Parse Error: {str(e)}")
                    continue
            else:
                errors_log.append(f"{model_name}: No JSON found.")
                continue
        except Exception as e:
            errors_log.append(f"{model_name} API Error: {str(e)}")
            continue 
            
    return {"error": "Thất bại toàn tập", "_meta_logs": errors_log}

# [PATCH V14.1] Logic Quét Ngược (Reverse Scan)
def auto_map_symbol(ocr_result, master_df):
    if not ocr_result or "error" in ocr_result: return None
    
    det_sym = str(ocr_result.get('symbol', '')).upper().strip()
    
    # Ưu tiên 1: Khớp chính xác Mã CW
    mask_exact = master_df['Mã CW'] == det_sym
    if mask_exact.any(): 
        return master_df.index[mask_exact].tolist()[0]
    
    # Ưu tiên 2: Quét ngược (Scan Underlying từ Master Data)
    unique_underlying = master_df['Mã CS'].unique()
    found_candidates = []
    for code in unique_underlying:
        if code in det_sym: # VD: Tìm thấy "VHM" trong "CVWHM"
            found_candidates.append(code)
    
    if found_candidates:
        best_match = found_candidates[0]
        mask_core = master_df['Mã CS'] == best_match
        if mask_core.any():
            return master_df.index[mask_core].tolist()[0]

    # Ưu tiên 3: Fix lỗi Typo phổ biến (W -> V)
    fixed_sym = det_sym.replace("W", "V").replace("CV", "") 
    mask_retry = master_df['Mã CS'].str.contains(fixed_sym)
    if len(fixed_sym) >= 3 and mask_retry.any():
        return master_df.index[mask_retry].tolist()[0]

    return None

# ==========================================
# 5. HELPER: PORTFOLIO & REPORT UI
# ==========================================
def add_to_portfolio(cw_row, qty, price):
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
    
    item = {
        "id": str(uuid.uuid4())[:8],
        "symbol": cw_row['Mã CW'],
        "underlying": cw_row['Mã CS'],
        "qty": float(qty),
        "cost_price": float(price),
        "exercise_price": float(cw_row['Giá thực hiện']),
        "ratio": float(cw_row['Tỷ lệ CĐ']),
        "maturity": str(cw_row['Ngày đáo hạn']),
        "market_price_cw": 0.0,
        "market_price_cs": 0.0
    }
    st.session_state['portfolio'].append(item)
    st.toast(f"✅ Đã thêm {item['symbol']} vào danh mục!")

def render_report_dashboard():
    pf = st.session_state.get('portfolio', [])
    if not pf:
        st.info("📭 Danh mục trống. Vui lòng thêm vị thế ở Tab 1.")
        return

    total_nav = 0
    total_cost = 0
    
    # Simulation Logic
    for item in pf:
        cs_price = DataManager.get_realtime_price_simulated(item['underlying'])
        item['market_price_cs'] = cs_price
        
        intrinsic = FinancialEngine.calc_intrinsic_value(cs_price, item['exercise_price'], item['ratio'])
        market_cw = intrinsic * 1.05 if intrinsic > 0 else 100 
        item['market_price_cw'] = market_cw
        
        total_nav += item['qty'] * market_cw
        total_cost += item['qty'] * item['cost_price']

    total_pnl = total_nav - total_cost
    pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    # --- SECTION 1: TỔNG QUAN ---
    st.markdown("### 1. TỔNG QUAN TÀI SẢN")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="report-card">
            <div class="report-label">GIÁ TRỊ RÒNG (NAV)</div>
            <div class="report-value" style="color:#1A237E">{total_nav:,.0f} VND</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        color = "#2E7D32" if total_pnl >= 0 else "#C62828"
        st.markdown(f"""
        <div class="report-card">
            <div class="report-label">TỔNG LÃI/LỖ</div>
            <div class="report-value" style="color:{color}">{total_pnl:,.0f} VND</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        color = "#2E7D32" if pnl_pct >= 0 else "#C62828"
        st.markdown(f"""
        <div class="report-card">
            <div class="report-label">HIỆU SUẤT</div>
            <div class="report-value" style="color:{color}">{pnl_pct:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 2: CHI TIẾT DANH MỤC ---
    st.markdown("### 2. CHI TIẾT DANH MỤC")
    
    display_data = []
    for item in pf:
        val_now = item['qty'] * item['market_price_cw']
        val_cost = item['qty'] * item['cost_price']
        pnl = val_now - val_cost
        pct = (pnl / val_cost) if val_cost > 0 else 0
        
        display_data.append({
            "Mã": item['symbol'],
            "SL": item['qty'],
            "Giá Vốn": item['cost_price'],
            "Giá CS": item['market_price_cs'],
            "Giá trị TT": val_now,
            "Lãi/Lỗ": pnl,
            "%": pct
        })
    
    df_display = pd.DataFrame(display_data)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        column_config={
            "SL": st.column_config.NumberColumn(format="%,.0f"),
            "Giá Vốn": st.column_config.NumberColumn(format="%,.0f"),
            "Giá CS": st.column_config.NumberColumn(format="%,.0f"),
            "Giá trị TT": st.column_config.NumberColumn(format="%,.0f"),
            "Lãi/Lỗ": st.column_config.NumberColumn(format="%,.0f"),
            "%": st.column_config.NumberColumn(format="%.2%"),
        },
        hide_index=True
    )

    # --- SECTION 3: PHÂN TÍCH VỊ THẾ & RỦI RO ---
    st.markdown("### 3. PHÂN TÍCH VỊ THẾ & RỦI RO")
    risk_data = []
    for item in pf:
        bep = FinancialEngine.calc_bep(item['exercise_price'], item['cost_price'], item['ratio'])
        dist = ((item['market_price_cs'] - bep) / bep) if bep > 0 else 0
        days = DataManager.calc_days_to_maturity(item['maturity'])
        
        status_icon = "🟢" if dist > 0 else "🔴" if dist < -0.1 else "🟡"
        
        risk_data.append({
            "Mã": item['symbol'],
            "Hòa vốn (BEP)": bep,
            "Khoảng cách": dist,
            "Đáo hạn": item['maturity'],
            "Còn lại": f"{days} ngày",
            "Trạng thái": status_icon
        })
        
    df_risk = pd.DataFrame(risk_data)
    st.dataframe(
        df_risk,
        use_container_width=True,
        column_config={
            "Hòa vốn (BEP)": st.column_config.NumberColumn(format="%,.0f"),
            "Khoảng cách": st.column_config.NumberColumn(format="%.1%"),
        },
        hide_index=True
    )

# ==========================================
# 6. MAIN APP
# ==========================================
def main():
    st.title("💎 LPBS CW Portfolio Master")
    st.caption(f"System: V14.2 | Clean Build | Model: Gemini 3 Flash Preview")

    # [CLEAN] Khởi tạo giá trị bằng 0 hoặc Rỗng
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
    if 'ocr_result' not in st.session_state: st.session_state['ocr_result'] = None
    if 'temp_qty' not in st.session_state: st.session_state['temp_qty'] = 0.0 # Clean
    if 'temp_price' not in st.session_state: st.session_state['temp_price'] = 0.0 # Clean
    if 'temp_index' not in st.session_state: st.session_state['temp_index'] = 0

    master_df = DataManager.get_default_master_data()
    master_df["Giá thực hiện"] = master_df["Giá thực hiện"].apply(DataManager.clean_number_value)
    master_df["Tỷ lệ CĐ"] = master_df["Tỷ lệ CĐ"].apply(DataManager.clean_number_value)

    with st.sidebar:
        if not SYSTEM_API_KEY:
            st.warning("⚠️ Chưa cấu hình Secrets.")
            active_key = st.text_input("Nhập Key:", type="password")
        else:
            active_key = SYSTEM_API_KEY
            
        st.info(f"📁 Danh mục: {len(st.session_state['portfolio'])} mã")
        if st.button("🗑️ Xóa danh mục"):
            st.session_state['portfolio'] = []
            st.rerun()

    tab_input, tab_report, tab_sim = st.tabs(["1️⃣ NHẬP LIỆU", "2️⃣ BÁO CÁO DANH MỤC", "3️⃣ GIẢ LẬP"])

    # --- TAB 1: INPUT ---
    with tab_input:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### 📥 Thêm Vị Thế Mới")
            mode = st.radio("Chế độ:", ["📸 Quét OCR", "✍️ Nhập Tay"], horizontal=True)
            
            if mode == "📸 Quét OCR":
                uploaded_file = st.file_uploader("Upload ảnh (Lệnh mua/Biên lai)", type=['png', 'jpg'])
                if uploaded_file and active_key:
                    if st.button("🚀 Phân Tích (Gemini 3)", use_container_width=True):
                        with st.spinner("Đang xử lý với Gemini 3 Flash Preview..."):
                            image = Image.open(uploaded_file)
                            result = process_image_with_gemini(image, active_key, mode="BUY_ORDER")
                            st.session_state['ocr_result'] = result
                            
                            if "error" not in result:
                                price = 0.0
                                if result.get('price'):
                                    price = float(result['price'])
                                elif result.get('total_amount') and result.get('qty'):
                                    try:
                                        price = float(result['total_amount']) / float(result['qty'])
                                        st.toast(f"ℹ️ Auto Calc: {price:,.0f}")
                                    except: pass
                                
                                if price < 1000 and price > 0: price *= 1000
                                
                                st.session_state['temp_price'] = price
                                if result.get('qty'): st.session_state['temp_qty'] = float(result['qty'])
                                
                                # [PATCH] Gọi hàm Auto Map V14.1
                                idx = auto_map_symbol(result, master_df)
                                if idx is not None: st.session_state['temp_index'] = idx

            cw_list = master_df["Mã CW"].unique()
            selected_cw = st.selectbox("Mã CW", cw_list, index=int(st.session_state['temp_index']))
            
            # [CLEAN] Default value = 0.0
            qty = st.number_input("Số lượng", value=st.session_state['temp_qty'], step=100.0)
            cost = st.number_input("Giá vốn", value=st.session_state['temp_price'], step=50.0)
            
            if st.button("💾 Lưu vào Danh mục", type="primary", use_container_width=True):
                if qty <= 0 or cost <= 0:
                    st.error("Số lượng và Giá vốn phải lớn hơn 0")
                else:
                    row = master_df[master_df['Mã CW'] == selected_cw].iloc[0]
                    add_to_portfolio(row, qty, cost)
                    st.success("Đã lưu thành công!")
                    # Reset input sau khi lưu
                    st.session_state['temp_qty'] = 0.0
                    st.session_state['temp_price'] = 0.0
                    st.rerun()

        with c2:
            if st.session_state['ocr_result']:
                res = st.session_state['ocr_result']
                st.markdown("#### 🔍 Glass Box Debug")
                with st.expander("Chi tiết xử lý AI", expanded=True):
                    st.markdown(f"**Model:** `{res.get('_meta_model', 'N/A')}`")
                    st.markdown("**Raw Output:**")
                    st.markdown(f"""<div class="debug-box">{res.get('_meta_raw_text', 'No Text')}</div>""", unsafe_allow_html=True)
                    st.json(res)

    with tab_report:
        render_report_dashboard()

    with tab_sim:
        if not st.session_state['portfolio']:
            st.info("Vui lòng thêm vị thế vào danh mục trước.")
        else:
            pf_df = pd.DataFrame(st.session_state['portfolio'])
            sim_cw = st.selectbox("Chọn mã để giả lập:", pf_df['symbol'].unique())
            item = next(x for x in st.session_state['portfolio'] if x['symbol'] == sim_cw)
            
            curr_cs = item['market_price_cs'] if item['market_price_cs'] > 0 else 20000
            st.info(f"Giả lập cho **{sim_cw}** (Giá vốn: {item['cost_price']:,.0f})")
            
            target_cs = st.slider("Giá Cơ sở Tương lai:", int(curr_cs * 0.8), int(curr_cs * 1.5), int(curr_cs))
            
            sim_val = FinancialEngine.calc_intrinsic_value(target_cs, item['exercise_price'], item['ratio'])
            sim_pnl = (sim_val - item['cost_price']) * item['qty']
            
            c1, c2 = st.columns(2)
            c1.metric("Giá CW Lý thuyết", f"{sim_val:,.0f} đ")
            c2.metric("Lãi/Lỗ Dự kiến", f"{sim_pnl:,.0f} đ", delta_color="normal")

if __name__ == "__main__":
    main()
