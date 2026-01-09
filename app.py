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
            {"Mã CW": "CWMWG2519", "Mã CS": "MWG", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 88000, "Ngày đáo hạn": "2026-06-29"},
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
# 4. AI SERVICE LAYER (V15.0 - BATCH CORE)
# ==========================================
def process_image_with_gemini(image, api_key, mode="ALL"):
    genai.configure(api_key=api_key)
    generation_config = {"temperature": 0.0}
    
    priority_models = [
        'gemini-2.0-flash-exp',    
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ]
    
    if mode == "BATCH_IMPORT":
        prompt = """
        Bạn là một chuyên gia nhập liệu (OCR). Nhiệm vụ: Trích xuất TOÀN BỘ các dòng trong bảng danh sách mua chứng quyền.
        
        Các cột cần lấy:
        1. raw_cw: Mã Chứng quyền đầy đủ (ví dụ: STB/LPBS/CALL/EU/CASH/6M/05).
        2. underlying: Mã CKCS (ví dụ: STB, HPG).
        3. qty: Khối lượng mua (cột KL mua).
        4. price: Giá khớp/Giá mua (cột Giá).
        
        QUAN TRỌNG:
        - Trả về định dạng JSON là một DANH SÁCH (LIST) các đối tượng.
        - Số liệu phải bỏ dấu phẩy ngăn cách (ví dụ: 2,000 -> 2000).
        - Chỉ lấy số liệu, không lấy text thừa.
        
        Output mẫu:
        [
          {"raw_cw": "STB/LPBS/...", "underlying": "STB", "qty": 2000, "price": 1468},
          {"raw_cw": "HPG/LPBS/...", "underlying": "HPG", "qty": 1000, "price": 2168}
        ]
        """
    elif mode == "BUY_ORDER":
        prompt = f"""
        Bạn là một trợ lý tài chính (OCR). Nhiệm vụ: Trích xuất thông tin LỆNH MUA đơn lẻ.
        Các trường cần tìm:
        1. Mã chứng khoán (Symbol): Tìm mã Chứng quyền (CW...) hoặc mã Cơ sở.
        2. Số lượng (Qty): Khối lượng mua.
        3. Giá vốn (Price): Giá khớp lệnh/đơn giá.
        4. Tổng tiền (Total Amount): Tổng giá trị giao dịch (nếu có).
        5. Giá thị trường (Market Price): Giá hiện tại trên bảng điện.

        Trả về JSON (chỉ số): 
        {{"symbol": "XXX", "qty": 1000, "price": 2168, "total_amount": 65040000, "market_price": 52000}}
        """
    else:
        prompt = "Trích xuất dữ liệu tài chính."
    
    errors_log = [] 

    for model_name in priority_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image], generation_config=generation_config)
            text = response.text.strip()
            
            # Xử lý cắt chuỗi JSON linh hoạt
            start_char = '[' if mode == "BATCH_IMPORT" else '{'
            end_char = ']' if mode == "BATCH_IMPORT" else '}'
            
            start_idx = text.find(start_char)
            end_idx = text.rfind(end_char) + 1
            
            if start_idx != -1 and end_idx != -1:
                try:
                    raw_json = text[start_idx:end_idx]
                    # Dùng json.loads cho an toàn với cả list và dict
                    json_data = json.loads(raw_json)
                    
                    if mode == "BATCH_IMPORT":
                         # Với list, trả về structure bọc ngoài để dễ xử lý
                         if isinstance(json_data, list):
                            return {"data": json_data, "_meta_model": model_name}
                         else:
                            errors_log.append(f"{model_name}: Expected List but got Dict")
                            continue
                    else:
                        # Với Single Object
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

# [PATCH V14.1] Logic Quét Ngược (Single)
def auto_map_symbol(ocr_result, master_df):
    if not ocr_result or "error" in ocr_result: return None
    det_sym = str(ocr_result.get('symbol', '')).upper().strip()
    
    mask_exact = master_df['Mã CW'] == det_sym
    if mask_exact.any(): return master_df.index[mask_exact].tolist()[0]
    
    unique_underlying = master_df['Mã CS'].unique()
    found_candidates = []
    for code in unique_underlying:
        if code in det_sym: found_candidates.append(code)
    
    if found_candidates:
        best_match = found_candidates[0]
        mask_core = master_df['Mã CS'] == best_match
        if mask_core.any(): return master_df.index[mask_core].tolist()[0]

    fixed_sym = det_sym.replace("W", "V").replace("CV", "") 
    mask_retry = master_df['Mã CS'].str.contains(fixed_sym)
    if len(fixed_sym) >= 3 and mask_retry.any(): return master_df.index[mask_retry].tolist()[0]
    return None

# [NEW V15.0] Logic Mapping Batch
def map_batch_data(ocr_list, master_df):
    mapped_results = []
    
    for item in ocr_list:
        raw_cw = item.get('raw_cw', '')
        underlying = item.get('underlying', '')
        
        candidates = master_df[master_df['Mã CS'] == underlying]
        matched_symbol = None
        
        if not candidates.empty:
            # Logic: Thử match 2 số cuối của raw string (VD: .../05) với mã CW (VD: CWSTB2505)
            suffix_match = re.search(r'/(\d{2})$', raw_cw.strip())
            if suffix_match:
                suffix = suffix_match.group(1)
                for idx, row in candidates.iterrows():
                    if row['Mã CW'].endswith(suffix):
                        matched_symbol = row['Mã CW']
                        break
            
            # Fallback: Lấy mã đầu tiên nếu ko match đuôi
            if not matched_symbol:
                matched_symbol = candidates.iloc[0]['Mã CW']
        
        mapped_results.append({
            "Chốt": True, 
            "Mã CW (Gợi ý)": matched_symbol if matched_symbol else "???",
            "Mã Gốc": raw_cw,
            "KL": float(item.get('qty', 0)),
            "Giá Vốn": float(item.get('price', 0))
        })
        
    return pd.DataFrame(mapped_results)

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
    st.caption(f"System: V15.0 | Batch Import Core | Model: Gemini 2.0 Flash Exp")

    # [CLEAN] Khởi tạo giá trị
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
    if 'ocr_result' not in st.session_state: st.session_state['ocr_result'] = None
    if 'temp_qty' not in st.session_state: st.session_state['temp_qty'] = 0.0 
    if 'temp_price' not in st.session_state: st.session_state['temp_price'] = 0.0
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

    # --- TAB 1: INPUT (UPGRADE V15) ---
    with tab_input:
        st.markdown("#### 📥 Nhập Liệu Danh Mục")
        mode = st.radio("Chế độ:", ["📑 Quét Hàng Loạt (Danh sách)", "📸 Quét Đơn (1 Lệnh)", "✍️ Nhập Tay"], horizontal=True)
        
        # === MODE 1: QUÉT HÀNG LOẠT (BATCH) ===
        if mode == "📑 Quét Hàng Loạt (Danh sách)":
            st.info("💡 Tip: Chụp ảnh bảng danh sách lệnh đã khớp (như hình mẫu Webview).")
            uploaded_file = st.file_uploader("Upload ảnh Danh sách", type=['png', 'jpg', 'jpeg'], key="batch_upl")
            
            if uploaded_file and active_key:
                if st.button("🚀 Phân Tích Danh Sách", type="primary", use_container_width=True):
                    with st.spinner("Đang đọc từng dòng với Gemini 2.0..."):
                        image = Image.open(uploaded_file)
                        result = process_image_with_gemini(image, active_key, mode="BATCH_IMPORT")
                        
                        if "data" in result:
                            # Auto Map
                            df_preview = map_batch_data(result['data'], master_df)
                            st.session_state['batch_preview'] = df_preview
                            st.success(f"Đã tìm thấy {len(df_preview)} dòng lệnh!")
                        else:
                            st.error("Không đọc được dữ liệu nào hợp lệ.")
                            with st.expander("Log lỗi"):
                                st.write(result)

            # Hiển thị bảng Review nếu có dữ liệu
            if 'batch_preview' in st.session_state and not st.session_state['batch_preview'].empty:
                st.markdown("---")
                st.markdown("#### 📝 Duyệt & Chỉnh Sửa")
                
                # Cấu hình cột cho Data Editor
                edited_df = st.data_editor(
                    st.session_state['batch_preview'],
                    column_config={
                        "Chốt": st.column_config.CheckboxColumn("Import?", help="Chọn để nhập dòng này", default=True),
                        "Mã CW (Gợi ý)": st.column_config.SelectboxColumn(
                            "Mã CW",
                            options=master_df["Mã CW"].unique(),
                            required=True,
                            width="medium"
                        ),
                        "KL": st.column_config.NumberColumn("Khối Lượng", format="%d"),
                        "Giá Vốn": st.column_config.NumberColumn("Giá Mua", format="%d"),
                        "Mã Gốc": st.column_config.TextColumn("Raw Data (Tham chiếu)", disabled=True)
                    },
                    use_container_width=True,
                    num_rows="dynamic"
                )
                
                c_act1, c_act2 = st.columns([1, 3])
                with c_act1:
                    if st.button("✅ THỰC THI IMPORT", type="primary", use_container_width=True):
                        count = 0
                        for index, row in edited_df.iterrows():
                            if row['Chốt'] and row['Mã CW (Gợi ý)'] != "???":
                                master_info = master_df[master_df['Mã CW'] == row['Mã CW (Gợi ý)']]
                                if not master_info.empty:
                                    master_row = master_info.iloc[0]
                                    add_to_portfolio(master_row, row['KL'], row['Giá Vốn'])
                                    count += 1
                        
                        st.success(f"Đã nhập thành công {count} lệnh vào danh mục!")
                        del st.session_state['batch_preview'] 
                        st.rerun()

        # === MODE 2 & 3: QUÉT ĐƠN & NHẬP TAY ===
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                if mode == "📸 Quét Đơn (1 Lệnh)":
                    uploaded_file = st.file_uploader("Upload ảnh (Lệnh mua/Biên lai)", type=['png', 'jpg'])
                    if uploaded_file and active_key:
                        if st.button("🚀 Phân Tích (Gemini)", use_container_width=True):
                            with st.spinner("Đang xử lý..."):
                                image = Image.open(uploaded_file)
                                result = process_image_with_gemini(image, active_key, mode="BUY_ORDER")
                                st.session_state['ocr_result'] = result
                                
                                if "error" not in result:
                                    price = 0.0
                                    if result.get('price'): price = float(result['price'])
                                    elif result.get('total_amount') and result.get('qty'):
                                        try: price = float(result['total_amount']) / float(result['qty'])
                                        except: pass
                                    
                                    if price < 1000 and price > 0: price *= 1000
                                    
                                    st.session_state['temp_price'] = price
                                    if result.get('qty'): st.session_state['temp_qty'] = float(result['qty'])
                                    
                                    idx = auto_map_symbol(result, master_df)
                                    if idx is not None: st.session_state['temp_index'] = idx

                # Form Nhập Liệu Chung
                cw_list = master_df["Mã CW"].unique()
                curr_idx = int(st.session_state.get('temp_index', 0))
                if curr_idx >= len(cw_list): curr_idx = 0

                selected_cw = st.selectbox("Mã CW", cw_list, index=curr_idx)
                qty = st.number_input("Số lượng", value=st.session_state.get('temp_qty', 0.0), step=100.0)
                cost = st.number_input("Giá vốn", value=st.session_state.get('temp_price', 0.0), step=50.0)
                
                if st.button("💾 Lưu vào Danh mục", type="primary", use_container_width=True):
                    if qty <= 0 or cost <= 0:
                        st.error("Số lượng và Giá vốn phải lớn hơn 0")
                    else:
                        row = master_df[master_df['Mã CW'] == selected_cw].iloc[0]
                        add_to_portfolio(row, qty, cost)
                        st.success("Đã lưu thành công!")
                        st.session_state['temp_qty'] = 0.0
                        st.session_state['temp_price'] = 0.0
                        st.rerun()

            with c2:
                if mode == "📸 Quét Đơn (1 Lệnh)" and st.session_state.get('ocr_result'):
                    res = st.session_state['ocr_result']
                    st.markdown("#### 🔍 Glass Box Debug")
                    with st.expander("Chi tiết xử lý AI", expanded=True):
                        st.markdown(f"**Model:** `{res.get('_meta_model', 'N/A')}`")
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
