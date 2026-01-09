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
    
    .report-card {
        background: white; padding: 20px; border-radius: 12px; 
        border: 1px solid #E0E0E0; border-top: 5px solid #3949AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #263238; margin-bottom: 10px;
    }
    .report-value { font-size: 1.8em; font-weight: bold; margin: 5px 0; }
    .report-label { font-size: 0.9em; color: #78909C; text-transform: uppercase; letter-spacing: 0.5px; }
    
    .debug-box { background-color: #263238; color: #ECEFF1; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER
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
            {"Mã CW": "CWVPB2511", "Mã CS": "VPB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 30000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CWVIB2510", "Mã CS": "VIB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 23000, "Ngày đáo hạn": "2026-06-29"}
        ]
        return pd.DataFrame(data)

    @staticmethod
    def get_realtime_price_simulated(symbol):
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

# ==========================================
# 4. AI SERVICE LAYER (V16.0 - HYBRID 2026)
# ==========================================
def process_image_with_gemini(image, api_key, mode="ALL"):
    genai.configure(api_key=api_key)
    
    # Mặc định
    base_config = {"temperature": 0.0}
    priority_models = []
    prompt = ""
    start_char = '{'
    end_char = '}'

    # --- STRATEGY SELECTION ---
    if mode == "BATCH_IMPORT":
        # [CHIẾN LƯỢC 1: ROBOT MODE CHO BẢNG GIÁ]
        # Sử dụng Gemini 2.5 Flash để thay thế 2.0 Exp (sắp bị khai tử)
        # BẮT BUỘC: Tắt Thinking để tránh model "tự suy diễn" số liệu
        
        priority_models = [
            'gemini-2.5-flash',       # Primary (2026 Standard)
            'gemini-2.0-flash-001',   # Fallback 1 (Stable)
            'gemini-1.5-flash'        # Fallback 2 (Legacy)
        ]
        
        # Cấu hình đặc biệt: Ép kiểu Robot (No thoughts)
        generation_config = {
            "temperature": 0.0,
            "thinking_config": {"include_thoughts": False, "thinking_budget": 0}, 
            "response_mime_type": "application/json"
        }
        
        prompt = """
        Extract stock data as JSON list. 
        NO reasoning. NO rounding. Exact pixels only.
        
        Required fields per item:
        1. raw_cw: Full CW code (e.g., STB/LPBS/...).
        2. underlying: Underlying stock (e.g., STB).
        3. qty: Volume (Remove commas).
        4. price: Match Price.
        """
        start_char = '['
        end_char = ']'
        
    elif mode == "BUY_ORDER":
        # [CHIẾN LƯỢC 2: INTELLIGENT AGENT CHO BIÊN LAI]
        # Cần Gemini 3.0 để hiểu ngữ cảnh phức tạp (Nộp tiền, Phí...)
        
        priority_models = [
            'gemini-3.0-flash-preview', # Primary (Context Aware)
            'gemini-2.0-flash-exp',     # Fallback
            'gemini-1.5-pro'            # Deep Reasoning Fallback
        ]
        
        generation_config = base_config
        prompt = """
        Extract SINGLE Buy Order/Receipt details.
        Return JSON: {"symbol": "XXX", "qty": 1000, "price": 2168, "total_amount": 0}
        """
    else:
        # Default
        priority_models = ['gemini-2.0-flash-exp', 'gemini-1.5-flash']
        generation_config = base_config
        prompt = "Extract financial data."

    errors_log = [] 

    # --- EXECUTION LOOP ---
    for model_name in priority_models:
        try:
            # Xử lý config riêng cho từng model (Tránh lỗi API nếu model cũ ko hỗ trợ thinking_config)
            current_config = generation_config.copy()
            if "thinking_config" in current_config and "gemini-2.5" not in model_name:
                 del current_config["thinking_config"]

            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image], generation_config=current_config)
            text = response.text.strip()
            
            start_idx = text.find(start_char)
            end_idx = text.rfind(end_char) + 1
            
            if start_idx != -1 and end_idx != -1:
                try:
                    raw_json = text[start_idx:end_idx]
                    json_data = json.loads(raw_json)
                    
                    if mode == "BATCH_IMPORT":
                         if isinstance(json_data, list):
                            return {"data": json_data, "_meta_model": model_name}
                         else:
                            errors_log.append(f"{model_name}: Expected List but got Dict")
                            continue
                    else:
                        json_data['_meta_model'] = model_name
                        json_data['_meta_raw_text'] = text
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

    return None

def map_batch_data(ocr_list, master_df):
    mapped_results = []
    
    for item in ocr_list:
        raw_cw = item.get('raw_cw', '')
        underlying = item.get('underlying', '')
        
        candidates = master_df[master_df['Mã CS'] == underlying]
        matched_symbol = None
        
        if not candidates.empty:
            suffix_match = re.search(r'/(\d{2})$', raw_cw.strip())
            if suffix_match:
                suffix = suffix_match.group(1)
                for idx, row in candidates.iterrows():
                    if row['Mã CW'].endswith(suffix):
                        matched_symbol = row['Mã CW']
                        break
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
# 5. HELPER
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
        st.info("📭 Danh mục trống.")
        return

    total_nav = 0
    total_cost = 0
    
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

    st.markdown("### 1. TỔNG QUAN")
    c1, c2, c3 = st.columns(3)
    c1.metric("NAV", f"{total_nav:,.0f} đ")
    c2.metric("Lãi/Lỗ", f"{total_pnl:,.0f} đ")
    c3.metric("Hiệu suất", f"{pnl_pct:+.2f}%")

    st.markdown("### 2. CHI TIẾT")
    df_display = pd.DataFrame(pf)
    if not df_display.empty:
        st.dataframe(
            df_display[["symbol", "qty", "cost_price", "market_price_cw", "market_price_cs"]],
            use_container_width=True
        )

# ==========================================
# 6. MAIN APP
# ==========================================
def main():
    st.title("💎 LPBS CW Portfolio Master")
    st.caption(f"System: V16.0 | Hybrid 2026 Ready | 2.5 Flash & 3.0 Preview")

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

    with tab_input:
        st.markdown("#### 📥 Nhập Liệu Danh Mục")
        mode = st.radio("Chế độ:", ["📑 Quét Hàng Loạt (Danh sách)", "📸 Quét Đơn (1 Lệnh)", "✍️ Nhập Tay"], horizontal=True)
        
        # === MODE 1: QUÉT HÀNG LOẠT (V16.0) ===
        if mode == "📑 Quét Hàng Loạt (Danh sách)":
            st.info("💡 Tip: Dùng cho ảnh danh sách nhiều mã. Engine: Gemini 2.5 Flash (Robot Mode).")
            uploaded_file = st.file_uploader("Upload ảnh Danh sách", type=['png', 'jpg', 'jpeg'], key="batch_upl")
            
            if uploaded_file and active_key:
                if st.button("🚀 Phân Tích Danh Sách", type="primary", use_container_width=True):
                    with st.spinner("Đang kích hoạt Gemini 2.5 Flash (No Thinking)..."):
                        image = Image.open(uploaded_file)
                        result = process_image_with_gemini(image, active_key, mode="BATCH_IMPORT")
                        
                        if "data" in result:
                            df_preview = map_batch_data(result['data'], master_df)
                            st.session_state['batch_preview'] = df_preview
                            st.success(f"Tìm thấy {len(df_preview)} dòng!")
                        else:
                            st.error("Lỗi đọc dữ liệu.")
                            st.write(result)

            if 'batch_preview' in st.session_state and not st.session_state['batch_preview'].empty:
                st.markdown("---")
                st.markdown("#### 📝 Duyệt & Chỉnh Sửa")
                
                safe_options = master_df["Mã CW"].unique().tolist()
                safe_options.append("???")
                
                edited_df = st.data_editor(
                    st.session_state['batch_preview'],
                    column_config={
                        "Chốt": st.column_config.CheckboxColumn("Import?", default=True),
                        "Mã CW (Gợi ý)": st.column_config.SelectboxColumn("Mã CW", options=safe_options, required=True, width="medium"),
                        "KL": st.column_config.NumberColumn("Khối Lượng", format="%d"),
                        "Giá Vốn": st.column_config.NumberColumn("Giá Mua", format="%d"),
                        "Mã Gốc": st.column_config.TextColumn("Raw Data", disabled=True)
                    },
                    use_container_width=True, num_rows="dynamic"
                )
                
                if st.button("✅ THỰC THI IMPORT", type="primary", use_container_width=True):
                    count = 0
                    for index, row in edited_df.iterrows():
                        if row['Chốt'] and row['Mã CW (Gợi ý)'] != "???":
                            master_info = master_df[master_df['Mã CW'] == row['Mã CW (Gợi ý)']]
                            if not master_info.empty:
                                add_to_portfolio(master_info.iloc[0], row['KL'], row['Giá Vốn'])
                                count += 1
                    
                    st.success(f"Đã nhập {count} lệnh!")
                    del st.session_state['batch_preview'] 
                    st.rerun()

        # === MODE 2: QUÉT ĐƠN (V16.0) ===
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                if mode == "📸 Quét Đơn (1 Lệnh)":
                    st.info("💡 Engine: Gemini 3.0 Preview (Reasoning Mode)")
                    uploaded_file = st.file_uploader("Upload ảnh Lệnh/Biên lai", type=['png', 'jpg'])
                    if uploaded_file and active_key:
                        if st.button("🚀 Phân Tích", use_container_width=True):
                            with st.spinner("Gemini 3.0 đang suy luận..."):
                                image = Image.open(uploaded_file)
                                result = process_image_with_gemini(image, active_key, mode="BUY_ORDER")
                                st.session_state['ocr_result'] = result
                                
                                if "error" not in result:
                                    price = float(result.get('price', 0))
                                    if price < 1000 and price > 0: price *= 1000
                                    st.session_state['temp_price'] = price
                                    st.session_state['temp_qty'] = float(result.get('qty', 0))
                                    idx = auto_map_symbol(result, master_df)
                                    if idx is not None: st.session_state['temp_index'] = idx

                cw_list = master_df["Mã CW"].unique()
                curr_idx = int(st.session_state.get('temp_index', 0))
                if curr_idx >= len(cw_list): curr_idx = 0

                selected_cw = st.selectbox("Mã CW", cw_list, index=curr_idx)
                qty = st.number_input("Số lượng", value=st.session_state.get('temp_qty', 0.0), step=100.0)
                cost = st.number_input("Giá vốn", value=st.session_state.get('temp_price', 0.0), step=50.0)
                
                if st.button("💾 Lưu vào Danh mục", type="primary", use_container_width=True):
                    row = master_df[master_df['Mã CW'] == selected_cw].iloc[0]
                    add_to_portfolio(row, qty, cost)
                    st.rerun()

            with c2:
                if mode == "📸 Quét Đơn (1 Lệnh)" and st.session_state.get('ocr_result'):
                    st.markdown("#### 🔍 Glass Box Debug")
                    st.json(st.session_state['ocr_result'])

    with tab_report:
        render_report_dashboard()

    with tab_sim:
        if st.session_state['portfolio']:
            pf_df = pd.DataFrame(st.session_state['portfolio'])
            sim_cw = st.selectbox("Chọn mã:", pf_df['symbol'].unique())
            item = next(x for x in st.session_state['portfolio'] if x['symbol'] == sim_cw)
            st.info(f"Giả lập: **{sim_cw}**")
            target_cs = st.slider("Giá Cơ sở:", int(item['market_price_cs']*0.8), int(item['market_price_cs']*1.5), int(item['market_price_cs']))
            sim_val = FinancialEngine.calc_intrinsic_value(target_cs, item['exercise_price'], item['ratio'])
            st.metric("Giá CW Lý thuyết", f"{sim_val:,.0f} đ")

if __name__ == "__main__":
    main()
