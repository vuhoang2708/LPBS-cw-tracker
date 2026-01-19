import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from json import JSONDecoder
import json  # Added for Batch Import
from datetime import datetime, timedelta
from PIL import Image
import uuid
import re

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
    [data-testid="stSidebar"] { background-color: #E8EAF6; border-right: 1px solid #C5CAE9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #FFF; border-radius: 6px; color: #5C6BC0; font-weight: 600; border: 1px solid #E8EAF6; }
    .stTabs [aria-selected="true"] { background-color: #3949AB !important; color: white !important; border-color: #3949AB; }
    .report-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #E0E0E0; border-top: 5px solid #3949AB; box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #263238; margin-bottom: 10px; }
    .report-value { font-size: 1.8em; font-weight: bold; margin: 5px 0; }
    .report-label { font-size: 0.9em; color: #78909C; text-transform: uppercase; letter-spacing: 0.5px; }
    .debug-box { background-color: #263238; color: #ECEFF1; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER (UPDATED)
# ==========================================
class DataManager:
    @staticmethod
    def get_default_master_data():
        # [UPDATE 19/01/2026] 13 Mã Chứng quyền mới (Thay thế list cũ)
        data = [
            {"Mã CW": "CACB2604", "Mã CS": "ACB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 26000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CMBB2605", "Mã CS": "MBB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 27000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CSTB2605", "Mã CS": "STB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 60000, "Ngày đáo hạn": "2026-06-29"},
            {"Mã CW": "CTCB2602", "Mã CS": "TCB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 36000, "Ngày đáo hạn": "2026-06-29"},
            {"Mã CW": "CVIB2601", "Mã CS": "VIB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 18000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CVPB2604", "Mã CS": "VPB", "Tỷ lệ CĐ": "3:1", "Giá thực hiện": 30000, "Ngày đáo hạn": "2026-06-29"},
            {"Mã CW": "CFPT2604", "Mã CS": "FPT", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 96000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CHPG2605", "Mã CS": "HPG", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 27000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CMSN2601", "Mã CS": "MSN", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 80000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CMWG2605", "Mã CS": "MWG", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 88000, "Ngày đáo hạn": "2026-06-29"},
            {"Mã CW": "CVHM2604", "Mã CS": "VHM", "Tỷ lệ CĐ": "10:1", "Giá thực hiện": 106000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CVNM2601", "Mã CS": "VNM", "Tỷ lệ CĐ": "5:1", "Giá thực hiện": 64000, "Ngày đáo hạn": "2026-12-28"},
            {"Mã CW": "CSHB2601", "Mã CS": "SHB", "Tỷ lệ CĐ": "2:1", "Giá thực hiện": 18000, "Ngày đáo hạn": "2026-06-29"}
        ]
        return pd.DataFrame(data)

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
# 4. AI SERVICE LAYER (HYBRID ENGINE - V15.7 + BATCH EXTENSION)
# ==========================================
def process_receipt_with_gemini(image, api_key):
    """
    [KEPT ORIGINAL] Xử lý Lệnh mua/Biên lai (Single Item)
    """
    genai.configure(api_key=api_key)
    generation_config = {"temperature": 0.0}
    priority_models = ['gemini-3-flash-preview', 'gemini-2.0-flash-exp']
    
    prompt = f"""
    Bạn là một trợ lý tài chính (OCR). Nhiệm vụ: Trích xuất thông tin LỆNH MUA / BIÊN LAI NỘP TIỀN.
    
    Các trường cần tìm:
    1. Mã chứng khoán (Symbol): Tìm mã Chứng quyền (CW...) hoặc mã Cơ sở.
    2. Số lượng (Qty): Khối lượng mua.
    3. Giá vốn (Price): Giá khớp lệnh/đơn giá (hoặc Tổng tiền chia Số lượng).
    4. Tổng tiền (Total Amount): Tổng giá trị giao dịch.
    5. Giá thị trường (Market Price): Nếu là biên lai mua/nộp tiền, mặc định bằng 0.

    Trả về JSON (chỉ số): 
    {{"symbol": "CWSTB", "qty": 1000, "price": 2168, "total_amount": 65040000, "market_price": 0}}
    """
    
    errors_log = [] 
    for model_name in priority_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image], generation_config=generation_config)
            text = response.text.strip()
            
            start_idx = text.find('{')
            if start_idx != -1:
                try:
                    json_data, _ = JSONDecoder().raw_decode(text[start_idx:])
                    json_data['_meta_model'] = model_name
                    return json_data
                except Exception as e:
                    errors_log.append(f"{model_name} Parse Error: {str(e)}")
        except Exception as e:
            errors_log.append(f"{model_name} API Error: {str(e)}")
            
    return {"error": "Thất bại toàn tập", "_meta_logs": errors_log}

def process_batch_list_with_gemini(image, api_key):
    """
    [NEW ADDITION] Xử lý Danh sách Import (Batch Items)
    Model: Gemini 2.5 Flash (Robot Mode - No Thinking)
    """
    genai.configure(api_key=api_key)
    priority_models = ['gemini-2.5-flash', 'gemini-2.0-flash-exp']
    
    # Cấu hình Robot Mode (Tắt suy luận để tránh bịa số)
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
    
    for model_name in priority_models:
        try:
            current_config = generation_config.copy()
            if "gemini-2.5" not in model_name: del current_config["thinking_config"]

            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image], generation_config=current_config)
            text = response.text.strip()
            
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != 0:
                try:
                    return json.loads(text[start:end])
                except: pass
        except Exception:
            continue
    return []

def map_batch_data(ocr_list, master_df):
    """ [NEW ADDITION] Mapping logic cho Batch Import """
    mapped_results = []
    for item in ocr_list:
        raw_cw = item.get('raw_cw', '')
        underlying = item.get('underlying', '')
        candidates = master_df[master_df['Mã CS'] == underlying]
        matched_symbol = None
        
        if not candidates.empty:
            # Logic: Match 2 số cuối (VD: .../05 -> CSTB2605)
            # Với mã mới C[Sym]26[xx], logic này vẫn hoạt động tốt
            suffix_match = re.search(r'/(\d{2})$', raw_cw.strip())
            if suffix_match:
                suffix = suffix_match.group(1)
                for idx, row in candidates.iterrows():
                    if row['Mã CW'].endswith(suffix):
                        matched_symbol = row['Mã CW']
                        break
            if not matched_symbol: matched_symbol = candidates.iloc[0]['Mã CW']
        
        mapped_results.append({
            "Chốt": True, 
            "Mã CW (Gợi ý)": matched_symbol if matched_symbol else "???",
            "Mã Gốc": raw_cw,
            "KL": float(item.get('qty', 0)),
            "Giá Vốn": float(item.get('price', 0))
        })
    return pd.DataFrame(mapped_results)

def scan_market_board(image, api_key):
    """
    [KEPT ORIGINAL] Xử lý Bảng giá (Batch Items)
    """
    genai.configure(api_key=api_key)
    
    target_model = 'gemini-2.5-flash' 
    fallback_models = ['gemini-2.0-flash-exp', 'gemini-1.5-flash']
    
    prompt = """
    SYSTEM: RAW_DATA_EXTRACTOR
    MODE: STRICT_PIXEL_TO_JSON
    CONSTRAINTS: NO REASONING. NO ROUNDING. EXACT DIGITS ONLY.
    TASK: EXTRACT PAIRS [SYMBOL, MATCHING_PRICE]
    TARGETS: UNDERLYING (e.g. VHM) AND WARRANTS (e.g. CW..., CV...)
    OUTPUT SCHEMA: [{"symbol": "STR", "price": FLOAT}]
    """
    
    all_models = [target_model] + fallback_models
    for model_name in all_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image])
            text = response.text.strip()
            
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != 0:
                result = JSONDecoder().raw_decode(text[start:end])[0]
                if isinstance(result, list) and len(result) > 0:
                    return result
        except Exception as e:
            print(f"OCR Board Error ({model_name}): {e}")
            continue
            
    return []

def auto_map_symbol(ocr_result, master_df):
    if not ocr_result or "error" in ocr_result: return None
    det_sym = str(ocr_result.get('symbol', '')).upper().strip()
    
    # 1. Exact Match
    mask_exact = master_df['Mã CW'] == det_sym
    if mask_exact.any(): return master_df.index[mask_exact].tolist()[0]
    
    # 2. Reverse Scan Underlying
    unique_underlying = master_df['Mã CS'].unique()
    found = [code for code in unique_underlying if code in det_sym]
    if found:
        mask_core = master_df['Mã CS'] == found[0]
        if mask_core.any(): return master_df.index[mask_core].tolist()[0]

    # 3. Typo Fix
    fixed_sym = det_sym.replace("W", "V").replace("CV", "") 
    mask_retry = master_df['Mã CS'].str.contains(fixed_sym)
    if len(fixed_sym) >= 3 and mask_retry.any(): return master_df.index[mask_retry].tolist()[0]

    return None

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

# ==========================================
# 5. MAIN APP
# ==========================================
def main():
    st.title("💎 LPBS CW Portfolio Master")
    st.caption(f"System: V15.8 | Stable V10 + Batch Core | Data Updated 19/01")

    # State Management (Clean Init)
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
    if 'ocr_result' not in st.session_state: st.session_state['ocr_result'] = None
    if 'temp_qty' not in st.session_state: st.session_state['temp_qty'] = 0.0
    if 'temp_price' not in st.session_state: st.session_state['temp_price'] = 0.0
    if 'temp_index' not in st.session_state: st.session_state['temp_index'] = None

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

    tab_input, tab_report, tab_sim = st.tabs(["1️⃣ NHẬP LIỆU", "2️⃣ CẬP NHẬT GIÁ & BÁO CÁO", "3️⃣ GIẢ LẬP"])

    # --- TAB 1: INPUT ---
    with tab_input:
        # [MODIFIED UI START] Thêm "📑 Quét Danh Sách" vào Radio
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### 📥 Thêm Vị Thế Mới")
            mode = st.radio("Chế độ:", ["📑 Quét Danh Sách (Batch)", "📸 Quét OCR (Lệnh mua/Biên lai)", "✍️ Nhập Tay"], horizontal=True)
            
            # --- FEATURE 1: BATCH IMPORT (NEW) ---
            if mode == "📑 Quét Danh Sách (Batch)":
                st.info("💡 Engine: Gemini 2.5 Flash (Robot Mode) - Dành cho danh sách nhiều mã.")
                uploaded_file = st.file_uploader("Upload ảnh Danh sách", type=['png', 'jpg', 'jpeg'], key="batch_upl")
                
                if uploaded_file and active_key:
                    if st.button("🚀 Phân Tích Danh Sách", type="primary", use_container_width=True):
                        with st.spinner("Đang kích hoạt Gemini 2.5 Flash (No Thinking)..."):
                            image = Image.open(uploaded_file)
                            result = process_batch_list_with_gemini(image, active_key)
                            
                            if result:
                                df_preview = map_batch_data(result, master_df)
                                st.session_state['batch_preview'] = df_preview
                                st.success(f"Tìm thấy {len(df_preview)} dòng!")
                            else:
                                st.error("Lỗi đọc dữ liệu hoặc không tìm thấy JSON.")
                
                # Bảng Preview & Import (Chỉ hiện khi ở Mode Batch)
                if 'batch_preview' in st.session_state and not st.session_state['batch_preview'].empty:
                    st.markdown("---")
                    
                    # Fix lỗi option "???" tránh ValueError
                    safe_options = master_df["Mã CW"].unique().tolist()
                    safe_options.append("???")

                    edited_df = st.data_editor(
                        st.session_state['batch_preview'],
                        column_config={
                            "Chốt": st.column_config.CheckboxColumn("Import?", default=True),
                            "Mã CW (Gợi ý)": st.column_config.SelectboxColumn("Mã CW", options=safe_options, required=True),
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
                        st.success(f"Đã nhập thành công {count} lệnh!")
                        del st.session_state['batch_preview']
                        st.rerun()

            # --- FEATURE 2: SINGLE OCR (KEPT ORIGINAL LOGIC) ---
            elif mode.startswith("📸"):
                uploaded_file = st.file_uploader("Upload ảnh Biên lai", type=['png', 'jpg'])
                if uploaded_file and active_key:
                    if st.button("🚀 Phân Tích (Gemini 3)", use_container_width=True):
                        with st.spinner("Đang đọc biên lai..."):
                            image = Image.open(uploaded_file)
                            result = process_receipt_with_gemini(image, active_key)
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
            
            # --- FORM NHẬP LIỆU CHUNG (KEPT ORIGINAL LOGIC) ---
            should_show_form = False
            # Nếu đang ở mode Nhập Tay -> Luôn hiện
            if "Nhập Tay" in mode:
                should_show_form = True
            # Nếu đang ở mode Single OCR -> Chỉ hiện khi đã có kết quả
            elif mode.startswith("📸") and st.session_state.get('ocr_result'):
                should_show_form = True
            
            # Khóa input nếu đang ở chế độ xem kết quả OCR (như code cũ)
            is_locked = True if (mode.startswith("📸") and "Batch" not in mode) else False

            if should_show_form:
                st.divider()
                if is_locked:
                    st.caption("🔒 Chế độ Xem: Dữ liệu từ AI. Muốn sửa đổi, vui lòng chọn chế độ 'Nhập Tay'.")

                cw_list = master_df["Mã CW"].unique()
                current_idx = st.session_state['temp_index']
                if current_idx is not None and (current_idx < 0 or current_idx >= len(cw_list)):
                     current_idx = None

                selected_cw = st.selectbox(
                    "Mã CW", 
                    cw_list, 
                    index=current_idx, 
                    placeholder="Chọn mã CW...",
                    disabled=is_locked 
                )
                
                qty = st.number_input(
                    "Số lượng", 
                    value=st.session_state['temp_qty'], 
                    step=100.0,
                    disabled=is_locked 
                )
                
                cost = st.number_input(
                    "Giá vốn", 
                    value=st.session_state['temp_price'], 
                    step=50.0,
                    disabled=is_locked 
                )
                
                if st.button("💾 Lưu vào Danh mục", type="primary", use_container_width=True):
                    if not selected_cw:
                        st.error("⚠️ Vui lòng chọn Mã CW!")
                    elif qty <= 0 or cost <= 0:
                        st.error("⚠️ Số lượng và Giá vốn phải lớn hơn 0")
                    else:
                        row = master_df[master_df['Mã CW'] == selected_cw].iloc[0]
                        add_to_portfolio(row, qty, cost)
                        st.success("Đã lưu thành công!")
                        st.session_state['temp_qty'] = 0.0
                        st.session_state['temp_price'] = 0.0
                        st.session_state['temp_index'] = None
                        st.session_state['ocr_result'] = None
                        st.rerun()

            elif mode.startswith("📸") and not st.session_state.get('ocr_result') and "Batch" not in mode:
                st.info("👈 Vui lòng Upload ảnh và bấm 'Phân Tích' để hiển thị thông tin.")
        # [MODIFIED UI END]

        with c2:
            if st.session_state['ocr_result']:
                res = st.session_state['ocr_result']
                st.markdown("#### 🔍 Glass Box Debug")
                with st.expander("Chi tiết xử lý AI", expanded=True):
                    st.markdown(f"**Model:** `{res.get('_meta_model', 'N/A')}`")
                    st.json(res)

    # --- TAB 2: UPDATE PRICE (KEPT ORIGINAL V15.7) ---
    with tab_report:
        pf = st.session_state.get('portfolio', [])
        if not pf:
            st.info("📭 Danh mục trống. Vui lòng thêm vị thế ở Tab 1.")
        else:
            st.markdown("### 🛠️ CẬP NHẬT GIÁ")
            
            # 1. CONTROL MODE
            update_mode = st.radio(
                "Phương thức cập nhật:", 
                ["📸 Quét Bảng Giá (Batch OCR)", "✍️ Chỉnh Sửa Thủ Công"], 
                horizontal=True,
                key="t2_mode"
            )
            is_view_only = True if update_mode.startswith("📸") else False

            # 2. OCR TOOL
            if update_mode.startswith("📸"):
                with st.expander("📸 Khu vực Upload Ảnh", expanded=True):
                    col_up, col_act = st.columns([3, 1])
                    with col_up:
                        img_file = st.file_uploader("Chụp ảnh bảng giá", type=['png', 'jpg'], key="board_upload")
                    with col_act:
                        st.write("") 
                        st.write("")
                        if img_file and active_key:
                            if st.button("🚀 Quét Ngay"):
                                with st.spinner("Đang quét với Gemini 2.5 Robot Mode..."):
                                    raw_data = scan_market_board(Image.open(img_file), active_key)
                                    if not raw_data:
                                        st.error("Không tìm thấy giá nào.")
                                    else:
                                        count = 0
                                        for price_item in raw_data:
                                            p_sym = str(price_item.get('symbol', '')).upper()
                                            p_val = float(price_item.get('price', 0))
                                            if p_val < 1000: p_val *= 1000
                                            
                                            for pf_item in st.session_state['portfolio']:
                                                if p_sym == pf_item['underlying']:
                                                    pf_item['market_price_cs'] = p_val
                                                    count += 1
                                                elif p_sym == pf_item['symbol']: 
                                                    pf_item['market_price_cw'] = p_val
                                                    count += 1
                                                elif (p_sym in pf_item['symbol']) and len(p_sym) > 4:
                                                    pf_item['market_price_cw'] = p_val
                                                    count += 1
                                        st.success(f"Đã cập nhật giá cho {count} mã!")
                                        st.rerun()

            # [OPTION B] AUTO-THEORETICAL FALLBACK
            for item in pf:
                curr_cw = item.get('market_price_cw', 0.0)
                curr_cs = item.get('market_price_cs', 0.0)
                if curr_cw <= 0 and curr_cs > 0:
                     intrinsic = FinancialEngine.calc_intrinsic_value(curr_cs, item['exercise_price'], item['ratio'])
                     item['market_price_cw'] = intrinsic

            # 3. DATA EDITOR (SECURE)
            st.divider()
            if is_view_only:
                st.caption("🔒 Chế độ Xem: Bảng giá đang bị khóa để bảo vệ dữ liệu AI. Chọn 'Chỉnh Sửa Thủ Công' để thay đổi.")

            input_data = []
            for item in pf:
                input_data.append({
                    "Mã CW": item['symbol'],
                    "Mã CS": item['underlying'],
                    "Giá TT (CW)": item.get('market_price_cw', 0.0),
                    "Giá CS (Gốc)": item.get('market_price_cs', 0.0)
                })
            
            edited_df = st.data_editor(
                pd.DataFrame(input_data),
                column_config={
                    "Giá TT (CW)": st.column_config.NumberColumn(format="%.0f", min_value=0),
                    "Giá CS (Gốc)": st.column_config.NumberColumn(format="%.0f", min_value=0),
                },
                use_container_width=True,
                key="price_editor",
                hide_index=True,
                disabled=is_view_only
            )

            # 4. CORE CALCULATION
            total_nav, total_cost = 0, 0
            price_map = edited_df.set_index("Mã CW").to_dict(orient="index")
            
            for item in pf:
                user_input = price_map.get(item['symbol'], {})
                mkt_cw = user_input.get("Giá TT (CW)", 0.0)
                mkt_cs = user_input.get("Giá CS (Gốc)", 0.0)
                
                # Update State nếu đang ở Mode Thủ công
                if not is_view_only:
                    item['market_price_cw'] = mkt_cw
                    item['market_price_cs'] = mkt_cs
                
                total_nav += item['qty'] * item['market_price_cw']
                total_cost += item['qty'] * item['cost_price']

            total_pnl = total_nav - total_cost
            pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            # 5. DASHBOARD UI
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("NAV", f"{total_nav:,.0f} đ")
            c2.metric("Tổng Lãi/Lỗ", f"{total_pnl:,.0f} đ", delta_color="normal")
            c3.metric("Hiệu suất", f"{pnl_pct:+.2f}%", delta_color="normal")

            st.markdown("### 2. CHI TIẾT DANH MỤC")
            display_data = []
            for item in pf:
                val_now = item['qty'] * item['market_price_cw']
                val_cost = item['qty'] * item['cost_price']
                display_data.append({
                    "Mã": item['symbol'], "SL": item['qty'], "Giá Vốn": item['cost_price'],
                    "Giá TT (CW)": item['market_price_cw'], "Giá trị TT": val_now,
                    "Lãi/Lỗ": val_now - val_cost, "%": (val_now - val_cost)/val_cost if val_cost>0 else 0
                })
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

            st.markdown("### 3. PHÂN TÍCH RỦI RO")
            risk_data = []
            for item in pf:
                bep = FinancialEngine.calc_bep(item['exercise_price'], item['cost_price'], item['ratio'])
                curr_cs = item.get('market_price_cs', 0)
                dist = ((curr_cs - bep) / bep) if bep > 0 and curr_cs > 0 else 0
                days = DataManager.calc_days_to_maturity(item['maturity'])
                status = "🟢" if dist > 0 else "🔴" if dist < -0.1 else "🟡"
                if curr_cs == 0: status = "⚪"
                
                risk_data.append({
                    "Mã": item['symbol'], "Hòa vốn (BEP)": bep, "Giá CS": curr_cs,
                    "Khoảng cách BEP": dist, "Còn lại": f"{days} ngày", "Trạng thái": status
                })
            st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    # --- TAB 3: SIMULATOR (KEPT ORIGINAL V15.7) ---
    with tab_sim:
        if not st.session_state['portfolio']:
            st.info("Vui lòng thêm vị thế trước.")
        else:
            pf_df = pd.DataFrame(st.session_state['portfolio'])
            sim_cw = st.selectbox("Chọn mã giả lập:", pf_df['symbol'].unique())
            item = next(x for x in st.session_state['portfolio'] if x['symbol'] == sim_cw)
            
            curr_cs = item.get('market_price_cs', 20000)
            if curr_cs == 0: curr_cs = 20000
            
            st.info(f"Giả lập cho **{sim_cw}** (Giá vốn: {item['cost_price']:,.0f})")
            target_cs = st.slider("Giá Cơ sở Tương lai:", int(curr_cs * 0.8), int(curr_cs * 1.5), int(curr_cs))
            
            sim_val = FinancialEngine.calc_intrinsic_value(target_cs, item['exercise_price'], item['ratio'])
            sim_pnl = (sim_val - item['cost_price']) * item['qty']
            
            c1, c2 = st.columns(2)
            c1.metric("Giá CW Lý thuyết", f"{sim_val:,.0f} đ")
            c2.metric("Lãi/Lỗ Dự kiến", f"{sim_pnl:,.0f} đ", delta_color="normal")

if __name__ == "__main__":
    main()
