import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import google.generativeai as genai
import json
from datetime import datetime, timedelta
from PIL import Image

# ==========================================
# 1. CONFIG & BRANDING
# ==========================================
st.set_page_config(page_title="LPBS CW Tracker & Simulator", layout="wide", page_icon="🔶")

vn_time = datetime.utcnow() + timedelta(hours=7)
build_time_str = vn_time.strftime("%H:%M:%S - %d/%m/%Y")

st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    h1, h2, h3 { color: #5D4037 !important; font-family: 'Segoe UI', sans-serif; }
    
    [data-testid="stSidebar"] {
        background-color: #FFF8E1;
        border-right: 1px solid #FFECB3;
    }
    
    .metric-card {
        background: white;
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #EEE;
        border-left: 5px solid #FF8F00;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #4E342E;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    
    .cw-profile-box {
        background-color: #E3F2FD;
        border: 1px solid #90CAF9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        color: #0D47A1;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; 
        background-color: #FFF; 
        border-radius: 4px; 
        color: #666;
        font-weight: 600;
        border: 1px solid #EEE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF8F00 !important;
        color: white !important;
        border-color: #FF8F00;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER (Dữ liệu CW Chuẩn của bạn)
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
        """Giả lập giá nền"""
        base_prices = {
            "HPG": 28500, "MWG": 48200, "VHM": 41800, "STB": 30500, "VNM": 66000,
            "FPT": 95000, "MBB": 18500, "TCB": 33000, "VPB": 19200, "MSN": 62000,
            "VIB": 21500, "SHB": 11200, "ACB": 24500
        }
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
        except:
            return 0

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
        if price_underlying > price_exercise:
            return "ITM (Có lời)", "green"
        elif price_underlying < price_exercise:
            return "OTM (Chưa lời)", "red"
        else:
            return "ATM (Ngang giá)", "orange"

# ==========================================
# 4. AI SERVICE LAYER (V11.5 - GEMINI 3.0 IS KING)
# ==========================================
def process_image_with_gemini(image, api_key):
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(temperature=0.0)
    
    # --- DANH SÁCH ƯU TIÊN CHUẨN 2026 ---
    # 1. gemini-3-flash-preview: Ưu tiên số 1 (Bạn đã kiểm chứng).
    # 2. gemini-1.5-pro: Backup tin cậy.
    # 3. gemini-1.5-flash: Backup cuối cùng.
    priority_models = ['gemini-3-flash-preview', 'gemini-1.5-pro', 'gemini-1.5-flash']
    
    prompt = """
    Bạn là một trợ lý tài chính (OCR). Nhiệm vụ: Trích xuất dữ liệu từ ảnh (Biên lai, Danh mục, hoặc Bảng giá/Popup).
    
    Hãy tìm các trường sau (nếu có):
    1. Mã chứng khoán (Symbol): 
       - Ưu tiên tìm mã CW (bắt đầu bằng CW...).
       - Nếu là Bảng giá (chỉ có VHM, MWG...), hãy lấy mã đó và thêm ghi chú là "Cơ sở".
    2. Số lượng (Qty): Khối lượng mua/nắm giữ (Nếu có).
    3. Giá vốn (Price): Giá khớp lệnh, giá trung bình (Nếu có).
       - Lưu ý: Bảng giá thị trường sẽ KHÔNG có giá này -> Trả về null.
    4. Giá thị trường (Market Price): Cột "Last", "Giá hiện tại", "Khớp lệnh", hoặc con số to nhất trên bảng giá.

    Trả về JSON: 
    {"symbol": "XXX", "qty": 1000, "price": 50000, "market_price": 52000}
    (Trường nào không thấy thì để null).
    """
    
    last_error = ""
    for model_name in priority_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image], generation_config=generation_config)
            text = response.text.strip()
            
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_data = json.loads(match.group(0))
                json_data['_processed_by'] = model_name 
                return json_data
            else:
                last_error = f"Model {model_name} trả về sai định dạng."
                continue 
                
        except Exception as e:
            last_error = f"Lỗi với {model_name}: {str(e)}"
            continue 
            
    return {"error": f"Tất cả model đều thất bại. Lỗi cuối: {last_error}"}

# ==========================================
# 5. UI HELPER
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
    st.caption(f"System: V11.5 | Build: {build_time_str} | Gemini 3.0 Lead")

    if 'ocr_result' not in st.session_state:
        st.session_state['ocr_result'] = None

    # --- SESSIONS CHO INPUT TAY ---
    if 'user_qty' not in st.session_state: st.session_state['user_qty'] = 1000.0
    if 'user_price' not in st.session_state: st.session_state['user_price'] = 1000.0
    if 'user_index' not in st.session_state: st.session_state['user_index'] = 0

    # --- SIDEBAR ---
    with st.sidebar:
        with st.expander("🔑 Cấu hình AI", expanded=True):
            api_key = st.text_input("API Key", type="password", placeholder="AIzaSy...")
            st.markdown("[👉 Lấy Key miễn phí](https://aistudio.google.com/app/apikey)")

        st.header("📸 AI Quét Lệnh")
        uploaded_img = st.file_uploader("Tải ảnh biên lai/SMS/Bảng giá", type=["png", "jpg", "jpeg"])
        
        if uploaded_img and api_key:
            if st.button("🚀 Phân tích ngay"):
                with st.spinner("Đang xử lý (Ưu tiên Gemini 3.0)..."):
                    image = Image.open(uploaded_img)
                    result = process_image_with_gemini(image, api_key)
                    
                    if "error" in result: 
                        st.error(result['error'])
                    else: 
                        st.session_state['ocr_result'] = result
                        
                        # --- LOGIC XỬ LÝ KẾT QUẢ THÔNG MINH ---
                        msg_parts = []
                        msg_parts.append(f"Model: {result.get('_processed_by', 'Unknown')}")
                        
                        if result.get('symbol'): msg_parts.append(f"Mã: {result['symbol']}")
                        
                        # 1. Cập nhật Số lượng & Giá vốn (Nếu có)
                        if result.get('qty'): 
                            st.session_state['user_qty'] = float(result['qty'])
                            msg_parts.append("SL ✅")
                        if result.get('price'): 
                            st.session_state['user_price'] = float(result['price'])
                            msg_parts.append("Giá vốn ✅")
                        else:
                            msg_parts.append("Giá vốn ❌ (Giữ cũ)")

                        # 2. Cập nhật Giá thị trường (Nếu có)
                        if result.get('market_price'):
                            st.session_state['temp_ocr_market_price'] = float(result['market_price'])
                            msg_parts.append(f"Giá TT: {result['market_price']:,.0f}")
                        
                        st.toast(" | ".join(msg_parts))

            if st.session_state['ocr_result']:
                with st.expander("👁️ Debug Info", expanded=True):
                    st.json(st.session_state['ocr_result'])
        
        st.divider()
        master_df = DataManager.get_default_master_data()
        
        if "Giá thực hiện" in master_df.columns:
            master_df["Giá thực hiện"] = master_df["Giá thực hiện"].apply(DataManager.clean_number_value)
            master_df["Tỷ lệ CĐ"] = master_df["Tỷ lệ CĐ"].apply(DataManager.clean_number_value)

        # Logic chọn mã từ OCR
        if st.session_state['ocr_result']:
            res = st.session_state['ocr_result']
            det_sym = str(res.get('symbol', '')).upper().strip()
            if det_sym:
                mask_exact = master_df['Mã CW'] == det_sym
                mask_contains = master_df['Mã CW'].str.contains(det_sym) | master_df['Mã CS'].str.contains(det_sym)
                
                core_sym = re.sub(r'[^A-Z]', '', det_sym).replace("CW", "").replace("CV", "")
                mask_core = master_df['Mã CS'].str.contains(core_sym) if len(core_sym) >= 3 else mask_contains

                if mask_exact.any(): st.session_state['user_index'] = master_df.index[mask_exact].tolist()[0]
                elif mask_core.any(): st.session_state['user_index'] = master_df.index[mask_core].tolist()[0]

        st.header("🛠️ Nhập liệu")
        cw_list = master_df["Mã CW"].unique()
        selected_cw = st.selectbox("Chọn Mã CW", cw_list, index=int(st.session_state.get('user_index', 0)))
        
        cw_info = master_df[master_df["Mã CW"] == selected_cw].iloc[0]
        val_exercise = float(cw_info.get("Giá thực hiện", 0))
        val_ratio = float(cw_info.get("Tỷ lệ CĐ", 0))
        val_underlying_code = str(cw_info.get("Mã CS", "UNKNOWN"))
        val_maturity_date = str(cw_info.get("Ngày đáo hạn", ""))
        
        # Input có State
        qty = st.number_input("Số lượng", value=st.session_state['user_qty'], step=100.0)
        cost_price = st.number_input("Giá vốn (VND)", value=st.session_state['user_price'], step=50.0)
        
        st.session_state['user_qty'] = qty
        st.session_state['user_price'] = cost_price

    # --- MAIN LOGIC ---
    days_left = DataManager.calc_days_to_maturity(val_maturity_date)
    render_cw_profile(selected_cw, val_underlying_code, val_exercise, val_ratio, val_maturity_date, days_left)
    
    manual_key = f"manual_price_{val_underlying_code}"
    if manual_key not in st.session_state:
        st.session_state[manual_key] = float(DataManager.get_realtime_price_simulated(val_underlying_code))
        
    if 'temp_ocr_market_price' in st.session_state:
        st.session_state[manual_key] = st.session_state['temp_ocr_market_price']
        del st.session_state['temp_ocr_market_price']

    st.markdown("---")
    c_p1, c_p2 = st.columns([1, 2])
    with c_p1:
        st.info("📡 Giá thị trường (Live)")
        if st.button("🔄 Reset giá giả lập"):
            st.session_state[manual_key] = float(DataManager.get_realtime_price_simulated(val_underlying_code))
            st.rerun()
            
    with c_p2:
        current_real_price = st.number_input(
            f"Giá {val_underlying_code} hiện tại (VND):", 
            value=float(st.session_state[manual_key]),
            step=100.0,
            format="%.0f",
            help="Tự động điền từ OCR (nếu quét bảng giá) hoặc Giả lập. Bạn có thể sửa tay."
        )
        st.session_state[manual_key] = current_real_price

    engine = FinancialEngine()
    bep = engine.calc_bep(val_exercise, cost_price, val_ratio)
    cw_intrinsic = engine.calc_intrinsic_value(current_real_price, val_exercise, val_ratio)
    
    if 'anchor_cw' not in st.session_state or st.session_state['anchor_cw'] != selected_cw:
        st.session_state['anchor_cw'] = selected_cw
        st.session_state['anchor_price'] = current_real_price
        st.session_state['sim_target_price'] = int(current_real_price)
    anchor_price = st.session_state['anchor_price']

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎲 Simulator", "📉 Chart P/L"])

    with tab1:
        moneyness_label, moneyness_color = FinancialEngine.get_moneyness(current_real_price, val_exercise)
        c1, c2, c3 = st.columns(3)
        with c1: render_metric_card(f"Giá {val_underlying_code}", f"{current_real_price:,.0f} ₫", moneyness_label, moneyness_color)
        with c2: 
            diff_pct = ((bep - current_real_price) / current_real_price) * 100
            status_text = f"Cần tăng {diff_pct:.1f}% để hòa vốn" if diff_pct > 0 else "Đã vượt BEP"
            render_metric_card("Điểm Hòa Vốn (BEP)", f"{bep:,.0f} ₫", status_text, "#E65100")
        with c3: render_metric_card("Giá CW Lý thuyết", f"{cw_intrinsic:,.0f} ₫", "Intrinsic Value", "#1565C0")
        
        if days_left < 30 and days_left > 0:
            st.warning(f"⚠️ CẢNH BÁO: Mã sắp đáo hạn ({days_left} ngày).")
        elif days_left <= 0:
            st.error("⛔ Mã ĐÃ ĐÁO HẠN.")

    with tab2:
        st.info("Kéo thanh trượt để giả lập:")
        slider_min = int(anchor_price * 0.5)
        slider_max = int(max(anchor_price * 1.5, bep * 1.5)) 
        
        target_price = st.slider("Giá Cơ sở Tương lai:", slider_min, slider_max, st.session_state['sim_target_price'], 100)
        
        sim_cw = engine.calc_intrinsic_value(target_price, val_exercise, val_ratio)
        sim_pnl = (sim_cw - cost_price) * qty
        sim_pnl_pct = (sim_pnl / (cost_price * qty) * 100) if cost_price > 0 else 0
        
        c1, c2 = st.columns(2)
        with c1: render_metric_card("Giá CW Dự kiến", f"{sim_cw:,.0f} ₫")
        with c2: 
            color = "green" if sim_pnl >= 0 else "red"
            st.markdown(f"### Lãi/Lỗ: :{color}[{sim_pnl:,.0f} VND ({sim_pnl_pct:.2f}%)]")

    with tab3:
        plot_max = max(current_real_price * 1.2, bep * 1.2)
        x_vals = np.linspace(current_real_price * 0.8, plot_max, 50)
        y_vals = [(engine.calc_intrinsic_value(x, val_exercise, val_ratio) - cost_price)*qty for x in x_vals]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='P/L Profile', line=dict(color='#FF8F00', width=3)))
        fig.add_vline(x=bep, line_dash="dash", line_color="#5D4037", annotation_text=f"BEP: {bep:,.0f}")
        fig.add_hline(y=0, line_color="gray")
        
        curr_pnl = (cw_intrinsic - cost_price) * qty
        fig.add_trace(go.Scatter(x=[current_real_price], y=[curr_pnl], mode='markers', name='Hiện tại', marker=dict(color='red', size=12)))
        
        fig.update_layout(template="plotly_white", yaxis_title="Lãi/Lỗ (VND)", xaxis_title=f"Giá {val_underlying_code}")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
