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
st.set_page_config(page_title="LPBS CW Tracker", layout="wide", page_icon="🔶")

# UPDATE: Giờ chuẩn hệ thống lúc tôi đang viết dòng này
build_time_str = "17:55:00 - 05/01/2026" 

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
    .metric-card:hover { transform: translateY(-2px); }
    
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
    
    .ocr-box {
        border: 2px dashed #FF8F00;
        padding: 15px;
        border-radius: 12px;
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
# 4. AI SERVICE LAYER (V8.9)
# ==========================================
def process_image_with_gemini(image, api_key):
    try:
        genai.configure(api_key=api_key)
        model_name = 'gemini-3-flash-preview' 
        model = genai.GenerativeModel(model_name)
        
        prompt = """
        Bạn là một trợ lý nhập liệu tài chính (OCR). Nhiệm vụ:
        1. Nhìn vào ảnh biên lai chuyển tiền hoặc màn hình đặt lệnh chứng khoán.
        2. Tìm Mã chứng khoán (ví dụ: MWG, HPG, VHM...).
        3. Tìm Số lượng (Quantity/Khối lượng).
        4. Tìm Giá khớp/Giá vốn (Price).
        
        Trả về kết quả CHỈ LÀ JSON thuần túy, không có markdown, theo định dạng:
        {"symbol": "XXX", "qty": 1000, "price": 50000}
        
        Nếu không tìm thấy trường nào thì trả về null.
        """
        response = model.generate_content([prompt, image])
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:-3]
        elif text.startswith("```"): text = text[3:-3]
        return json.loads(text) 
    except Exception as e:
        return {"error": str(e)}

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
    st.caption(f"System: V8.9 | Build: {build_time_str} | Gemini 3.0 Ready")

    if 'ocr_result' not in st.session_state:
        st.session_state['ocr_result'] = None

    # --- SIDEBAR ---
    with st.sidebar:
        with st.expander("🔑 Cấu hình AI", expanded=True):
            api_key = st.text_input("API Key", type="password", placeholder="AIzaSy...")
            st.markdown("[👉 Lấy Key miễn phí](https://aistudio.google.com/app/apikey)")

        st.header("📸 AI Quét Lệnh")
        uploaded_img = st.file_uploader("Tải ảnh biên lai/SMS", type=["png", "jpg", "jpeg"])
        
        if uploaded_img and api_key and st.button("🚀 Phân tích ngay"):
            with st.spinner("Đang xử lý..."):
                image = Image.open(uploaded_img)
                result = process_image_with_gemini(image, api_key)
                if "error" in result: st.error(result['error'])
                else: 
                    st.session_state['ocr_result'] = result
                    st.success("Xong!")
        
        st.divider()
        master_df = DataManager.get_default_master_data()
        
        # Clean Data
        if "Giá thực hiện" in master_df.columns:
            master_df["Giá thực hiện"] = master_df["Giá thực hiện"].apply(DataManager.clean_number_value)
            master_df["Tỷ lệ CĐ"] = master_df["Tỷ lệ CĐ"].apply(DataManager.clean_number_value)

        # Auto-Fill
        default_qty, default_price, default_index = 1000.0, 1000.0, 0
        if st.session_state['ocr_result']:
            res = st.session_state['ocr_result']
            if res.get('qty'): default_qty = float(res['qty'])
            if res.get('price'): default_price = float(res['price'])
            det_sym = str(res.get('symbol', '')).upper()
            if det_sym:
                mask = master_df['Mã CW'].str.contains(det_sym) | master_df['Mã CS'].str.contains(det_sym)
                found = master_df.index[mask].tolist()
                if found: default_index = found[0]

        st.header("🛠️ Nhập liệu")
        cw_list = master_df["Mã CW"].unique()
        selected_cw = st.selectbox("Chọn Mã CW", cw_list, index=int(default_index))
        
        cw_info = master_df[master_df["Mã CW"] == selected_cw].iloc[0]
        val_exercise = float(cw_info.get("Giá thực hiện", 0))
        val_ratio = float(cw_info.get("Tỷ lệ CĐ", 0))
        val_underlying_code = str(cw_info.get("Mã CS", "UNKNOWN"))
        val_maturity_date = str(cw_info.get("Ngày đáo hạn", ""))
        
        qty = st.number_input("Số lượng", value=default_qty, step=100.0)
        cost_price = st.number_input("Giá vốn (VND)", value=default_price, step=50.0)

    # --- MAIN DISPLAY ---
    # 1. Profile CW
    days_left = DataManager.calc_days_to_maturity(val_maturity_date)
    render_cw_profile(selected_cw, val_underlying_code, val_exercise, val_ratio, val_maturity_date, days_left)
    
    # 2. Logic Calc
    current_real_price = DataManager.get_realtime_price(val_underlying_code)
    engine = FinancialEngine()
    bep = engine.calc_bep(val_exercise, cost_price, val_ratio)
    cw_intrinsic = engine.calc_intrinsic_value(current_real_price, val_exercise, val_ratio)
    
    # Snapshot
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
