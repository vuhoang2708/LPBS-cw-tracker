import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import google.generativeai as genai
from datetime import datetime, timedelta
from PIL import Image

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
        margin-bottom: 10px;
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
    
    .ocr-box {
        border: 2px dashed #FF8F00;
        padding: 10px;
        border-radius: 10px;
        background-color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #FF8F00 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER
# ==========================================
class DataManager:
    @staticmethod
    def get_default_master_data():
        """Dữ liệu lõi 13 mã CW mới nhất của LPBS"""
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
# 4. AI SERVICE LAYER (NEW)
# ==========================================
def process_image_with_gemini(image, api_key):
    """
    Gửi ảnh lên Gemini Flash để trích xuất dữ liệu.
    Trả về dict: {'symbol': '...', 'qty': ..., 'price': ...}
    """
    try:
        genai.configure(api_key=api_key)
        # Sử dụng model Flash cho tốc độ nhanh và rẻ
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        Bạn là một trợ lý nhập liệu tài chính (OCR). Nhiệm vụ:
        1. Nhìn vào ảnh biên lai chuyển tiền hoặc màn hình đặt lệnh chứng khoán.
        2. Tìm Mã chứng khoán (ví dụ: MWG, HPG, VHM...).
        3. Tìm Số lượng (Quantity/Khối lượng).
        4. Tìm Giá khớp/Giá vốn (Price).
        
        Trả về kết quả CHỈ LÀ JSON thuần túy, không có markdown, theo định dạng:
        {"symbol": "XXX", "qty": 1000, "price": 50000}
        
        Nếu không tìm thấy trường nào thì để null.
        """
        
        response = model.generate_content([prompt, image])
        
        # Xử lý chuỗi trả về để lấy JSON sạch
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
            
        return eval(text) # Chuyển string thành dict
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 5. UI HELPER
# ==========================================
def render_metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.9em; color:#666;">{label}</div>
        <div style="font-size:1.5em; font-weight:bold; color:#E65100;">{value}</div>
        <div style="font-size:0.8em; color:#888;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. MAIN APP
# ==========================================
def main():
    st.title("🔶 LPBS CW Tracker & Simulator")
    st.caption(f"Credit: VuHoang | Build: {build_time_str} | Status: Stable V8.0 (AI Inside)")

    # Init Session State cho AI
    if 'ocr_result' not in st.session_state:
        st.session_state['ocr_result'] = None

    # --- SIDEBAR ---
    with st.sidebar:
        # A. CẤU HÌNH AI (Quan trọng)
        with st.expander("🔑 Cấu hình AI (Bước 1)", expanded=True):
            st.caption("Nhập Google AI Key để kích hoạt tính năng đọc ảnh.")
            api_key = st.text_input("API Key", type="password", placeholder="AIzaSy...")
            st.markdown("[👉 Lấy Key miễn phí tại đây](https://aistudio.google.com/app/apikey)")

        # B. OCR SECTION
        st.header("📸 AI Quét Lệnh")
        st.markdown('<div class="ocr-box">', unsafe_allow_html=True)
        uploaded_img = st.file_uploader("Tải ảnh biên lai/SMS", type=["png", "jpg", "jpeg"])
        
        if uploaded_img:
            if not api_key:
                st.warning("⚠️ Vui lòng nhập API Key ở trên trước!")
            else:
                if st.button("🚀 Phân tích ngay"):
                    with st.spinner("Gemini đang đọc ảnh..."):
                        try:
                            image = Image.open(uploaded_img)
                            result = process_image_with_gemini(image, api_key)
                            
                            if "error" in result:
                                st.error(f"Lỗi AI: {result['error']}")
                            else:
                                st.session_state['ocr_result'] = result
                                st.success("✅ Đã trích xuất xong!")
                                st.json(result) # Hiển thị kết quả thô để check
                        except Exception as e:
                            st.error(f"Lỗi xử lý ảnh: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()

        # C. DATA & INPUT
        master_df = DataManager.get_default_master_data()
        
        # Admin Import (Giấu gọn)
        with st.expander("⚙️ Admin: Upload CSV"):
            uploaded_csv = st.file_uploader("File danh sách mã", type=["csv"])
            if uploaded_csv:
                try:
                    temp = pd.read_csv(uploaded_csv)
                    temp.columns = temp.columns.str.strip()
                    master_df = temp
                    st.success("Updated CSV!")
                except: pass

        # Clean Data
        if "Giá thực hiện" in master_df.columns:
            master_df["Giá thực hiện"] = master_df["Giá thực hiện"].apply(DataManager.clean_number_value)
            master_df["Tỷ lệ CĐ"] = master_df["Tỷ lệ CĐ"].apply(DataManager.clean_number_value)

        # D. AUTO-FILL LOGIC (Điền form tự động từ kết quả AI)
        default_qty = 1000.0
        default_price = 1000.0
        default_index = 0 # Mặc định chọn mã đầu tiên
        
        if st.session_state['ocr_result']:
            res = st.session_state['ocr_result']
            # 1. Fill Số lượng & Giá
            if res.get('qty'): default_qty = float(res['qty'])
            if res.get('price'): default_price = float(res['price'])
            
            # 2. Fill Mã (Tìm tương đối)
            detected_sym = str(res.get('symbol', '')).upper()
            if detected_sym:
                # Tìm xem mã AI đọc được có nằm trong cột Mã CW hoặc Mã CS không
                mask = master_df['Mã CW'].str.contains(detected_sym) | master_df['Mã CS'].str.contains(detected_sym)
                found_idx = master_df.index[mask].tolist()
                if found_idx:
                    default_index = found_idx[0]
                    st.toast(f"🤖 AI đã chọn mã: {master_df.iloc[default_index]['Mã CW']}")

        # E. MANUAL INPUT FORM
        st.header("🛠️ Nhập liệu")
        cw_list = master_df["Mã CW"].unique()
        
        # Selectbox có index động
        selected_cw = st.selectbox("Chọn Mã CW", cw_list, index=int(default_index))
        
        cw_info = master_df[master_df["Mã CW"] == selected_cw].iloc[0]
        val_exercise = float(cw_info.get("Giá thực hiện", 0))
        val_ratio = float(cw_info.get("Tỷ lệ CĐ", 0))
        val_underlying_code = str(cw_info.get("Mã CS", "UNKNOWN"))
        
        qty = st.number_input("Số lượng", value=default_qty, step=100.0)
        cost_price = st.number_input("Giá vốn (VND)", value=default_price, step=50.0)

    # --- MAIN PROCESS ---
    current_real_price = DataManager.get_realtime_price(val_underlying_code)
    
    # Snapshot State
    if 'anchor_cw' not in st.session_state or st.session_state['anchor_cw'] != selected_cw:
        st.session_state['anchor_cw'] = selected_cw
        st.session_state['anchor_price'] = current_real_price
        st.session_state['sim_target_price'] = int(current_real_price)

    anchor_price = st.session_state['anchor_price']
    engine = FinancialEngine()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎲 Simulator (Giả lập)", "📉 Biểu đồ Hòa vốn"])

    with tab1:
        bep = engine.calc_bep(val_exercise, cost_price, val_ratio)
        cw_price_theory = engine.calc_intrinsic_value(current_real_price, val_exercise, val_ratio)
        
        c1, c2, c3 = st.columns(3)
        with c1: render_metric_card(f"Giá {val_underlying_code}", f"{current_real_price:,.0f} ₫", "Thị trường")
        with c2: render_metric_card("Giá CW Lý thuyết", f"{cw_price_theory:,.0f} ₫", "Intrinsic Value")
        with c3: render_metric_card("Điểm Hòa Vốn", f"{bep:,.0f} ₫", "Break-even Point")
        
        if current_real_price < bep:
             diff = ((bep - current_real_price) / current_real_price) * 100
             st.warning(f"⚠️ Cần {val_underlying_code} tăng **{diff:.2f}%** để hòa vốn.")
        else:
             st.success(f"🚀 Đã có lãi! (Thị giá > BEP)")

    with tab2:
        st.subheader("Giả lập Lợi nhuận")
        st.info(f"Giả định giá tương lai cho: {val_underlying_code} (Hiện tại: {anchor_price:,.0f})")
        
        target_price = st.slider(
            "Kéo giá mục tiêu:", 
            min_value=int(anchor_price*0.5), 
            max_value=int(anchor_price*1.5), 
            value=st.session_state['sim_target_price'], 
            step=100
        )
        
        sim_cw = engine.calc_intrinsic_value(target_price, val_exercise, val_ratio)
        sim_pnl = (sim_cw - cost_price) * qty
        sim_pnl_pct = (sim_pnl / (cost_price*qty) * 100) if cost_price > 0 else 0
        
        c1, c2 = st.columns(2)
        with c1: st.metric("Giá CW Dự kiến", f"{sim_cw:,.0f} ₫")
        with c2: 
            color = "green" if sim_pnl >= 0 else "red"
            st.markdown(f"Lãi/Lỗ: :{color}[**{sim_pnl:,.0f} VND ({sim_pnl_pct:.2f}%)**]")

    with tab3:
        st.subheader("Biểu đồ P/L")
        x_vals = np.linspace(current_real_price*0.8, current_real_price*1.2, 50)
        y_vals = [(engine.calc_intrinsic_value(x, val_exercise, val_ratio) - cost_price)*qty for x in x_vals]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='P/L Profile', line=dict(color='#FF8F00', width=3)))
        fig.add_vline(x=bep, line_dash="dash", line_color="#5D4037", annotation_text="Hòa Vốn")
        fig.add_hline(y=0, line_color="gray")
        
        # Điểm hiện tại
        curr_pnl = (cw_price_theory - cost_price) * qty
        fig.add_trace(go.Scatter(x=[current_real_price], y=[curr_pnl], mode='markers', name='Hiện tại', marker=dict(color='red', size=12)))
        
        fig.update_layout(template="plotly_white", yaxis_title="Lãi/Lỗ (VND)")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
