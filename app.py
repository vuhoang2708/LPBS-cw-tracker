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

# CSS TÙY BIẾN THEO MÀU THƯƠNG HIỆU LPBS (CAM - VÀNG - NÂU)
st.markdown("""
<style>
    /* 1. Tổng thể */
    .main { background-color: #FFFFFF; }
    h1, h2, h3 { color: #5D4037 !important; } /* Màu Nâu đậm thương hiệu */
    
    /* 2. Sidebar (Màu kem sáng) */
    [data-testid="stSidebar"] {
        background-color: #FFF8E1; /* Light Cream */
        border-right: 1px solid #FFECB3;
    }
    
    /* 3. Metric Card (Thẻ chỉ số) */
    .metric-card {
        background: linear-gradient(to right, #FFF3E0, #FFFFFF);
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #FF8F00; /* Cam đậm LPBS */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #4E342E;
    }
    
    /* 4. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; 
        background-color: #FFF8E1; 
        border-radius: 5px 5px 0px 0px; 
        color: #5D4037;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF8F00 !important; /* Màu Cam khi chọn */
        color: white !important;
    }

    /* 5. Debug Box & Info Box */
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

    /* 6. Custom Button & Slider colors (Hack nhẹ Streamlit) */
    div.stSlider > div[data-baseweb = "slider"] > div > div > div[role="slider"]{
        background-color: #FF8F00 !important;
    }
    div.stSlider > div[data-baseweb = "slider"] > div > div {
        background-color: #FFECB3 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER
# ==========================================
class DataManager:
    @staticmethod
    def get_default_master_data():
        return pd.DataFrame([
            {"Mã CW": "CHPG2316", "Mã CS": "HPG", "Tỷ lệ CĐ": 2, "Giá thực hiện": 28000, "Ngày đáo hạn": "2026-06-01", "Trạng thái": "Listed"},
            {"Mã CW": "CMWG2305", "Mã CS": "MWG", "Tỷ lệ CĐ": 5, "Giá thực hiện": 45000, "Ngày đáo hạn": "2026-12-31", "Trạng thái": "Pre-listing"},
        ])

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
    # --- HEADER & CREDIT ---
    st.title("🔶 LPBS CW Tracker & Simulator")
    st.caption(f"Credit: VuHoang | Build: {build_time_str}")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📂 Dữ liệu Nguồn")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        col_exercise = "Giá thực hiện"
        col_ratio = "Tỷ lệ CĐ"
        col_code = "Mã CW"
        col_underlying = "Mã CS"

        if uploaded_file is not None:
            try:
                master_df = pd.read_csv(uploaded_file)
                master_df.columns = master_df.columns.str.strip()
                
                found_exercise = DataManager.smart_find_column(master_df, ['thực hiện', 'exercise', 'strike', 'giá th'])
                found_ratio = DataManager.smart_find_column(master_df, ['tỷ lệ', 'ratio', 'conversion', 'cđ'])
                found_code = DataManager.smart_find_column(master_df, ['mã cw', 'cw code', 'symbol'])
                found_underlying = DataManager.smart_find_column(master_df, ['mã cs', 'underlying', 'cơ sở'])

                if found_exercise: col_exercise = found_exercise
                if found_ratio: col_ratio = found_ratio
                if found_code: col_code = found_code
                if found_underlying: col_underlying = found_underlying
                
                for col in [col_exercise, col_ratio]:
                    if col in master_df.columns:
                        master_df[col] = master_df[col].apply(DataManager.clean_number_value)
                
                st.success(f"✅ Đã map cột: {col_exercise} & {col_ratio}")
            except Exception as e:
                st.error(f"Lỗi file: {e}")
                master_df = DataManager.get_default_master_data()
        else:
            master_df = DataManager.get_default_master_data()

        st.divider()
        if master_df.empty: st.stop()

        cw_list = master_df[col_code].unique()
        selected_cw = st.selectbox("Chọn Mã CW", cw_list)
        
        cw_info = master_df[master_df[col_code] == selected_cw].iloc[0]
        
        val_exercise = float(cw_info.get(col_exercise, 0))
        val_ratio = float(cw_info.get(col_ratio, 0))
        val_underlying_code = str(cw_info.get(col_underlying, "UNKNOWN"))
        
        qty = st.number_input("Số lượng", value=1000, step=100)
        cost_price = st.number_input("Giá vốn (VND)", value=1000, step=50)

    # --- MAIN PROCESS ---
    current_real_price = DataManager.get_realtime_price(val_underlying_code)
    
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
        
        # Custom Metric Card HTML
        def card(label, value, sub=""):
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9em; color:#666;">{label}</div>
                <div style="font-size:1.5em; font-weight:bold; color:#E65100;">{value}</div>
                <div style="font-size:0.8em; color:#888;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: card(f"Giá {val_underlying_code}", f"{current_real_price:,.0f} ₫", "Thị trường (Real-time)")
        with c2: card("Giá CW Lý thuyết", f"{cw_price_theory:,.0f} ₫", "Intrinsic Value")
        with c3: card("Điểm Hòa Vốn (BEP)", f"{bep:,.0f} ₫", "Break-even Point")

    with tab2:
        st.subheader("Kiểm tra thông số đầu vào (Debug)")
        st.markdown(f"""
        <div class="debug-box">
            <b>Đang tính toán với thông số:</b><br>
            - Giá thực hiện: <b>{val_exercise:,.0f} VND</b><br>
            - Tỷ lệ chuyển đổi: <b>{val_ratio} : 1</b><br>
            - Công thức: Max((Giá Mục Tiêu - {val_exercise:,.0f}) / {val_ratio}, 0)
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        
        target_price = st.slider(
            f"Giá mục tiêu {val_underlying_code}", 
            min_value=int(anchor_price * 0.5), 
            max_value=int(anchor_price * 2.0), 
            key="sim_target_price",
            step=100
        )
        
        sim_cw_price = engine.calc_intrinsic_value(target_price, val_exercise, val_ratio)
        sim_pnl = (sim_cw_price - cost_price) * qty
        sim_pnl_pct = (sim_pnl / (cost_price * qty) * 100) if cost_price > 0 else 0
        
        c1, c2 = st.columns(2)
        with c1: st.info(f"Giá CW Lý thuyết: **{sim_cw_price:,.0f} VND**")
        with c2:
            color = "#2E7D32" if sim_pnl >= 0 else "#C62828" # Xanh đậm / Đỏ đậm
            st.markdown(f"Lãi/Lỗ dự kiến: :**<span style='color:{color}'>{sim_pnl:,.0f} VND ({sim_pnl_pct:.2f}%)</span>**", unsafe_allow_html=True)

    with tab3:
        st.subheader("Phân tích Điểm Hòa Vốn (Break-even Analysis)")
        
        # --- HƯỚNG DẪN SỬ DỤNG (NEW) ---
        st.markdown("""
        <div class="guide-box">
            <b>💡 Hướng dẫn đọc biểu đồ:</b>
            <ul style="margin-top:5px; margin-bottom:0;">
                <li><b>Đường màu xanh (P/L Profile):</b> Biểu diễn Lãi/Lỗ của bạn tương ứng với giá Cổ phiếu cơ sở.</li>
                <li><b>Đường đứt đoạn màu cam (BEP):</b> Là mức giá Cổ phiếu cơ sở cần đạt để bạn hòa vốn.</li>
                <li><b>Điểm màu đỏ:</b> Vị trí giá hiện tại. Nếu điểm đỏ nằm bên phải đường cam -> Bạn đang Lãi.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # -------------------------------

        x_values = np.linspace(current_real_price * 0.8, current_real_price * 1.2, 50)
        y_pnl = []
        for x in x_values:
            cw_val = engine.calc_intrinsic_value(x, val_exercise, val_ratio)
            y_pnl.append((cw_val - cost_price) * qty)
            
        fig = go.Figure()
        # Đổi màu line chart sang màu Cam/Vàng thương hiệu
        fig.add_trace(go.Scatter(x=x_values, y=y_pnl, mode='lines', name='Lợi nhuận dự kiến', line=dict(color='#FF8F00', width=3)))
        fig.add_vline(x=bep, line_width=2, line_dash="dash", line_color="#5D4037", annotation_text="Hòa Vốn")
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.add_trace(go.Scatter(x=[current_real_price], y=[(engine.calc_intrinsic_value(current_real_price, val_exercise, val_ratio) - cost_price) * qty], mode='markers', name='Hiện tại', marker=dict(color='#D32F2F', size=12)))
        
        fig.update_layout(
            title=f"Biểu đồ P/L: {selected_cw} vs {val_underlying_code}",
            xaxis_title=f"Giá Cổ phiếu {val_underlying_code}",
            yaxis_title="Lãi/Lỗ (VND)",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
