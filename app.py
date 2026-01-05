import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# ==========================================
# 1. CONFIG & SYSTEM SETTINGS
# ==========================================
st.set_page_config(page_title="LPBank CW Tracker", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;}
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; border-bottom: 2px solid #4CAF50; }
    .debug-box { background-color: #e3f2fd; color: #0d47a1; padding: 10px; border-radius: 5px; font-size: 0.9em; margin-bottom: 10px; border: 1px solid #90caf9; }
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
        if ':' in s: s = s.split(':')[0] # Xử lý 5:1 -> lấy 5
        s = re.sub(r'[^\d.]', '', s)     # Xóa ký tự lạ
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
    st.title("📈 LPBank Invest - CW Tracker & Simulator")
    
    # --- HIỂN THỊ ĐÚNG YÊU CẦU ---
    st.caption("System Architect: AI Guardian | Build: 15:30 05/01/2026")

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
                
                # Smart Mapping
                found_exercise = DataManager.smart_find_column(master_df, ['thực hiện', 'exercise', 'strike', 'giá th'])
                found_ratio = DataManager.smart_find_column(master_df, ['tỷ lệ', 'ratio', 'conversion', 'cđ'])
                found_code = DataManager.smart_find_column(master_df, ['mã cw', 'cw code', 'symbol'])
                found_underlying = DataManager.smart_find_column(master_df, ['mã cs', 'underlying', 'cơ sở'])

                if found_exercise: col_exercise = found_exercise
                if found_ratio: col_ratio = found_ratio
                if found_code: col_code = found_code
                if found_underlying: col_underlying = found_underlying
                
                # Clean Data
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
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎲 Simulator (Giả lập)", "📉 Biểu đồ BEP"])

    with tab1:
        bep = engine.calc_bep(val_exercise, cost_price, val_ratio)
        cw_price_theory = engine.calc_intrinsic_value(current_real_price, val_exercise, val_ratio)
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Giá {val_underlying_code}", f"{current_real_price:,.0f}")
        c2.metric("Giá CW Lý thuyết", f"{cw_price_theory:,.0f}")
        c3.metric("Điểm Hòa Vốn", f"{bep:,.0f}")

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
        with c1:
            st.info(f"Giá CW Lý thuyết: **{sim_cw_price:,.0f} VND**")
        with c2:
            color = "green" if sim_pnl >= 0 else "red"
            st.markdown(f"Lãi/Lỗ dự kiến: :**{color}[{sim_pnl:,.0f} VND ({sim_pnl_pct:.2f}%)]**")

    with tab3:
        st.subheader("Phân tích Điểm Hòa Vốn Trực quan")
        x_values = np.linspace(current_real_price * 0.8, current_real_price * 1.2, 50)
        y_pnl = []
        for x in x_values:
            cw_val = engine.calc_intrinsic_value(x, val_exercise, val_ratio)
            y_pnl.append((cw_val - cost_price) * qty)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=y_pnl, mode='lines', name='P/L Profile', line=dict(color='blue', width=3)))
        fig.add_vline(x=bep, line_width=2, line_dash="dash", line_color="orange", annotation_text="Điểm Hòa Vốn")
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.add_trace(go.Scatter(x=[current_real_price], y=[(engine.calc_intrinsic_value(current_real_price, val_exercise, val_ratio) - cost_price) * qty], mode='markers', name='Hiện tại', marker=dict(color='red', size=12)))
        
        fig.update_layout(
            title=f"Biểu đồ P/L của {selected_cw} theo giá {val_underlying_code}",
            xaxis_title=f"Giá Cổ phiếu {val_underlying_code}",
            yaxis_title="Lãi/Lỗ (VND)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
