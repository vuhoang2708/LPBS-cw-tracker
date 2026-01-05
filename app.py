import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER
# ==========================================
class DataManager:
    @staticmethod
    def get_master_data():
        return pd.DataFrame([
            {"Mã CW": "CHPG2301", "Mã CS": "HPG", "Tỷ lệ CĐ": 2, "Giá thực hiện": 20000, "Ngày đáo hạn": "2026-06-01", "Trạng thái": "Listed"},
            {"Mã CW": "CMWG2305", "Mã CS": "MWG", "Tỷ lệ CĐ": 5, "Giá thực hiện": 45000, "Ngày đáo hạn": "2026-12-31", "Trạng thái": "Pre-listing"},
            {"Mã CW": "CVHM2302", "Mã CS": "VHM", "Tỷ lệ CĐ": 4, "Giá thực hiện": 40000, "Ngày đáo hạn": "2026-08-15", "Trạng thái": "Listed"},
        ])

    @staticmethod
    def get_realtime_price(symbol):
        # Giả lập giá có biến động nhẹ
        base_prices = {
            "HPG": 28500, "MWG": 48200, "VHM": 41800,
            "CHPG2301": 4300, "CVHM2302": 550
        }
        noise = np.random.uniform(0.99, 1.01)
        return base_prices.get(symbol, 0) * noise

# ==========================================
# 3. LOGIC LAYER
# ==========================================
class FinancialEngine:
    @staticmethod
    def calc_intrinsic_value(price_underlying, price_exercise, ratio):
        return max((price_underlying - price_exercise) / ratio, 0)

    @staticmethod
    def calc_bep(price_exercise, price_cost, ratio):
        return price_exercise + (price_cost * ratio)

# ==========================================
# 4. UI PRESENTATION
# ==========================================
def main():
    st.title("📈 LPBank Invest - CW Tracker & Simulator")
    st.caption("System Architect: AI Guardian | Version: 3.1 (Stable Slider)")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🛠️ Cấu hình Danh mục")
        master_df = DataManager.get_master_data()
        selected_cw = st.selectbox("Chọn Mã CW", master_df["Mã CW"].unique())
        cw_info = master_df[master_df["Mã CW"] == selected_cw].iloc[0]
        
        qty = st.number_input("Số lượng sở hữu", value=1000, step=100)
        cost_price = st.number_input("Giá vốn bình quân (VND)", value=1000, step=50)
        
        st.info(f"ℹ️ **Thông tin {selected_cw}**\n\n- Mã CS: {cw_info['Mã CS']}\n- Giá TH: {cw_info['Giá thực hiện']:,}\n- Tỷ lệ: {cw_info['Tỷ lệ CĐ']}:1")

    # --- DATA PROCESSING ---
    price_underlying = DataManager.get_realtime_price(cw_info["Mã CS"])
    
    # --- FIX: SESSION STATE FOR SLIDER ---
    # Logic: Chỉ reset thanh trượt về giá thị trường khi người dùng đổi Mã CW
    if "last_cw" not in st.session_state or st.session_state["last_cw"] != selected_cw:
        st.session_state["last_cw"] = selected_cw
        st.session_state["sim_slider_val"] = int(price_underlying) # Reset slider về giá hiện tại

    # --- CORE CALCULATION ---
    engine = FinancialEngine()
    bep = engine.calc_bep(cw_info["Giá thực hiện"], cost_price, cw_info["Tỷ lệ CĐ"])
    
    if cw_info['Trạng thái'] == 'Pre-listing':
        current_cw_price = engine.calc_intrinsic_value(price_underlying, cw_info["Giá thực hiện"], cw_info["Tỷ lệ CĐ"])
        note = "⚠️ Giá trị nội tại (Pre-listing)"
    else:
        market_cw_price = DataManager.get_realtime_price(selected_cw)
        current_cw_price = market_cw_price if market_cw_price > 0 else engine.calc_intrinsic_value(price_underlying, cw_info["Giá thực hiện"], cw_info["Tỷ lệ CĐ"])
        note = "✅ Giá thị trường (Listed)"

    pnl = (current_cw_price - cost_price) * qty
    pnl_pct = (pnl / (cost_price * qty) * 100) if cost_price > 0 else 0

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard P/L", "🎲 Simulator (Giả lập)", "📉 Biểu đồ BEP"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Giá {cw_info['Mã CS']}", f"{price_underlying:,.0f} ₫")
        col2.metric("Giá CW Hiện tại", f"{current_cw_price:,.0f} ₫", delta=note, delta_color="off")
        col3.metric("Điểm Hòa Vốn (BEP)", f"{bep:,.0f} ₫")
        col4.metric("Lãi/Lỗ (P/L)", f"{pnl:,.0f} ₫", f"{pnl_pct:.2f}%")

        if price_underlying < bep:
            diff = ((bep - price_underlying) / price_underlying) * 100
            st.warning(f"📉 Cần **{cw_info['Mã CS']}** tăng thêm **{diff:.2f}%** (lên mức {bep:,.0f}) để về bờ.")
        else:
            st.success(f"🎉 Đã về bờ! Bạn đang lãi trên mỗi biến động của {cw_info['Mã CS']}.")

    with tab2:
        st.subheader("Giả lập Lợi nhuận theo Kỳ vọng")
        st.write("Kéo thanh trượt để thay đổi giá Cổ phiếu cơ sở tương lai:")
        
        # SLIDER ĐÃ FIX: Dùng key="sim_slider_val" để liên kết với Session State
        target_price = st.slider(
            f"Giá mục tiêu {cw_info['Mã CS']}", 
            min_value=int(price_underlying * 0.8), 
            max_value=int(price_underlying * 1.5), 
            key="sim_slider_val", # Quan trọng: Key này giúp Streamlit nhớ vị trí
            step=100
        )
        
        sim_cw_price = engine.calc_intrinsic_value(target_price, cw_info["Giá thực hiện"], cw_info["Tỷ lệ CĐ"])
        sim_pnl = (sim_cw_price - cost_price) * qty
        sim_pnl_pct = (sim_pnl / (cost_price * qty) * 100) if cost_price > 0 else 0
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"Giá CW Lý thuyết dự kiến: **{sim_cw_price:,.0f} VND**")
        with c2:
            color = "green" if sim_pnl >= 0 else "red"
            st.markdown(f"Lãi/Lỗ dự kiến: :**{color}[{sim_pnl:,.0f} VND ({sim_pnl_pct:.2f}%)]**")

    with tab3:
        st.subheader("Phân tích Điểm Hòa Vốn Trực quan")
        x_values = np.linspace(price_underlying * 0.8, price_underlying * 1.2, 50)
        y_pnl = []
        for x in x_values:
            cw_val = engine.calc_intrinsic_value(x, cw_info["Giá thực hiện"], cw_info["Tỷ lệ CĐ"])
            y_pnl.append((cw_val - cost_price) * qty)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=y_pnl, mode='lines', name='P/L Profile', line=dict(color='blue', width=3)))
        fig.add_vline(x=bep, line_width=2, line_dash="dash", line_color="orange", annotation_text="Điểm Hòa Vốn")
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.add_trace(go.Scatter(x=[price_underlying], y=[pnl], mode='markers', name='Hiện tại', marker=dict(color='red', size=12)))
        
        fig.update_layout(
            title=f"Biểu đồ P/L của {selected_cw} theo giá {cw_info['Mã CS']}",
            xaxis_title=f"Giá Cổ phiếu {cw_info['Mã CS']}",
            yaxis_title="Lãi/Lỗ (VND)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
