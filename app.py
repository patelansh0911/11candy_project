import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from utils.data_loader import load_data, compute_lead_times
from utils.factory_info import PRODUCT_FACTORY, FACTORY_COORDS, FACTORY_COLORS
from models.predictor import FactoryPredictor

st.set_page_config(
    page_title="11 Candy — Factory Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #212529; }
    .rec-card {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: .75rem 1rem;
        margin-bottom: .5rem;
    }
    .warn-card {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: .75rem 1rem;
        margin-bottom: .5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_data("data/Nassau_Candy_Distributor.csv")
    df = compute_lead_times(df)
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    df["Margin"] = (df["Gross Profit"] / df["Sales"] * 100).round(1)
    return df

@st.cache_resource
def get_predictor(df):
    p = FactoryPredictor()
    p.fit(df)
    return p

df = get_data()
predictor = get_predictor(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1a1a2e/ffffff?text=🍬+11+Candy", use_container_width=True)
    st.markdown("## Filters")
    sel_region = st.multiselect("Region", sorted(df["Region"].unique()), default=list(df["Region"].unique()))
    sel_ship = st.multiselect("Ship Mode", sorted(df["Ship Mode"].unique()), default=list(df["Ship Mode"].unique()))
    sel_division = st.multiselect("Division", sorted(df["Division"].unique()), default=list(df["Division"].unique()))
    opt_priority = st.slider("Optimization Priority", 0, 100, 50, help="0 = Speed first | 100 = Profit first")
    st.caption("Speed ←——→ Profit")

fdf = df[df["Region"].isin(sel_region) & df["Ship Mode"].isin(sel_ship) & df["Division"].isin(sel_division)]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🍬 11 Candy — Factory Reallocation & Shipping Optimizer")
st.caption(f"Analyzing {len(fdf):,} orders across {fdf['Product Name'].nunique()} products and 5 factories")
st.divider()

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Sales", f"${fdf['Sales'].sum():,.0f}")
with k2:
    st.metric("Gross Profit", f"${fdf['Gross Profit'].sum():,.0f}")
with k3:
    margin = fdf['Gross Profit'].sum() / fdf['Sales'].sum() * 100
    st.metric("Avg Margin", f"{margin:.1f}%")
with k4:
    st.metric("Avg Lead Time", f"{fdf['Lead_Time'].mean():.0f} days")
with k5:
    st.metric("Total Orders", f"{len(fdf):,}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Factory Simulator",
    "🔀 Scenario Analysis",
    "🏆 Recommendations",
    "📊 Analytics"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Factory Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Factory Optimization Simulator")
    st.caption("Select a product and see its predicted performance if assigned to each factory.")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        sel_prod = st.selectbox("Product", sorted(df["Product Name"].unique()))
        sel_region_sim = st.selectbox("Target Region", ["All"] + sorted(df["Region"].unique()))
        sel_ship_sim = st.selectbox("Ship Mode", ["All"] + sorted(df["Ship Mode"].unique()))

    prod_data = df[df["Product Name"] == sel_prod]
    current_factory = PRODUCT_FACTORY.get(sel_prod, "Unknown")

    factories = list(FACTORY_COORDS.keys())
    rows = []
    for f in factories:
        pred_lt = predictor.predict_lead_time(sel_prod, f, sel_region_sim, sel_ship_sim, opt_priority)
        pred_profit_delta = predictor.estimate_profit_impact(sel_prod, f)
        is_current = f == current_factory
        rows.append({
            "Factory": f,
            "Predicted Lead Time (days)": pred_lt,
            "Profit Impact (%)": pred_profit_delta,
            "Current": "★ Current" if is_current else ""
        })

    sim_df = pd.DataFrame(rows).sort_values("Predicted Lead Time (days)")

    with col_r:
        fig = px.bar(
            sim_df,
            x="Factory",
            y="Predicted Lead Time (days)",
            color="Factory",
            color_discrete_map=FACTORY_COLORS,
            text="Predicted Lead Time (days)",
            title=f"Predicted Lead Time by Factory — {sel_prod}"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=350, xaxis_title="", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Factory Comparison Table")
    display_df = sim_df.copy()
    display_df["Profit Impact (%)"] = display_df["Profit Impact (%)"].apply(lambda x: f"{x:+.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    best = sim_df.iloc[0]
    if best["Factory"] != current_factory:
        st.markdown(f"""
        <div class="rec-card">
        ✅ <strong>Recommendation:</strong> Reassigning <em>{sel_prod}</em> from
        <strong>{current_factory}</strong> → <strong>{best['Factory']}</strong>
        could reduce lead time by ~<strong>{sim_df[sim_df['Factory']==current_factory]['Predicted Lead Time (days)'].values[0] - best['Predicted Lead Time (days)']:.0f} days</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ **{sel_prod}** is already optimally assigned to **{current_factory}**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Scenario Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("What-If Scenario Analysis")
    st.caption("Compare current assignment vs. a simulated reassignment side-by-side.")

    sc1, sc2 = st.columns(2)
    with sc1:
        scen_prod = st.selectbox("Product to Reassign", sorted(df["Product Name"].unique()), key="scen_prod")
    with sc2:
        scen_factory = st.selectbox("Reassign To Factory", [f for f in factories if f != PRODUCT_FACTORY.get(scen_prod)], key="scen_fac")

    cur_fac = PRODUCT_FACTORY.get(scen_prod, "Unknown")
    cur_lt = predictor.predict_lead_time(scen_prod, cur_fac, "All", "All", 50)
    new_lt = predictor.predict_lead_time(scen_prod, scen_factory, "All", "All", 50)
    cur_profit = df[df["Product Name"]==scen_prod]["Gross Profit"].sum()
    new_profit = cur_profit * (1 + predictor.estimate_profit_impact(scen_prod, scen_factory)/100)
    lt_delta = ((cur_lt - new_lt) / cur_lt * 100)
    profit_delta = ((new_profit - cur_profit) / cur_profit * 100)

    cs1, cs2 = st.columns(2)
    with cs1:
        st.markdown("##### 📍 Current Assignment")
        st.metric("Factory", cur_fac)
        st.metric("Avg Lead Time", f"{cur_lt:.0f} days")
        st.metric("Gross Profit", f"${cur_profit:,.0f}")
    with cs2:
        st.markdown("##### 🔀 Simulated Assignment")
        st.metric("Factory", scen_factory)
        st.metric("Avg Lead Time", f"{new_lt:.0f} days", delta=f"{new_lt-cur_lt:.0f}d", delta_color="inverse")
        st.metric("Gross Profit", f"${new_profit:,.2f}", delta=f"{profit_delta:+.1f}%")

    st.divider()
    col_gauge1, col_gauge2 = st.columns(2)
    with col_gauge1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_lt,
            delta={"reference": cur_lt, "valueformat": ".0f"},
            title={"text": "Lead Time (days)"},
            gauge={"axis": {"range": [900, 1700]},
                   "bar": {"color": "#22c55e" if new_lt < cur_lt else "#ef4444"},
                   "threshold": {"line": {"color": "gray", "width": 2}, "value": cur_lt}}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    with col_gauge2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(new_profit, 2),
            delta={"reference": cur_profit, "valueformat": ".2f"},
            title={"text": "Gross Profit ($)"},
            gauge={"axis": {"range": [0, max(cur_profit, new_profit)*1.2]},
                   "bar": {"color": "#22c55e" if new_profit >= cur_profit else "#ef4444"}}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    if lt_delta > 0:
        st.markdown(f'<div class="rec-card">✅ <strong>Gain:</strong> Lead time improves by <strong>{lt_delta:.1f}%</strong>. Profit impact: <strong>{profit_delta:+.1f}%</strong>.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-card">⚠️ <strong>Caution:</strong> Lead time increases by <strong>{abs(lt_delta):.1f}%</strong>. Profit impact: <strong>{profit_delta:+.1f}%</strong>.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Recommendations
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Ranked Factory Reassignment Recommendations")
    st.caption("Products sorted by potential lead-time improvement with profit risk assessment.")

    recs = predictor.generate_recommendations(df, opt_priority)
    rec_df = pd.DataFrame(recs)

    for _, row in rec_df.iterrows():
        risk_icon = "🟢" if row["Profit Risk"] == "Low" else "🟡" if row["Profit Risk"] == "Medium" else "🔴"
        card_class = "rec-card" if row["Profit Risk"] == "Low" else "warn-card"
        st.markdown(f"""
        <div class="{card_class}">
        <strong>{row['Product']}</strong><br>
        {row['Current Factory']} → <strong>{row['Recommended Factory']}</strong>&nbsp;&nbsp;
        Lead time: <strong>-{row['LT Gain %']:.1f}%</strong> &nbsp;|&nbsp;
        Profit impact: <strong>{row['Profit Impact %']:+.1f}%</strong> &nbsp;|&nbsp;
        Risk: {risk_icon} {row['Profit Risk']} &nbsp;|&nbsp;
        Confidence: <strong>{row['Confidence %']:.0f}%</strong>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Full Recommendation Table")
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    fig = px.scatter(
        rec_df,
        x="LT Gain %",
        y="Profit Impact %",
        size="Confidence %",
        color="Profit Risk",
        hover_name="Product",
        color_discrete_map={"Low": "#22c55e", "Medium": "#f97316", "High": "#ef4444"},
        title="Lead Time Gain vs. Profit Impact"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Data Analytics")

    a1, a2 = st.columns(2)
    with a1:
        div_sales = fdf.groupby("Division")["Sales"].sum().reset_index()
        fig = px.pie(div_sales, values="Sales", names="Division", title="Sales by Division",
                     color_discrete_sequence=["#3266ad","#1D9E75","#D85A30"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        reg_sales = fdf.groupby("Region")["Sales"].sum().reset_index()
        fig = px.bar(reg_sales, x="Region", y="Sales", title="Sales by Region",
                     color="Region", color_discrete_sequence=["#378ADD","#1D9E75","#D85A30","#7F77DD"])
        fig.update_layout(height=300, showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    lt_prod = fdf.groupby("Product Name")["Lead_Time"].mean().reset_index().sort_values("Lead_Time", ascending=False)
    fig = px.bar(lt_prod, x="Lead_Time", y="Product Name", orientation="h",
                 title="Average Lead Time by Product (days)",
                 color="Lead_Time", color_continuous_scale="RdYlGn_r")
    fig.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    margin_prod = fdf.groupby("Product Name").apply(
        lambda x: x["Gross Profit"].sum() / x["Sales"].sum() * 100
    ).reset_index(name="Margin %").sort_values("Margin %", ascending=False)
    fig = px.bar(margin_prod, x="Product Name", y="Margin %",
                 title="Gross Profit Margin by Product (%)",
                 color="Margin %", color_continuous_scale="RdYlGn")
    fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                      xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    ship_lt = fdf.groupby("Ship Mode")["Lead_Time"].mean().reset_index()
    fig = px.bar(ship_lt, x="Ship Mode", y="Lead_Time", title="Avg Lead Time by Ship Mode",
                 color="Ship Mode", color_discrete_sequence=["#378ADD","#1D9E75","#D85A30","#7F77DD"])
    fig.update_layout(height=300, showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
