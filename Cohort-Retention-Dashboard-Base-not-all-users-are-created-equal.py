# retention_dashboard_10.py
# -------------------------------------------------------------
# Base Blockchain Retention Dashboard – Cohort, KPI & Weekly Views
# -------------------------------------------------------------

from __future__ import annotations
import streamlit as st
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# -------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Base Blockchain Retention Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Inject Inter font via CSS
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# 🎨 Header Section: Logo & Contact (left) and Highlights & Overview (right)
# -------------------------------------------------------------
logo_path = Path(__file__).parent / "Base_Wordmark_Blue.svg"

col_left, col_desc = st.columns([2, 5])

with col_left:
    # Logo (scaled up to fit)
    if logo_path.exists():
        svg = logo_path.read_text()
        st.markdown(
            '<div style="width:200px; margin-bottom:20px;">'
            + svg +
            '</div>',
            unsafe_allow_html=True,
        )

    # Contact box beneath logo
    st.markdown(
    """
    <div style="padding:20px; background-color:#1e1117; border-radius:8px;">
      <h4 style="color:white; margin:0 0 8px;">Contact & Links</h4>
      <ul style="color:white; list-style:none; padding-left:0; margin:0;">
        <li>
          <a href="https://x.com/ARr00gant" target="_blank" style="color:#BBDEFB; text-decoration:none;">
            𝕏 Twitter @ARr00gant
          </a>
        </li>
        <li>
          <a href="mailto:arr00gant.research@gmail.com" style="color:#BBDEFB; text-decoration:none;">
            📧 Email
          </a>
        </li>
        <li>
          <a href="https://dune.com/arr0gant/base-master-dashboard" target="_blank" style="color:#BBDEFB; text-decoration:none;">
            📊 Dune Base Dashboard
          </a>
        </li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)


with col_desc:
    # Main headline
    st.markdown(
        """
        <h2 style="color:white; margin-bottom:16px;">
          Cohort Retention Dashboard Base: not all users are created equal
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # Detailed overview & highlights
    st.markdown(
        """
        <div style="padding:20px; background-color:#1e1117; border-radius:8px;">
          <p style="color:white; margin:0 0 12px;">
            Every protocol, dApp, or business in crypto chases “more users” and higher activity—after all, address counts and engagement often correlate with valuations at coefficients above 0.8 in mature networks. But raw user numbers alone don’t tell the whole story: not all users are created equal. Degenerates, early adopters, and mainstream adopters each exhibit distinct profiles, transaction patterns, and fee-generation behaviors. This dashboard brings those differences to light by grouping users into cohorts based on the month of their first transaction, then tracking their engagement through transaction volume, total value transacted, and both aggregated and disaggregated fee components for the Base layer-2 blockchain.
          </p>
          <p style="color:white; margin:0 0 12px;">
            Use the KPI selector on the right to update the heatmap—showing your chosen metric by cohort month and retention window, and refer to the accompanying bar chart for the underlying counts or aggregated values per cohort. The transaction & activity graphs, cohort retention overview, and daily fees chart are included for additional context.
          </p>
          <h4 style="color:white; margin:16px 0 8px;">Highlights:</h4>
          <ul style="color:white; padding-left:20px; margin:0;">
            <li><strong>March 2024 cohort punches above its weight.</strong> Despite being smaller in size, it has generated far more in total fees—especially in L2 base and priority fees—than much larger cohorts from July 2024 onward.</li>
            <li><strong>Early cohorts drive higher per-user value.</strong> Cohorts formed through March 2024 consistently show higher transaction volumes and fee-generation per user across all retention windows—even when normalized to a daily rate.</li>
            <li><strong>Newer cohorts are catching up in aggregate.</strong> While more recent cohorts boast much larger headcounts and are beginning to close the gap in aggregate metrics—particularly total L2 fees—they still lag on a per-user basis, underscoring the lasting value of your earliest adopters.</li>
          </ul>

          <br><br>
          <p style="color:white; margin:0;">
            A detailed breakdown and explanation of the metrics included in this dashboard, as well as the SQL queries used to retrieve them, can be found in the Dune dashboard linked on the left-hand side. Please note that the cohort retention chart in the second row enforces progressive dependencies, whereas the heatmap does not.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------

# -------------------------------------------------------------

# -------------------------------------------------------------
# 1️⃣ Load & preprocess cohort data
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading cohort data…")
def load_data() -> pd.DataFrame:
    path = Path(__file__).parent
    df = pd.read_csv(path / "Base_dash_clean.csv", parse_dates=["cohort_month"])
    for col in [
        "avg_tx_fee_eth",
        "avg_fee_per_user",
        "daily_fee_eth_per_user",
        "daily_l1_fee_per_user",
        "daily_l2_base_fee_per_user",
        "daily_l2_priority_fee_per_user",
    ]:
        if col in df.columns:
            df[col] *= 1e6
    return df

df = load_data()

# Exclude April 2025 & drop immature cohorts
report_date = pd.to_datetime("2025-03-31")
thresholds = {
    "1-7 days": 7,
    "8-30 days": 30,
    "31-90 days": 90,
    "91-180 days": 180,
    "181-360 days": 360,
}
df = df[~((df.cohort_month.dt.year == 2025) & (df.cohort_month.dt.month == 4))]
df["cohort_end_date"] = df.cohort_month + MonthEnd(0)
df["days_since_end"] = (report_date - df.cohort_end_date).dt.days
df = df[df.days_since_end >= df.retention_category.map(thresholds)]
df.drop(columns=["cohort_end_date", "days_since_end"], inplace=True)

# -------------------------------------------------------------
# 2️⃣ KPI selector & ordering
# -------------------------------------------------------------
kpi_options = {
    "Total Fee (ETH)": "total_fee_eth",
    "Average Tx Fee (μETH)": "avg_tx_fee_eth",
    "Average Fee per User (μETH)": "avg_fee_per_user",
    "Total L1 Fee (ETH)": "total_l1_fee",
    "Total L2 Base Fee (ETH)": "total_l2_base_fee",
    "Total L2 Priority Fee (ETH)": "total_l2_priority_fee",
    "Daily Fee per User (μETH)": "daily_fee_eth_per_user",
    "Daily L1 Fee per User (μETH)": "daily_l1_fee_per_user",
    "Daily L2 Base Fee per User (μETH)": "daily_l2_base_fee_per_user",
    "Daily L2 Priority Fee per User (μETH)": "daily_l2_priority_fee_per_user",
}
retention_order = ["1-7 days", "8-30 days", "31-90 days", "91-180 days", "181-360 days"]
df.retention_category = pd.Categorical(
    df.retention_category, categories=retention_order, ordered=True
)

st.sidebar.header("Choose KPI for Heatmap")
selected_kpi = st.sidebar.selectbox("KPI", list(kpi_options))
kpi_col = kpi_options[selected_kpi]

# -------------------------------------------------------------
# 3️⃣ Heatmap trace
# -------------------------------------------------------------
pivot = (
    df.pivot_table(
        index="cohort_month",
        columns="retention_category",
        values=kpi_col,
        aggfunc="mean",
    )
    .sort_index()
)
pivot.index = pivot.index.strftime("%b %Y")
z = pivot.values
text = np.where(pd.notna(z), np.round(z, 2).astype(str), "")
mask = np.ma.masked_invalid(z)
heatmap = go.Heatmap(
    z=mask,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    text=text,
    texttemplate="%{text}",
    hovertemplate="%{x}<br>%{y}<br><b>%{z:.2f}</b><extra></extra>",
    colorscale="Viridis",
    xgap=2,
    ygap=2,
    hoverongaps=False,
    colorbar=dict(
        title=selected_kpi,
        orientation="h",
        x=0,
        y=-0.15,
        xanchor="left",
        yanchor="top",
        len=0.5,
        thickness=10,
    ),
)

# -------------------------------------------------------------
# 4️⃣ Bar chart traces (heatmap right)
# -------------------------------------------------------------
latest = df.cohort_month.max()
df_f = df[df.cohort_month < latest].copy()
df_f["label"] = df_f.cohort_month.dt.strftime("%b %Y")
order = (
    df_f[["cohort_month", "label"]]
    .drop_duplicates()
    .sort_values("cohort_month")["label"]
    .tolist()
)
color_seq = ["#FFD700", "#FFA500", "#FF6347", "#BA55D3", "#1E90FF"]

if selected_kpi in [
    "Total Fee (ETH)",
    "Daily Fee per User (μETH)",
    "Daily L1 Fee per User (μETH)",
    "Daily L2 Base Fee per User (μETH)",
    "Daily L2 Priority Fee per User (μETH)",
]:
    bar_fig = px.bar(
        df_f,
        y="label",
        x=kpi_col,
        color="retention_category",
        orientation="h",
        category_orders={"retention_category": retention_order, "label": order},
        labels={kpi_col: selected_kpi, "label": "Cohort Month"},
        color_discrete_sequence=color_seq,
    )
elif selected_kpi == "Average Tx Fee (μETH)":
    # plotting transaction counts, not fee
    bar_fig = px.bar(
        df_f,
        y="label",
        x="tx_count",
        color="retention_category",
        orientation="h",
        category_orders={"retention_category": retention_order, "label": order},
        title="Transactions by Cohort & Retention Window",
        labels={"tx_count": "# Transactions", "label": "Cohort Month"},
        color_discrete_sequence=color_seq,
    )

elif selected_kpi == "Average Fee per User (μETH)":
    # plotting unique-user counts
    bar_fig = px.bar(
        df_f,
        y="label",
        x="unique_users",
        color="retention_category",
        orientation="h",
        category_orders={"retention_category": retention_order, "label": order},
        title="Unique Users by Cohort & Retention Window",
        labels={"unique_users": "Unique Users", "label": "Cohort Month"},
        color_discrete_sequence=color_seq,
    )
else:
    fee_map = {
        "Total L1 Fee (ETH)": ("total_l1_fee", "Total L1 Fees by Cohort"),
        "Total L2 Base Fee (ETH)": ("total_l2_base_fee", "Total L2 Base Fees by Cohort"),
        "Total L2 Priority Fee (ETH)": ("total_l2_priority_fee",        "Total L2 Priority Fees by Cohort"),
    }
    col, _ = fee_map[selected_kpi]
    tmp = (
        df_f.groupby("cohort_month", observed=True)
        .agg(**{col: (col, "sum")})
        .reset_index()
    )
    tmp["label"] = tmp.cohort_month.dt.strftime("%b %Y")
    tmp = tmp.sort_values("cohort_month")
    bar_fig = px.bar(
        tmp,
        y="label",
        x=col,
        labels={col: selected_kpi, "label": "Cohort Month"},
        category_orders={"label": tmp["label"].tolist()},
    )
bar_traces = list(bar_fig.data)

# -------------------------------------------------------------
# 5️⃣ Combine Heatmap + Bar with two separate titles
# -------------------------------------------------------------
combined = make_subplots(
    rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.02, column_widths=[0.6, 0.4]
)
combined.add_trace(heatmap, row=1, col=1)
for tr in bar_traces:
    combined.add_trace(tr, row=1, col=2)

combined.update_layout(
    font=dict(family="Inter", color="white"),
    xaxis=dict(title="", side="top", showgrid=False, ticks=""),
    yaxis=dict(
        title="Cohort Month",
        autorange="reversed",
        showgrid=False,
        ticks="",
        categoryorder="array",
        categoryarray=list(pivot.index),
    ),
    xaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    margin=dict(t=140, b=40, l=30, r=30),
    barmode="stack",
    legend=dict(orientation="h", y=-0.2, x=0.6, title=None),
    height=700,
)
# left chart title + subtitle
combined.add_annotation(
    text=selected_kpi,
    xref="paper",
    yref="paper",
    x=0.22,
    y=1.23,
    showarrow=False,
    font=dict(family="Inter", size=20, color="white"),
)
combined.add_annotation(
    text="Retention Period",
    xref="paper",
    yref="paper",
    x=0.22,
    y=1.13,
    showarrow=False,
    font=dict(family="Inter", size=14, color="white"),
)
# right chart title (now uses the PX-built-in title)
combined.add_annotation(
    text=bar_fig.layout.title.text or selected_kpi,
    xref="paper",
    yref="paper",
    x=0.78,
    y=1.23,
    showarrow=False,
    font=dict(family="Inter", size=20, color="white"),
)


# — equal spacing & first row —
st.markdown("<br>", unsafe_allow_html=True)
st.plotly_chart(combined, use_container_width=True, config={"displayModeBar": False})

# -------------------------------------------------------------
# 6️⃣ Cohort Retention Progressive
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading cohort retention data…")
def load_cohort_retention() -> pd.DataFrame:
    path = Path(__file__).parent
    return pd.read_csv(path / "Base cohort retention progressive.csv", parse_dates=["cohort_month"])

df_cr = load_cohort_retention()

def cohort_retention_chart(df: pd.DataFrame) -> go.Figure:
    order_days = ["30d", "90d", "180d", "360d"]
    bar_colors  = ["#FFA500", "#FF6347", "#BA55D3", "#1E90FF"]
    rate_colors = ["#FFFFFF", "#CCCCCC", "#00CED1", "#32CD32"]

    fig = go.Figure()

    # bars: match the heatmap palette
    for d, color in zip(order_days, bar_colors):
        fig.add_bar(
            x=df["cohort_month"],
            y=df[f"retained_users_{d}"],
            name=d,
            marker_color=color
        )

    # lines: high-contrast accents
    for d, color in zip(order_days, rate_colors):
        fig.add_scatter(
            x=df["cohort_month"],
            y=df[f"retention_rate_{d}"],
            mode="markers+lines",
            name=f"{d} rate",
            line=dict(color=color, width=2),
            yaxis="y2",
        )

    # fix x-axis range
    fig.update_xaxes(
        type="date",
        range=[df["cohort_month"].min(), df["cohort_month"].max()],
        autorange=False,
    )

    # layout without built-in title
    fig.update_layout(
        font=dict(family="Inter", color="white"),
        xaxis_title="Cohort Month",
        yaxis_title="Retained Users",
        yaxis2=dict(title="Retention Rate", overlaying="y", side="right", tickformat=".0%"),
        barmode="stack",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2),
        margin=dict(t=80, b=40, l=40, r=40),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        height=400,
    )

    # annotation-style title (normal weight, 20px)
    fig.add_annotation(
        text="Cohort Retention Progressive Dependencies",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(family="Inter", size=20, color="white"),
    )

    return fig



# -------------------------------------------------------------
# 7️⃣ Weekly Transactions & Active Users
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading weekly stats…")
def load_weekly_data() -> pd.DataFrame:
    path = Path(__file__).parent
    df_w = pd.read_csv(path / "Base_weekly_tx_adresses_4wMA.csv", parse_dates=["week"])
    df_w = df_w[df_w["week"] <= pd.to_datetime("2025-03-31")].sort_values("week")
    return df_w.rename(columns={
        "four_week_ma_transactions": "ma_txn_4w",
        "four_week_ma_active_users": "ma_users_4w",
    }).reset_index(drop=True)

weekly_df = load_weekly_data()

def weekly_tx_active_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    # area for active users on secondary y-axis
    fig.add_scatter(
        x=df["week"], y=df["users_count"],
        mode="none", fill="tozeroy", fillcolor="rgba(46,204,113,0.3)",
        name="Active Users", yaxis="y2"
    )
    # bar for transactions
    fig.add_bar(
        x=df["week"], y=df["transactions_count"],
        name="Transactions"
    )
    # 4-week moving average for transactions
    fig.add_scatter(
    x=df["week"], y=df["ma_txn_4w"],
    mode="lines", name="4-week MA Txn",
    line=dict(width=2, color="rgba(255,215,0,1)")   # ← add this
)
    # 4-week moving average for users on secondary y-axis
    fig.add_scatter(
        x=df["week"], y=df["ma_users_4w"],
        mode="lines", line=dict(dash="dot"), name="4-week MA Users",
        yaxis="y2"
    )

    # fix x-axis range
    fig.update_xaxes(
        type="date",
        range=[df["week"].min(), df["week"].max()],
        autorange=False,
    )

    # layout without built-in title
    fig.update_layout(
        font=dict(family="Inter", color="white"),
        xaxis_title="Week",
        yaxis=dict(title="# Transactions"),
        yaxis2=dict(title="# Active Users", overlaying="y", side="right", showgrid=False),
        barmode="stack",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2),
        margin=dict(t=80, b=40, l=40, r=40),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        height=400,
    )

    # annotation-style title (normal weight, 20px)
    fig.add_annotation(
        text="Weekly Transactions & Active Users",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(family="Inter", size=20, color="white"),
    )

    return fig


# — equal spacing & middle row —
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    wk_fig = weekly_tx_active_chart(weekly_df)
    st.plotly_chart(wk_fig, use_container_width=False, width=600, config={"displayModeBar": False})
with col2:
    cr_fig = cohort_retention_chart(df_cr)
    st.plotly_chart(cr_fig, use_container_width=False, width=600, config={"displayModeBar": False})

# — equal spacing before the last row —
st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 8️⃣ Daily Fees & Cumulative USD
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading daily fees USD data…")
def load_fee_usd_data() -> pd.DataFrame:
    path = Path(__file__).parent
    df_fee = pd.read_csv(path / "Base fees usd.csv", parse_dates=["day"])
    return df_fee.sort_values("day")

df_fee_usd = load_fee_usd_data()

def daily_fees_cum_usd_chart(df: pd.DataFrame) -> go.Figure:
    # prepare bases so L2 bars never fall below zero
    l1 = df["l1_fee_usd"]
    base_for_l2 = l1.clip(lower=0)
    base_for_priority = base_for_l2 + df["base_l2_fee_usd"]

    fig = go.Figure()

    # 1️⃣ L1 Fee from zero (can go negative)
    fig.add_bar(
        x=df["day"], y=l1,
        name="L1 Fee (USD)",
        marker_color="#FF0000"
    )
    # 2️⃣ Base L2 Fee stacked on top of non-neg L1
    fig.add_bar(
        x=df["day"], y=df["base_l2_fee_usd"],
        name="Base L2 Fee (USD)",
        marker_color="#808080",
        base=base_for_l2
    )
    # 3️⃣ Priority L2 Fee stacked on top of (non-neg L1 + Base L2)
    fig.add_bar(
        x=df["day"], y=df["priority_fee_l2_usd"],
        name="Priority L2 Fee (USD)",
        marker_color="#0000FF",
        base=base_for_priority
    )
    # 4️⃣ cumulative L2 priority fee line
    fig.add_scatter(
        x=df["day"], y=df["cum_priority_fee_l2_usd"],
        mode="lines", name="Cumulative Priority L2 Fee (USD)",
        line=dict(width=2, color="#FF1493"),
        yaxis="y2"
    )

    # time axis
    fig.update_xaxes(
        type="date",
        range=[df["day"].min(), df["day"].max()],
        autorange=False,
    )

    # overlay mode so our manual bases are respected
    fig.update_layout(
        font=dict(family="Inter", color="white"),
        xaxis_title="Date",
        yaxis=dict(title="Daily Fee (USD)"),
        yaxis2=dict(
            title="Cumulative L2 Priority Fee (USD)",
            overlaying="y", side="right", showgrid=False
        ),
        barmode="overlay",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25),
        margin=dict(t=80, b=40, l=40, r=40),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        height=400,
    )
    fig.update_traces(selector=dict(type="bar"), opacity=0.6)

    # title annotation
    fig.add_annotation(
        text="Daily Fees and Cumulative L2 Priority Fee (USD)",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(family="Inter", size=20, color="white"),
    )

    return fig




# — bottom row full width —
st.markdown("<br>", unsafe_allow_html=True)
st.plotly_chart(
    daily_fees_cum_usd_chart(df_fee_usd),
    use_container_width=True,
    config={"displayModeBar": False},
)
