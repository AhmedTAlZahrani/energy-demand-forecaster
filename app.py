import streamlit as st
import plotly.express as px

from forecaster.data_loader import load_energy_data, split_by_date
from forecaster.feature_engineering import DemandFeatures

st.set_page_config(page_title="Energy Demand Forecaster", page_icon="⚡", layout="wide")
st.title("Energy Demand Forecaster")

# ── Sidebar ───────────────────────────────────
st.sidebar.title("Settings")
data_path = st.sidebar.text_input("Data path", "data/energy.csv")
test_days = st.sidebar.slider("Test set (days)", 7, 90, 30)
seq_length = st.sidebar.slider("LSTM sequence length", 24, 336, 168, 24)

# ── Load Data ─────────────────────────────────
try:
    df = load_energy_data(data_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info("Place your energy consumption CSV in the data/ directory.")
    st.stop()

train, test = split_by_date(df, test_days=test_days)

# ── Tabs ──────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["Data Explorer", "Seasonal Patterns", "Forecasts", "Comparison"])

# ── Data Explorer ─────────────────────────────
with t1:
    st.subheader("Raw Consumption Data")
    fig = px.line(df, x="timestamp", y="consumption", title="Energy Consumption Over Time")
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(df["consumption"].describe().round(2).to_frame().T, use_container_width=True)

# ── Seasonal Patterns ────────────────────────
with t2:
    st.subheader("Hourly Pattern")
    hourly = df.groupby("hour")["consumption"].mean().reset_index()
    fig_h = px.bar(hourly, x="hour", y="consumption", title="Average Consumption by Hour")
    fig_h.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("Day of Week Pattern")
    daily = df.groupby("day_of_week")["consumption"].mean().reset_index()
    daily["day_name"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_d = px.bar(daily, x="day_name", y="consumption", title="Average Consumption by Day")
    fig_d.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("Monthly Pattern")
    monthly = df.groupby("month")["consumption"].mean().reset_index()
    fig_m = px.bar(monthly, x="month", y="consumption", title="Average Consumption by Month")
    fig_m.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_m, use_container_width=True)

# ── Forecasts ─────────────────────────────────
with t3:
    st.subheader("Train / Test Split")
    fig_split = px.line(title="Train (blue) vs Test (red) Split")
    fig_split.add_scatter(x=train["timestamp"], y=train["consumption"], name="Train")
    fig_split.add_scatter(x=test["timestamp"], y=test["consumption"], name="Test")
    fig_split.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_split, use_container_width=True)

    st.info("Use the comparison tab to run models and see forecasts vs actuals.")

# ── Comparison ────────────────────────────────
with t4:
    st.subheader("Model Comparison")
    st.markdown(
        "To compare models, run the comparison script:\n"
        "```python\n"
        "from forecaster.comparison import ModelComparison\n"
        "from forecaster.data_loader import load_energy_data, split_by_date\n\n"
        "df = load_energy_data('data/energy.csv')\n"
        "train, test = split_by_date(df, test_days=30)\n"
        "comp = ModelComparison()\n"
        "results = comp.run_all_models(train, test)\n"
        "print(results)\n"
        "```"
    )
