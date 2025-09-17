import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, datetime

CSV_FILE = "alert_log.csv"

st.set_page_config(page_title="Stock Predictions", layout="wide")

st.title("📈 Stock Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_FILE)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "prediction", "current_price", "predicted_price", "news"])

df = load_data()

if df.empty:
    st.warning("No prediction data found yet.")
else:
    # --- Calendar Heatmap ---
    st.subheader("📅 Prediction Calendar")
    date_counts = df["date"].value_counts().reset_index()
    date_counts.columns = ["date", "count"]

    # Plotly calendar-like heatmap
    fig = px.density_heatmap(
        date_counts,
        x="date",
        y=["Predictions"] * len(date_counts),  # Fake y to make single-row calendar
        z="count",
        color_continuous_scale=["red", "green"],  # Red = no data, Green = more data
        nbinsx=len(date_counts),
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="",
        yaxis=dict(showticklabels=False),
        height=150,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Date Picker ---
    st.subheader("🔎 Filter by Date")
    min_date = min(df["date"])
    max_date = max(df["date"])
    selected_date = st.date_input("Select a date", value=max_date, min_value=min_date, max_value=max_date)

    # Filter data by selected date
    filtered = df[df["date"] == selected_date]

    if filtered.empty:
        st.error(f"No predictions found for {selected_date}")
    else:
        st.success(f"Showing predictions for {selected_date}")
        st.dataframe(
            filtered[["date", "ticker", "prediction", "current_price", "predicted_price", "news"]],
            use_container_width=True
        )
