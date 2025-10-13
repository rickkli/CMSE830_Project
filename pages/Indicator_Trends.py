import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# load cleaned datasets
@st.cache_data
def load_data():
    disaster_df = pd.read_csv('datasets/disaster_df.csv')
    indicators_df_imputed = pd.read_csv('datasets/indicators_df_imputed.csv')
    return disaster_df, indicators_df_imputed

disaster_df, indicators_df_imputed = load_data()

st.header("Country Economic Indicator Trends")
st.markdown("Select a country and indicator below to view trends over time.")

# ---- Dropdowns ----
countries = sorted(indicators_df_imputed["country_name"].unique())
indicators = [col for col in indicators_df_imputed.columns if col not in ["country_name", "year"]]

col1, col2 = st.columns(2)
with col1:
    selected_country = st.selectbox("Select a Country", countries, index=countries.index("United States") if "United States" in countries else 0)
with col2:
    selected_indicator = st.selectbox("Select an Indicator", indicators)

# ---- Filter data ----
country_data = indicators_df_imputed[indicators_df_imputed["country_name"] == selected_country]

# ---- Plotly figure ----
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=country_data["year"],
        y=country_data[selected_indicator],
        mode="lines+markers",
        name=selected_indicator,
        hovertemplate=f"{selected_indicator}: %{{y:.2f}}<extra></extra>",
    )
)

fig.update_layout(
    title=f"{selected_indicator} for {selected_country}",
    xaxis_title="Year",
    yaxis_title=selected_indicator,
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

