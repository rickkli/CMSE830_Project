import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="Economic Indicator Trends", layout="wide")

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    disaster_df = pd.read_csv("datasets/disaster_df.csv")
    indicators_df_imputed = pd.read_csv("datasets/indicators_df_imputed.csv")
    return disaster_df, indicators_df_imputed

disaster_df, indicators_df_imputed = load_data()

# ------------------------------
# PAGE HEADER
# ------------------------------
st.header("Country Economic Indicator Trends")
st.markdown(
"""
Explore how key **economic indicators** have evolved across countries over time.  
Select a **country** and an **indicator** from the dropdowns below to visualize trends.
""" 
)

# ------------------------------
# DROPDOWNS
# ------------------------------
countries = sorted(indicators_df_imputed["country_name"].unique())
indicators = [col for col in indicators_df_imputed.columns if col not in ["country_name", "year"]]

col1, col2 = st.columns(2, gap="large")

with col1:
    selected_country = st.selectbox(
        "Select a Country",
        countries,
        index=countries.index("United States") if "United States" in countries else 0,
    )

with col2:
    selected_indicator = st.selectbox("Select an Indicator", indicators)

st.write("")

# ------------------------------
# FILTER DATA
# ------------------------------
country_data = indicators_df_imputed[indicators_df_imputed["country_name"] == selected_country]

# ------------------------------
# PLOTLY LINE CHART
# ------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=country_data["year"],
        y=country_data[selected_indicator],
        mode="lines+markers",
        name=selected_indicator,
        hovertemplate="%{y:.2f}<extra></extra>",  # clean hover — just the value
        #marker=dict(color="#636EFA"),
        line=dict(width=3),
    )
)

fig.update_layout(
    title=dict(
        text=f"{selected_indicator} for {selected_country}",
        xanchor="left",
        font=dict(size=20)
    ),
    xaxis_title="Year",
    yaxis_title=selected_indicator,
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=80, b=40),
)

# ------------------------------
# DISPLAY CHART
# ------------------------------
st.plotly_chart(fig, use_container_width=True)

st.caption("""
Hover over a datapoint to see the **indicator value** for that year.
""")