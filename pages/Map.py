import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="Global Disaster Map", layout="wide")

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
# PAGE TITLE & INTRODUCTION
# ------------------------------
st.header("Global Disaster Map Explorer")
st.markdown(
"""
Visualize the **geographic distribution** and **human impact** of natural disasters around the world through the:

- **Heatmap:** Highlights physical locations of natural disasters with higher disaster density and affected populations for a selected year.
- **Choropleth Map:** Summarizes total affected populations by country for a selected year.
"""
)

st.divider()

# ------------------------------
# YEAR SELECTION
# ------------------------------
years = sorted(disaster_df["year"].unique())
selected_year = st.selectbox("Select a Year", years, index=len(years) - 1)

year_data = disaster_df[disaster_df["year"] == selected_year].copy()
year_data["affected_population"] = pd.to_numeric(year_data["affected_population"], errors="coerce").fillna(0)

# Scale radius for scatter points
MAX_RADIUS = 50000
year_data["radius"] = (
    year_data["affected_population"] / year_data["affected_population"].max() * MAX_RADIUS
)
year_data["scatter_color"] = [[220, 220, 220]] * len(year_data)

# Format affected population with commas for hover
year_data["affected_population_str"] = (
    year_data["affected_population"]
    .apply(lambda x: f"{int(x):,}")
)

st.write("")

# ------------------------------
# HEATMAP LAYER
# ------------------------------
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=year_data,
    get_position=["longitude", "latitude"],
    get_weight="affected_population",
    radiusPixels=50,
    intensity=1,
    threshold=0.03,
    opacity=0.35,
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=year_data,
    get_position=["longitude", "latitude"],
    get_radius="radius",
    get_fill_color="scatter_color",
    pickable=True,
    auto_highlight=True,
    opacity=0.6,
)

view_state = pdk.ViewState(longitude=0, latitude=0, zoom=1.4, pitch=0)

r = pdk.Deck(
    layers=[heatmap_layer, scatter_layer],
    initial_view_state=view_state,
    tooltip={
    "html": "<b>Event Type:</b> {event_type}<br/><b>Affected:</b> {affected_population_str}",
    "style": {"color": "white", "font-size": "12px"}
    }
)

# ------------------------------
# DISPLAY HEATMAP
# ------------------------------
st.subheader(f"Disaster Event Heatmap")
st.pydeck_chart(r, use_container_width=True)

st.caption("""
Hover over points on the map to view the **event type** and **affected population** details.
""")

st.divider()

# ------------------------------
# CHOROPLETH MAP
# ------------------------------
country_data = (
    year_data.groupby("country")["affected_population"].sum().reset_index()
)

fig = px.choropleth(
    country_data,
    locations="country",
    locationmode="country names",
    color="affected_population",
    color_continuous_scale="YlOrRd",
    hover_data={"affected_population": ":,f"},
)

fig.update_layout(
    title=dict(
        text=f"Total Affected Population by Country",
        xanchor="left",
        font=dict(size=20),
    ),
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",
        fitbounds="locations",
    ),
    template="plotly_dark",
    height=550,
    margin=dict(l=30, r=30, t=60, b=30),
    coloraxis_colorbar=dict(title="Affected Population"),
)

fig.update_traces(
    hovertemplate="<b>%{location}</b><br>Affected Population: %{z:,.0f}<extra></extra>"
)

st.plotly_chart(fig, use_container_width=True)

st.caption("""
Hover over a country to see the total **affected population** for the year.
""")