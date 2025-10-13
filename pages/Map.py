import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

# --------------------
# Load data
# --------------------
@st.cache_data
def load_data():
    disaster_df = pd.read_csv('datasets/disaster_df.csv')
    indicators_df_imputed = pd.read_csv('datasets/indicators_df_imputed.csv')
    return disaster_df, indicators_df_imputed

disaster_df, indicators_df_imputed = load_data()

# --------------------
# Page title and description
# --------------------
st.title("Global Disaster Dashboard Map")
st.markdown(
    """
    Explore disaster events across the globe. 
    - **Heatmap:** Shows event density and affected population for the selected year.
    - **Choropleth:** Aggregate affected population by country for the same year.
    """
)
st.markdown("---")

# --------------------
# Year selection
# --------------------
years = sorted(disaster_df['year'].unique())
selected_year = st.selectbox("Select Year", years)
year_data = disaster_df[disaster_df['year'] == selected_year].copy()

# --------------------
# Prepare data
# --------------------
year_data = disaster_df[disaster_df['year'] == selected_year].copy()
year_data['affected_population'] = pd.to_numeric(year_data['affected_population'], errors='coerce').fillna(0)

# Scale radius for scatter points
MAX_RADIUS = 50000
year_data['radius'] = year_data['affected_population'] / year_data['affected_population'].max() * MAX_RADIUS

# Neutral gray color for scatter points
year_data['scatter_color'] = [[200, 200, 200]] * len(year_data)

# --------------------
# Layers
# --------------------
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=year_data,
    get_position=["longitude", "latitude"],
    get_weight="affected_population",
    radiusPixels=50,
    intensity=1,
    threshold=0.03,
    opacity=0.3
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=year_data,
    get_position=["longitude", "latitude"],
    get_radius="radius",
    get_fill_color="scatter_color",
    pickable=True,           # enables hover
    auto_highlight=True,
    opacity=0.6
)

# --------------------
# View state
# --------------------
view_state = pdk.ViewState(
    longitude=0,
    latitude=0,
    zoom=1.5,
    pitch=0
)

# --------------------
# Deck
# --------------------
r = pdk.Deck(
    layers=[heatmap_layer, scatter_layer],
    initial_view_state=view_state,
    tooltip={
        "html": "<b>Event Type:</b> {event_type} <br/><b>Affected:</b> {affected_population}",
        "style": {"color": "white"}
    }
)

st.subheader(f"Disaster Heatmap - {selected_year}")
st.pydeck_chart(r)
st.markdown("---")

# --------------------
# Choropleth Map
# --------------------
country_data = year_data.groupby("country")["affected_population"].sum().reset_index()

fig = px.choropleth(
    country_data,
    locations="country",
    locationmode="country names",
    color="affected_population",
    color_continuous_scale="YlOrRd",
    hover_data={"affected_population": ":,f"}  # format with commas
)

# Dark / blended style
fig.update_layout(
    template="plotly_dark",
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        bgcolor='rgba(0,0,0,0)',  # transparent
        lakecolor='rgba(0,0,0,0)',
        fitbounds="locations"  # makes map fill figure
    ),
    margin={"r":10,"t":10,"l":10,"b":10},  # small margins
    height=500,  # increase height to fill more space
    coloraxis_colorbar=dict(
        title="Affected Population"
    )
)

# Resize colorbar
fig.update_coloraxes(
    colorbar=dict(
        thickness=20,   # thinner
        len=0.6,        # scale of colorbar relative to figure
        yanchor="middle",
        y=0.5
    )
)

fig.update_traces(
    hovertemplate='<b>%{location}</b><br>Affected Population: %{z:,.0f}<extra></extra>'
)

st.subheader(f"Total Affected Population by Country - {selected_year}")
st.plotly_chart(fig, use_container_width=True)
