import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import plotly.express as px

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
   page_title="Exploratory Data Analysis",
   layout="wide"
)

# ------------------------------
# MAIN TITLE
# ------------------------------
st.title("Exploratory Data Analysis")

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
   disaster_df = pd.read_csv('datasets/disaster_df.csv')
   indicators_df_imputed = pd.read_csv('datasets/indicators_df_imputed.csv')
   demographic_df_imputed = pd.read_csv('datasets/demographic_df_imputed.csv')
   return disaster_df, indicators_df_imputed, demographic_df_imputed

disaster_df, indicators_df_imputed, demographic_df_imputed = load_data()

# ------------------------------
# TAB SETUP
# ------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Event Type Counts",
    "Economic Indicator Trends",
    "Impact Choropleth Map",
    "Disaster Heatmap",
    "Statistical Summaries"
])

# ------------------------------
# TAB 1
# ------------------------------
with tab1:
    st.subheader("Disaster Event Type Counts and Impacts")

    st.markdown("""
    This section provides an overview of global natural disasters, highlighting how different disaster types vary in frequency and human impact.

    Use the selector below to explore:
    - How often each disaster type occurred.
    - Total affected population.
    - Total casualties.

    This view helps identify which disaster types were most disruptive in a given year.
    """)

    st.divider()

    # ------------------------------
    # YEAR SELECTION + DATA PREP
    # ------------------------------

    # Group and aggregate totals per year + event_type
    disaster_counts = (
        disaster_df.groupby(['year', 'event_type'])
        .agg(
            count=('event_type', 'size'),
            affected_population=('affected_population', 'sum'),
            total_casualties=('total_casualties', 'sum')
        )
        .reset_index()
    )

    # Dropdown for year selection
    years = sorted(disaster_counts['year'].unique())
    selected_year = st.selectbox("Select a Year", years, index=len(years)-1)

    # Filter data for selected year
    subset = disaster_counts[disaster_counts['year'] == selected_year]

    # Sort filtered data by count descending
    subset = subset.sort_values(by='count', ascending=False)

    st.write("")

    # ------------------------------
    # VISUALIZATION
    # ------------------------------
    # Hover text
    hover_text = [
        f"Count: {count:,}<br>"
        f"Affected Population: {affected_pop:,}<br>"
        f"Casualties: {casualties:,}"
        for count, affected_pop, casualties in zip(
            subset['count'],
            subset['affected_population'],
            subset['total_casualties']
        )
    ]

    # Bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=subset['event_type'],
                y=subset['count'],
                hovertext=hover_text,
                hoverinfo='text'
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"", #Disaster Type Counts
            xanchor="left",
            font=dict(size=20)
        ),
        xaxis_title="Disaster Event Type",
        yaxis_title="Count",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=60)
    )

    # Display chart
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    Hover over each bar to view additional information including affected population and casualty totals for the selected year.
    """)

# ------------------------------
# TAB 2
# ------------------------------
with tab2:
    st.subheader("Country Economic Indicator Trends")
    st.markdown(
    """
    Explore how key economic indicators have evolved across countries over time.

    Select a country and an indicator from the dropdowns below to visualize trends.
    """)

    st.divider()

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
            hovertemplate="%{y:.2f}<extra></extra>",
            #marker=dict(color="#636EFA"),
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"", #{selected_indicator} for {selected_country}
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

    st.info("""
    Hover over a datapoint to see the indicator value for that year.
    """)

# ------------------------------
# TAB 3
# ------------------------------
with tab3:
    st.subheader("Impact Choropleth Map")

    st.markdown("""
    This map visualizes the total affected population by country for the selected year.
    It highlights where disasters had the greatest human impact, helping identify regional
    vulnerability and geographic clustering of high-impact events.
    """)

    st.divider()

    # ------------------------------
    # YEAR SELECTION
    # ------------------------------
    years = sorted(disaster_df["year"].unique())
    selected_year = st.selectbox("Select a Year", years, index=len(years) - 1, key="year_select_tab_3")

    year_data = disaster_df[disaster_df["year"] == selected_year].copy()
    year_data["affected_population"] = pd.to_numeric(
        year_data["affected_population"], errors="coerce"
    ).fillna(0)

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
            text=f"", #Total Affected Population by Country
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

    st.info("""
    Hover over a country to view its total affected population for the selected year.
    """)

# ------------------------------
# TAB 4
# ------------------------------
with tab4:
    st.subheader("Disaster Heatmap")
    st.markdown(
    """
    This map visualizes the geographic distribution and intensity of disaster events for the selected year.
    Using a heatmap layer, it highlights physical locations of natural disasters with higher disaster density and affected populations.
    """)

    st.divider()

    # ------------------------------
    # YEAR SELECTION
    # ------------------------------
    years = sorted(disaster_df["year"].unique())
    selected_year = st.selectbox("Select a Year", years, index=len(years) - 1, key="year_select_tab_4")

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
    #st.subheader(f"Disaster Event Heatmap")
    st.pydeck_chart(r, use_container_width=True)

    st.info("""
    Hover over points on the map to view the event type and affected population details.
    """)

# ------------------------------
# TAB 5
# ------------------------------
with tab5:
    st.subheader("Statistical Summaries of Datasets")
    st.markdown("""
    This section provides statistical summaries of all three datasets used in the analysis to establish an understanding of their structure, distributions, and variability.
                
    Descriptive statistics such as mean, median, standard deviation, and range are presented for key numerical variables in each dataset, which help characterize the data before advanced exploration
    """)

    st.divider()

    # --- Natural Disaster Dataset Summary ---
    #st.subheader("Natural Disasters Dataset")
    st.markdown("""
    **Natural Disasters Dataset**
    """)

    st.dataframe(disaster_df.describe().T, use_container_width=True)

    st.divider()

    # --- Economic Indicators Dataset Summary ---
    st.markdown("""
    **Economic Indicators Dataset**
    """)

    st.dataframe(indicators_df_imputed.describe().T, use_container_width=True)

    st.divider()

    # --- Demographics Dataset Summary ---
    st.markdown("""
    **Demographics Dataset**
    """)

    st.dataframe(demographic_df_imputed.describe().T, use_container_width=True)

    st.divider()

    st.markdown("""
    Together, these summaries provide a quantitative foundation for deeper analysis, helping to identify variables with wide ranges or potential outliers that may warrant further investigation in the visualization and modeling stages.
    """)
