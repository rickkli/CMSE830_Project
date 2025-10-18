import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="Disaster Type Counts",
    layout="wide"
)

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    disaster_df = pd.read_csv('datasets/disaster_df.csv')
    indicators_df_imputed = pd.read_csv('datasets/indicators_df_imputed.csv')
    return disaster_df, indicators_df_imputed

disaster_df, indicators_df_imputed = load_data()

# ------------------------------
# PAGE TITLE & INTRO
# ------------------------------
st.header("Disaster Type Counts and Impacts")

st.markdown("""
### Overview  
Explore how different **types of natural disasters** have occurred globally between 2020 and 2025.  
Use the dropdown below to select a specific year to view:
- The **number of recorded disasters** by type  
- The **total affected population**  
- The **total number of casualties**

The visualization below helps identify which types of disasters are most frequent or have the greatest human impact in a given year.
""")

st.divider()

# ------------------------------
# YEAR SELECTION + DATA PREP
# ------------------------------
st.markdown("Select a year below to view disaster counts and impacts.")

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
            #marker_color=dict(color="#636EFA"),
            hovertext=hover_text,
            hoverinfo='text'
        )
    ]
)

fig.update_layout(
    title=dict(
        text=f"Disaster Type Counts",
        xanchor="left",
        font=dict(size=20)
    ),
    xaxis_title="Disaster Type",
    yaxis_title="Count",
    showlegend=False,
    template="plotly_white",
    margin=dict(l=40, r=40, t=60, b=60)
)

# Display chart
st.plotly_chart(fig, use_container_width=True)

st.caption("""
Each bar represents the total number of recorded disaster events for the selected year.  
Hover over a bar to see the **affected population** and **casualty totals** for that disaster type.
""")