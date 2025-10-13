import streamlit as st
import plotly.graph_objects as go

# load cleaned datasets
@st.cache_data
def load_data():
    disaster_df = pd.read_csv('datasets/disaster_df.csv')
    indicators_df_imputed = pd.read_csv('datasets/indicators_df_imputed.csv')
    return disaster_df, indicators_df_imputed

disaster_df, indicators_df_imputed = load_data()

# --- visualization code ---
st.header("Disaster Trends by Year")
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

# --- Streamlit dropdown for year selection ---
years = sorted(disaster_counts['year'].unique())
selected_year = st.selectbox("Select a Year", years, index=len(years)-1)

# Filter data for selected year
subset = disaster_counts[disaster_counts['year'] == selected_year]

# --- Create hover text for better readability ---
hover_text = [
    f"Count: {count:,} <br>"
    f"Affected Population: {affected_pop:,} <br>"
    f"Total Casualties: {casualties:,} <br>"
    for count, affected_pop, casualties in zip(
        subset['count'],
        subset['affected_population'],
        subset['total_casualties']
    )
]

# --- Create bar chart ---
fig = go.Figure(
    data=[
        go.Bar(
            x=subset['event_type'],
            y=subset['count'],
            marker_color='steelblue',
            hovertext=hover_text,
            hoverinfo='text'
        )
    ]
)

fig.update_layout(
    title=f"Disaster Type Counts - {selected_year}",
    xaxis_title="Disaster Type",
    yaxis_title="Count",
    showlegend=False
)

# --- Display chart ---
st.plotly_chart(fig, use_container_width=True)