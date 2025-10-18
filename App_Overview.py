import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="Global Disaster and Economic Impact Explorer",
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
# TITLE & INTRO
# ------------------------------
st.title("Global Disaster and Economic Impact Explorer (2020 – 2025)")
st.markdown("""
### Overview  
Natural disasters can leave lasting impacts not only on communities and ecosystems, but also on the **economic stability and development** of countries.  
This project aims to **analyze and visualize the relationship between natural disasters and key economic indicators** from 2020 – 2025.  

By integrating data from global disaster records and socioeconomic indicators, we seek to uncover:  
- How disaster frequency and severity correlate with changes in GDP, income, and employment.  
- Whether certain countries/regions show higher economic vulnerability to natural disasters.  
- How recovery patterns differ across countries and time periods.

The ultimate goal is to build **data-driven insights** that help policymakers, analysts, and researchers understand the 
            **economic resilience** of nations in the face of climate-related challenges.
""")

st.divider()

# ------------------------------
# PROJECT GOAL / NEXT STEPS
# ------------------------------
st.header("Project Goals")
st.markdown("""
1. **Integrate** natural disaster and economic indicator datasets to create a unified, analyzable framework.  
2. **Explore correlations** between disaster intensity/frequency and economic performance metrics.  
3. **Visualize patterns and trends** through interactive charts and geographic visualizations.  
4. **Identify vulnerable regions** that show strong economic declines following major disasters.  
5. **Support future predictive modeling** for estimating economic losses or resilience levels based on disaster characteristics.
""")

st.info("""
💡 *Use the sidebar to navigate through sections for data exploration, visualization, and analysis.*
""")

st.divider()

# ------------------------------
# DATASET DESCRIPTIONS
# ------------------------------
st.header("Dataset Overview")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Natural Disasters Dataset")
    st.markdown("""
    **Description:**  
    This dataset contains information on **natural disaster events** worldwide from 2020 – 2025.  
    It includes attributes describing the **type, location, severity, and human/economic impact** of each event.

    **Variables:**
    - `date` — Date of occurrence  
    - `year` — Year of occurrence  
    - `month` — Month of occurrence  
    - `country` — Country where the event occurred  
    - `event_type` — Type of climate event  
    - `severity` — Severity of the event (1–10 scale)  
    - `duration_days` — Duration of the event in days  
    - `affected_population` — Number of people affected  
    - `deaths` — Number of deaths  
    - `injuries` — Number of injuries  
    - `economic_impact_million_usd` — Economic impact in millions of USD  
    - `infrastructure_damage_million_score` — Infrastructure damage (1–100 scale)  
    - `response_time_hours` — Emergency response time in hours  
    - `international_aid_million_usd` — International aid received in millions of USD  
    - `latitude` — Latitude of the event location  
    - `longitude` — Longitude of the event location  
    - `impact_per_capita` — Economic impact per affected person  
    - `aid_percentage` — Percentage of economic impact covered by aid  
    """)
    
    st.write("")
    st.dataframe(disaster_df.head())

with col2:
    st.subheader("Economic Indicators Dataset")
    st.markdown("""
    **Description:**  
    This dataset contains **country-level economic indicators** used to measure the **economic health, performance, and resilience** of nations.  
    It provides annual data from **2010 to 2025**, including key macroeconomic variables such as GDP, inflation, unemployment, and public debt.  
    Missing values have been imputed to ensure consistency across all countries and years.

    **Variables:**
    - `country_name` — Full name of the country
    - `year` — The year the data corresponds to (from 2010 to 2025)  
    - `Inflation (CPI %)` — Annual consumer price inflation  
    - `GDP (Million USD)` — Gross Domestic Product in current millions USD  
    - `GDP per Capita (Current USD)` — GDP divided by total population in current USD  
    - `Unemployment Rate (%)` — Percentage of labor force unemployed  
    - `Interest Rate (Real, %)` — Lending interest rate adjusted for inflation  
    - `Inflation (GDP Deflator, %)` — Inflation based on the GDP deflator  
    - `GDP Growth (% Annual)` — Year-over-year GDP growth rate
    - `Current Account Balance (% GDP)` — Net flow of current transactions as a percentage of GDP
    - `Government Expense (% of GDP)` — Total government expenditure as a share of GDP
    - `Government Revenue (% of GDP)` — Total government revenue as a share of GDP
    - `Tax Revenue (% of GDP)` — Share of GDP collected in taxes
    - `Gross National Income (Million USD)` — Total income received by residents of a country in millions USD
    - `Public Debt (% of GDP)` — Total government debt as a percentage of GDP  
    """)

    st.write("")
    st.dataframe(indicators_df_imputed.head())