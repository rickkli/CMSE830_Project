import streamlit as st
import pandas as pd

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
   page_title="Global Disaster, Economic, and Demographic Explorer",
   layout="wide"
)

# ------------------------------
# MAIN TITLE
# ------------------------------
st.title("Global Disaster, Economic, and Demographic Explorer (2020–2025)")

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
tab1, tab2 = st.tabs([
   "Overview and Goals",
   "Datasets Overview"
])

# ------------------------------
# TAB 1 – APP OVERVIEW AND GOALS
# ------------------------------
with tab1:
   st.header("Overview")
   st.markdown("""
   Assessing the global impact of natural disasters requires an integrated view of environmental events, 
   economic systems, and population characteristics. Disasters influence countries differently depending on 
   their economic structure, demographic composition, and overall exposure to hazard-prone conditions.

   This application brings together three interconnected datasets covering the years 2020–2025:  
   global disaster event records, key socioeconomic indicators, and demographic profiles.  
   By combining these sources, the platform provides a comprehensive foundation for examining how disasters 
   interact with economic performance and population vulnerability at a global scale.

   The analytical framework of this project supports examination of questions such as:
   - How disaster frequency and severity correspond to fluctuations in economic output, employment, or income levels.
   - How demographic attributes, including population density, age distribution, and urbanization, influence 
   disaster exposure and recovery capacity.
   - Which countries or regions exhibit elevated vulnerability when economic and demographic risk factors overlap.
   - How patterns of resilience and long-term recovery vary across geographic regions and over time.

   This integrated approach offers a clearer understanding of the multifaceted impacts of natural disasters 
   and the social and economic contexts in which they occur.
   """)

   st.divider()

   st.header("Project Goals")
   st.markdown("""
   1. Develop a unified analytical framework that merges disaster, economic indicator, and demographic datasets.  
   2. Assess how demographic characteristics influence exposure, impact severity, and post-disaster economic outcomes.  
   3. Identify statistical relationships among disaster intensity, economic performance, and population vulnerability.  
   4. Provide interactive visual tools, including geographic maps and temporal trend analyses, to support exploratory research.  
   5. Detect high-risk regions by evaluating combined demographic and economic sensitivity to disaster events.  
   6. Enable machine learning tasks such as severity prediction and vulnerability clustering using the integrated dataset.  
   """)

# ------------------------------
# TAB 2 – DATASETS OVERVIEW
# ------------------------------
with tab2:
    
   col1, col2, col3 = st.columns(3, gap="large")

   # -----------------------------------
   # COLUMN 1 – DISASTERS
   # -----------------------------------
   with col1:
      st.subheader("Natural Disasters Dataset")
      st.markdown("""
      Description:
      This dataset contains information on natural disaster events worldwide from 2020 – 2025.  
      It includes attributes describing the type, location, severity, and human/economic impact of each event.

      Variables:
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

   # -----------------------------------
   # COLUMN 2 – ECONOMIC INDICATORS
   # -----------------------------------
   with col2:
      st.subheader("Economic Indicators Dataset")
      st.markdown("""
      Description:
      This dataset contains country-level economic indicators used to measure the economic health, performance, and resilience** of nations.  
      It provides annual data from 2010 to 2025, including key macroeconomic variables such as GDP, inflation, unemployment, and public debt.  

      Variables:
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


   # -----------------------------------
   # COLUMN 3 – DEMOGRAPHICS
   # -----------------------------------
   with col3:
      st.subheader("Demographics Dataset")
      st.markdown("""
      Description:  
      This dataset provides annual demographic information for each country, including population size, 
      growth dynamics, age structure, and mortality patterns.  
      These characteristics are essential for understanding population vulnerability, exposure to hazards, 
      and the capacity for long-term recovery.

      Variables:
      - `Name` — Country name  
      - `Year` — Reference year of demographic statistics  
      - `Total Population` — Total national population  
      - `Growth Rate` — Annual population growth rate  
      - `Population Density (per sq km)` — Number of individuals per square kilometer  
      - `Total Fertility Rate` — Average number of births per woman  
      - `Life Expectancy at Birth` — Average life expectancy at birth  
      - `Under-5 Mortality Rate` — Mortality rate for children under age five  
      - `Sex Ratio of the Population` — Ratio of males to females  
      - `Youth and Old Age (0–14 and 65+)` — Percentage of the population in dependent age groups  
      - `Youth (0–14)` — Population percentage aged 0–14  
      - `Old Age (65+)` — Population percentage aged 65 or older  
      - `Both Sexes` — Percentage of population across both sexes
      - `Male` — Percentage of the population that is male  
      - `Female` — Percentage of the population that is female  
      """)

      st.write("")
      st.dataframe(demographic_df_imputed.head())
