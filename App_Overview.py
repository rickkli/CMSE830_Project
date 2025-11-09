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
# MAIN TITLE
# ------------------------------
st.title("Global Disaster and Economic Impact Explorer (2020 – 2025)")

# ------------------------------
# TAB SETUP
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "App Overview and Goals",
    "Initial Data Analysis",
    "Exploratory Data Analysis",
    "Datasets Overview"
])

# ------------------------------
# TAB 1 – APP OVERVIEW AND GOALS
# ------------------------------
with tab1:
    st.header("Overview")
    st.markdown("""
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

    st.header("Project Goals")
    st.markdown("""
    1. **Integrate** natural disaster and economic indicator datasets to create a unified, analyzable framework.  
    2. **Explore correlations** between disaster intensity/frequency and economic performance metrics.  
    3. **Visualize patterns and trends** through interactive charts and geographic visualizations.  
    4. **Identify vulnerable regions** that show strong economic declines following major disasters.  
    5. **Support future predictive modeling** for estimating economic losses or resilience levels based on disaster characteristics.
    """)

# ------------------------------
# TAB 2 – INITIAL DATA ANALYSIS
# ------------------------------
with tab2:
    # -------------------------------------
    # SECTION 1: Data Cleaning & Preprocessing
    # -------------------------------------

    # --- Natural Disasters Dataset ---
    st.markdown("### Natural Disasters Dataset Cleaning and Preprocessing")
    st.markdown("""
    The **natural disaster dataset** was cleaned and standardized to ensure consistency and usability.  
    The following preprocessing steps were applied:

    1. **Irrelevant Columns Removed**  
       - Dropped metadata columns such as `event_id` and other non-essential fields.

    2. **Date Standardization**  
       - Converted date strings into proper `datetime` objects.

    3. **Categorical Variable Preparation**  
       - Cleaned and standardized categorical variables like `event_type` and `country`.  
       - Prepared these variables for encoding in later analysis.

    4. **Numeric Data Consistency**  
       - Ensured all numeric variables (e.g., `economic_impact_million_usd`, `infrastructure_damage_million_score`) were correctly typed as numeric values.
    """)

    st.divider()

    # --- Economic Indicators Dataset ---
    st.markdown("### Economic Indicators Dataset Cleaning and Preprocessing")
    st.markdown("""
    The **economic indicators dataset** was also cleaned and standardized to align with the disaster dataset.  
    Key preprocessing steps included:

    1. **Column Pruning**  
       - Removed unnecessary columns such as redundant identifiers and metadata.

    2. **Monetary Value Standardization**  
       - Converted variables like `GDP` and `Gross National Income` to **millions of USD** for consistency.

    3. **Column Renaming**  
       - Renamed columns for improved readability and standardized units.

    4. **Data Type Correction**  
       - Ensured numeric, percentage, and string fields were properly typed.
    """)

    st.divider()

    # --- Dataset Alignment ---
    st.markdown("### Dataset Alignment and Country Harmonization")
    st.markdown("""
    To ensure consistency between the **disaster** and **economic indicator** datasets, country names and records were aligned through the following process:

    1. **Country Name Comparison**  
       - Identified countries present in one dataset but missing in the other.

    2. **Mapping Dictionary Creation**  
       - Built a mapping dictionary to align country names (e.g., `"UAE"` → `"United Arab Emirates"`).  
       - Applied this mapping to standardize naming conventions.

    3. **Filtering and Harmonization**  
       - Filtered datasets to retain only shared countries.  
       - Ensured both datasets cover a **common set of nations** for joint analysis.
    """)

    st.divider()

    # -------------------------------------
    # SECTION 2: Missingness Analysis
    # -------------------------------------
    st.subheader("Missingness Analysis")

    st.markdown("""
    Understanding missing data patterns is essential for making valid assumptions during imputation and modeling.  
    The following visualizations and analyses explore missingness across both datasets.
    """)

    st.markdown("### Visualizing Missing Data Patterns")

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/missing_values_heatmap.png", caption="Heatmap of Missing Values Across Indicators", use_container_width=True)
    with col2:
        st.image("images/missingness_correlation.png", caption="Correlation Between Missingness Flags and Observed Values", use_container_width=True)

    st.markdown("### Missingness Insights and Interpretation")
    st.markdown("""
    From the visualizations, several important patterns emerged:

    **1. Temporal and Variable-Level Patterns**  
    - **Horizontal bands** in the heatmap show countries with missing values across multiple indicators—especially in **2024–2025**, likely due to incomplete reporting.  
    - **Vertical bands** reveal indicators missing across many countries, reflecting potential reporting bias.

    **2. Correlation Between Missingness and Observed Values**  
    - Missingness flags (binary variables marking NaNs) were correlated with observed values.  
    - Several moderate correlations suggest **informative missingness**, where missing values depend on economic conditions.

    **3. Missingness Mechanism**  
    - The observed structure supports a **Missing At Random (MAR)** mechanism:  
      Missingness depends on observed variables (e.g., reporting completeness) rather than the unobserved values.  
    - This justifies using **statistical imputation methods** like regression or KNN without introducing major bias.

    **Key Takeaway:**  
    The MAR assumption supports imputation approaches that rely on observed data, ensuring statistical validity.
    """)

    st.divider()

    # -------------------------------------
    # SECTION 3: Imputation Strategy
    # -------------------------------------
    st.subheader("Imputation Strategy")

    st.markdown("""
    Based on the MAR assumption, we applied a **two-stage imputation process** combining temporal regression and similarity-based methods.
    """)

    st.markdown("#### 1. Stochastic Regression Imputation")
    st.markdown("""
    - Modeled each indicator as a function of **time** within each country using linear regression.  
    - Predicted missing values based on regression trends and added random noise to preserve natural variability.  
    - Skipped country-indicator pairs with insufficient data (fewer than two valid points), deferring them to the next stage.
    """)

    st.markdown("#### 2. K-Nearest Neighbors (KNN) Imputation")
    st.markdown("""
    - Applied **KNN imputation** to fill remaining gaps.  
    - Used economic similarity between countries within the same year to infer missing values.  
    - Combined with regression, this hybrid approach leverages both **temporal structure** (within-country trends) and **cross-sectional relationships** (between-country similarities).
    """)

    st.divider()

    # -------------------------------------
    # SECTION 4: Post-Imputation Evaluation
    # -------------------------------------
    st.subheader("Post-Imputation Evaluation")

    st.markdown("""
    To evaluate imputation quality, we compared the **correlation structure** of indicators **before and after** imputation.
    """)

    st.image('images/correlation_structure_comparison.png', 
         caption='Comparison of Correlation Structures Before and After Imputation', 
         use_container_width=True)

    st.markdown("""
    The two heatmaps displayed nearly identical correlation patterns, suggesting the imputation preserved multivariate relationships.  
    Minor variations reflect the stochastic component and interpolation effects but no structural distortion was observed.

    **Conclusion:**  
    The hybrid imputation strategy (**stochastic regression + KNN**) successfully restored missing values while maintaining the dataset’s original statistical structure and avoiding artificial correlations.
    """)


# ------------------------------
# TAB 3 – EXPLORATORY DATA ANALYSIS
# ------------------------------
with tab3:
    st.markdown("""
    This section provides **statistical summaries** of both datasets used in the analysis — the **Natural Disasters dataset** and the **Economic Indicators dataset** — to establish an initial understanding of their structure, distributions, and variability.

    While interactive and graphical analyses are presented in other dedicated sidebar tabs, this section focuses on core **descriptive statistics** such as mean, median, standard deviation, and range, which help characterize the data before advanced exploration.
    """)

    st.divider()

    # --- Natural Disaster Dataset Summary ---
    st.subheader("Natural Disasters Dataset – Statistical Summary")
    st.markdown("""
    The summary below presents descriptive statistics for key numerical variables in the disaster dataset, 
    including counts, central tendency (mean/median), dispersion (standard deviation), and range.
    """)

    st.dataframe(disaster_df.describe().T, use_container_width=True)

    st.divider()

    # --- Economic Indicators Dataset Summary ---
    st.subheader("Economic Indicators Dataset – Statistical Summary")
    st.markdown("""
    The summary below provides key descriptive statistics for the imputed economic indicators dataset.  
    These values capture the general economic scale, variability, and relative differences across countries and years.
    """)

    st.dataframe(indicators_df_imputed.describe().T, use_container_width=True)

    st.divider()

    st.markdown("""
    Together, these summaries provide a quantitative foundation for deeper analysis, helping to identify variables with wide ranges or potential outliers that may warrant further investigation in the visualization and modeling stages.
    """)

    st.info("""
    **Note:** Use the sidebar tabs to explore interactive EDA visualizations.
    """)

# ------------------------------
# TAB 4 – DATASETS OVERVIEW
# ------------------------------
with tab4:
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